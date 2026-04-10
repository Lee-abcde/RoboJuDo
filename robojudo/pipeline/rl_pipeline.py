import logging
import time

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class PolicyWrapper:
    """A wrapper for Policy to handle observation and action adaptation."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs):
        action = self.policy.get_action(obs)
        pd_target = action + self.policy.default_pos
        return self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)

    def get_init_dof_pos(self):
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name):
        """Fallback: delegate other func to the wrapped policy."""
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )

        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        self.reset()
        self.self_check()

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        # self.env.reborn(init_qpos=[0.2, 0.2, 0.8] + [ 0.707, 0, 0, 0.707]) # FOR SIM DEBUG
        self.policy.reset()
        self.ctrl_manager.reset()

        # Blend-out state: transitions policy → init pose (frame 0) at end of motion.
        self._blend_out_active = False
        self._blend_out_step = 0
        self._blend_out_duration = int(5.0 * self.freq)  # 5 seconds
        self._init_dof_pos = np.asarray(self.policy.get_init_dof_pos(), dtype=np.float32)
        self._pending_blend_in = False
        self._blend_in_completed = False
        self._user_fade_out = False  # True when fade-out was user-triggered (not auto)
        self._prepare_seconds = None  # set by prepare() for re-use on reset
        # Cache original gantry gains for ramping.
        if hasattr(self.env, "_gantry_stiffness"):
            self._gantry_orig_stiffness = self.env._gantry_stiffness
            self._gantry_orig_damping = self.env._gantry_damping

    def safety_check(self):
        if not self.do_safety_check:
            return
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            if hasattr(self.env, "reborn"):
                self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                self.policy.reset_alignment()
            else:
                self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                        self.policy.reset_alignment()
                case "[MOTION_RESET]" | "[MOTION_FADE_IN]":
                    logger.info(f"{command} — re-entering blend-in phase")
                    self._blend_out_active = False
                    self._blend_out_step = 0
                    self._user_fade_out = False
                    # Re-run phase 2 of prepare (blend default → policy at frame 0).
                    # Gantry stays active — _run_blend_in will fade it out.
                    # The policy reset happens inside post_step_callback below.
                    self._pending_blend_in = True
                case "[MOTION_FADE_OUT]":
                    if not self._blend_out_active:
                        logger.info("Fade out — blending to default pose")
                        self._blend_out_active = True
                        self._blend_out_step = 0
                        self._user_fade_out = True
                        # Pause frame advancement in the policy.
                        inner = getattr(self.policy, "policy", self.policy)
                        if hasattr(inner, "_paused"):
                            inner._paused = True
                        # Activate gantry at current position.
                        has_gantry = hasattr(self.env, "enable_gantry")
                        if has_gantry:
                            self.env.enable_gantry()
                            self.env._gantry_stiffness = 0.0
                            self.env._gantry_damping = 0.0

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)

        # -- Detect motion done, start blend-out --
        has_gantry = hasattr(self.env, "enable_gantry")
        callbacks = extras.get("CALLBACK", [])
        if "[MOTION_DONE]" in callbacks and not self._blend_out_active:
            logger.info("Motion done — blending out to default pose")
            self._blend_out_active = True
            self._blend_out_step = 0
            # Activate gantry at current position so it can ramp up.
            if has_gantry:
                self.env.enable_gantry()
                self.env._gantry_stiffness = 0.0
                self.env._gantry_damping = 0.0

        # -- Blend policy output → init pose (frame 0) + ramp gantry --
        if self._blend_out_active:
            alpha = min(self._blend_out_step / max(self._blend_out_duration, 1), 1.0)
            pd_target = (1 - alpha) * pd_target + alpha * self._init_dof_pos
            # Ramp gantry support from 0 → full
            if has_gantry and self.env._gantry_enabled:
                self.env._gantry_stiffness = self._gantry_orig_stiffness * alpha
                self.env._gantry_damping = self._gantry_orig_damping * alpha
            self._blend_out_step += 1

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

        # Handle pending blend-in (after MOTION_RESET / FADE_IN).
        if self._pending_blend_in:
            self._pending_blend_in = False
            self._run_blend_in()
            self._blend_in_completed = True

    def _run_blend_in(self):
        """Phase 2 of prepare: blend from init pose (frame 0) to policy output.

        If gantry is active (from blend-out), fade it out in sync.
        Policy runs but frame stays at 0 (no post_step_callback).
        """
        secs = self._prepare_seconds or 3.0
        blend_steps = int(secs * self.freq)
        has_gantry = hasattr(self.env, "enable_gantry")
        fading_gantry = has_gantry and self.env._gantry_enabled

        logger.warning(
            f"Blend-in: init DOF → policy ({blend_steps} steps, {secs:.1f}s)"
            + (" + gantry fade-out" if fading_gantry else "")
        )
        pbar = ProgressBar("Blend in", blend_steps)

        last_step_time = time.time()
        for t in range(blend_steps):
            alpha = t / max(blend_steps - 1, 1)

            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            policy_pd = self.policy.get_pd_target(obs)

            action = (1 - alpha) * self._init_dof_pos + alpha * policy_pd

            # Fade gantry out: full → 0
            if fading_gantry:
                self.env._gantry_stiffness = self._gantry_orig_stiffness * (1 - alpha)
                self.env._gantry_damping = self._gantry_orig_damping * (1 - alpha)

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Release gantry, restore original gains.
        if fading_gantry:
            self.env._gantry_stiffness = self._gantry_orig_stiffness
            self.env._gantry_damping = self._gantry_orig_damping
            self.env.disable_gantry()

        logger.warning("Blend-in done — motion starting")

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        self._prepare_seconds = prepare_seconds

        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        else:
            desired_motor_angle = self.policy.get_init_dof_pos()

        has_gantry = hasattr(self.env, "enable_gantry")

        # Convert seconds to steps (at policy frequency).
        # Default: 3s ramp + 5s blend.  CLI --prepare-seconds overrides both.
        if prepare_seconds is not None:
            ramp_steps = int(prepare_seconds * self.freq)
            blend_steps = int(prepare_seconds * self.freq)
        else:
            ramp_steps = int(3.0 * self.freq)
            blend_steps = int(5.0 * self.freq)

        # ── Phase 1: Gantry holds robot, ramp joints to init pose ──
        logger.warning(
            f"prepare: phase 1 — ramp joints ({ramp_steps} steps, "
            f"{ramp_steps / self.freq:.1f}s, gantry support)"
        )
        pbar = ProgressBar("Prepare: ramp joints", ramp_steps)

        if has_gantry:
            self.env.enable_gantry()

        last_step_time = time.time()
        for t in range(ramp_steps):
            current_motor_angle = np.array(self.env.dof_pos)
            alpha = min(t / max(ramp_steps - 1, 1), 1.0)
            action = (1 - alpha) * current_motor_angle + alpha * desired_motor_angle

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Reset policy for a clean start — frame goes back to 0.
        self.reset()

        # ── Phase 2: Blend in policy + lower gantry ──
        # Policy runs but frame stays at 0 (no post_step_callback).
        # Actions blend from init DOF to policy output.
        # Gantry support fades out linearly.
        logger.warning(
            f"prepare: phase 2 — blend policy ({blend_steps} steps, "
            f"{blend_steps / self.freq:.1f}s, gantry lowering)"
        )
        pbar = ProgressBar("Prepare: blend policy", blend_steps)

        if has_gantry:
            orig_stiffness = self.env._gantry_stiffness
            orig_damping = self.env._gantry_damping

        last_step_time = time.time()
        for t in range(blend_steps):
            alpha = t / max(blend_steps - 1, 1)

            # Run policy observation + action (frame stays at 0).
            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            policy_pd = self.policy.get_pd_target(obs)

            # Blend: init DOF → policy output
            action = (1 - alpha) * desired_motor_angle + alpha * policy_pd

            # Fade gantry support
            if has_gantry:
                self.env._gantry_stiffness = orig_stiffness * (1 - alpha)
                self.env._gantry_damping = orig_damping * (1 - alpha)

            self.env.step(action)

            # Do NOT call post_step_callback — frame stays at 0.

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # ── Release ──
        if has_gantry:
            self.env._gantry_stiffness = orig_stiffness
            self.env._gantry_damping = orig_damping
            self.env.disable_gantry()

        logger.warning("prepare_done")


if __name__ == "__main__":
    pass
