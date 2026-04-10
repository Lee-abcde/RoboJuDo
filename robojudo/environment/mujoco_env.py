import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()

        # Virtual gantry: spring-damper harness.
        # Displacement is computed from the ROOT body (pelvis / free joint).
        # Force is applied at the attachment bodies (shoulders preferred).
        # A rest length provides slack so the spring only engages on large drops.
        self._gantry_enabled = False
        self._gantry_anchor = None  # target position (from root body) [3]
        self._gantry_stiffness = 2000.0  # N/m
        self._gantry_damping = 400.0  # Ns/m
        self._gantry_rest_length = 0.05  # meters of slack before spring engages

        # Root body (pelvis) — used for displacement computation.
        self._gantry_root_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # Attachment bodies — where force is applied.
        _gantry_body_sets = [
            ["left_shoulder_pitch_link", "right_shoulder_pitch_link"],
            ["torso_link"],
            ["pelvis"],
        ]
        self._gantry_body_ids = []
        self._gantry_body_names = []
        for body_set in _gantry_body_sets:
            bids = []
            for name in body_set:
                bid = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                if bid >= 0:
                    bids.append(bid)
            if len(bids) == len(body_set):
                self._gantry_body_ids = bids
                self._gantry_body_names = body_set
                break
        if not self._gantry_body_ids:
            logger.warning("Virtual gantry: no suitable bodies found, gantry disabled")

        self.update()  # get initial state

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def enable_gantry(self):
        """Attach gantry at the root body's current position."""
        if not self._gantry_body_ids:
            logger.warning("No gantry bodies — skipping enable")
            return
        # Anchor is the root body (pelvis) position — displacement computed from here.
        self._gantry_anchor = self.data.xpos[self._gantry_root_id].copy()
        self._gantry_enabled = True
        logger.info(
            f"Virtual gantry enabled: root=pelvis, apply={self._gantry_body_names}, "
            f"anchor={self._gantry_anchor.round(3).tolist()}, "
            f"k={self._gantry_stiffness}, c={self._gantry_damping}, "
            f"rest_len={self._gantry_rest_length}"
        )

    def disable_gantry(self):
        """Release all gantry bodies to free dynamics."""
        self._gantry_enabled = False
        for bid in self._gantry_body_ids:
            self.data.xfrc_applied[bid, :] = 0.0
        logger.info("Virtual gantry disabled")

    def _apply_gantry_force(self):
        """Compute spring-damper force from ROOT displacement, apply at attachment bodies.

        Like holosoma: displacement and velocity are read from the root body (pelvis).
        The resulting force is split equally across the attachment bodies (shoulders).
        A rest length provides slack — the spring only engages beyond that distance.
        """
        # Compute displacement from root body to anchor.
        root_pos = self.data.xpos[self._gantry_root_id]
        root_vel = self.data.cvel[self._gantry_root_id, 3:]  # linear velocity

        dx = self._gantry_anchor - root_pos
        distance = np.linalg.norm(dx)

        if distance < 1e-8:
            force = -self._gantry_damping * root_vel
        else:
            direction = dx / distance
            v_radial = np.dot(root_vel, direction)
            # Spring only engages beyond rest length (slack).
            stretch = max(distance - self._gantry_rest_length, 0.0)
            force = (
                self._gantry_stiffness * stretch - self._gantry_damping * v_radial
            ) * direction

        # Split force equally across attachment bodies.
        n = len(self._gantry_body_ids)
        per_body_force = force / n
        for bid in self._gantry_body_ids:
            self.data.xfrc_applied[bid, :3] = per_body_force
            self.data.xfrc_applied[bid, 3:] = 0.0

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            if self._gantry_enabled:
                self._apply_gantry_force()

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
