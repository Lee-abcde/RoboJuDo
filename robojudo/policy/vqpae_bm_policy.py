"""ProtoMotions VQ-PAE BM policy for RoboJuDo."""

from __future__ import annotations

import importlib
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

logger = logging.getLogger(__name__)


def _find_protomotions_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2].parent
    candidates = [
        repo_root / "ProtoMotions",
        repo_root / "protomotions",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Missing ProtoMotions source repository. Expected a sibling directory named "
        f"'ProtoMotions' or 'protomotions' next to this repo under {repo_root}."
    )


_PROTO_ROOT = str(_find_protomotions_root())
if _PROTO_ROOT not in sys.path:
    sys.path.insert(0, _PROTO_ROOT)


try:
    importlib.import_module("deployment")
except ModuleNotFoundError:
    logger.error(
        "Missing ProtoMotions deployment module under %s. Cannot import 'deployment'.",
        _PROTO_ROOT,
    )
    raise RuntimeError("Cannot import 'deployment' from ProtoMotions repo!") from None

from deployment.motion_utils import MotionPlayer  # noqa: E402
from deployment.state_utils import apply_heading_offset_np, compute_yaw_offset_np  # noqa: E402


def _quat_rotate_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32)
    vec = np.asarray(v, dtype=np.float32)
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_vec, vec)
    return vec + q_w * t + np.cross(q_vec, t)


@policy_registry.register
class VQPAEBMPolicy(Policy):
    """Policy that runs a ProtoMotions VQ-PAE unified ONNX pipeline."""

    cfg_policy: PolicyCfg

    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        onnx_path = cfg_policy.policy_file
        yaml_path = onnx_path.replace(".onnx", ".yaml")

        with open(yaml_path) as f:
            self._meta = yaml.safe_load(f)

        robot_meta = self._meta["robot"]
        control_meta = self._meta["control"]
        motion_meta = self._meta["motion"]
        runtime = self._meta["_runtime"]

        joint_names = robot_meta["joint_names"]
        num_dofs = robot_meta["num_dofs"]
        stiffness = control_meta["stiffness"]
        damping = control_meta["damping"]
        effort_limits = control_meta.get("effort_limits")

        dof_cfg = DoFConfig(
            joint_names=joint_names,
            default_pos=[0.0] * num_dofs,
            stiffness=stiffness,
            damping=damping,
            torque_limits=effort_limits,
        )
        cfg_policy_updated = cfg_policy.model_copy()
        cfg_policy_updated.obs_dof = dof_cfg
        cfg_policy_updated.action_dof = dof_cfg

        super().__init__(cfg_policy=cfg_policy_updated, device="cpu")

        logger.info("[VQPAEBMPolicy] Loading ONNX: %s", onnx_path)
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._onnx_in_names = [inp.name for inp in self._session.get_inputs()]
        self._onnx_out_names = [out.name for out in self._session.get_outputs()]
        self._onnx_name_to_key = runtime["onnx_name_to_in_key"]
        self._out_name_to_idx = {
            out.name: idx for idx, out in enumerate(self._session.get_outputs())
        }

        motion_path = getattr(cfg_policy, "motion_path", None)
        if not motion_path:
            raise ValueError("VQPAEBMPolicyCfg must set motion_path")
        motion_index = getattr(cfg_policy, "motion_index", 0)
        timing = self._meta["timing"]
        self._player = MotionPlayer(
            motion_path, motion_index=motion_index, control_dt=timing["control_dt"]
        )

        self._anchor_idx = robot_meta["anchor_body_index"]
        self._root_idx = robot_meta["root_body_index"]
        self._anchor_body_name = robot_meta.get("anchor_body_name")
        self._future_step_indices = motion_meta["future_step_indices"]
        self._history_steps = self._infer_history_steps(self._meta)

        self._pd_target_max_accel = control_meta.get("pd_target_max_accel")
        self._action_ema_alpha = control_meta.get("action_ema_alpha", 1.0)
        self._joint_targets_out_idx = self._out_name_to_idx.get(
            "joint_pos_targets", 1
        )

        logger.info(
            "[VQPAEBMPolicy] %s DOFs, %s motion frames, history_steps=%s, "
            "anchor=%s(idx=%s), root_idx=%s",
            num_dofs,
            self._player.total_frames,
            self._history_steps,
            self._anchor_body_name or "pelvis",
            self._anchor_idx,
            self._root_idx,
        )

        self._heading_offset = None
        self.reset()

    @staticmethod
    def _infer_history_steps(meta: dict) -> int:
        history_steps = 1
        for policy_input in meta.get("policy_inputs", []):
            key = policy_input.get("key")
            shape = policy_input.get("shape")
            if key and key.startswith("historical.") and shape and len(shape) >= 3:
                history_steps = max(history_steps, int(shape[1]))
        return history_steps

    def reset(self):
        self._frame = 0
        self._prev_pd = None
        self._prev_prev_pd = None
        self._ema_prev = None
        self._stashed_pd_targets = np.zeros(self.num_actions, dtype=np.float32)
        self._prev_actions = np.zeros(self.num_actions, dtype=np.float32)
        self._history_dof_pos = None
        self._history_dof_vel = None
        self._history_root_local_ang_vel = None
        self._history_processed_actions = None
        self._motion_done = False
        self._paused = False
        self._last_semantic_inputs: dict[str, np.ndarray] = {}
        self._last_onnx_inputs: dict[str, np.ndarray] = {}
        self._last_onnx_outputs: dict[str, np.ndarray] = {}

    def reset_alignment(self):
        self._heading_offset = None

    def post_step_callback(self, commands=None):
        if not self._paused:
            self._frame += 1
            if self._frame >= self._player.total_frames:
                self._frame = self._player.total_frames - 1
                self._motion_done = True
        for cmd in commands or []:
            if cmd in ("[MOTION_RESET]", "[MOTION_FADE_IN]"):
                self.reset()

    def _init_history_buffers(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        root_local_ang_vel: np.ndarray,
    ) -> None:
        self._history_dof_pos = deque(maxlen=self._history_steps)
        self._history_dof_vel = deque(maxlen=self._history_steps)
        self._history_root_local_ang_vel = deque(maxlen=self._history_steps)
        self._history_processed_actions = deque(maxlen=self._history_steps)

        zeros_actions = np.zeros(self.num_actions, dtype=np.float32)
        for _ in range(self._history_steps):
            self._history_dof_pos.append(dof_pos.copy())
            self._history_dof_vel.append(dof_vel.copy())
            self._history_root_local_ang_vel.append(root_local_ang_vel.copy())
            self._history_processed_actions.append(zeros_actions.copy())

    def _export_history(self) -> dict[str, np.ndarray]:
        assert self._history_dof_pos is not None
        assert self._history_dof_vel is not None
        assert self._history_root_local_ang_vel is not None
        assert self._history_processed_actions is not None
        return {
            "historical.dof_pos": np.stack(list(self._history_dof_pos), axis=0)[None],
            "historical.dof_vel": np.stack(list(self._history_dof_vel), axis=0)[None],
            "historical.root_local_ang_vel": np.stack(
                list(self._history_root_local_ang_vel), axis=0
            )[None],
            "historical.processed_actions": np.stack(
                list(self._history_processed_actions), axis=0
            )[None],
        }

    def _append_history(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        root_local_ang_vel: np.ndarray,
        processed_actions: np.ndarray,
    ) -> None:
        assert self._history_dof_pos is not None
        assert self._history_dof_vel is not None
        assert self._history_root_local_ang_vel is not None
        assert self._history_processed_actions is not None
        self._history_dof_pos.append(dof_pos.copy())
        self._history_dof_vel.append(dof_vel.copy())
        self._history_root_local_ang_vel.append(root_local_ang_vel.copy())
        self._history_processed_actions.append(processed_actions.copy())

    def get_observation(self, env_data, ctrl_data):
        if self._heading_offset is None:
            motion_anchor_rot = self._player.get_state_at_frame(0)["body_rot"][
                self._anchor_idx
            ]
            robot_anchor_rot = self._get_anchor_quat(env_data)
            self._heading_offset = compute_yaw_offset_np(
                robot_anchor_rot, motion_anchor_rot
            )

        anchor_rot = self._get_anchor_quat(env_data)
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        root_local_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)

        if self._history_dof_pos is None:
            self._init_history_buffers(dof_pos, dof_vel, root_local_ang_vel)

        future_refs = self._player.get_future_references(
            self._frame, self._future_step_indices
        )
        future_body_rot = apply_heading_offset_np(
            self._heading_offset, future_refs["body_rot"]
        )
        future_body_ang_vel = _quat_rotate_np(
            np.broadcast_to(
                self._heading_offset,
                future_refs["body_ang_vel"].shape[:-1] + (4,),
            ),
            future_refs["body_ang_vel"],
        ).astype(np.float32)

        key_to_array = {
            "current.anchor_rot": anchor_rot[None],
            "current.dof_pos": dof_pos[None],
            "current.dof_vel": dof_vel[None],
            "current.root_local_ang_vel": root_local_ang_vel[None],
            "mimic.future_anchor_rot": future_body_rot[:, self._anchor_idx, :][None],
            "mimic.future_anchor_ang_vel": future_body_ang_vel[
                :, self._anchor_idx, :
            ][None],
            "mimic.future_dof_pos": future_refs["dof_pos"][None],
            "mimic.future_dof_vel": future_refs["dof_vel"][None],
            **self._export_history(),
        }
        self._last_semantic_inputs = {
            key: value.copy() for key, value in key_to_array.items()
        }

        onnx_inputs = {}
        for onnx_name in self._onnx_in_names:
            sem_key = self._onnx_name_to_key.get(onnx_name)
            if sem_key and sem_key in key_to_array:
                onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)
        self._last_onnx_inputs = {
            key: value.copy() for key, value in onnx_inputs.items()
        }

        ort_out = self._session.run(self._onnx_out_names, onnx_inputs)
        self._last_onnx_outputs = {
            name: np.asarray(value).copy()
            for name, value in zip(self._onnx_out_names, ort_out)
        }
        pd_targets = ort_out[self._joint_targets_out_idx].squeeze().copy()

        if (
            self._pd_target_max_accel is not None
            and self._prev_pd is not None
            and self._prev_prev_pd is not None
        ):
            delta = pd_targets - self._prev_pd
            prev_delta = self._prev_pd - self._prev_prev_pd
            accel = delta - prev_delta
            pd_targets = self._prev_pd + prev_delta + np.clip(
                accel, -self._pd_target_max_accel, self._pd_target_max_accel
            )
        self._prev_prev_pd = self._prev_pd
        self._prev_pd = pd_targets.copy()

        alpha = self._action_ema_alpha
        if alpha < 1.0:
            if self._ema_prev is None:
                self._ema_prev = pd_targets.copy()
            pd_targets = alpha * pd_targets + (1.0 - alpha) * self._ema_prev
            self._ema_prev = pd_targets.copy()

        self._append_history(dof_pos, dof_vel, root_local_ang_vel, pd_targets)
        self._stashed_pd_targets = pd_targets
        self._prev_actions = pd_targets.copy()

        extras = {
            "CALLBACK": ["[MOTION_DONE]"] if self._motion_done else [],
        }
        dummy_obs = np.zeros(1, dtype=np.float32)
        return dummy_obs, extras

    def _get_anchor_quat(self, env_data) -> np.ndarray:
        name = self._anchor_body_name
        if name is not None and name not in (None, "pelvis"):
            fk = env_data.fk_info
            if fk is not None and name in fk:
                return np.asarray(fk[name]["quat"], dtype=np.float32)
            if name == "torso_link" and env_data.torso_quat is not None:
                return np.asarray(env_data.torso_quat, dtype=np.float32)
        return np.asarray(env_data.base_quat, dtype=np.float32)

    def get_action(self, obs):
        return self._stashed_pd_targets

    def get_init_dof_pos(self):
        return self._player.get_state_at_frame(0)["dof_pos"].copy()

    def get_init_motion_state(self, zero_velocity: bool = True) -> dict[str, np.ndarray]:
        frame0 = self._player.get_state_at_frame(0)
        root_lin_vel = np.zeros(3, dtype=np.float32)
        root_ang_vel = np.zeros(3, dtype=np.float32)
        dof_vel = np.zeros(self.num_dofs, dtype=np.float32)
        if not zero_velocity:
            root_lin_vel = frame0["body_vel"][self._root_idx].copy()
            root_ang_vel = frame0["body_ang_vel"][self._root_idx].copy()
            dof_vel = frame0["dof_vel"].copy()
        return {
            "root_pos": frame0["body_pos"][self._root_idx].copy(),
            "root_quat": frame0["body_rot"][self._root_idx].copy(),
            "dof_pos": frame0["dof_pos"].copy(),
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel": dof_vel,
        }

    @property
    def last_semantic_inputs(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._last_semantic_inputs.items()}

    @property
    def last_onnx_inputs(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._last_onnx_inputs.items()}

    @property
    def last_onnx_outputs(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._last_onnx_outputs.items()}
