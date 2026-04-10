"""Motion playback utilities for deployment (bundled from ProtoMotions).

:class:`MotionPlayer` loads **a single motion clip** and provides per-frame
state (joint positions, velocities, body rotations, etc.) at a fixed control
rate.  It accepts three input formats:

- A single ``.motion`` file (RobotState dict with ``fps``, ``dof_pos``, ...).
- A packaged ``.pt`` library (multi-motion file with ``length_starts``,
  ``gts``, ``grs``, ...) -- requires an explicit ``motion_index`` to select
  which clip to load.
- A pre-resampled cache previously written by :meth:`cache_to_file`.

Two runtime modes
-----------------

**Raw mode** (first run)
    Loads one of the raw formats above and resamples to the control rate
    using the same interpolation code as training (SLERP for quaternions,
    linear for positions).  Call :meth:`cache_to_file` afterwards to write
    a pre-resampled ``.pt`` so that future runs load faster.

**Cached mode** (subsequent runs)
    Loads a cache written by :meth:`cache_to_file`.  All queries become
    plain NumPy array indexing.

All interpolation code is bundled here -- no external dependencies beyond
PyTorch and NumPy.

Original source: https://github.com/NVlabs/ProtoMotions (deployment/motion_utils.py)
"""

from __future__ import annotations

import enum
import sys
from typing import Dict, List

import numpy as np

__all__ = ["MotionPlayer"]

# Keys present in every state/future-reference dict
_STATE_KEYS = ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel")


def _is_cache_file(data: dict) -> bool:
    """Return True if *data* looks like a pre-resampled cache (not a raw motion)."""
    return "control_dt" in data and "body_rot" in data


# ---------------------------------------------------------------------------
# Stub for torch.load unpickling
# ---------------------------------------------------------------------------
# ProtoMotions .motion files are pickled dicts that reference
# ``protomotions.simulator.base_simulator.simulator_state.StateConversion``
# (a simple Enum).  We register a stub so torch.load works without
# installing protomotions.


class _StateConversion(enum.Enum):
    """Stub for protomotions.simulator.base_simulator.simulator_state.StateConversion."""
    SIMULATOR = "simulator"
    COMMON = "common"


def _register_unpickle_stub():
    """Make torch.load find StateConversion without protomotions installed."""
    if "protomotions" not in sys.modules:
        # Create the module hierarchy as simple namespace objects
        import types
        for name in [
            "protomotions",
            "protomotions.simulator",
            "protomotions.simulator.base_simulator",
            "protomotions.simulator.base_simulator.simulator_state",
        ]:
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
        sys.modules[
            "protomotions.simulator.base_simulator.simulator_state"
        ].StateConversion = _StateConversion


# ---------------------------------------------------------------------------
# Bundled interpolation utilities (from protomotions)
# ---------------------------------------------------------------------------


def _slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions (PyTorch).

    Args:
        q0, q1: Quaternions [..., 4].
        t: Blend factor [..., 1] where 0=q0, 1=q1.

    Returns:
        Interpolated quaternions, same shape.
    """
    import torch

    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


def _interpolate_pos(pos0, pos1, blend):
    """Linear interpolation between position tensors."""
    if pos1.dim() == 2:
        blend = blend.unsqueeze(-1)
    elif pos1.dim() == 3:
        blend = blend.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"pos1 has {pos1.dim()} dimensions, expected 2 or 3")
    return (1.0 - blend) * pos0 + blend * pos1


def _interpolate_quat(rot0, rot1, blend):
    """SLERP between quaternion tensors."""
    if rot1.dim() == 2:
        blend = blend.unsqueeze(-1)
    elif rot1.dim() == 3:
        blend = blend.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"rot1 has {rot1.dim()} dimensions, expected 2 or 3")
    return _slerp(rot0, rot1, blend)


def _calc_frame_blend(time, length, num_frames, dt):
    """Calculate frame indices and blend factor for interpolation."""
    import torch

    phase = time / length
    phase = torch.clip(phase, 0.0, 1.0)

    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0 * dt) / dt

    return frame_idx0, frame_idx1, blend


# ---------------------------------------------------------------------------
# MotionPlayer
# ---------------------------------------------------------------------------


class MotionPlayer:
    """Lightweight player for a **single** motion clip at a fixed control rate.

    Accepts three input formats (auto-detected):

    1. **Single ``.motion`` file** -- a RobotState dict saved via
       ``torch.save`` with keys ``fps``, ``dof_pos``, ``rigid_body_pos``, etc.
    2. **Packaged ``.pt`` library** -- a multi-motion file with
       ``length_starts``, ``gts``, ``grs``, ... keys.  You **must** pass
       ``motion_index`` to select which clip to extract.
    3. **Pre-resampled cache** -- written by :meth:`cache_to_file`, containing
       NumPy arrays at the control rate.  Auto-detected by the presence of
       a ``control_dt`` key.

    Parameters
    ----------
    motion_file:
        Path to any of the three formats above.
    motion_index:
        Index of the clip to extract from a packaged ``.pt`` library.
        **Required** for packaged files, ignored for ``.motion`` and cache
        files.
    control_dt:
        Control period in seconds (default 0.02 s = 50 Hz).  Determines the
        resampling rate when loading raw motion data.  Ignored when loading
        from a cache file (the cache stores its own dt).
    """

    def __init__(
        self,
        motion_file: str,
        motion_index: int = 0,
        control_dt: float = 0.02,
    ):
        import torch

        _register_unpickle_stub()

        self._torch = torch
        motion_file = str(motion_file)
        data = torch.load(motion_file, map_location="cpu", weights_only=False)

        if _is_cache_file(data):
            self._load_cache(data)
        else:
            self._load_raw(data, motion_index, control_dt)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Total number of frames available at the control rate."""
        return self._num_frames

    @property
    def num_bodies(self) -> int:
        return self._body_rot.shape[1]

    @property
    def num_dofs(self) -> int:
        return self._dof_pos.shape[1]

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def get_state_at_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return the motion state at *frame_idx*.

        Frame index is clamped to ``[0, total_frames - 1]``.

        Returns
        -------
        dict with keys ``dof_pos``, ``dof_vel``, ``body_rot``, ``body_pos``,
        ``body_vel``, ``body_ang_vel``.  All arrays have no batch dimension.
        """
        idx = int(np.clip(frame_idx, 0, self._num_frames - 1))
        return {
            "dof_pos":      self._dof_pos[idx],
            "dof_vel":      self._dof_vel[idx],
            "body_rot":     self._body_rot[idx],
            "body_pos":     self._body_pos[idx],
            "body_vel":     self._body_vel[idx],
            "body_ang_vel": self._body_ang_vel[idx],
        }

    def get_future_references(
        self,
        frame_idx: int,
        step_indices: List[int],
    ) -> Dict[str, np.ndarray]:
        """Return stacked future motion states.

        Each entry in *step_indices* is a positive 1-indexed offset (e.g.
        ``step_indices=[1, 25]`` means ``frame_idx + 1`` and
        ``frame_idx + 25``).  Future frames beyond the last available frame
        are clamped to the last frame.

        Returns
        -------
        dict with keys ``dof_pos``, ``dof_vel``, ``body_rot``, ``body_pos``,
        ``body_vel``, ``body_ang_vel``.  Arrays have shape
        ``[len(step_indices), ...]``.
        """
        future_states = [
            self.get_state_at_frame(frame_idx + s) for s in step_indices
        ]
        return {
            key: np.stack([s[key] for s in future_states], axis=0)
            for key in _STATE_KEYS
        }

    def cache_to_file(self, output_path: str) -> None:
        """Write a pre-resampled cache file at the current control rate.

        The cache contains NumPy arrays (not tensors) so it can be loaded
        without any PyTorch computation on subsequent runs.

        Args:
            output_path: Destination path, e.g. ``walk.50fps.pt``.
        """
        import torch

        cache = {
            "dof_pos":      self._dof_pos,
            "dof_vel":      self._dof_vel,
            "body_rot":     self._body_rot,
            "body_pos":     self._body_pos,
            "body_vel":     self._body_vel,
            "body_ang_vel": self._body_ang_vel,
            "control_dt":   self._control_dt,
            "num_frames":   self._num_frames,
        }
        torch.save(cache, output_path)
        print(
            f"[MotionPlayer] Cached {self._num_frames} frames @ "
            f"{1.0 / self._control_dt:.0f} Hz -> {output_path}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self, data: dict) -> None:
        """Load from a pre-resampled cache dict."""
        self._dof_pos      = np.asarray(data["dof_pos"],      dtype=np.float32)
        self._dof_vel      = np.asarray(data["dof_vel"],      dtype=np.float32)
        self._body_rot     = np.asarray(data["body_rot"],     dtype=np.float32)
        self._body_pos     = np.asarray(data["body_pos"],     dtype=np.float32)
        self._body_vel     = np.asarray(data["body_vel"],     dtype=np.float32)
        self._body_ang_vel = np.asarray(data["body_ang_vel"], dtype=np.float32)
        self._control_dt   = float(data["control_dt"])
        self._num_frames   = int(data["num_frames"])
        self._cached = True
        print(
            f"[MotionPlayer] Loaded cache: {self._num_frames} frames "
            f"@ {1.0 / self._control_dt:.0f} Hz"
        )

    def _load_raw(self, data: dict, motion_index: int, control_dt: float) -> None:
        """Load from a raw ProtoMotions motion file and resample.

        All interpolation code is bundled -- no protomotions install needed.
        """
        import torch

        self._control_dt = control_dt
        self._cached = False

        if "length_starts" in data:
            # ---- Multi-motion packaged library (.pt) ----
            length_starts     = data["length_starts"]
            motion_num_frames = data["motion_num_frames"]
            motion_dt_all     = data["motion_dt"]

            start  = int(length_starts[motion_index].item())
            nf     = int(motion_num_frames[motion_index].item())
            end    = start + nf
            src_dt = float(motion_dt_all[motion_index].item())

            gts  = data["gts"][start:end]
            grs  = data["grs"][start:end]
            gvs  = data["gvs"][start:end]
            gavs = data["gavs"][start:end]
            dps  = data["dps"][start:end]
            dvs  = data["dvs"][start:end]
            motion_length = src_dt * (nf - 1)

        elif "rigid_body_pos" in data:
            # ---- Single-motion file (.motion / .npy via torch.load) ----
            fps    = float(data["fps"])
            src_dt = 1.0 / fps

            gts  = data["rigid_body_pos"]
            grs  = data["rigid_body_rot"]
            gvs  = data["rigid_body_vel"]
            gavs = data["rigid_body_ang_vel"]
            dps  = data["dof_pos"]
            dvs  = data["dof_vel"]
            nf   = gts.shape[0]
            motion_length = src_dt * (nf - 1)
        else:
            raise ValueError(
                "Unrecognised raw motion format.  Expected either:\n"
                "  - packaged library: keys 'length_starts', 'gts', 'grs', ...\n"
                "  - single-motion:   keys 'rigid_body_pos', 'fps', 'dof_pos', ..."
            )

        # ---- resample to control rate via training-identical interpolation ----
        num_ctrl_frames = max(1, int(round(motion_length / control_dt)) + 1)
        ctrl_times = torch.linspace(0.0, motion_length, num_ctrl_frames)

        motion_len_t  = torch.tensor([motion_length])
        num_frames_t  = torch.tensor([nf])
        motion_dt_t   = torch.tensor([src_dt])

        f0_list, f1_list, blend_list = [], [], []
        for t in ctrl_times:
            t_t = t.unsqueeze(0)
            f0, f1, bl = _calc_frame_blend(t_t, motion_len_t, num_frames_t, motion_dt_t)
            f0_list.append(f0)
            f1_list.append(f1)
            blend_list.append(bl)

        f0    = torch.cat(f0_list)
        f1    = torch.cat(f1_list)
        blend = torch.cat(blend_list)

        def _interp_pos(src):
            s0 = src[f0]
            s1 = src[f1]
            return _interpolate_pos(s0, s1, blend)

        def _interp_quat(src):
            s0 = src[f0]
            s1 = src[f1]
            return _interpolate_quat(s0, s1, blend)

        body_pos     = _interp_pos(gts)
        body_rot     = _interp_quat(grs)
        body_vel     = _interp_pos(gvs)
        body_ang_vel = _interp_pos(gavs)
        dof_pos      = _interp_pos(dps)
        dof_vel      = _interp_pos(dvs)

        self._dof_pos      = dof_pos.numpy().astype(np.float32)
        self._dof_vel      = dof_vel.numpy().astype(np.float32)
        self._body_rot     = body_rot.numpy().astype(np.float32)
        self._body_pos     = body_pos.numpy().astype(np.float32)
        self._body_vel     = body_vel.numpy().astype(np.float32)
        self._body_ang_vel = body_ang_vel.numpy().astype(np.float32)
        self._num_frames   = num_ctrl_frames

        print(
            f"[MotionPlayer] Loaded raw motion #{motion_index}: "
            f"{nf} source frames @ {1.0 / src_dt:.1f} Hz -> "
            f"{num_ctrl_frames} resampled frames @ {1.0 / control_dt:.0f} Hz"
        )
