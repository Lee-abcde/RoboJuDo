"""Configuration for ProtoMotions VQ-PAE BM policy."""

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class VQPAEBMPolicyCfg(PolicyCfg):
    """Config for :class:`VQPAEBMPolicy`."""

    policy_type: str = "VQPAEBMPolicy"
    robot: str = "g1"
    disable_autoload: bool = True

    onnx_name: str = "unified_pipeline"
    onnx_path: str | None = None
    motion_path: str = ""
    motion_index: int = 0

    @property
    def policy_file(self) -> str:
        if self.onnx_path is not None:
            return self.onnx_path
        return (
            ASSETS_DIR / f"models/{self.robot}/vqpae_bm/{self.onnx_name}.onnx"
        ).as_posix()

    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    obs_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
    action_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
