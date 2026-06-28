"""Play a HIM velocity task with an ONNX policy.

Usage:
    python scripts/play_onnx.py Boying-HIM-Rough \\
        --onnx_file logs/rsl_rl/boying_him_velocity/2026-06-21_16-41-29/policy_1180.onnx

All other flags mirror scripts/play.py:
    --num_envs 1
    --viewer   auto | native | viser
    --video
    --no_terminations
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDict

# ── onnxruntime is optional; fall back to a clear error ──────────────────────
try:
    import onnxruntime as ort
except ImportError:
    sys.exit(
        "[ERROR] onnxruntime is not installed.\n"
        "  conda install -n unitree_rl_mjlab onnxruntime  (or pip install onnxruntime)"
    )

import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


# ─────────────────────────────────────────────────────────────────────────────
# ONNX policy wrapper
# ─────────────────────────────────────────────────────────────────────────────

class HIMOnnxPolicy:
    """Wraps an ONNX _HIMExportModel for use in the mjlab viewer loop.

    The viewer calls ``policy(obs: TensorDict)`` each step.
    The HIM ONNX model expects a single tensor:

        obs_input = concat([current_frame(S), history(H*S)])   shape [B, S+H*S]

    ``S`` and ``H*S`` are inferred from the ONNX model's input shape.
    The obs groups required are ``proprio_current`` and ``proprio_history``.
    """

    def __init__(self, onnx_path: str, device: str = "cpu") -> None:
        self.device = device

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Infer input size from ONNX model metadata.
        inp = self.session.get_inputs()[0]
        # inp.shape is [batch, S + H*S]
        self.input_size: int = inp.shape[1]

        # Read HIM dims from ONNX metadata if available (written by HIMVelocityOnPolicyRunner).
        meta = {p.key: p.value for p in self.session.get_modelmeta().custom_metadata_map.items()}
        if "him_num_one_step_obs" in meta and "him_history_length" in meta:
            self.num_one_step_obs: int = int(meta["him_num_one_step_obs"])
            self.history_length: int   = int(meta["him_history_length"])
        else:
            # Fall back: assume default H=6, infer S from input_size.
            self.history_length = 6
            self.num_one_step_obs = self.input_size // (self.history_length + 1)

        self.history_dim = self.num_one_step_obs * self.history_length
        assert self.num_one_step_obs + self.history_dim == self.input_size, (
            f"HIM input size mismatch: {self.num_one_step_obs} + {self.history_dim} "
            f"!= {self.input_size}"
        )

        print(
            f"[OnnxPolicy] Loaded {Path(onnx_path).name}  "
            f"| S={self.num_one_step_obs}  H={self.history_length}  "
            f"input={self.input_size}"
        )

    # ── PolicyProtocol ────────────────────────────────────────────────────────

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        """Build the ONNX input from obs groups and run inference."""
        current = obs["proprio_current"]   # [B, S]
        history = obs["proprio_history"]   # [B, H*S]

        obs_np = torch.cat([current, history], dim=-1).cpu().numpy().astype(np.float32)
        actions_np = self.session.run(None, {"obs": obs_np})[0]   # [B, 12]

        return torch.from_numpy(actions_np).to(self.device)

    def reset(self) -> None:
        """Called by the viewer on episode reset (stateless model → no-op)."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlayOnnxConfig:
    onnx_file: str
    """Path to the .onnx policy file."""

    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run_play_onnx(task_id: str, cfg: PlayOnnxConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    onnx_path = Path(cfg.onnx_file)
    if not onnx_path.exists():
        sys.exit(f"[ERROR] ONNX file not found: {onnx_path}")

    env_cfg = load_env_cfg(task_id, play=True)

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled")

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if cfg.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if cfg.video:
        log_dir = onnx_path.parent / "videos" / "play_onnx"
        env = VideoRecorder(
            env,
            video_folder=log_dir,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    policy = HIMOnnxPolicy(str(onnx_path), device=device)

    # Viewer selection.
    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401 — populate task registry
    import src.tasks    # noqa: F401

    all_tasks = list_tasks()

    import mjlab
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayOnnxConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_play_onnx(chosen_task, args)


if __name__ == "__main__":
    main()
