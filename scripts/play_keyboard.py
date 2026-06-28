"""Interactive play script with terminal keyboard velocity control.

Controls (type in the terminal — no Enter needed, works without MuJoCo window focus):
  w / s  — increase / decrease forward velocity (vx, ±0.3 m/s per press)
  a / d  — strafe left / right (vy, ±0.2 m/s per press)
  q / e  — turn left / right (wz, ±0.3 rad/s per press)
  x      — zero all velocity commands (stop in place)
  9      — reset environment + zero commands
  ESC / Ctrl-C — quit cleanly

Terrain: pyramid_stairs (upward) + hf_discrete_obstacles only.
heading_command disabled so wz is directly keyboard-controlled.
"""

import os
import select
import sys
import termios
import time
import tty
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains import (
  BoxPyramidStairsTerrainCfg,
  HfDiscreteObstaclesTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

# ── Terrain ───────────────────────────────────────────────────────────────────
_KEYBOARD_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=10.0,
  num_rows=5,
  num_cols=10,
  curriculum=False,
  sub_terrains={
    "pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.05, 0.15),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
      proportion=0.5,
      obstacle_height_range=(0.05, 0.12),
      obstacle_width_range=(0.4, 0.8),
      num_obstacles=20,
      platform_width=1.0,
    ),
  },
  add_lights=True,
)

_VX_MAX = 2.0
_VY_MAX = 1.0
_WZ_MAX = 1.0
_FREEZE  = 1e6   # large value → CommandTerm.compute() never resamples


# ── Velocity state ────────────────────────────────────────────────────────────
class _Ctrl:
  def __init__(self) -> None:
    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0

  def zero(self) -> None:
    self.vx = self.vy = self.wz = 0.0


# ── Viewer subclass ───────────────────────────────────────────────────────────
class _KeyboardViewer(NativeMujocoViewer):
  """Viewer that reads keyboard from the terminal (raw mode + select poll)."""

  def __init__(self, env: RslRlVecEnvWrapper, policy, ctrl: _Ctrl) -> None:
    super().__init__(env, policy)
    self._ctrl = ctrl
    self._env_ref = env

  # Called every iteration — keeps command tensor frozen at keyboard state.
  # This survives env.reset() which would otherwise re-randomise time_left.
  def _inject_cmd(self) -> None:
    cmd = self._env_ref.unwrapped.command_manager.get_term("twist")
    cmd.vel_command_b[:, 0] = self._ctrl.vx
    cmd.vel_command_b[:, 1] = self._ctrl.vy
    cmd.vel_command_b[:, 2] = self._ctrl.wz
    cmd.time_left[:] = _FREEZE

  def _handle_key(self, ch: str) -> None:
    c = self._ctrl
    changed = True
    if ch == "w":
      c.vx = min(c.vx + 0.3, _VX_MAX)
    elif ch == "s":
      c.vx = max(c.vx - 0.3, -_VX_MAX)
    elif ch == "a":
      c.vy = min(c.vy + 0.2, _VY_MAX)
    elif ch == "d":
      c.vy = max(c.vy - 0.2, -_VY_MAX)
    elif ch == "q":
      c.wz = min(c.wz + 0.3, _WZ_MAX)
    elif ch == "e":
      c.wz = max(c.wz - 0.3, -_WZ_MAX)
    elif ch == "x":
      c.zero()
    elif ch == "9":
      c.zero()
      self.request_reset()
    elif ch in ("\x03", "\x1b"):   # Ctrl-C or ESC
      self.close()
      return
    else:
      changed = False

    if changed:
      _print_cmd(c, reset=(ch == "9"))

  def run(self) -> None:  # type: ignore[override]
    """Override run(): interleave select-based stdin poll with each tick."""
    self.setup()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    # Set terminal to raw mode on the MAIN thread — reliable.
    tty.setraw(fd)
    try:
      while self.is_running():
        # Drain all pending stdin bytes before stepping (non-blocking).
        while True:
          r, _, _ = select.select([sys.stdin], [], [], 0)
          if not r:
            break
          raw = os.read(fd, 1)
          ch = raw.decode("utf-8", errors="ignore").lower()
          self._handle_key(ch)

        # Step the simulation one tick.
        if not self.tick():
          time.sleep(0.001)

        # Re-inject keyboard command every tick so env.reset() can't override it.
        self._inject_cmd()

    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
      sys.stdout.write("\n")
      sys.stdout.flush()
      self.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _print_cmd(c: _Ctrl, reset: bool = False) -> None:
  tag = "RESET" if reset else "CMD  "
  sys.stdout.write(
    f"\r[{tag}] vx={c.vx:+.2f}  vy={c.vy:+.2f}  wz={c.wz:+.2f}"
    "    w/s=前后  a/d=左右  q/e=转向  x=停  9=重置  ESC=退出  "
  )
  sys.stdout.flush()


def _print_help() -> None:
  print(
    "\n"
    "┌──────────────────────────────────────────────────────┐\n"
    "│       Terminal Keyboard Velocity Control             │\n"
    "│  w / s  : 前进 / 后退       +/- 0.30 m/s           │\n"
    "│  a / d  : 左平移 / 右平移   +/- 0.20 m/s           │\n"
    "│  q / e  : 左转  / 右转      +/- 0.30 rad/s         │\n"
    "│  x      : 速度清零（停止原地）                     │\n"
    "│  9      : 重置环境 + 速度清零                      │\n"
    "│  ESC / Ctrl-C : 退出                               │\n"
    "│                                                      │\n"
    "│  在此终端输入，无需切换到 MuJoCo 窗口             │\n"
    "└──────────────────────────────────────────────────────┘"
  )


# ── CLI ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Cfg:
  checkpoint_file: str
  """Path to model checkpoint (.pt)."""
  num_envs: int = 1
  """Number of parallel environments."""
  device: str | None = None
  """Torch device (default: cuda:0 if available, else cpu)."""


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks    # noqa: F401

  from mjlab.tasks.registry import list_tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  cfg = tyro.cli(
    _Cfg,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # ── Env config ───────────────────────────────────────────────────
  env_cfg = load_env_cfg(chosen_task, play=True)
  agent_cfg = load_rl_cfg(chosen_task)
  env_cfg.scene.num_envs = cfg.num_envs

  # Terrain: stairs + obstacles only.
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.terrain_generator = _KEYBOARD_TERRAIN_CFG

  # Disable heading_command — wz is directly keyboard-controlled.
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
  twist_cmd = env_cfg.commands.get("twist")
  if isinstance(twist_cmd, UniformVelocityCommandCfg):
    twist_cmd.heading_command = False
    twist_cmd.ranges.heading = None

  # Free exploration — no forced terminations.
  env_cfg.terminations = {}

  # ── Build env + policy ───────────────────────────────────────────
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  print(f"[INFO] Loading: {resume_path.name}")

  runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  # ── Launch ───────────────────────────────────────────────────────
  ctrl = _Ctrl()
  _print_help()
  _KeyboardViewer(env, policy, ctrl).run()
  env.close()


if __name__ == "__main__":
  main()
