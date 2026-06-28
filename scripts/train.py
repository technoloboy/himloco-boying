"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


def _patch_mjlab_termination_obs():
  """Monkey-patch ManagerBasedRlEnv.step() to capture pre-reset obs.

  Mirrors HIMLoco's termination_privileged_obs mechanism. The estimator
  needs s_{t+1} BEFORE reset_idx() clears observation history. We snapshot
  obs_manager.compute() right after termination is decided but BEFORE
  _reset_idx is called, then inject into extras["termination_obs"] so
  HIMPPO can patch transition.next_observations for terminated envs.

  WARNING: this duplicates the body of ManagerBasedRlEnv.step(). If mjlab
  upgrades the step() implementation, this patch must be re-aligned.
  """
  import mjlab.envs.manager_based_rl_env as _m

  def patched_step(self, action: torch.Tensor):
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    self.episode_length_buf += 1
    self.common_step_counter += 1

    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # *** Capture obs BEFORE reset_idx — this is s_{t+1} for terminated envs ***
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    termination_obs_dict = None
    if len(reset_env_ids) > 0:
      # Refresh derived quantities so obs reflects post-physics terminal state
      self.sim.forward()
      self.sim.sense()
      termination_obs_full = self.observation_manager.compute(update_history=False)
      termination_obs_dict = {
        k: v[reset_env_ids].clone() for k, v in termination_obs_full.items()
      }

      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    self.sim.forward()
    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if termination_obs_dict is not None:
      self.extras["termination_obs"] = termination_obs_dict
      self.extras["termination_env_ids"] = reset_env_ids

    return (
      self.obs_buf, self.reward_buf,
      self.reset_terminated, self.reset_time_outs, self.extras,
    )

  _m.ManagerBasedRlEnv.step = patched_step
  print("[patch] ManagerBasedRlEnv.step() patched for termination_obs", flush=True)


class NaNGuardWrapper:
  """Wraps RslRlVecEnvWrapper to detect & force-reset envs with NaN physics state.

  Reads robot.root_link_lin_vel_b after step() — if any env has NaN/inf or
  unrealistic velocity (hfield collision overflow), force-reset that env to
  clean qpos/qvel and mark dones=1 so PPO ignores the corrupted batch.
  """

  def __init__(self, env):
    self.env = env
    self._unwrapped = env.unwrapped
    self._asset = self._unwrapped.scene["robot"]

  def __getattr__(self, name):
    # Forward all other attrs to wrapped env (after self.env is set in __init__).
    if name in ("env", "_unwrapped", "_asset"):
      raise AttributeError(name)
    return getattr(self.env, name)

  def step(self, actions):
    obs, rew, dones, extras = self.env.step(actions)
    lin_vel = self._asset.data.root_link_lin_vel_b  # [N, 3]
    bad = (~torch.isfinite(lin_vel).all(dim=-1)) | (lin_vel.norm(dim=-1) > 50.0)
    if bad.any():
      bad_ids = torch.nonzero(bad, as_tuple=False).flatten().to(dtype=torch.int32)
      print(f"[NaNGuard] resetting {bad_ids.numel()} envs with NaN/extreme physics", flush=True)
      self._unwrapped.reset(env_ids=bad_ids)
      dones = dones.clone()
      dones[bad_ids.long()] = 1
      rew = rew.clone()
      rew[bad_ids.long()] = 0.0
    return obs, rew, dones, extras


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  env = NaNGuardWrapper(env)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Apply mjlab termination_obs patch before any env construction.
  _patch_mjlab_termination_obs()

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
