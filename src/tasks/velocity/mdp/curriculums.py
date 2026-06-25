from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  max_level_per_episode: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Limit the fraction of envs that can advance per step to prevent terrain
  # levels from spiking faster than the policy can adapt.
  if max_level_per_episode < 1.0:
    gate = torch.rand(move_up.shape, device=move_up.device) < max_level_per_episode
    move_up = move_up & gate

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class commands_vel_adaptive:
  """Smoothly expand velocity command ranges as tracking reward improves.

  Uses EMA of track_linear_velocity reward as mastery signal.
  Only expands when ema > mastery_threshold, at rate expand_rate per call.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    self._reward_name = cfg.params["reward_name"]
    self._ema = 0.0
    self._ema_alpha = cfg.params.get("ema_alpha", 0.02)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids,
    command_name: str,
    reward_name: str,
    target_lin_vel_x: tuple[float, float],
    target_lin_vel_y: tuple[float, float],
    target_ang_vel_z: tuple[float, float],
    mastery_threshold: float = 0.7,
    expand_rate: float = 0.005,
    ema_alpha: float = 0.02,
  ) -> dict[str, torch.Tensor]:
    del env_ids  # Unused.
    rm = env.reward_manager
    if reward_name in rm._term_names:
      idx = rm._term_names.index(reward_name)
      current_reward = rm._step_reward[:, idx].mean().item()
    else:
      current_reward = 0.0
    self._ema = ema_alpha * current_reward + (1.0 - ema_alpha) * self._ema

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    ranges = command_term.cfg.ranges

    if self._ema > mastery_threshold:
      def _expand(current, target):
        lo = current[0] + expand_rate * (target[0] - current[0])
        hi = current[1] + expand_rate * (target[1] - current[1])
        lo = max(lo, target[0]) if target[0] < current[0] else min(lo, target[0])
        hi = min(hi, target[1]) if target[1] > current[1] else max(hi, target[1])
        return (lo, hi)

      ranges.lin_vel_x = _expand(ranges.lin_vel_x, target_lin_vel_x)
      ranges.lin_vel_y = _expand(ranges.lin_vel_y, target_lin_vel_y)
      ranges.ang_vel_z = _expand(ranges.ang_vel_z, target_ang_vel_z)

    return {
      "reward_ema":   torch.tensor(self._ema),
      "lin_vel_x_lo": torch.tensor(ranges.lin_vel_x[0]),
      "lin_vel_x_hi": torch.tensor(ranges.lin_vel_x[1]),
      "lin_vel_y_lo": torch.tensor(ranges.lin_vel_y[0]),
      "lin_vel_y_hi": torch.tensor(ranges.lin_vel_y[1]),
      "ang_vel_z_lo": torch.tensor(ranges.ang_vel_z[0]),
      "ang_vel_z_hi": torch.tensor(ranges.ang_vel_z[1]),
    }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
