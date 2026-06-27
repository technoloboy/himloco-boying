"""Boying HIM velocity environment configurations.

Mirrors the Go2-HIM pattern:
- Actor: HIM observation groups (proprio_history / proprio_current / estimator_vel)
  — blind proprioceptive policy, no height scan.
- Critic: inherits the full privileged observation group from the base boying config,
  including height_scan on rough terrain.

Boying single-step proprioceptive dim S = 50:
    base_ang_vel(3) + projected_gravity(3) + command(3) + phase(2)
  + joint_pos(12) + joint_vel(12) + actions(12) + base_lin_vel(3) = 50

Mirrors HIMLoco's original obs design where history includes lin_vel as an
auxiliary signal for the estimator encoder.

So: proprio_history = 50 × 6 = 300, proprio_current = 50, estimator_vel = 3.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.boying.env_cfgs import (
  boying_flat_env_cfg,
  boying_rough_env_cfg,
)

HISTORY_LENGTH = 6


def _proprio_terms() -> dict[str, ObservationTermCfg]:
  """Boying proprioceptive single-step terms.

  Identical structure to Go2-HIM; noise scales match the base velocity task.
  Includes base_lin_vel to mirror HIMLoco's original obs design, where the
  encoder history contains true velocity as a supervision auxiliary signal.
  """
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.8, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
  }


def _him_observations(cfg: ManagerBasedRlEnvCfg, play: bool) -> dict:
  """Build the four HIM observation groups, preserving the base critic group."""
  critic_group = cfg.observations["critic"]
  return {
    "proprio_history": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "proprio_current": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
    "estimator_vel": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_lin_vel"},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
    "critic": critic_group,
  }


def boying_him_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Boying flat terrain HIM velocity configuration."""
  cfg = boying_flat_env_cfg(play=play)
  cfg.observations = _him_observations(cfg, play)
  return cfg


def boying_him_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Boying rough terrain HIM velocity configuration.

  The actor remains blind (no height scan). The critic retains privileged
  observations including height_scan from the base rough config.
  """
  cfg = boying_rough_env_cfg(play=play)
  cfg.observations = _him_observations(cfg, play)
  return cfg
