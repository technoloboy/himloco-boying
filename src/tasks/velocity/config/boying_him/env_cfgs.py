"""Boying HIM velocity environment configurations.

Mirrors HIMLoco Go1 obs layout exactly:
  - Actor history (proprio_history): 45D × 6 = 270D (phase OFF)
      command(3,×[2,2,0.25]) + ang_vel(3,×0.25) + gravity(3)
    + dof_pos(12) + dof_vel(12,×0.05) + actions(12) = 45D
  - Actor history (phase ON): 47D × 6 = 282D (add phase(2))
  - Critic (privileged): 238D = 45D + lin_vel(3,×2.0) + ext_force(3) + height(187,×5.0)
    (no foot obs — HIMLoco Go1 critic has none)
  - estimator_vel: 3D ×2.0 (matches HIMLoco obs_scales.lin_vel=2.0)

lin_vel is NOT in the actor history — it belongs in critic privileged obs only.
History T = 6. Network dims auto-inferred from obs shape.
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import src.tasks.velocity.mdp as mdp
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.config.boying.env_cfgs import (
  boying_flat_env_cfg,
  boying_rough_env_cfg,
)

HISTORY_LENGTH = 6
PHASE_ENABLED = os.environ.get("MJLAB_PHASE_ENABLED", "0") == "1"


def _proprio_terms() -> dict[str, ObservationTermCfg]:
  """Boying HIM proprioceptive single-step terms.

  Mirrors HIMLoco Go1 45D actor history obs exactly:
    command(3,×[2,2,0.25]) + ang_vel(3,×0.25) + gravity(3)
    + dof_pos(12) + dof_vel(12,×0.05) + actions(12) = 45D (phase OFF)
    = 47D (phase ON with extra phase(2))

  lin_vel is NOT here — it belongs in critic privileged obs only (HIMLoco parity).
  Noise values are final (noise_level × noise_scale × obs_scale).
  """
  terms: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=1.0,
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands_scaled,
      params={"command_name": "twist", "scale": (2.0, 2.0, 0.25)},
    ),
  }
  if PHASE_ENABLED:
    terms["phase"] = ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.5, "command_name": "twist"},
    )
  terms.update({
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      scale=1.0,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.075, n_max=0.075),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # NO base_lin_vel — belongs in critic privileged obs only (HIMLoco parity)
  })
  return terms


def _him_observations(cfg: ManagerBasedRlEnvCfg, play: bool) -> dict:
  """Build the four HIM observation groups with HIMLoco Go1 exact parity.

  Critic is rebuilt explicitly to match HIMLoco 238D:
    45D proprio + lin_vel(3,×2.0) + ext_force(3) + height(187,×5.0)
  Foot obs are excluded (HIMLoco critic has none).
  """
  # Build HIM-specific critic: 238D = 45D + lin_vel(3) + ext_force(3) + height(187)
  him_critic_terms = {
    # ── 45D proprio (same scales/noise as proprio_terms) ──
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=1.0,
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands_scaled,
      params={"command_name": "twist", "scale": (2.0, 2.0, 0.25)},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      scale=1.0,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.075, n_max=0.075),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # ── 3D lin_vel (×2.0, noise±0.2) ──
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=2.0,
    ),
    # ── 3D ext_force (body frame, no scale) ──
    "external_force": ObservationTermCfg(
      func=mdp.external_body_force,
      params={"body_name": "base", "asset_cfg": SceneEntityCfg("robot")},
    ),
    # ── 187D height_scan (×5.0, noise±0.5) ──
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      scale=5.0,
    ),
  }
  if PHASE_ENABLED:
    # phase(2D) placed after command, before joint_pos — same order as proprio_terms
    him_critic_terms_with_phase = {}
    for k, v in him_critic_terms.items():
      him_critic_terms_with_phase[k] = v
      if k == "command":
        him_critic_terms_with_phase["phase"] = ObservationTermCfg(
          func=mdp.phase,
          params={"period": 0.5, "command_name": "twist"},
        )
    him_critic_terms = him_critic_terms_with_phase

  return {
    "proprio_history": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=HISTORY_LENGTH,
      flatten_history_dim=True,
      nan_policy="sanitize",  # hfield collision overflow guard
    ),
    "proprio_current": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    "estimator_vel": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_lin_vel"},
          scale=2.0,  # Match HIMLoco obs_scales.lin_vel=2.0 (privileged_obs[:, 45:48])
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=him_critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      nan_policy="sanitize",
    ),
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
