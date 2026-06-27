from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def generated_commands_scaled(
    env: "ManagerBasedRlEnv",
    command_name: str,
    scale: tuple[float, ...] = (1.0,),
) -> torch.Tensor:
    """generated_commands with per-element scaling.

    Mirrors HIMLoco's commands_scale=[lin_vel, lin_vel, ang_vel] = [2.0, 2.0, 0.25].
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None
    scale_t = torch.tensor(scale, device=command.device, dtype=command.dtype)
    return command * scale_t


def external_body_force(
    env: "ManagerBasedRlEnv",
    body_name: str = "base",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """External force applied to a body in the body's local frame.

    Mirrors HIMLoco's self.disturbance[:, 0, :] which stores forces in
    LOCAL_SPACE (body frame). Converts world-frame xfrc_applied to body frame
    using the body's quaternion.

    Returns shape [num_envs, 3].
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(body_name)
    # World-frame external force: [B, 1, 3]
    force_w = asset.data.body_external_force[:, body_ids, :]
    # Body quaternion in world frame: [B, 1, 4]
    quat_w = asset.data.body_com_quat_w[:, body_ids, :]
    # Rotate world force into body frame: [B, 3]
    return quat_apply_inverse(quat_w.squeeze(1), force_w.squeeze(1))

