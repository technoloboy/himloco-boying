from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base XY linear velocity (Go1 parity).

  Z error removed — lin_vel_z_l2 handles z separately.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  return torch.exp(-xy_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded yaw angular velocity (Go1 parity).

  XY error removed — body_ang_vel handles xy separately.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  return torch.exp(-z_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot clearance deviation in body frame (HIMLoco parity).

  Transforms foot positions/velocities from world frame to base body frame
  via root_link orientation inverse. Computes |z_b - target| × ||v_xy_b||,
  promoting consistent foot height relative to base (robust on slopes/stairs).

  target_height is in body frame z (typically -0.20m: feet 20cm below base).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # World-frame foot pose and velocity
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]      # [B, N, 3]
  foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]  # [B, N, 3]

  # Base pose in world frame
  base_pos_w = asset.data.root_link_pos_w        # [B, 3]
  base_quat_w = asset.data.root_link_quat_w      # [B, 4]

  # Translate foot positions to base origin
  foot_pos_translated = foot_pos_w - base_pos_w.unsqueeze(1)  # [B, N, 3]

  # Expand quat to per-foot, then rotate world → body frame
  N = foot_pos_translated.shape[1]
  quat_expanded = base_quat_w.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4)
  foot_pos_b = quat_apply_inverse(
    quat_expanded, foot_pos_translated.reshape(-1, 3)
  ).reshape(-1, N, 3)
  foot_vel_b = quat_apply_inverse(
    quat_expanded, foot_vel_w.reshape(-1, 3)
  ).reshape(-1, N, 3)

  # Cost: |z_body - target| × ||v_xy_body||
  foot_z_b = foot_pos_b[:, :, 2]                 # [B, N]
  foot_vel_xy_b = foot_vel_b[:, :, :2]           # [B, N, 2]
  delta = torch.abs(foot_z_b - target_height)    # [B, N]
  vel_norm = torch.norm(foot_vel_xy_b, dim=-1)   # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)      # [B]
  cost = cost.clamp(max=2.0)  # prevent gradient spikes on rough terrain
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


class energy_efficiency:
  """Reward energy-efficient locomotion; trot emerges as energy-optimal gait.

  reward = exp(-P / (sigma_x * |vx_cmd| + sigma_z * |wz_cmd| + eps)) * (1 - slip_factor)
  P = sum(clamp(tau * qd, min=0)) — positive mechanical power only.
  slip_factor = clamp(mean_contact_foot_xy_vel / slip_scale, 0, 1) penalizes sliding.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, _ = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    actuator_ids, _ = asset.find_actuators(cfg.params["asset_cfg"].joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)

    self._slip_sensor_name: str | None = cfg.params.get("slip_sensor_name", None)
    if self._slip_sensor_name is not None:
      site_ids, _ = asset.find_sites(cfg.params["asset_cfg"].site_names)
      self._site_ids = torch.tensor(site_ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sigma_x: float = 300.0,
    sigma_z: float = 150.0,
    eps: float = 1.0,
    slip_sensor_name: str | None = None,
    slip_scale: float = 0.5,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    tau = asset.data.actuator_force[:, self._actuator_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    power = torch.sum(torch.clamp(tau * qd, min=0.0), dim=1)  # [B]

    vx_abs = torch.abs(command[:, 0])
    wz_abs = torch.abs(command[:, 2])
    denom = sigma_x * vx_abs + sigma_z * wz_abs + eps

    base_reward = torch.exp(-power / denom)

    if self._slip_sensor_name is not None:
      contact_sensor: ContactSensor = env.scene[self._slip_sensor_name]
      assert contact_sensor.data.found is not None
      in_contact = (contact_sensor.data.found > 0).float()              # [B, N]
      foot_vel_xy = asset.data.site_lin_vel_w[:, self._site_ids, :2]   # [B, N, 2]
      vel_norm = torch.norm(foot_vel_xy, dim=-1)                        # [B, N]
      num_contact = in_contact.sum(dim=1).clamp(min=1.0)
      mean_slip = (vel_norm * in_contact).sum(dim=1) / num_contact      # [B]
      slip_factor = (mean_slip / slip_scale).clamp(0.0, 1.0)
      base_reward = base_reward * (1.0 - slip_factor)

    return base_reward


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize action rate, clamped to prevent gradient explosion."""
  raw = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )
  return raw.clamp(max=10.0)


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward


def lin_vel_z_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize Z-direction linear velocity (Go1 lin_vel_z reward)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_height_l2(
    env: "ManagerBasedRlEnv",
    target_height: float,
    sensor_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize squared deviation from target base height (Go1 base_height reward).

    When ``sensor_name`` is None (flat terrain), the absolute world-frame base
    height is used. When ``sensor_name`` names a height-scan RayCastSensor
    (rough terrain), the base height is measured *relative to the terrain
    directly beneath the robot* (root_z minus the mean of the ray-hit terrain
    heights). This is the legged_gym/HIMLoco convention: it prevents the term
    from penalizing the robot for legitimately climbing onto steps/obstacles
    (where absolute base_z rises with the terrain).
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]
    if sensor_name is not None:
        sensor = env.scene[sensor_name]
        # Mean terrain height under the robot from valid ray hits.
        hit_z = sensor.data.hit_pos_w[..., 2]          # [B, N]
        valid = sensor.data.distances >= 0             # [B, N], exclude misses
        valid_f = valid.float()
        denom = valid_f.sum(dim=1).clamp(min=1.0)
        terrain_z = (hit_z * valid_f).sum(dim=1) / denom
        base_z = base_z - terrain_z
    return torch.square(base_z - target_height)


class smoothness:
    """Penalize ||a_t - 2*a_{t-1} + a_{t-2}||^2 (Go1 smoothness reward)."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        self.prev_prev_action = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim,
            device=env.device,
        )

    def __call__(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        a = env.action_manager.action
        a_prev = env.action_manager.prev_action
        diff = a - 2.0 * a_prev + self.prev_prev_action
        cost = torch.sum(torch.square(diff), dim=1).clamp(max=10.0)
        self.prev_prev_action = a_prev.clone()
        return cost


def hip_joint_deviation_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize hip joint deviation from default position (continuous).

    Mild posture regularization (Walk-These-Ways style) — discourages
    hip adduction (legs collapsing inward). Active at all times regardless
    of command. Used with weight ≈ -0.05 for gentle constraint.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]            # [B, N_hip]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]    # [B, N_hip]
    deviation = current_pos - default_pos                                  # [B, N_hip]
    return torch.sum(torch.square(deviation), dim=1)                       # [B]


def action_symmetry_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize left-right action AMPLITUDE asymmetry (MoE-CTS action_mirror style).

  Uses |a_L| - |a_R| (phase-agnostic). A trot gait keeps the left and right legs
  in antiphase, so the raw signed difference (a_L - a_R) is large even for a
  perfectly symmetric gait — penalizing it would fight normal locomotion. Taking
  magnitudes measures true limping (one side working harder) without penalizing
  gait phase. Gated by uprightness so falls/large disturbances aren't penalized.

  Joint index layout (FL→FR→RL→RR, hip→thigh→calf per leg):
    FL: 0,1,2  FR: 3,4,5  RL: 6,7,8  RR: 9,10,11
  Pairs: FL↔FR, RL↔RR.
  """
  a = env.action_manager.action.abs()  # [B, 12]
  error = (
    (a[:, 0] - a[:, 3]) ** 2    # hip   FL-FR
    + (a[:, 1] - a[:, 4]) ** 2  # thigh FL-FR
    + (a[:, 2] - a[:, 5]) ** 2  # calf  FL-FR
    + (a[:, 6] - a[:, 9]) ** 2  # hip   RL-RR
    + (a[:, 7] - a[:, 10]) ** 2 # thigh RL-RR
    + (a[:, 8] - a[:, 11]) ** 2 # calf  RL-RR
  ) / 12.0
  return error * _upright_gate(env, asset_cfg)


# =============================================================================
# MoE-CTS-inspired terms (RSS 2026, go2_rl_robotlab) for Boying obstacle traversal
# =============================================================================


def _upright_gate(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Uprightness gate in [0, 1] (MoE-CTS trick).

  Returns clamp(-projected_gravity_z, 0, 0.7) / 0.7. Upright robot → gravity_z≈-1
  → gate≈1 (penalty full strength). Tipping/fallen robot → gravity_z→0 or positive
  → gate→0 (penalty vanishes). Multiplying penalty terms by this gate removes the
  huge gradient spike at the moment of falling, which destabilises training.
  """
  g_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]  # [B]
  return torch.clamp(-g_z, 0.0, 0.7) / 0.7


def _terrain_relative_base_height(
  env: "ManagerBasedRlEnv",
  sensor_name: str | None,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Base height relative to terrain beneath the robot (or absolute if no sensor)."""
  asset: Entity = env.scene[asset_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  if sensor_name is None:
    return base_z
  sensor = env.scene[sensor_name]
  hit_z = sensor.data.hit_pos_w[..., 2]  # [B, N]
  valid = sensor.data.distances >= 0
  valid_f = valid.float()
  denom = valid_f.sum(dim=1).clamp(min=1.0)
  terrain_z = (hit_z * valid_f).sum(dim=1) / denom
  return base_z - terrain_z


def lin_vel_z_l2_gated(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """lin_vel_z_l2 multiplied by the uprightness gate."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = torch.square(asset.data.root_link_lin_vel_b[:, 2])
  return cost * _upright_gate(env, asset_cfg)


def body_orientation_l2_gated(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """body_orientation_l2 multiplied by the uprightness gate."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    xy_squared = torch.sum(
      torch.square(asset.data.projected_gravity_b[:, :2]), dim=1
    )
  return xy_squared * _upright_gate(env, asset_cfg)


def body_angular_velocity_penalty_gated(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """body_angular_velocity_penalty multiplied by the uprightness gate."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
  return cost * _upright_gate(env, asset_cfg)


def feet_regulation(
  env: "ManagerBasedRlEnv",
  base_height_target: float,
  sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize fast horizontal foot motion near the ground (MoE-CTS feet_regulation).

  Feet close to the ground get a much larger penalty for lateral speed; lifted
  (swing) feet are barely penalized. This discourages foot scuffing/dragging and
  encourages the robot to lift its feet before moving them — exactly the clean
  high-stepping gait needed for blind obstacle traversal. Gated by uprightness.

  reward = Σ_feet ( ||v_xy_foot||² · exp(-foot_height / (0.025·base_target)) )
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]        # [B, N, 3]
  feet_xy_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  base_pos_w = asset.data.root_link_pos_w.unsqueeze(1)               # [B, 1, 3]

  base_height = _terrain_relative_base_height(env, sensor_name, asset_cfg)  # [B]
  # Foot height above terrain ≈ base_height - (base_z - foot_z).
  feet_below_base = base_pos_w[:, :, 2] - feet_pos_w[:, :, 2]         # [B, N]
  feet_height = torch.clamp(base_height.unsqueeze(1) - feet_below_base, min=0.0)

  scale = 0.025 * base_height_target
  cost = (
    feet_xy_vel_w.pow(2).sum(dim=-1) * torch.exp(-feet_height / scale)
  ).sum(dim=-1)  # [B]
  return cost * _upright_gate(env, asset_cfg)


def undesired_contacts(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  threshold: float = 5.0,
) -> torch.Tensor:
  """Penalize thigh/calf contacts above a force threshold (MoE-CTS undesired_contacts).

  Counts, per env, how many monitored geoms have contact force magnitude above
  ``threshold`` Newtons. Discourages the robot from scuffing thighs/shanks on
  steps/obstacles instead of lifting the leg clear.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  force = contact_sensor.data.force  # [B, N, 3]
  assert force is not None
  force_norm = torch.norm(force, dim=-1)  # [B, N]
  is_contact = force_norm > threshold
  return torch.sum(is_contact.float(), dim=1)  # [B]


def joint_pos_limits(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint positions exceeding the soft limits (MoE-CTS joint_pos_limits)."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]                     # [B, N]
  limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]  # [B, N, 2]
  out_of_lower = (limits[..., 0] - q).clamp(min=0.0)
  out_of_upper = (q - limits[..., 1]).clamp(min=0.0)
  return torch.sum(out_of_lower + out_of_upper, dim=1)  # [B]


def joint_pos_penalty_l1(
  env: "ManagerBasedRlEnv",
  command_name: str,
  stand_still_scale: float = 1.0,
  velocity_threshold: float = 0.1,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Command-aware L1 deviation of joints from default (MoE-CTS joint_pos_penalty_l1).

  When the robot is commanded to move (or is moving), the L1 deviation from the
  default pose is penalized at weight 1×; when standing still (small command and
  low body velocity) the penalty is scaled by ``stand_still_scale``. Intended for
  thigh/calf joints (hip already handled by hip_joint_deviation).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = torch.linalg.norm(command, dim=1)
  body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  deviation = torch.linalg.norm(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
    dim=1,
    ord=1,
  )
  moving = torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold)
  return torch.where(moving, deviation, stand_still_scale * deviation)


