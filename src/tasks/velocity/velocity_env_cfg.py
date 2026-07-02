"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg  # 基于管理器的RL环境配置类，定义整个环境结构
from mjlab.envs import mdp as envs_mdp  # 环境内置的MDP模块（动作、观测等）
from mjlab.envs.mdp import dr  # 数据随机化模块（对传感器等添加噪声）
from mjlab.envs.mdp.actions import JointPositionActionCfg  # 关节位置动作配置
from mjlab.managers.action_manager import ActionTermCfg  # 动作项配置，定义动作空间
from mjlab.managers.command_manager import CommandTermCfg  # 命令项配置（如速度指令）
from mjlab.managers.curriculum_manager import CurriculumTermCfg  # 课程学习项配置（难度递增）
from mjlab.managers.event_manager import EventTermCfg  # 事件项配置（如重置、随机扰动）
from mjlab.managers.metrics_manager import MetricsTermCfg  # 指标项配置（用于评估训练效果）
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # 观测组/项配置
from mjlab.managers.reward_manager import RewardTermCfg  # 奖励项配置（定义奖励函数）
from mjlab.managers.scene_entity_config import SceneEntityCfg  # 场景实体配置（指定实体名称和site ID）
from mjlab.managers.termination_manager import TerminationTermCfg  # 终止项配置（定义episode结束条件）
from mjlab.scene import SceneCfg  # 场景配置（包含机器人、地形、传感器等）
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg  # 传感器配置（网格模式、物体参考、射线投射）
from mjlab.sim import MujocoCfg, SimulationCfg  # MuJoCo仿真配置和通用仿真配置
from mjlab.tasks.velocity import mdp  # 速度任务的MDP模块（奖励、观测、命令等）
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # 均匀速度命令配置（随机生成速度指令）
from mjlab.terrains import (
  TerrainEntityCfg,  # 地形实体配置
  TerrainGeneratorCfg,  # 地形生成器配置（组合多种地形）
  BoxFlatTerrainCfg,  # 平坦地形配置
  BoxPyramidStairsTerrainCfg,  # 金字塔阶梯地形配置
  BoxInvertedPyramidStairsTerrainCfg,  # 倒金字塔阶梯地形配置
  HfPyramidSlopedTerrainCfg,  # 金字塔斜坡地形配置（高度场）
  HfPerlinNoiseTerrainCfg,  # Perlin噪声地形配置（高度场）
  HfDiscreteObstaclesTerrainCfg,  # 离散障碍物地形配置（高度场）
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # 均匀噪声配置（添加到观测/动作）
from mjlab.viewer import ViewerConfig  # 查看器配置（分辨率、帧率等）

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.terrains_custom import HfRoughSlopedTerrainCfg


# Terrain generator mirroring HIMLoco's 8-terrain structure, parameters scaled
# for Boying (leg length ~470mm vs A1's ~280mm; step height ceiling 0.20m vs 0.23m).
# mjlab scales each sub-terrain parameter by difficulty = row/num_rows when
# curriculum=True. HfWaveTerrainCfg dropped; stepping stones + discrete obstacles added.
_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  curriculum=False,  # boying/env_cfgs.py sets curriculum=True for training
  sub_terrains={
    "flat": BoxFlatTerrainCfg(proportion=0.05),
    "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      horizontal_scale=0.2,
    ),
    "hf_pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 0.4),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
      horizontal_scale=0.2,
    ),
    "pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=0.20,  # 0.15→0.20: stronger obstacle-traversal focus (MoE-CTS)
      step_height_range=(0.05, 0.15),  # lower for Boying (28cm stand height)
      step_width=0.5,                  # wider tread → 3 steps, more foothold room
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.20,  # 0.15→0.20
      step_height_range=(0.05, 0.15),
      step_width=0.5,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
      proportion=0.15,  # 0.20→0.15
      obstacle_height_range=(0.05, 0.15),
      obstacle_width_range=(1.0, 2.0),  # 0.4-0.8→1.0-2.0: wider, easier to cross (MoE-CTS)
      num_obstacles=20,
      platform_width=1.0,
      horizontal_scale=0.2,
    ),
    # rough_slope: pyramid slope + difficulty-scaled roughness (MoE-CTS rough_slope).
    "rough_slope": HfRoughSlopedTerrainCfg(
      proportion=0.10,
      slope_range=(0.0, 0.4),
      noise_range=(0.0, 0.06),
      noise_step=0.005,
      downsampled_scale=0.2,
      platform_width=2.0,
      border_width=0.25,
      horizontal_scale=0.2,
    ),
    "hf_perlin_noise": HfPerlinNoiseTerrainCfg(
      proportion=0.10,
      height_range=(0.0, 0.08),
      horizontal_scale=0.2,
    ),
  },
  add_lights=True,
)


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # Observations
  ##

  actor_terms = {
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
    # height_scan present for non-HIM tasks (flat tasks delete it via del cfg.observations["actor"].terms["height_scan"])
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }

  critic_terms = {
    # ── 45D proprio (same as actor, with scale) ──
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
    # ── 3D lin_vel (privileged, HIMLoco ×2.0, noise±0.2) ──
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=2.0,
    ),
    # ── 3D ext_force (privileged, HIMLoco no scale) ──
    "external_force": ObservationTermCfg(
      func=mdp.external_body_force,
      params={
        "body_name": "base",
        "asset_cfg": SceneEntityCfg("robot"),  # body_name set per-robot if needed
      },
    ),
    # ── 187D height_scan (privileged, HIMLoco scale=5.0, noise±0.5) ──
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      scale=5.0,
    ),
    # ── foot observations (used by non-HIM tasks, overridden per-robot) ──
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
      nan_policy="warn",  # hfield collision overflow guard
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      nan_policy="warn",
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.05,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-3.14, 3.14),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        # Go1 init state: zero velocity on reset.
        "velocity_range": {
          "x":     (0.0, 0.0),
          "y":     (0.0, 0.0),
          "z":     (0.0, 0.0),
          "roll":  (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw":   (0.0, 0.0),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_scale,
      mode="reset",
      params={
        # Mirrors HIMLoco: dof_pos = default_dof_pos * uniform(0.5, 1.5).
        "position_range": (0.5, 1.5),
        "velocity_range": (0.0, 0.0),  # HIMLoco zeros joint velocities on reset.
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(16.0, 16.0),
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-1.0, 1.0),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.2, 1.25),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (0.0, 0.0),   # Go1 has no encoder_bias DR.
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.03, 0.03),
          1: (-0.03, 0.03),
          2: (-0.03, 0.03),
        },
      },
    ),
    # Randomize the payload mass of the base link, matching HIMLoco's
    # payload_mass_range = [−1, 2] kg applied per reset.
    # body_mass(operation="add") models a point mass at the CoM, leaving
    # the inertia tensor unchanged — physically appropriate for a payload.
    # Body name must be set per-robot (e.g. "base_link" for Go2).
    "payload_mass": EventTermCfg(
      mode="reset",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": (-1.0, 2.0),
      },
    ),
    # Randomize PD gains by ±10%, matching HIMLoco's kp_range / kd_range = [0.9, 1.1].
    # Targets all actuators of the robot (actuator_ids=slice(None) by default).
    "pd_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    # Apply random external force on base body every ~0.04s (Go1: ±30N every 8 sim steps).
    "external_force": EventTermCfg(
      func=envs_mdp.apply_external_force_torque,
      mode="interval",
      interval_range_s=(0.04, 0.04),
      params={
        "force_range": (-30.0, 30.0),
        "torque_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),  # Set per-robot if needed.
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-0.2,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2, 
      weight=-2.5e-7
      ),
    "joint_power": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-2e-5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    "base_height_l2": RewardTermCfg(
      func=mdp.base_height_l2,
      weight=-1.0,
      params={"target_height": 0.30},
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.01,
      params={
        "target_height": -0.20,  # body frame: feet 20cm below base (HIMLoco parity)
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    # "action_symmetry_l2": RewardTermCfg(func=mdp.action_symmetry_l2, weight=-0.015),
    "smoothness": RewardTermCfg(func=mdp.smoothness, weight=-0.01),
    "hip_joint_deviation": RewardTermCfg(
      func=mdp.hip_joint_deviation_l2,
      weight=-0.15,  # was -0.05, increased to stronger constrain hip adduction
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_joint",)),
      },
    ),
    # weight=0 — kept for per-robot override (go2/a2/boying use pose/foot_slip)
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},   # Set per-robot.
        "std_running": {},   # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "energy_efficiency": RewardTermCfg(
      func=mdp.energy_efficiency,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*", site_names=()),
        "command_name": "twist",
        "sigma_x": 300.0,
        "sigma_z": 150.0,
        "eps": 1.0,
        "slip_sensor_name": "feet_ground_contact",
        "slip_scale": 0.5,
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist", "max_level_per_episode": 1.0},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel_him,
      params={
        "command_name": "twist",
        "reward_name": "track_linear_velocity",
        "max_curriculum": 2.0,
        "expand_step": 0.2,
        "tracking_threshold": 0.8,
      },
    ),
    # Reward-weight curricula (MoE-CTS gradual_reward_weight_modification).
    # Steps = iteration × num_steps_per_env(100). Stepwise approx of linear ramp.
    # lin_vel_z_l2: -2.0 → 0 over first ~1500 iter (early anti-bounce, then release
    # so the robot dares to climb; uprightness gate + other terms keep z controlled).
    "reward_weight_lin_vel_z": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "lin_vel_z_l2",
        "weight_stages": [
          {"step": 0, "weight": -2.0},
          {"step": 50000, "weight": -1.3},
          {"step": 100000, "weight": -0.6},
          {"step": 150000, "weight": 0.0},
        ],
      },
    ),
    # base_height_l2: gentle ramp -1.0 → -1.3 → -1.6 (avoid the huge -3.0/-10.0
    # jumps that previously caused Value-loss blow-up + policy collapse).
    "reward_weight_base_height": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "base_height_l2",
        "weight_stages": [
          {"step": 0, "weight": -1.0},
          {"step": 250000, "weight": -1.3},
          {"step": 500000, "weight": -1.6},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(_ROUGH_TERRAIN_CFG),
        max_init_terrain_level=3,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
