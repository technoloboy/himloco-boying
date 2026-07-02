"""Boying velocity environment configurations."""

from src.assets.robots import get_boying_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.velocity.mdp as boying_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def boying_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Boying rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 512  # was 512, additional safety margin for L5 hfield collision overflow

  cfg.scene.entities = {"robot": get_boying_robot_cfg()}

  # Set raycast sensor frame to boying base body.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base"

  # boying foot geom names follow the same pattern as Go2/A2.
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Monitors only base and head collision geoms — mirrors HIMLoco's terminate_after_contacts_on=["base"].
  # Hip/thigh/calf contact with ground does NOT trigger termination.
  base_head_ground_cfg = ContactSensorCfg(
    name="base_head_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=(
        "base1_collision", "base2_collision",
        "head1_collision", "head2_collision",
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Thigh/calf ground contacts — monitored so undesired_contacts can penalize
  # the robot for scuffing legs on steps/obstacles instead of lifting clear.
  # These contacts do NOT terminate (only base/head do); they are penalized.
  thigh_calf_geoms = tuple(
    f"{leg}_{seg}{i}_collision"
    for leg in ("FR", "FL", "RR", "RL")
    for seg in ("thigh", "calf")
    for i in (1, 2, 3)
  )
  thigh_calf_contact_cfg = ContactSensorCfg(
    name="thigh_calf_contact",
    primary=ContactMatch(mode="geom", entity="robot", pattern=thigh_calf_geoms),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=3,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    base_head_ground_cfg,
    thigh_calf_contact_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 5  # HIMLoco default: start up to level 5

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
  cfg.events["payload_mass"].params["asset_cfg"].body_names = ("base",)

  # Leg-link mass/inertia randomization (mirrors go2_rl_robotlab's
  # randomize_rigid_body_mass_others: scale non-base links by 0.9~1.1x).
  # Use pseudo_inertia (not body_mass) so mass AND inertia scale together for a
  # physically consistent density change. alpha_range=(-0.05, 0.05) gives a mass
  # scale of e^(2*alpha) in [e^-0.1, e^0.1] = [0.905, 1.105] ~ [0.9, 1.1].
  cfg.events["leg_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(".*_hip", ".*_thigh", ".*_calf")
      ),
      "alpha_range": (-0.05, 0.05),
    },
  )

  # Pose reward: boying hip limits are tighter (±0.681 rad) than Go2 (±1.047 rad),
  # so we use the same relative std as Go2.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint":   0.05,
    r".*(FR|FL|RR|RL)_thigh_joint": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint":  0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint":   0.15,
    r".*(FR|FL|RR|RL)_thigh_joint": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint":  0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint":   0.15,
    r".*(FR|FL|RR|RL)_thigh_joint": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint":  0.5,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  # base_height: measure relative to the terrain beneath the robot (height_scan),
  # so climbing onto steps/obstacles is not penalized. target 0.30 → 0.28 to match
  # Boying's true FK standing height (~0.281m, base above feet at nominal pose).
  cfg.rewards["base_height_l2"].params["sensor_name"] = "terrain_scan"
  cfg.rewards["base_height_l2"].params["target_height"] = 0.28

  # hip deviation: relax -0.15 → -0.05 (HIMLoco/default magnitude). The stronger
  # pull constrained the lateral re-stepping/hip-swing needed to balance on rough
  # terrain; symmetry is already covered by action_symmetry_l2.
  cfg.rewards["hip_joint_deviation"].weight = -0.05

  # ── MoE-CTS (RSS2026) inspired obstacle-traversal improvements (Boying only) ──
  # (1) Uprightness gate: swap 3 penalty terms to gated versions so the huge
  #     gradient spike at the moment of falling vanishes (removes training blowups).
  cfg.rewards["lin_vel_z_l2"].func = boying_mdp.lin_vel_z_l2_gated
  cfg.rewards["body_orientation_l2"].func = boying_mdp.body_orientation_l2_gated
  cfg.rewards["body_ang_vel"].func = boying_mdp.body_angular_velocity_penalty_gated

  # (2) feet_regulation: penalize fast horizontal foot motion near ground → high
  #     stepping gait for blind obstacle traversal (weight -0.05, MoE-CTS parity).
  cfg.rewards["feet_regulation"] = RewardTermCfg(
    func=boying_mdp.feet_regulation,
    weight=-0.05,
    params={
      "base_height_target": 0.28,
      "sensor_name": "terrain_scan",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )
  # (3a) undesired_contacts: penalize thigh/calf scuffing on obstacles (weight -0.5).
  cfg.rewards["undesired_contacts"] = RewardTermCfg(
    func=boying_mdp.undesired_contacts,
    weight=-0.5,
    params={"sensor_name": "thigh_calf_contact", "threshold": 5.0},
  )
  # (3b) joint_pos_limits: penalize joints hitting soft limits (weight -0.5).
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=boying_mdp.joint_pos_limits,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_joint",))},
  )
  # (3c) joint_pos_penalty_l1: command-aware thigh/calf deviation (weight -0.01).
  cfg.rewards["joint_pos_penalty_l1"] = RewardTermCfg(
    func=boying_mdp.joint_pos_penalty_l1,
    weight=-0.01,
    params={
      "command_name": "twist",
      "stand_still_scale": 1.0,
      "velocity_threshold": 0.1,
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_(thigh|calf)_joint",)),
    },
  )

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": base_head_ground_cfg.name, "force_threshold": 1.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def boying_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Boying flat terrain velocity configuration."""
  cfg = boying_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Flat terrain has no terrain_scan sensor: revert base_height to absolute
  # world height (target 0.28 from the rough override still applies).
  cfg.rewards["base_height_l2"].params["sensor_name"] = None

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
