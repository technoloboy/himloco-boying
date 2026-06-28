"""Boying quadruped robot constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH  # 项目源码根路径，用于定位MJCF模型文件
from mjlab.actuator import BuiltinPositionActuatorCfg  # MuJoCo内置位置控制器配置，支持PD增益和力矩限制
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg  # 实体配置：EntityCfg定义机器人初始状态/碰撞/规格，EntityArticulationInfoCfg配置关节驱动器
from mjlab.utils.os import update_assets  # 更新资源字典，将mesh/texture等文件读入字节流
from mjlab.utils.spec_config import CollisionCfg  # 碰撞检测配置，控制geom碰撞类型、优先级、摩擦系数等

##
# MJCF and assets.
##

BOYING_XML: Path = (
  SRC_PATH / "assets" / "robots" / "boying" / "xmls" / "boying.xml"
)
assert BOYING_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, BOYING_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(BOYING_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config (from 关节名轴MJCF限位rad力矩限位.txt).
#
# All joints share the same kp/kd/armature; calf has a higher torque limit.
##

BOYING_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_joint",
  ),
  stiffness=60.0,
  damping=4.5,
  effort_limit=60.0,
  armature=0.0396,
)
BOYING_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_joint",
  ),
  stiffness=60.0,
  damping=4.5,
  effort_limit=60.0,
  armature=0.0396,
)
BOYING_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_joint",
  ),
  stiffness=60.0,
  damping=4.5,
  effort_limit=90.0,
  armature=0.0396,
)

##
# Keyframes / initial state (from 关节名轴MJCF限位rad力矩限位.txt).
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.35),
  joint_pos={
    ".*hip_joint":   0.0,
    ".*thigh_joint": 0.9,
    ".*calf_joint":  -1.8,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
#
# Foot geom names follow the pattern ^[FR][LR]_foot_collision$ (matching Go2/A2).
# All other collision geoms (base, head, hip, thigh, calf) match .*_collision\d*$
# and are excluded from foot-only sensors / marked as illegal contact bodies.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# Enables all collisions (no self-collision), with feet getting priority contact params.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Articulation config.
##

BOYING_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BOYING_ACTUATOR_HIP,
    BOYING_ACTUATOR_THIGH,
    BOYING_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_boying_robot_cfg() -> EntityCfg:
  """Get a fresh Boying robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=BOYING_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_boying_robot_cfg())
  viewer.launch(robot.spec.compile())
