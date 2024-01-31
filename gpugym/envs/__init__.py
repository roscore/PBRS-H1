
from gpugym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from gpugym.utils.task_registry import task_registry
from .PBRS.humanoid import Humanoid
from .PBRS.humanoid_config import HumanoidCfg, HumanoidCfgPPO


task_registry.register("pbrs:humanoid", Humanoid, HumanoidCfg(), HumanoidCfgPPO())

# register H1 robot
from .PBRS.H1_config import H1Cfg, H1CfgPPO
from .PBRS.H1 import H1
task_registry.register("pbrs:H1", H1, H1Cfg(), H1CfgPPO())
