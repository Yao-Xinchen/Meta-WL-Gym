from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

class MetaWLCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "l_leg": 0.5,
            "l_wheel": 0.0,
            "r_leg": -0.5,
            "r_wheel": 0.0,
        }

    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/meta_wl/urdf/meta_wl.urdf"
        name = "MetaWl"

class MetaWLCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "meta_wl"