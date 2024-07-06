import torch

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

from enum import Enum


class ActionIdx(Enum):
    l_leg = 0
    r_leg = 1
    l_wheel = 2
    r_wheel = 3


class JointIdx(Enum):
    l_hip = 0
    l_knee = 1
    l_wheel = 2
    r_hip = 3
    r_knee = 4
    r_wheel = 5


class MetaWLVMCCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_observations = 18  # YXC: defined in _compute_proprioception_observations()
        num_privileged_obs = num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3
        num_actions = len(ActionIdx)  # YXC: 4

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/meta_wl/urdf/meta_wl.urdf"
        name = "MetaWL"
        penalize_contacts_on = ["upper", "lower", "base"]
        terminate_after_contacts_on = ["base"]

        # YXC: make sure the following parameters are consistent with the URDF file
        upper_len = 0.15
        lower_len = 0.25

        # links
        l1 = 0.07  # fixed link
        l2 = 0.1566  # driving link
        l3 = 0.1447  # follower link
        l4 = 0.0375  # output link
        l5 = 0.180 - l4  # output link extension

    class control(LeggedRobotCfg.control):
        action_scale_pos = 0.5
        action_scale_vel = 10.0

        # PD Drive parameters:
        stiffness = {"hip": 100.0, "knee": 100.0, "wheel": 0}  # [N*m/rad]
        damping = {"hip": 1.0, "knee": 1.0, "wheel": 0.5}  # [N*m*s/rad]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]  # x,y,z [m]
        init_hip = torch.pi / 4  # [rad]
        init_knee = 0.45
        default_joint_angles = {
            "l_hip": init_hip,
            "l_knee": init_knee,
            "l_wheel": 0.0,
            "r_hip": init_hip,
            "r_knee": init_knee,
            "r_wheel": 0.0,
        }


class MetaWLVMCCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        kl_decay = (
                           LeggedRobotCfgPPO.algorithm.desired_kl - 0.002
                   ) / LeggedRobotCfgPPO.runner.max_iterations

    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "meta_wl_vmc"
