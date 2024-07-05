from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

from enum import Enum


def calc_knee(hip):
    # YXC: TODO: implement this function
    return 0.5 * hip


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
        num_observations = 19  # YXC: defined in _compute_proprioception_observations()
        num_privileged_obs = num_observations + 7 * 11 + 3 + 6 * 5 + 3 + 3
        num_actions = len(ActionIdx)  # YXC: 4

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/meta_wl/urdf/meta_wl_vmc.urdf"
        name = "MetaWL"
        penalize_contacts_on = ["upper", "lower", "base"]
        terminate_after_contacts_on = ["base"]

        # YXC: make sure the following parameters are consistent with the URDF file
        upper_len = 0.15
        lower_len = 0.25

    class control(LeggedRobotCfg.control):
        action_scale_pos = 0.5
        action_scale_vel = 10.0

        feedforward_force = 40.0  # [N]

        kp_l0 = 900.0  # [N/m]
        kd_l0 = 20.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"hip": 0.0, "knee": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"hip": 0.0, "knee": 0.0, "wheel": 0.5}  # [N*m*s/rad]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]  # x,y,z [m]
        init_hip = 0.5  # [rad]
        init_knee = calc_knee(init_hip)
        default_joint_angles = {
            "l_hip": init_hip,
            "l_knee": init_knee,
            "l_wheel": 0.0,
            "r_hip": init_hip,
            "r_knee": init_knee,
            "r_wheel": 0.0,
        }

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25

    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1


class MetaWLVMCCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        kl_decay = (
                           LeggedRobotCfgPPO.algorithm.desired_kl - 0.002
                   ) / LeggedRobotCfgPPO.runner.max_iterations

    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "meta_wl_vmc"
