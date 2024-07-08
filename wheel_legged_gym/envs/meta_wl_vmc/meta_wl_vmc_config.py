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
        num_envs = 1024
        num_observations = 18  # YXC: defined in _compute_proprioception_observations()
        num_privileged_obs = 142 - 6
        num_actions = len(ActionIdx)  # YXC: 4
        # YXC: remember to update policy accordingly

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/meta_wl/urdf/meta_wl.urdf"
        name = "meta_wl"
        penalize_contacts_on = ["shin", "thigh", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0
        flip_visual_attachments = False
        default_dof_drive_mode = 2  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)

        # YXC: make sure the following parameters are consistent with the URDF file
        # links
        l1 = 0.07  # fixed link
        l2 = 0.1566  # driving link
        l3 = 0.1447  # follower link
        l4 = 0.0375  # output link
        l5 = 0.180 - l4  # output link extension

    class control(LeggedRobotCfg.control):
        action_joint_scale = 2.0
        action_wheel_scale = 30.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]  # x,y,z [m]
        # rot = [-0.2, 0, 0, 0.96]  # quaternion [x,y,z,w]
        init_hip = 0
        init_knee = 0
        default_joint_angles = {
            "l_hip": init_hip,
            "l_knee": init_knee,
            "l_wheel": 0.0,
            "r_hip": init_hip,
            "r_knee": init_knee,
            "r_wheel": 0.0,
        }

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2.0
        randomize_Kp = True
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = True
        randomize_Kd_range = [0.9, 1.1]
        # randomize_motor_torque = True
        # randomize_motor_torque_range = [0.9, 1.1]
        randomize_motor_velocity = True
        randomize_motor_velocity_range = [0.9, 1.1]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        delay_ms_range = [0, 10]


class MetaWLVMCCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # only for ActorCriticSequence
        num_encoder_obs = (
                MetaWLVMCCfg.env.obs_history_length * MetaWLVMCCfg.env.num_observations
        )
        latent_dim = 3  # at least 3 to estimate base linear velocity
        encoder_hidden_dims = [128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        kl_decay = (
                           LeggedRobotCfgPPO.algorithm.desired_kl - 0.002
                   ) / LeggedRobotCfgPPO.runner.max_iterations

    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "meta_wl_vmc"
        policy_class_name = (
            "ActorCriticSequence"  # could be ActorCritic, ActorCriticSequence
        )
        algorithm_class_name = "PPO"
        num_steps_per_env = 48  # per iteration
        max_iterations = 5000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
