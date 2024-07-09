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
        num_privileged_obs = 142
        num_actions = len(ActionIdx)  # YXC: 4
        # YXC: remember to update policy accordingly

    # class terrain(LeggedRobotCfg.terrain):
    #     mesh_type = "plane"

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/meta_wl/urdf/meta_wl.urdf"
        name = "meta_wl"
        penalize_contacts_on = ["shin", "thigh", "base"]
        # terminate_after_contacts_on = ["base"]
        self_collisions = 0
        flip_visual_attachments = False

        # YXC: make sure the following parameters are consistent with the URDF file
        origin_to_hip = 0.022  # [m]
        thigh_len = 0.157  # [m]
        shin_len = 0.143  # [m]

        hip_offset = 1.3090
        knee_offset = 1.6515

        theta_offset = -(torch.pi / 2 - 0.973)
        leg_offset = 0.15

    class control(LeggedRobotCfg.control):
        action_scale_leg = 0.2
        action_scale_wheel = 17.0

        # PD Drive parameters:
        stiffness = {"hip": 5.0, "knee": 5.0, "wheel": 0}  # [N*m/rad]
        damping = {"hip": 0.2, "knee": 0.2, "wheel": 0.5}  # [N*m*s/rad]

        kp_leg = 900.0  # [N*m/rad]
        kd_leg = 5.0  # [N*m*s/rad]
        kp_theta = 10.0  # [N*m/rad]
        kd_theta = 0.0  # [N*m*s/rad]

        feedforward_force = 0.0  # [N]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
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

    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.0

            base_height = 5.0
            nominal_state = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -4.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.0001
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_single_reward = 1
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.30
        max_contact_force = 100.0  # forces above this value are penalized

class MetaWLVMCCfgPPO(LeggedRobotCfgPPO):
    seed = 10
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
