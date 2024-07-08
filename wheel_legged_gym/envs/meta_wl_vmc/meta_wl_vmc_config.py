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
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = False

        # YXC: make sure the following parameters are consistent with the URDF file
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
        stiffness = {"hip": 5.0, "knee": 3.0, "wheel": 0}  # [N*m/rad]
        damping = {"hip": 0.5, "knee": 0.5, "wheel": 0.2}  # [N*m*s/rad]

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
