from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.utils.terrain import Terrain
from wheel_legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from wheel_legged_gym.utils.helpers import class_to_dict
from .meta_wl_vmc_config import MetaWLVMCCfg, ActionIdx, JointIdx


class MetaWLVMC(LeggedRobot):
    def __init__(
            self, cfg: MetaWLVMCCfg, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self._leg_post_physics_step()
            self.envs_steps_buf += 1
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

    def _leg_post_physics_step(self):
        # YXC: store feedback info to the buffer
        self.hip_pos = torch.cat(
            (self.dof_pos[:, JointIdx.l_hip].unsqueeze(1), self.dof_pos[:, JointIdx.r_hip].unsqueeze(1)), dim=1
        )  # YXC: TODO: add to init_buffers
        self.knee_pos = torch.cat(
            (self.dof_pos[:, JointIdx.l_knee].unsqueeze(1), self.dof_pos[:, JointIdx.r_knee].unsqueeze(1)), dim=1
        )
        self.hip_vel = torch.cat(
            (self.dof_vel[:, JointIdx.l_hip].unsqueeze(1), self.dof_vel[:, JointIdx.r_hip].unsqueeze(1)), dim=1
        )
        self.knee_vel = torch.cat(
            (self.dof_vel[:, JointIdx.l_knee].unsqueeze(1), self.dof_vel[:, JointIdx.r_knee].unsqueeze(1)), dim=1
        )

    def _compute_proprioception_observations(self):
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.dof_pos[:, [JointIdx.l_hip, JointIdx.r_hip]] * self.obs_scales.dof_pos,
                self.dof_vel[:, [JointIdx.l_hip, JointIdx.r_hip]] * self.obs_scales.dof_vel,
                self.dof_vel[:, [JointIdx.l_wheel, JointIdx.r_wheel]] * self.obs_scales.dof_vel,
                self.actions
            ),
            dim=-1,
        )
        # YXC: 3 + 3 + 3 + 2 + 2 + 2 + 4 = 19
        return obs_buf

    def compute_observations(self):
        self.obs_buf = self._compute_proprioception_observations()

        if self.cfg.env.num_privileged_obs is not None:
            heights = (
                    torch.clip(
                        self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                        -1,
                        1.0,
                    )
                    * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.obs_buf,
                    self.last_actions[:, :, 0],
                    self.last_actions[:, :, 1],
                    self.dof_acc * self.obs_scales.dof_acc,
                    self.dof_pos * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    heights,
                    self.torques * self.obs_scales.torque,
                    (self.base_mass - self.base_mass.mean()).view(self.num_envs, 1),
                    self.base_com,
                    self.default_dof_pos - self.raw_default_dof_pos,
                    self.friction_coef.view(self.num_envs, 1),
                    self.restitution_coef.view(self.num_envs, 1),
                ),
                dim=-1,
            )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs:], self.obs_buf), dim=-1
        )

    def _compute_knee(self, hip):
        l1 = self.cfg.asset.l1
        l2 = self.cfg.asset.l2
        l3 = self.cfg.asset.l3
        l4 = self.cfg.asset.l4
        diagonal = l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * torch.cos(hip)
        diagonal = torch.sqrt(diagonal)
        angle1 = torch.acos((l4**2 + diagonal**2 - l3**2) / (2 * l4 * diagonal))
        angle2 = torch.acos((l2**2 + diagonal**2 - l1**2) / (2 * l2 * diagonal))
        return self.pi - angle1 - angle2

    def _compute_torques(self, actions):
        # YXC: hip uses position PD control according to the action
        hip_goal_pos = torch.cat(
            (
                actions[:, ActionIdx.l_leg].unsqueeze(1),
                actions[:, ActionIdx.r_leg].unsqueeze(1),
            ), axis=1,
        ) * self.cfg.control.action_scale_pos

        hip_j = [JointIdx.l_hip, JointIdx.r_hip]
        hip_torques = (self.p_gains[:, hip_j] * (hip_goal_pos - self.dof_pos[:, hip_j])
                       - self.d_gains[:, hip_j] * self.dof_vel[:, hip_j])

        # YXC: knee uses position PD control according to hip position
        knee_goal_pos = self._compute_knee(hip_goal_pos)
        knee_j = [JointIdx.l_knee, JointIdx.r_knee]
        knee_torques = (self.p_gains[:, knee_j] * (knee_goal_pos - self.dof_pos[:, knee_j])
                        - self.d_gains[:, knee_j] * self.dof_vel[:, knee_j])

        # YXC: wheel uses velocity PD control according to the action
        wheel_goal_vel = torch.cat(
            (
                actions[:, ActionIdx.l_wheel].unsqueeze(1),
                actions[:, ActionIdx.r_wheel].unsqueeze(1),
            ), axis=1,
        ) * self.cfg.control.action_scale_vel
        wheel_j = [JointIdx.l_wheel, JointIdx.r_wheel]
        wheel_torques = (self.p_gains[:, wheel_j] * (wheel_goal_vel - self.dof_vel[:, wheel_j])
                         - self.d_gains[:, wheel_j] * (
                                 self.dof_vel[:, wheel_j] - self.last_dof_vel[:, wheel_j]) / self.sim_params.dt)

        torque = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        torque[:, hip_j] = hip_torques
        torque[:, knee_j] = knee_torques
        torque[:, wheel_j] = wheel_torques
        return torque

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3]
        self.last_base_position = self.base_position.clone()
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands + 1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.height_measurements,
            ],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.command_ranges["lin_vel_x"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_x"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x
        )
        self.command_ranges["ang_vel_yaw"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["ang_vel_yaw"][:] = torch.tensor(
            self.cfg.commands.ranges.ang_vel_yaw
        )
        self.command_ranges["height"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["height"][:] = torch.tensor(self.cfg.commands.ranges.height)
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
        )
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        self.hip_pos = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.knee_pos = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.hip_vel = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.knee_vel = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )

        if self.cfg.domain_rand.randomize_Kp:
            (
                p_gains_scale_min,
                p_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.p_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_Kd:
            (
                d_gains_scale_min,
                d_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.d_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_motor_torque:
            (
                torque_scale_min,
                torque_scale_max,
            ) = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                torque_scale_min,
                torque_scale_max,
                self.torques_scale.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_action_delay:
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()
