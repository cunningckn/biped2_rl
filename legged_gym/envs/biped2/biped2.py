# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

from isaacgym import gymtorch

from legged_gym.envs.cassie.cassie import Cassie


class Biped2(Cassie):
    def _init_buffers(self):
        super()._init_buffers()
        self.feet_contact_time = torch.zeros_like(self.feet_air_time)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.feet_contact_time[env_ids] = 0.

    def _update_feet_contact_timers(self):
        """Track per-foot air/contact duration with contact filtering."""
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.feet_contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt

    def _reward_feet_air_time(self):
        """Reward single-stance duration; min(swing/support time) encourages alternating steps."""
        self._update_feet_contact_timers()

        cfg = self.cfg.rewards
        in_contact = self.last_contacts
        single_stance = torch.sum(in_contact.int(), dim=1) == 1

        in_mode_time = torch.where(in_contact, self.feet_contact_time, self.feet_air_time)
        mode_time = torch.where(
            single_stance.unsqueeze(-1),
            in_mode_time,
            torch.zeros_like(in_mode_time),
        )
        reward = torch.min(mode_time, dim=1)[0]
        reward = torch.clamp(reward, max=cfg.feet_air_time_threshold)
        reward *= torch.norm(self.commands[:, :2], dim=1) > cfg.feet_air_time_min_cmd_speed
        return reward

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.
        noise_vec[12:22] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[22:32] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[32:42] = 0.
        return noise_vec

    def _foot_contacts(self):
        th = self.cfg.rewards.gait.contact_threshold
        return self.contact_forces[:, self.feet_indices, 2] > th

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(contacts.float(), dim=1) == 1
        active = torch.norm(self.commands[:, :2], dim=1) > self.cfg.rewards.no_fly_min_cmd_speed
        return single_contact.float() * active

    def _reward_low_speed_upright(self):
        """Reward tracking default joint pose when commanded speed is low."""
        cfg = self.cfg.rewards
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = cmd_speed < cfg.low_speed_upright_max_cmd_speed
        pose_error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        upright = torch.exp(-pose_error / cfg.low_speed_upright_sigma)
        return upright * active.float()


class Biped2Gait(Biped2):
    """Biped2 with phase-based gait rewards."""

    @staticmethod
    def gait_clock(phase, air_ratio, delta_t):
        """Return swing-phase mask I_frc and support-phase mask I_spd."""
        swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
        stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))
        trans_flag1 = phase < delta_t
        trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
        trans_flag3 = phase > (1 - delta_t)

        I_frc = (
            1.0 * swing_flag
            + (0.5 + phase / (2 * delta_t)) * trans_flag1
            - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
            + 0.0 * stand_flag
            + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
        )
        I_spd = 1.0 - I_frc
        return I_frc, I_spd

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)

        self.gait_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self.avg_feet_force_per_step = torch.zeros(self.num_envs, 2, device=self.device)
        self.avg_feet_speed_per_step = torch.zeros(self.num_envs, 2, device=self.device)
        self._feet_force_accum = torch.zeros(self.num_envs, 2, device=self.device)
        self._feet_speed_accum = torch.zeros(self.num_envs, 2, device=self.device)

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        decimation = self.cfg.control.decimation
        self._feet_force_accum.zero_()
        self._feet_speed_accum.zero_()
        for _ in range(decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self._accumulate_feet_metrics()
        self.avg_feet_force_per_step[:] = self._feet_force_accum / decimation
        self.avg_feet_speed_per_step[:] = self._feet_speed_accum / decimation
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _accumulate_feet_metrics(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        foot_speeds = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:10], dim=-1)
        self._feet_force_accum[:, 0] += foot_forces[:, self._left_foot_idx]
        self._feet_force_accum[:, 1] += foot_forces[:, self._right_foot_idx]
        self._feet_speed_accum[:, 0] += foot_speeds[:, self._left_foot_idx]
        self._feet_speed_accum[:, 1] += foot_speeds[:, self._right_foot_idx]

    def _create_envs(self):
        super()._create_envs()
        body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        left_idx = None
        right_idx = None
        for i, idx in enumerate(self.feet_indices):
            name = body_names[int(idx)]
            if 'Left' in name:
                left_idx = i
            elif 'Right' in name:
                right_idx = i
        if left_idx is None or right_idx is None:
            raise ValueError('Could not identify left/right foot bodies for gait rewards.')
        self._left_foot_idx = left_idx
        self._right_foot_idx = right_idx

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        g = self.cfg.rewards.gait
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        max_cmd_speed = max(
            abs(self.command_ranges['lin_vel_x'][0]),
            abs(self.command_ranges['lin_vel_x'][1]),
        )
        speed_ratio = (cmd_speed / max_cmd_speed).clip(0., 1.5)
        t = self.episode_length_buf.float() * self.dt * speed_ratio
        self.gait_phase[:, 0] = (t / g.cycle + g.phase_offset_l) % 1.0
        self.gait_phase[:, 1] = (t / g.cycle + g.phase_offset_r) % 1.0

    def _gait_masks(self):
        g = self.cfg.rewards.gait
        I_frc_l, I_spd_l = self.gait_clock(self.gait_phase[:, 0], g.air_ratio_l, g.clock_delta_t)
        I_frc_r, I_spd_r = self.gait_clock(self.gait_phase[:, 1], g.air_ratio_r, g.clock_delta_t)
        return I_frc_l, I_spd_l, I_frc_r, I_spd_r

    def _reward_gait_contact(self):
        g = self.cfg.rewards.gait
        contacts = self._foot_contacts()
        _, I_spd_l, _, I_spd_r = self._gait_masks()
        desired = torch.stack([
            I_spd_r > 0.5,
            I_spd_l > 0.5,
        ], dim=1)
        match = (contacts == desired).float().mean(dim=1)
        return match * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_gait_feet_frc_perio(self):
        """Swing phase: foot force should be near zero."""
        g = self.cfg.rewards.gait
        I_frc_l, _, I_frc_r, _ = self._gait_masks()
        force = self.avg_feet_force_per_step
        rew_l = I_frc_l * torch.exp(-g.frc_swing_gain * torch.square(force[:, 0]))
        rew_r = I_frc_r * torch.exp(-g.frc_swing_gain * torch.square(force[:, 1]))
        return (rew_l + rew_r) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_gait_feet_spd_perio(self):
        """Support phase: foot speed should be near zero."""
        g = self.cfg.rewards.gait
        _, I_spd_l, _, I_spd_r = self._gait_masks()
        speed = self.avg_feet_speed_per_step
        rew_l = I_spd_l * torch.exp(-g.spd_support_gain * torch.square(speed[:, 0]))
        rew_r = I_spd_r * torch.exp(-g.spd_support_gain * torch.square(speed[:, 1]))
        return (rew_l + rew_r) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_gait_feet_frc_support_perio(self):
        """Support phase: foot should carry load."""
        g = self.cfg.rewards.gait
        _, I_spd_l, _, I_spd_r = self._gait_masks()
        force = self.avg_feet_force_per_step
        rew_l = I_spd_l * (1.0 - torch.exp(-g.frc_support_gain * torch.square(force[:, 0])))
        rew_r = I_spd_r * (1.0 - torch.exp(-g.frc_support_gain * torch.square(force[:, 1])))
        return (rew_l + rew_r) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_feet_air_time(self):
        """Biped single-stance reward gated by gait speed weight."""
        self._update_feet_contact_timers()

        cfg = self.cfg.rewards
        in_contact = self.last_contacts
        single_stance = torch.sum(in_contact.int(), dim=1) == 1

        in_mode_time = torch.where(in_contact, self.feet_contact_time, self.feet_air_time)
        mode_time = torch.where(
            single_stance.unsqueeze(-1),
            in_mode_time,
            torch.zeros_like(in_mode_time),
        )
        reward = torch.min(mode_time, dim=1)[0]
        reward = torch.clamp(reward, max=cfg.feet_air_time_threshold)
        reward *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        return reward
