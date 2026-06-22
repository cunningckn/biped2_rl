# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO


class Biped2FlatCfg(CassieRoughCfg):
    class env(CassieRoughCfg.env):
        num_envs = 4096
        num_observations = 42
        num_actions = 10

    class terrain(CassieRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(CassieRoughCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        default_joint_angles = {
            'RightHipYaw': 0.,
            'RightHipRoll': 0.,
            'RightHipPitch': 0.30,
            'RightKneePitch': -0.55,
            'RightAnklePitch': 0.25,
            'LeftHipYaw': 0.,
            'LeftHipRoll': 0.,
            'LeftHipPitch': 0.30,
            'LeftKneePitch': -0.55,
            'LeftAnklePitch': 0.25,
        }

    class control(CassieRoughCfg.control):
        stiffness = {
            'HipYaw': 100.0,
            'HipRoll': 100.0,
            'HipPitch': 200.0,
            'KneePitch': 200.0,
            'AnklePitch': 40.0,
        }
        damping = {
            'HipYaw': 3.0,
            'HipRoll': 3.0,
            'HipPitch': 6.0,
            'KneePitch': 6.0,
            'AnklePitch': 1.0,
        }
        action_scale = 0.25
        decimation = 4

    class asset(CassieRoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/biped2/urdf/biped2.urdf'
        name = 'biped2'
        foot_name = 'AnklePitch'
        terminate_after_contacts_on = ['base_link', 'HipRoll', 'HipPitch']
        # penalize_contacts_on = ['base_link', 'HipPitch', 'KneePitch', 'HipRoll']
        flip_visual_attachments = False
        self_collisions = 1

    class commands(CassieRoughCfg.commands):
        heading_command = False
        resampling_time = 4.0

        class ranges(CassieRoughCfg.commands.ranges):
            lin_vel_x = [0.0, 0.4]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.4, 0.4]

    class domain_rand(CassieRoughCfg.domain_rand):
        push_robots = False
        randomize_friction = True
        friction_range = [0.5, 1.25]

    class rewards(CassieRoughCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.6
        soft_dof_pos_limit = 0.95

        class scales(CassieRoughCfg.rewards.scales):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.


class Biped2FlatCfgPPO(CassieRoughCfgPPO):
    class policy(CassieRoughCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class runner(CassieRoughCfgPPO.runner):
        run_name = 'upright'
        experiment_name = 'flat_biped2'
        max_iterations = 10000
        save_interval = 500
