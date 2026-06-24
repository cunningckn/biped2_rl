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
        pos = [0.0, 0.0, 0.45]
        default_joint_angles = {
            'RightHipYaw': 0.,
            'RightHipRoll': 0.,
            'RightHipPitch': -0.0, #-0.1
            'RightKneePitch': -0.0,
            'RightAnklePitch': 0.0,
            'LeftHipYaw': 0.,
            'LeftHipRoll': 0.,
            'LeftHipPitch': 0.0, #0.1
            'LeftKneePitch': -0.0,
            'LeftAnklePitch': 0.0,
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

    # class commands(CassieRoughCfg.commands):
    #     heading_command = False
    #     resampling_time = 4.0

    #     class ranges(CassieRoughCfg.commands.ranges):
    #         lin_vel_x = [-0.4, 0.0]
    #         lin_vel_y = [-0.0, 0.0]
    #         ang_vel_yaw = [-0.4, 0.4]

    class domain_rand(CassieRoughCfg.domain_rand):
        push_robots = False
        randomize_friction = True
        friction_range = [0.5, 1.25]

    class rewards(CassieRoughCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.45
        soft_dof_pos_limit = 0.95
        no_fly_min_cmd_speed = 0.1  # no_fly 仅在 ||cmd_xy|| > 此值时生效 [m/s]
        low_speed_upright_max_cmd_speed = 0.1  # 低于此速度时奖励站直 [m/s]
        low_speed_upright_sigma = 0.25  # 默认姿态跟踪误差 exp 衰减系数
        feet_air_time_threshold = 0.5       # 单腿支撑奖励上限 [s]
        feet_air_time_min_cmd_speed = 0.1   # 低于此速度不奖励迈步 [m/s]
        feet_contact_threshold = 0.1        # 判定着地的接触力阈值 [N]

        class scales(CassieRoughCfg.rewards.scales):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 10
            dof_pos_limits = -1.
            no_fly = 0.25
            low_speed_upright = 1.5
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            action_rate = -0.03
            collision = 0

            tracking_lin_vel = 3.0
            tracking_ang_vel = 2.0

class Biped2FlatCfgPPO(CassieRoughCfgPPO):
    class policy(CassieRoughCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class runner(CassieRoughCfgPPO.runner):
        run_name = 'upright'
        experiment_name = 'flat_biped2'
        max_iterations = 4000
        save_interval = 250


class Biped2GaitFlatCfg(Biped2FlatCfg):
    class rewards(Biped2FlatCfg.rewards):
        class gait:
            cycle = 1.00              # 步态周期 [s]
            air_ratio_l = 0.38        # 左腿摆动相占比
            air_ratio_r = 0.38        # 右腿摆动相占比
            phase_offset_l = 0.38     # 左腿相位偏移
            phase_offset_r = 0.88       # 右腿相位偏移
            clock_delta_t = 0.02      # 相位边界平滑宽度
            contact_threshold = 1.0   # 判定着地的接触力阈值
            min_cmd_speed = 0.1       # 低于此速度不施加步态奖励
            frc_swing_gain = 200.     # 摆动相-力奖励指数系数
            spd_support_gain = 100.   # 支撑相-速度奖励指数系数
            frc_support_gain = 10.    # 支撑相-力奖励指数系数

        class scales(Biped2FlatCfg.rewards.scales):
            no_fly = 0.0
            gait_contact = 0.0
            gait_feet_frc_perio = 1.0
            gait_feet_spd_perio = 1.0
            gait_feet_frc_support_perio = 1.0
            feet_air_time = 2.0


class Biped2GaitFlatCfgPPO(Biped2FlatCfgPPO):
    class runner(Biped2FlatCfgPPO.runner):
        run_name = 'gait'
        experiment_name = 'flat_biped2_gait'

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
