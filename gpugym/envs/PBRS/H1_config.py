"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gpugym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotCfgPPO


class H1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 38
        num_actions = 10
        episode_length_s = 5

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.
        heading_command = False
        ang_vel_command = True

        class ranges:
            # TRAINING COMMAND RANGES #
            lin_vel_x = [0, 4.5]        # min max [m/s] 只能往前走
            lin_vel_y = [-0.75, 0.75]   # min max [m/s] y 表示往左或者往右走
            ang_vel_yaw = [-2., 2.]     # min max [rad/s]
            heading = [0., 0.]
            
    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_range'
        penetration_check = False
        pos = [0., 0., 1.05]        # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],
            [0., 0.],
            [1.02, 1.02],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10]
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5]
        ]

        default_joint_angles = {
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': -0.2,
            'left_hip_pitch_joint': -0.2,
            'left_knee_joint': 0.25,  # 0.6
            'left_ankle_joint': 0.0,

            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': -0.2,
            'right_hip_pitch_joint': -0.2,
            'right_knee_joint': 0.25,  # 0.6
            'right_ankle_joint': 0.0,
        }

        # dof_pos_range = {
        #     'left_hip_yaw_joint': [0.0, 0.0],
        #     'left_hip_roll_joint': [-0.2, -0.2],
        #     'left_hip_pitch_joint': [-0.2, -0.2],
        #     'left_knee_joint': [0.25, 0.25],
        #     'left_ankle_joint': [0.0, 0.0],

        #     'right_hip_yaw_joint': [0.0, 0.0],
        #     'right_hip_roll_joint': [-0.2, -0.2],
        #     'right_hip_pitch_joint': [-0.2, -0.2],
        #     'right_knee_joint': [0.25, 0.25],
        #     'right_ankle_joint': [0.0, 0.0],
        # }

        dof_pos_range = {
            'left_hip_yaw_joint': [-0.1, 0.1],
            'left_hip_roll_joint': [-0.2, 0.2],
            'left_hip_pitch_joint': [-0.2, 0.2],
            'left_knee_joint': [0.6, 0.7],
            'left_ankle_joint': [-0.3, 0.3],

            'right_hip_yaw_joint': [-0.1, 0.1],
            'right_hip_roll_joint': [-0.2, 0.2],
            'right_hip_pitch_joint': [-0.2, 0.2],
            'right_knee_joint': [0.6, 0.7],
            'right_ankle_joint': [-0.3, 0.3],
        }

        dof_vel_range = {
            'left_hip_yaw_joint': [0.0, 0.0],
            'left_hip_roll_joint': [0.0, 0.0],
            'left_hip_pitch_joint': [0.0, 0.0],
            'left_knee_joint': [0.0, 0.0],
            'left_ankle_joint': [0.0, 0.0],

            'right_hip_yaw_joint': [0.0, 0.0],
            'right_hip_roll_joint': [0.0, 0.0],
            'right_hip_pitch_joint': [0.0, 0.0],
            'right_knee_joint': [0.0, 0.0],
            'right_ankle_joint': [0.0, 0.0],
        }


    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints

        overall_stiffness = 60.

        stiffness = {
            'left_hip_yaw_joint': overall_stiffness,
            'left_hip_roll_joint': overall_stiffness,
            'left_hip_pitch_joint': overall_stiffness,
            'left_knee_joint': overall_stiffness,
            'left_ankle_joint': overall_stiffness,
            'right_hip_yaw_joint': overall_stiffness,
            'right_hip_roll_joint': overall_stiffness,
            'right_hip_pitch_joint': overall_stiffness,
            'right_knee_joint': overall_stiffness,
            'right_ankle_joint': overall_stiffness,
        }

        overall_dampling = 5.

        damping = {
            'left_hip_yaw_joint': overall_dampling,
            'left_hip_roll_joint': overall_dampling,
            'left_hip_pitch_joint': overall_dampling,
            'left_knee_joint': overall_dampling,
            'left_ankle_joint': overall_dampling,
            'right_hip_yaw_joint': overall_dampling,
            'right_hip_roll_joint': overall_dampling,
            'right_hip_pitch_joint': overall_dampling,
            'right_knee_joint': overall_dampling,
            'right_ankle_joint': overall_dampling
        }


        action_scale = 1.0
        exp_avg_decay = None
        decimation = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]

        randomize_base_mass = False
        added_mass_range = [-1., 1.]

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.5

    class asset(LeggedRobotCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}'\
        #     '/resources/robots/mit_humanoid/mit_humanoid_fixed_arms.urdf'

        h1_urdf_version = 7

        # file = '{LEGGED_GYM_ROOT_DIR}'\
        #     '/resources/h1_robot_res/h1_v4.urdf'

        file = '{LEGGED_GYM_ROOT_DIR}'\
            f'/resources/h1_robot_res/h1_v{h1_urdf_version}.urdf'
        
        # file = '{LEGGED_GYM_ROOT_DIR}'\
        #     '/resources/h1_robot_res/h1_v6.urdf'

        print("-" * 50)
        print(f"Using H1 v{h1_urdf_version}")
        print("-" * 50)


        keypoints = ["pelvis"]

        end_effectors = ["left_ankle_link", "right_ankle_link"]
        foot_name = 'ankle_link'

        terminate_after_contacts_on = [
            # 'pelvis',
            # "left_knee_link",
            # "left_hip_pitch_link",
            # "right_knee_link",
            # "right_hip_pitch_link",

            "left_elbow_link",
            "left_shoulder_yaw_link",
            "right_elbow_link",
            "right_shoulder_yaw_link",
        ]

        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

    class rewards(LeggedRobotCfg.rewards):
        # ! "Incorrect" specification of height
        # base_height_target = 0.7
        base_height_target = 0.92
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False
        tracking_sigma = 0.5

        class scales(LeggedRobotCfg.rewards.scales):
            # * "True" rewards * #
            action_rate = -1.e-3
            action_rate2 = -1.e-4
            tracking_lin_vel = 10.
            tracking_ang_vel = 5.
            torques = -1e-4
            dof_pos_limits = -10
            torque_limits = -1e-2
            termination = -100

            # * Shaping rewards * #
            # Sweep values: [0.5, 2.5, 10, 25., 50.]
            # Default: 5.0
            # orientation = 5.0

            # Sweep values: [0.2, 1.0, 4.0, 10., 20.]
            # Default: 2.0
            # base_height = 2.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # joint_regularization = 1.0

            # * PBRS rewards * #
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            ori_pb = 1.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # baseHeight_pb = 1.0
            # baseHeight_pb = 10.0
            baseHeight_pb = 10.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # jointReg_pb = 1.0
            jointReg_pb = 10.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            base_z = 1./0.6565

        clip_observations = 100.
        clip_actions = 10.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            base_z = 0.05
            dof_pos = 0.005
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            in_contact = 0.1
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]

        class physx:
            max_depenetration_velocity = 10.0


class H1CfgPPO(LeggedRobotCfgPPO):
    do_wandb = True
    seed = -1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.e-5
        schedule = 'adaptive'   # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 24
        max_iterations = 1000
        run_name = 'ICRA2023'
        experiment_name = 'PBRS_HumanoidLocomotion'
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = 'elu'
