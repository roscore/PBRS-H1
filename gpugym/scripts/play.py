from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch

from tqdm import tqdm


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False #True
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    # env_cfg.init_state.reset_ratio = 0.8 # seems useless

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy(ppo_runner.alg.actor_critic, path)
        print('Exported policy model to: ', path)

    # export critic as a jit module (used to run it from C++)
    if EXPORT_CRITIC:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'critics')
        export_critic(ppo_runner.alg.actor_critic, path)
        print('Exported critic model to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    # stop_state_log = 1000  # number of steps before plotting states
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    

    # camera_position = np.array([-7, -7, 3], dtype=np.float64)
    # camera_lookat = np.array([5., 5, 2.], dtype=np.float64)

    # cam_pos_x = 14
    # cam_tgt_x = cam_pos_x - 12

    camera_position = np.array([12, -3, 1.5], dtype=np.float64)
    camera_lookat = np.array([0, 7, 0], dtype=np.float64)

    camera_direction = camera_lookat - camera_position
    camera_vel = np.array([1, 0, 0.], dtype=np.float64)

    env.set_camera(camera_position, camera_position + camera_direction)
    
    img_idx = 0
    # img dir
    wandb_name = args.wandb_name

    img_dir_name = wandb_name
    if args.img_dir_name is not None:
        img_dir_name = args.img_dir_name

    img_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "imgs", img_dir_name)
    os.makedirs(img_dir, exist_ok=True)

    play_log = []
    env.max_episode_length = 1000./env.dt
    for i in tqdm(range(10*int(env.max_episode_length))):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(img_dir, f"{str(img_idx).zfill(3)}.png")

                # filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")

                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

                # terminate when img idx == 200
                if img_idx == 200:
                    print("img idx is 200")
                    break

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            ### Humanoid PBRS Logging ###
            # [ 1]  Timestep
            # [38]  Agent observations
            # [10]  Agent actions (joint setpoints)
            # [13]  Floating base states in world frame
            # [ 6]  Contact forces for feet
            # [10]  Joint torques
            play_log.append(
                [i*env.dt]
                + obs[robot_index, :].cpu().numpy().tolist()
                + actions[robot_index, :].detach().cpu().numpy().tolist()
                + env.root_states[0, :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env.torques[robot_index, :].detach().cpu().numpy().tolist()
            )
        elif i==stop_state_log:
            # np.savetxt('../analysis/data/play_log.csv', play_log, delimiter=',')
            # logger.plot_states()
            pass

        
        if  0 < i < stop_rew_log:

            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)

        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    EXPORT_CRITIC = False
    RECORD_FRAMES = True
    MOVE_CAMERA = True
    args = get_args()
    play(args)
