import numpy as np
import os
from datetime import datetime

import isaacgym
from gpugym.envs import *
from gpugym.utils import get_args, task_registry, wandb_helper
from gpugym import LEGGED_GYM_ROOT_DIR
import torch

import wandb

def save_conf(exp_name):
    # save config file
    save_conf_base = "/mnt/hypercube/zhsha/workspace/pbrs-humanoid/config_bakcup"
    save_conf_dir = os.path.join(save_conf_base, exp_name)

    conf_to_save_ls = [
        "/mnt/hypercube/zhsha/workspace/pbrs-humanoid/gpugym/envs/PBRS"
    ]

    for conf_dir in conf_to_save_ls:
        os.system("cp -r {} {}".format(conf_dir, save_conf_dir))

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    do_wandb = False

    if args.wandb_name:
        experiment_name = args.wandb_name
        do_wandb = True
    else:
        experiment_name = f'{args.task}'

    save_conf(experiment_name)

    # log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    # log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

    # # Check if we specified that we want to use wandb
    # do_wandb = train_cfg.do_wandb if hasattr(train_cfg, 'do_wandb') else False
    # # Do the logging only if wandb requirements have been fully specified
    # do_wandb = do_wandb and None not in (args.wandb_project, args.wandb_entity)

    if do_wandb:
        wandb.config = {}

        if hasattr(train_cfg, 'wandb'):
            what_to_log = train_cfg.wandb.what_to_log
            wandb_helper.craft_log_config(env_cfg, train_cfg, wandb.config, what_to_log)

        # print(f'Received WandB project name: {args.wandb_project}\nReceived WandB entitiy name: {args.wandb_entity}\n')
        wandb.init(project=args.wandb_project,
                #    entity=args.wandb_entity,
                #    group=args.wandb_group,
                   config=args,
                   name=experiment_name)

        ppo_runner.configure_wandb(wandb)
        ppo_runner.configure_learn(train_cfg.runner.max_iterations, True)
        ppo_runner.learn()

        wandb.finish()
    else:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
