
import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi, gymutil

from gpugym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):

    # hardcode fixed seed
    seed = 777

    # if seed == -1:
    #     seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = sorted(os.listdir(root),
                        key=lambda x: os.path.getctime(os.path.join(root, x)))
        #TODO sort by date to handle change of month
        # runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # this is a general function for updating both env_cfg and train_cfg
    # the program will first create env, therefore, env_cfg is always created first
    # after that, train_cfg is created


    # update env cfg from args
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs

        # more arugemnt
        if args.task == "pbrs:H1":

            print("-" * 50)

            # urdf version
            assert args.h1_urdf_version is not None, "Please specify the urdf version of H1 robot"
            env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}'\
                f'/resources/h1_robot_res/h1_v{args.h1_urdf_version}.urdf'
            print(env_cfg.asset.file)

            # action scale
            assert args.action_scale is not None, "Please specify the action scale of H1 robot"
            env_cfg.control.action_scale = args.action_scale
            print(f"action scale: {env_cfg.control.action_scale}")

            # gravity reset threshold
            assert args.ori_term_threshold is not None, "Please specify the orientation term threshold of H1 robot"
            env_cfg.rewards.ori_term_threshold = args.ori_term_threshold
            print(f"orientation term threshold: {env_cfg.rewards.ori_term_threshold}")

            # lin x range
            assert args.lin_vel_x_min is not None and args.lin_vel_x_max is not None, "Please specify the linear velocity range of H1 robot"
            env_cfg.commands.ranges.lin_vel_x = [args.lin_vel_x_min, args.lin_vel_x_max]
            print(f"lin x range: {env_cfg.commands.ranges.lin_vel_x}")

            # lin y range
            assert args.lin_vel_y_abs is not None, "Please specify the absolute value of linear velocity in y direction of H1 robot"
            env_cfg.commands.ranges.lin_vel_y = [-args.lin_vel_y_abs, args.lin_vel_y_abs]
            print(f"lin y range: {env_cfg.commands.ranges.lin_vel_y}")

            # ang yaw range
            assert args.ang_vel_yaw_abs is not None, "Please specify the absolute value of angular velocity in yaw direction of H1 robot"
            env_cfg.commands.ranges.ang_vel_yaw = [-args.ang_vel_yaw_abs, args.ang_vel_yaw_abs]
            print(f"ang yaw range: {env_cfg.commands.ranges.ang_vel_yaw}")

            # ankle stiffness
            assert args.ankle_stiffness is not None, "Please specify the ankle stiffness of H1 robot"
            env_cfg.control.stiffness["left_ankle_joint"] = args.ankle_stiffness
            env_cfg.control.stiffness["right_ankle_joint"] = args.ankle_stiffness
            print(f"ankle stiffness: {env_cfg.control.stiffness['left_ankle_joint']}")

            # hip stiffness
            assert args.hip_pitch_stiffness is not None, "Please specify the hip stiffness of H1 robot"
            env_cfg.control.stiffness["left_hip_pitch_joint"] = args.hip_pitch_stiffness
            env_cfg.control.stiffness["right_hip_pitch_joint"] = args.hip_pitch_stiffness
            print(f"hip stiffness: {env_cfg.control.stiffness['left_hip_pitch_joint']}")

            print("-" * 50)


        # we also need to parser argument for mit robot later
        elif args.task == "pbrs:humanoid":
            print("-" * 50)

             # lin x range
            assert args.lin_vel_x_min is not None and args.lin_vel_x_max is not None, "Please specify the linear velocity range of H1 robot"
            env_cfg.commands.ranges.lin_vel_x = [args.lin_vel_x_min, args.lin_vel_x_max]
            print(f"lin x range: {env_cfg.commands.ranges.lin_vel_x}")

            # lin y range
            assert args.lin_vel_y_abs is not None, "Please specify the absolute value of linear velocity in y direction of H1 robot"
            env_cfg.commands.ranges.lin_vel_y = [-args.lin_vel_y_abs, args.lin_vel_y_abs]
            print(f"lin y range: {env_cfg.commands.ranges.lin_vel_y}")

            # ang yaw range
            assert args.ang_vel_yaw_abs is not None, "Please specify the absolute value of angular velocity in yaw direction of H1 robot"
            env_cfg.commands.ranges.ang_vel_yaw = [-args.ang_vel_yaw_abs, args.ang_vel_yaw_abs]
            print(f"ang yaw range: {env_cfg.commands.ranges.ang_vel_yaw}")

            print("-" * 50)

        # env_cfg.asset.file

    # update train 
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--wandb_name", "type": str, "help": "Enter the group name of the runs."},
        {"name": "--wandb_project", "type": str, "help": "Enter the name of your project for better WandB tracking."},
        {"name": "--wandb_entity", "type": str, "help": "Enter your wandb entity username to track your experiment on your account."},
        {"name": "--wandb_group", "type": str, "default": "training_run", "help": "Enter the group name of the runs."},
        {"name": "--reward_scale", "type": float, "help": "value to override reward scale with (which reward hard-coded in train.py)"}, # ! hacky AF
        {"name": "--pbrs", "type": int, "help": "pbrs or not (1, 0))"}, # ! hacky AF
        # argument by szz
        {"name":"--h1_urdf_version", "type": int, "default": None},
        {"name":"--action_scale", "type" : float, "default" : None},
        {"name":"--ori_term_threshold", "type" : float, "default" : None},
        {"name":"--lin_vel_x_min", "type" :float, "default" : None},
        {"name":"--lin_vel_x_max", "type" :float, "default" : None},
        {"name" : "--lin_vel_y_abs", "type" : float, "default" : None},
        {"name" : "--ang_vel_yaw_abs", "type" : float, "default" : None},
        {"name" : "--ankle_stiffness", "type" : float, "default" : None},
        {"name" : "--hip_pitch_stiffness", "type" : float, "default" : None},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy(actor_critic, path):
    # setup file paths for saving policy modules
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)                                            # setup file paths for saving critic modules
        file_path_pt = os.path.join(path, 'policy_0.pt')
        file_path_jit = os.path.join(path, 'policy_0.jit')
        file_path_onnx = os.path.join(path, 'policy_0.onnx')
        i = 0
        while os.path.exists(file_path_jit):
            i += 1
            file_path_pt = os.path.join(path, f'policy_{i}.pt')
            file_path_jit = os.path.join(path, f'policy_{i}.jit')
            file_path_onnx = os.path.join(path, f'policy_{i}.onnx')
        model = copy.deepcopy(actor_critic.actor).to('cpu')                         # copy model to CPU for saving
        torch.save(model, file_path_pt)                                             # save model as Pytorch module
        traced_script_module = torch.jit.script(model)                              # save model as jitted module
        traced_script_module.save(file_path_jit)
        batch_size = 1
        actor_input = torch.rand(batch_size, actor_critic.num_actor_inputs, device='cuda')
        torch.onnx.export(actor_critic.actor, actor_input, file_path_onnx,          # save model as onnx module
            do_constant_folding=True,
            input_names = ['actor_observations'],
            output_names = ['actions'],
            dynamic_axes={
                'actor_observations' : {0 : 'batch_size'},
                'actions' : {0 : 'batch_size'}}
            )


def export_critic(actor_critic, path):
    os.makedirs(path, exist_ok=True)                                                # setup file paths for saving critic modules
    file_path_pt = os.path.join(path, 'critic_0.pt')
    file_path_jit = os.path.join(path, 'critic_0.jit')
    file_path_onnx = os.path.join(path, 'critic_0.onnx')
    i = 0
    while os.path.exists(file_path_jit):
        i += 1
        file_path_pt = os.path.join(path, f'critic_{i}.pt')
        file_path_jit = os.path.join(path, f'critic_{i}.jit')
        file_path_onnx = os.path.join(path, f'critic_{i}.onnx')
    model = copy.deepcopy(actor_critic.critic).to('cpu')                            # copy model to CPU for saving
    torch.save(model, file_path_pt)                                                 # save model as Pytorch module
    traced_script_module = torch.jit.script(model)                                  # save model as jitted module
    traced_script_module.save(file_path_jit)
    batch_size = 1
    critic_input = torch.rand(batch_size, actor_critic.num_critic_inputs, device='cuda')
    torch.onnx.export(actor_critic.critic, critic_input, file_path_onnx,            # save model as onnx module
        do_constant_folding=True,
        input_names = ['critic_observations'],
        output_names = ['value'],
        dynamic_axes={
            'critic_observations' : {0 : 'batch_size'},
            'value' : {0 : 'batch_size'}}
        )

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
