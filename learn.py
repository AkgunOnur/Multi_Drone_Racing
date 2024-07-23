"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch

from train_ppo import PPO
from common.env_util import make_vec_env
from common.callbacks_updated import EvalCallback, StopTrainingOnRewardThreshold
from common.evaluation import evaluate_policy
from common.monitor import Monitor
from common.vec_env.vec_monitor import VecMonitor
import torch as th

import sys
sys.path.append('/home/onur/Downloads/MultiDrone/gym_drones')
sys.path.append('/home/onur/Downloads/MultiDrone/sb3_selfplay')


from gym_pybullet_drones.envs.MultiGates_SB import MultiGates
from envs.MultiGates_SelfPlay_v0 import MultiGates_v0
from envs.MultiGates_SelfPlay_v1 import MultiGates_v1
# from gym_pybullet_drones.envs.MultiGatesCont import MultiGatesCont
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('vel') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
ALGO = PPO
DISCRETE_ACTION = True
DUMB_NO_MODEL = False
ENV_NAME = MultiGates_v1
N_ENVS = 4
MAX_TIMESTEPS = 5000
train_iter = 3


def get_unique_filename(base_filename):
    counter = 1
    filename = base_filename
    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}"
        counter += 1
    return filename
    

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO):

    reset_num_timesteps, tb_log_name, progress_bar = True, ALGO.__name__, False

    iteration, log_interval = 0, 100
    eval_freq = int(5000)

    # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[128, 128], vf=[128, 128]))
    policy_kwargs = dict(net_arch=[128, 128])

    current_date = datetime.now().strftime("%d%b")

    NUM_DRONES = 2
    NUM_DUMB_DRONES = 2
    DRONE_COLLISION = True
    total_timesteps = 9e7

    algo_name = ALGO.__name__.lower()
    action_type = DEFAULT_ACT.value
    action_space = "discrete" if DISCRETE_ACTION else "continuous"
    update = "no_update" if DUMB_NO_MODEL else "update"
    base_filename = f"{current_date}_noselfplay_agent_{NUM_DRONES}_dumb_agent_{NUM_DUMB_DRONES}_{update}_collision_{DRONE_COLLISION}_{algo_name}_{action_type}_{action_space}_iter_"
    

    for train_it in range(train_iter):
        filename = os.path.join(output_folder, base_filename + str(train_it))
        filename = get_unique_filename(filename)

        load_folder = "" #os.path.join(output_folder, previous_load_folder + str(train_it))
        model_to_be_loaded = os.path.join(load_folder, 'best_model.zip')
        
        if not os.path.exists(filename):
            os.makedirs(filename+'/')

    
        train_env = make_vec_env(ENV_NAME,
                                env_kwargs=dict(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, obs=DEFAULT_OBS,  act=DEFAULT_ACT, gui=gui, max_timesteps=MAX_TIMESTEPS, 
                                                drone_collision=DRONE_COLLISION, dumb_no_model=DUMB_NO_MODEL),
                                n_envs=N_ENVS,
                                seed=0,
                                monitor_dir=filename + "/train"
                                )
        
        # eval_env = make_vec_env(ENV_NAME,
        #                         env_kwargs=dict(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, obs=DEFAULT_OBS,  act=DEFAULT_ACT, gui=gui, max_timesteps=MAX_TIMESTEPS, dumb_no_model=DUMB_NO_MODEL),
        #                         n_envs=N_ENVS,
        #                         seed=0,
        #                         monitor_dir=filename + "/eval"
        #                         )

        eval_env = Monitor(ENV_NAME(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, obs=DEFAULT_OBS, act=DEFAULT_ACT, max_timesteps=MAX_TIMESTEPS, 
                                    drone_collision=DRONE_COLLISION, dumb_no_model=DUMB_NO_MODEL), filename=filename + "/eval")

        agent = ALGO('MlpPolicy',
                        train_env,
                        policy_kwargs=policy_kwargs,
                        tensorboard_log=filename+'/tb/',
                        verbose=1,
                        opponent_model=True)
        
        if os.path.exists(model_to_be_loaded):
            loaded_model = ALGO.load(model_to_be_loaded)
            # Set the mlp_extractor of the new model to the loaded mlp_extractor
            agent.policy.mlp_extractor = loaded_model.policy.mlp_extractor
            print(f"Loaded previous model from: {model_to_be_loaded}")
            
        
        eval_callback = EvalCallback(eval_env,
                                    verbose=1,
                                    best_model_save_path=filename+'/',
                                    log_path=filename+'/',
                                    eval_freq=eval_freq,
                                    deterministic=True,
                                    render=False,
                                    opponent_model_update=True)
    
        
        total_timesteps, callback = agent._setup_learn(total_timesteps, eval_callback, reset_num_timesteps, tb_log_name, progress_bar,)

        callback.on_training_start(locals(), globals())

        while agent.num_timesteps < total_timesteps:
            continue_training = agent.collect_rollouts(agent.env, callback, agent.rollout_buffer, n_rollout_steps=agent.n_steps)

            if not continue_training:
                break

            iteration += 1
            agent._update_current_progress_remaining(agent.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert agent.ep_info_buffer is not None
                agent._dump_logs(iteration)

            agent.train()

        callback.on_training_end()

    

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
