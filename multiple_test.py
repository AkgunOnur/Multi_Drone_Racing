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
from train_ppo import PPO
from train_td3 import TD3
from train_ddpg import DDPG
from train_sac import SAC
from common.env_util import make_vec_env
from common.callbacks_updated import EvalCallback, StopTrainingOnRewardThreshold
from common.evaluation import evaluate_policy
from common.monitor import Monitor
from common.vec_env.vec_monitor import VecMonitor
# from envs.MultiGates_SelfPlay_v0 import MultiGates_v0
# from envs.MultiGates_SelfPlay_v1 import MultiGates_v1
from envs.MultiGates_SelfPlay_v2 import MultiGates_v2
# from gym_pybullet_drones.envs.MultiGatesCont import MultiGatesCont
from utils.utils import sync, str2bool
from utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
MAX_TIMESTEPS = 8000
ENV_NAME = MultiGates_v2

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pos') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
# ALGO = DDPG
DISCRETE_ACTION = False
# NUM_DRONES = 2
# NUM_DUMB_DRONES = 2
TRACK = 1
# INIT_TYPE = "simple"
# MODE_TYPE = "normal"
N_EPISODES = 50

def extract_info(text):
    parts = text.split('_')
    num_agents = None
    num_dumb_agents = None
    algorithm = None

    for i in range(len(parts)):
        if parts[i] == 'dumb' and i+1 < len(parts):
            num_agents = int(parts[i-1])
            num_dumb_agents = int(parts[i+2])            
        elif parts[i] in ['ddpg', 'ppo', 'sac']:
            algorithm = parts[i]

    return num_agents, num_dumb_agents, algorithm


def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO):

    # algo_name = ALGO.__name__.lower()
    
    # filename = "25Jul_agent_2_dumb_agent_2_track_1_ddpg_pos_continuous_iter_0"

    init_list = ["simple", "normal"]

    mode_list = ["normal", "normal", "normal", "normal",  "planner"]

    model_list = ["24Jul_agent_2_dumb_agent_2_track_1_ddpg_pos_continuous_single_iter_0", "25Jul_agent_2_dumb_agent_2_track_1_ddpg_pos_continuous_iter_0", 
                  "25Jul_agent_2_dumb_agent_2_track_1_ppo_pos_continuous_iter_0", "25Jul_noselfplay_agent_4_dumb_agent_0_track_1_ppo_pos_continuous_iter_0", 
                  "25Jul_agent_2_dumb_agent_2_track_1_ppo_pos_continuous_iter_0", ]
    
    algo_dict = {"ddpg": DDPG,"ppo": PPO, "sac":SAC, "td3":TD3}

    for INIT_TYPE in init_list:
        for index, MODE_TYPE in enumerate(mode_list): 

            filename = model_list[index]

            NUM_DRONES, NUM_DUMB_DRONES, algo_name = extract_info(filename)
            ALGO = algo_dict.get(algo_name)

            model_to_be_loaded = os.path.join(output_folder, algo_name, filename, 'best_model.zip')

            env = ENV_NAME(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, 
                        gui=DEFAULT_GUI, obs=DEFAULT_OBS, act=DEFAULT_ACT, max_timesteps=MAX_TIMESTEPS, track=TRACK, mode=MODE_TYPE, init_type=INIT_TYPE)
            

            if os.path.exists(model_to_be_loaded):
                # Load the saved model
                agent = ALGO.load(model_to_be_loaded, env=env)

                # Set the mlp_extractor of the new model to the loaded mlp_extractor
                
                print(f"\n\nLoaded previous model from: {model_to_be_loaded} Mode: {MODE_TYPE} Init: {INIT_TYPE}")
                mean_reward, std_reward, mean_dumb_reward, std_dumb_reward, info_list = evaluate_policy(model=agent, env=env, n_eval_episodes=N_EPISODES, opponent_policy=agent)
                print (f"Mean Reward: {mean_reward:.4f} Std Reward: {std_reward:.4f} Mean Opponent Reward: {mean_dumb_reward:.4f} Std Opponent Reward: {std_dumb_reward:.4f} ")
                # print (info_list)

                result = {}

                for item in info_list:
                    for key, value in item.items():
                        if 'drone' in key and isinstance(value, dict):
                            if key not in result:
                                result[key] = {'lap_time': [], 'successful_flight': []}
                            if 'lap_time' in value:
                                result[key]['lap_time'].extend(value['lap_time'])
                            if 'successful_flight' in value:
                                result[key]['successful_flight'].append(value['successful_flight'])

                print(result)

            else:
                print(f"No such file found! ", model_to_be_loaded)
                continue
        
    

    

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
