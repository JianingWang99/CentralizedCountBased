from ma_envs.multiagent.environment import MultiAgentEnv
from ma_envs.multiagent.policy import InteractivePolicy, RandomPolicy
import ma_envs.multiagent.scenarios as scenarios
from importlib import import_module

import numpy as np
import pandas as pd
import os
from collections import Counter

from stable_baselines3.maddpg.maddpg import MADDPG

'''
scenario_name: Multi-agent environment, e.g. simple, simple_spread, simple_tag, simple_speaker_listener
share_view: True, all the agents show in a same viewer, False each agent has a viewer.
total_timesteps: Total time steps for training
horizon: Each episode has horizon time steps
tensorboard_log: TensorBoard log directory
tb_log_name: The name of the run for TensorBoard logging
save_model: Save the final model after learning.
load_model: Load the existing model.
countbased_beta: -1.0 means disable the count based
countbased_strategy: state function, 'None', 'decimals', 'simhash'
'''

seed = 0
scenario_name = "simple_spread"
shared_viewer = True
total_timesteps = 1000000
horizon = 20
eval_freq_timestep = horizon #how many timestep evaluate, -1 do not evaluate
eval_episodes = 1

#tensorboard
# tensorboard_log = "./maddpg_tensorboard_new/speaker_listener/"
tensorboard_log = "./maddpg_tensorboard_new/2agents_2Landmarks/"
tb_log_name="MADDPG_noBonus_seed_"+str(seed)
# tb_log_name="test"

#save csv for evaluation reward and success rate
csv_dir = f"./maddpg_csv_new/{scenario_name}"
file_name = f"{tb_log_name}.csv"
df = pd.DataFrame(list())
if not os.path.exists(csv_dir):
    os.mkdir(csv_dir)
csv_dir = os.path.join(csv_dir, file_name)
df.to_csv(csv_dir)

#save and load model directory
save_model = False
save_dir = "./maddpg_"+scenario_name+tb_log_name+"_policy/agent_"
load_model = False
load_dir = "./maddpg_"+scenario_name+"_policy/agent_"


# load scenario from script 
scenario = scenarios.load(scenario_name + ".py").Scenario() #change from scenario to env
# create world
world = scenario.make_world()
# create multiagent environment
info_callback = None
done_callback = None
if scenario_name == "simple_spread":
    info_callback = scenario.collision
    done_callback = scenario.done

if scenario_name == "simple_speaker_listener":
    info_callback = scenario.success_rate

env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=info_callback, done_callback=done_callback, shared_viewer = shared_viewer)
print('env_type: {}, n_agents: {}, action_space: {}, obs_space: {}'.format(scenario_name, env.n, env.action_space, env.observation_space))

maddpg = MADDPG(env, tensorboard_log=tensorboard_log, seed=seed, load_model=load_model)

if load_model:
    maddpg.load(load_dir = load_dir)

#log_interval set to 1, every horizon timesteps log one time.
maddpg.learn(total_timesteps=total_timesteps, log_interval=1, horizon=horizon, tb_log_name=tb_log_name, eval_freq_timestep=eval_freq_timestep, eval_episodes=eval_episodes, csv_dir=csv_dir)
#save the models
if save_model:
    for i in range(env.n):
        maddpg.policy_n[i].save(save_dir + str(i))

    
## execution loop
# obs_n = env.reset()
# for _ in range(100):
#     # query for action from each agent's policy
#     act_n = []
#     for i, policy in enumerate(simple_maddpg.policy_n):
#         act,_ = policy.predict(obs_n[i])
#         act_n.append(act)
#     # step environment
#     print("test_env: ", act_n[0].shape)
#     obs_n, reward_n, done_n, _ = env.step(act_n)
#     # print("action: ", act_n, "obs_n: ", obs_n, "reward_n: ", reward_n, "done_n: ", done_n)
#     # render all agent views
#     env.render()
#     if all(done_n):
#         break
#     # display rewards
#     for agent in env.world.agents:
#         print(agent.name + " reward: %0.3f" % env._get_reward(agent))