from stable_baselines3.exploration.count_based import CountBased
from stable_baselines3.exploration.simhash import Simhash
import numpy as np

from ma_envs.multiagent.environment import MultiAgentEnv
from ma_envs.multiagent.policy import InteractivePolicy, RandomPolicy
import ma_envs.multiagent.scenarios as scenarios
from importlib import import_module

scenario_name = "simple"
# load scenario from script 
scenario = scenarios.load(scenario_name + ".py").Scenario() #change from scenario to env
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=None, shared_viewer = True)
print('env_type: {}, n_agents: {}, action_space: {}, obs_space: {}'.format(scenario_name, env.n, env.action_space, env.observation_space))

sim_hash = Simhash(k=32, observation_space=env.observation_space[0])
# last_observation = np.array([-0.31478572,  0.27859437,  0.85169816, -0.3913897])
new_obs_1 = np.array([0.1,  0.28, 0.90, -0.40])
new_obs = np.array([0.1,  0.28,  0.85, -0.39])
a = sim_hash.conti_to_discre(new_obs_1)
b = sim_hash.conti_to_discre(new_obs)
print((a==b).all())
# count_based = CountBased(strategy='decimals')

# new_obs_1 = np.array([0.0,  0.27859437,  0.85169816, -0.3913897])
# new_obs = np.array([0.0,  0.27859438,  0.85169816, -0.3913897])


# count_based.update(last_observation)
# count_based.update(last_observation)
# count_based.update(new_obs_1)
# count_based.update(new_obs_1)
# count_based.update(new_obs)

# # c = np.fromiter(count_based.table.values(), dtype=int)
# # print(type(c))
# print(count_based.table)
# print(sum(map((1).__eq__, count_based.table.values())))
# print(count_based.reward(last_observation))
# print(count_based.reward(new_obs_1))
# print(count_based.reward(new_obs_0))
