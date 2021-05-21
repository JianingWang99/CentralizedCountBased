from ma_envs.multiagent.environment import MultiAgentEnv
from ma_envs.multiagent.policy import InteractivePolicy, RandomPolicy
import ma_envs.multiagent.scenarios as scenarios
from importlib import import_module

import numpy as np

from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# simple, simple_spread, simple_tag
scenario_name = "simple_spread"
# load scenario from script 
scenario = scenarios.load(scenario_name + ".py").Scenario() #change from scenario to env
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
# print("Before disable communiction dimension: ", env.world.dim_c)
# env.world.dim_c = 0 #disable communication
print('env_type: {}, n_agents: {}, action_space: {}, obs_space: {}'.format(scenario_name, env.n, env.action_space, env.observation_space))
print(env.discrete_action_input)
# render call to create viewer window (necessary only for interactive policies)

for _ in range(10000000):
    env.render()
# create interactive policies for each agent
# policies = [RandomPolicy(env,i) for i in range(env.n)]

# The noise objects for DDPG
# n_actions = [env.action_space[i].shape[-1] for i in range(env.n)]
# action_noise = [NormalActionNoise(mean=np.zeros(n_actions[i]), sigma=0.1 * np.ones(n_actions[i])) for i in range(env.n)]

# policies = [DDPG('MlpPolicy', env, action_noise=action_noise[i], verbose=1, agent_id=i) for i in range(env.n)] #add agent_id 

# # execution loop
# obs_n = env.reset()
# for _ in range(1000):
#     # query for action from each agent's policy
#     act_n = []
#     for i, policy in enumerate(policies):
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
#        print(agent.name + " reward: %0.3f" % env._get_reward(agent))


# The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")
# env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()