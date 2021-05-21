from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import csv

from stable_baselines3.common import logger
from stable_baselines3.maddpg.maddpg_algo import MADDPGAlgo
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise

class MADDPG:
    def __init__(
        self, 
        env: Union[GymEnv, str],
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        load_model: bool = False,
    ):
        self.env = env
        # The noise objects for DDPG
        n_actions = [self.env.action_space[i].shape[-1] for i in range(self.env.n)]
        self.action_noise_n = [NormalActionNoise(mean=np.zeros(n_actions[i]), sigma=0.1 * np.ones(n_actions[i])) for i in range(self.env.n)]

        if not load_model:
            self.policy_n = [MADDPGAlgo('MlpPolicy', self.env, action_noise=self.action_noise_n[i], n_episodes_rollout=-1, train_freq=1, verbose=1, tensorboard_log=tensorboard_log, seed=seed, agent_id=i) for i in range(self.env.n)] #add agent_id
            self.replay_buffer_n = [policy.replay_buffer for policy in self.policy_n]

        self.horizon_reward = 0

        self.horizon_success = 0

        self.horizon_info = 0

    
    def load(
        self,
        load_dir: str = "None",
    )-> None:
    
        self.policy_n = [DDPG.load(load_dir+str(i)) for i in range(self.env.n)]
        for i in range(self.env.n):
            self.policy_n[i].set_env(self.env)

        self.replay_buffer_n = [policy.replay_buffer for policy in self.policy_n]

        #TODO save and load the countbased dictionary
        return self


    def learn(
        self, 
        total_timesteps=10000, 
        horizon = -1,
        log_interval=10,
        tb_log_name = "Run",
        eval_freq_timestep= -1, 
        eval_episodes=5,
        csv_dir = "Run",
    ) -> None:
        
        #function learn
        total_timesteps_n, callback_n = [], []
        for policy in self.policy_n:
            total_timesteps, callback = policy._setup_learn(
                total_timesteps=total_timesteps, 
                horizon=horizon,
                eval_env=None, 
                callback=None, 
                eval_freq=-1, 
                n_eval_episodes=5, 
                log_path=None,  #eval_log_path
                reset_num_timesteps=True, 
                tb_log_name=tb_log_name
            )
            total_timesteps_n.append(total_timesteps)
            callback_n.append(callback)
        
        for callback in callback_n:
            callback.on_training_start(locals(), globals())

        while self.policy_n[0].num_timesteps < total_timesteps_n[0]:
            #collect data
            rollout = self.collect_rollouts(
                    env=self.policy_n[0].env,
                    n_episodes=self.policy_n[0].n_episodes_rollout,
                    n_steps=self.policy_n[0].train_freq,
                    horizon=horizon,
                    action_noise_n=self.action_noise_n,
                    callback_n=callback_n,
                    learning_starts=self.policy_n[0].learning_starts,
                    replay_buffer_n=self.replay_buffer_n,
                    log_interval=log_interval,
                )

            if rollout.continue_training is False:
                break
            #train the policies
            for i in range(len(self.policy_n)):
                if self.policy_n[i].num_timesteps > 0 and self.policy_n[i].num_timesteps > self.policy_n[i].learning_starts:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    gradient_steps = self.policy_n[i].gradient_steps if self.policy_n[i].gradient_steps > 0 else rollout.episode_timesteps
                    self.policy_n[i].train(batch_size=self.policy_n[i].batch_size, gradient_steps=gradient_steps, replay_buffer_n=self.replay_buffer_n, policy_n=self.policy_n)

            if eval_freq_timestep != -1:
                if self.policy_n[0].num_timesteps % eval_freq_timestep == 0:
                    eval_reward = []
                    eval_success_rate = []
                    eval_collision = []
                    for _ in range(eval_episodes):
                        eval_re, eval_sr, eval_co = self.evaluation(horizon=horizon)
                        eval_reward.append(eval_re)
                        eval_success_rate.append(eval_sr)
                        eval_collision.append(eval_co)

                    with open(csv_dir, 'a') as fn:
                        wn = csv.writer(fn, dialect='excel')
                        final_res = [self.policy_n[0].num_timesteps, np.array(eval_reward).mean(), np.array(eval_success_rate).mean(), sum(eval_collision)]
                        wn.writerow(final_res)

        for callback in callback_n:
            callback.on_training_end()

        return self

    def evaluation(
        self, 
        horizon=0,
    ):
        obs_n = self.env.reset()
        e_reward = 0
        e_success = 0
        e_co=0
        self.env.render()
        for _ in range(horizon):
            # query for action from each agent's policy
            act_n = []
            for i in range(len(self.policy_n)):
                act,_ = self.policy_n[i].predict(obs_n[i], deterministic=False)
                act_n.append(act)
            # step environment
            e_new_obs, e_reward_n, e_done_n, e_infos_n =  self.env.step(act_n)
            self.env.render()
            obs_n = e_new_obs
            print("eval === e_done_n, infos_n: ", e_done_n, e_infos_n)
            e_reward += sum(e_reward_n)
            if any(e_done_n) == True:
                e_success = 1
            if 1 in e_infos_n['n']:
                e_co += 1
        self.env.reset()
        return e_reward, e_success, e_co


    def _sample_action(
        self, learning_starts: int, action_noise_n: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.policy_n[0].num_timesteps < learning_starts: #and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action_n = []
            for i in range(len(self.policy_n)):
                unscaled_action_n.append(np.array([self.policy_n[i].action_space.sample()]))
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action_n = []
            # make the shape of predict actions and random action the same
            for i in range(len(self.policy_n)):
                unscaled_action, _ = self.policy_n[i].predict(self.policy_n[i]._last_obs[i], deterministic=False)
                unscaled_action = [unscaled_action]
                unscaled_action = np.array(unscaled_action)
                unscaled_action_n.append(unscaled_action)

        # Rescale the action from [low, high] to [-1, 1]
        action_n, buffer_action_n = [], []
        for i in range(len(self.policy_n)):
            if isinstance(self.policy_n[i].action_space, gym.spaces.Box):
                scaled_action = self.policy_n[i].policy.scale_action(unscaled_action_n[i])

                # Add noise to the action (improve exploration)
                if action_noise_n[i] is not None:
                    scaled_action = np.clip(scaled_action + action_noise_n[i](), -1, 1)

                # We store the scaled action in the buffer
                buffer_action = scaled_action
                action = self.policy_n[i].policy.unscale_action(scaled_action)
                action_n.append(action[0])
                buffer_action_n.append(buffer_action[0])
            else:
                # Discrete case, no need to normalize or clip
                buffer_action = unscaled_action_n[i]
                action = buffer_action
                action_n.append(action[0])
                buffer_action_n.append(buffer_action[0])
        return action_n, buffer_action_n


    def collect_rollouts(
        self,
        env: VecEnv,
        callback_n: [BaseCallback],
        n_episodes: int = 1,
        n_steps: int = -1,
        horizon: int = -1, #add for maximam timestep
        action_noise_n: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer_n: Optional[ReplayBuffer] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """

        episode_rewards, total_timesteps = [], []  # sum of rewards for all agents
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        # if self.use_sde:
        #     self.actor.reset_noise()

        for callback in callback_n:
            callback.on_rollout_start()
        continue_training = True
        while total_steps < n_steps or total_episodes < n_episodes:
            done_n = [False for i in range(len(self.policy_n))]
            episode_reward, episode_timesteps = 0.0, 0

            env.render()
            while not all(done_n):
                
                # if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                #     # Sample a new noise matrix
                #     self.actor.reset_noise()

                # Select action randomly or according to policy
                action_n, buffer_action_n = self._sample_action(learning_starts, action_noise_n)
                
                # Rescale and perform action
                new_obs_n, reward_n, done_n, infos_n = env.step(action_n)
                env.render()
                print("maddpg: ", action_n, new_obs_n, reward_n, done_n, infos_n)
                # print("maddpg: ", action_n, reward_n, infos_n)
                if any(done_n) == True:
                    self.horizon_success += 1

                for i in range(len(self.policy_n)):
                    self.policy_n[i].num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                #TODO: Env done when agent reached or timestep reached horizon
                if horizon and self.policy_n[0].num_timesteps % horizon == 0:
                        done_n = [True for _ in range(len(self.policy_n))]

                # Give access to local variables
                for callback in callback_n:
                    callback.update_locals(locals())
                    # Only stop training if return value is False, not when it is None.
                    if callback.on_step() is False:
                        return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)
                
                episode_reward += sum(reward_n)

                # Retrieve reward and episode length if using Monitor wrapper
                for i in range(len(self.policy_n)):
                    self.policy_n[i]._update_info_buffer(infos_n, done_n[i])

                #record the collision for simple_spread or reach for speaker_listener
                if 1 in infos_n[0]['n']:
                    self.horizon_info += 1

                # Store data in replay buffer
                if all(replay_buffer_n) is not None:
                    # Store only the unnormalized version
                    new_obs_n_ = []
                    for i in range(len(self.policy_n)):
                        if self.policy_n[i]._vec_normalize_env is not None:
                            new_obs_ = self.policy_n[i]._vec_normalize_env.get_original_obs()
                            reward_ = self.policy_n[i]._vec_normalize_env.get_original_reward()
                            new_obs_n_.append(new_obs_)
                        else:
                            # Avoid changing the original ones
                            self.policy_n[i]._last_original_obs, new_obs_, reward_ = self.policy_n[i]._last_obs, new_obs_n[i], reward_n[i]
                            new_obs_n_.append(new_obs_)
                        
                        replay_buffer_n[i].add(self.policy_n[i]._last_original_obs[i], new_obs_, buffer_action_n[i], reward_, done_n[i])
                
                for i in range(len(self.policy_n)):
                    self.policy_n[i]._last_obs = new_obs_n
                    # Save the unnormalized observation
                    if self.policy_n[i]._vec_normalize_env is not None:
                        self.policy_n[i]._last_original_obs = new_obs_n_[i]

                    self.policy_n[i]._update_current_progress_remaining(self.policy_n[i].num_timesteps, self.policy_n[i]._total_timesteps)

                    # For DQN, check if the target network should be updated
                    # and update the exploration schedule
                    # For SAC/TD3, the update is done as the same time as the gradient update
                    # see https://github.com/hill-a/stable-baselines/issues/900
                    self.policy_n[i]._on_step() #haven't check where it is!!!!!!!!!!!!!!!!
                
                if 0 < n_steps <= total_steps:
                    break
            
            
            self.horizon_reward += sum(reward_n)
            #Env Never Done!!! Change to every episode has horizon timestep.
            if horizon and self.policy_n[0].num_timesteps % horizon == 0:
                total_episodes += 1
                for i in range(len(self.policy_n)):
                    self.policy_n[i]._episode_num += 1
                    self.policy_n[i]._horizon_reward = self.horizon_reward
                    
                self.horizon_reward = 0
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                for i in range(len(self.policy_n)):
                    if action_noise_n[i] is not None:
                        action_noise_n[i].reset()
                
                if self.env.world.flags:
                    #tensorboard record success rate for speaker_listener
                    logger.record("reward/episode_success_rate", self.horizon_info/horizon)
                    self.horizon_info = 0
                else:
                    #tensorboard record collision
                    logger.record("count/episode_collision", self.horizon_info)
                    logger.record("reward/episode_success_rate", self.horizon_success/horizon)
                    self.horizon_success = 0
                
                # Log training infos
                for i in range(len(self.policy_n)):
                    if log_interval is not None and self.policy_n[i]._episode_num % log_interval == 0:
                        self.policy_n[i]._dump_logs()
                
                env.reset()
        

        # mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0
        
        for callback in callback_n:
            callback.on_rollout_end()
        return RolloutReturn(self.horizon_reward, total_steps, total_episodes, continue_training)


        


