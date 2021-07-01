from typing import Any, Dict, Optional, Type, Union, List, Tuple

import gym
from gym import spaces
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.maddpg.maddpg_policies import MADDPGPolicy
from stable_baselines3.td3.td3 import TD3

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.exploration.count_based import CountBased


class MADDPGAlgo(TD3):
    """
    Extension of Deep Deterministic Policy Gradient (DDPG) to Multi-Agent DDPG (MADDPG).
    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    Note: we treat DDPG as a special case of its successor TD3.
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[MADDPGPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = -1,
        gradient_steps: int = -1,
        n_episodes_rollout: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        agent_id: int = 0, # add for multi-agent environemnt
    ):

        super(MADDPGAlgo, self).__init__(
            policy=MADDPGPolicy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            n_episodes_rollout=n_episodes_rollout,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            agent_id=agent_id, # add for multi-agent environemnt
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        self.original_env = env
        self.agent_id = agent_id
       
        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model_maddpg()

    def _setup_model_maddpg(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        
        critic_obs_space = spaces.Box(low=self.observation_space.low, high=self.observation_space.high, shape=self.observation_space.shape, dtype=self.observation_space.dtype)
        obs_shape = self.original_env.observation_space[0].shape[0]
        for i in range(1, self.original_env.n):
            obs_shape += self.original_env.observation_space[i].shape[0]
        critic_obs_space.shape = (obs_shape, )

        critic_action_space = spaces.Box(low=self.action_space.low, high=self.action_space.high, shape=self.action_space.shape, dtype=self.action_space.dtype)
        act_shape = self.original_env.action_space[0].shape[0]
        for i in range(1, self.original_env.n):
            act_shape += self.original_env.action_space[i].shape[0]
        critic_action_space.shape = (act_shape, )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
            critic_obs_space = critic_obs_space,
            critic_action_space = critic_action_space
        )
        self.policy = self.policy.to(self.device)
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        print("actor: ", self.actor.features_dim,  self.actor.action_space)
        print("critic: ", self.critic.features_dim,  self.critic.action_space)

    def train(self, gradient_steps: int, replay_buffer_n: List, policy_n: List, batch_size: int = 100, count_based_n: Optional[CountBased] = None, countbased_joint:bool = True) -> None:
            # Update learning rate according to lr schedule
            self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])            
            actor_losses, critic_losses = [], []

            for gradient_step in range(gradient_steps):
                # Collect replay sample from all agents
                self.replay_sample_index = self.replay_buffer.make_index(batch_size)
                # self.replay_sample_index =np.sort(self.replay_sample_index)
                # self.replay_sample_index = np.array([0])
                # print("self.replay_sample_index:", self.replay_sample_index)
                replay_data_n = []
                for i in range(self.original_env.n):
                    replay_data_n.append(replay_buffer_n[i]._get_samples(self.replay_sample_index, env=self._vec_normalize_env))

                # Sample replay buffer of current agent
                # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                replay_data = self.replay_buffer._get_samples(self.replay_sample_index, env=self._vec_normalize_env)
                

                obs_n = replay_data_n[0].observations
                act_n = replay_data_n[0].actions
                for i in range(1, self.original_env.n):
                    obs_n = th.cat((obs_n, replay_data_n[i].observations), dim=1)
                    act_n = th.cat((act_n, replay_data_n[i].actions), dim=1)
                
                #add intrinsic reward:    
                if countbased_joint:            
#                     joint_obs_act_n = obs_n.clone()
#                     joint_obs_act_n_front = joint_obs_act_n[:, :3]
#                     joint_obs_act_n_back = joint_obs_act_n[:, 11:]
#                     joint_obs_act_n = th.cat((joint_obs_act_n_front, joint_obs_act_n_back), dim=1)
                    joint_obs_act_n = th.cat((obs_n, act_n), dim=1)
                   
                    #reward add bonus
                    bonus_n = []
                    for i in range(joint_obs_act_n.shape[0]):
                        bonus_n.append(count_based_n[0].reward(joint_obs_act_n[i].numpy().round(2)))
                        # bonus_n.append(count_based_n[self.agent_id].reward(joint_obs_act_n[i].numpy()))
                    # print("bonus_n:", bonus_n)
                else:
                    obs_i = replay_data.observations.clone()
                    act_i = replay_data.actions.clone()
                    joint_obs_act_i = th.cat((obs_i, act_i), dim=1)
                    #update count
                    # for i in range(joint_obs_act_i.shape[0]):
                    #     count_based_n[self.agent_id].update(joint_obs_act_i[i].numpy())
                    #reward add bonus
                    bonus_n = []
                    for i in range(joint_obs_act_i.shape[0]):
                        bonus_n.append(count_based_n[self.agent_id].reward(joint_obs_act_i[i].numpy()))
                # print("bonus_n:", bonus_n)
                bonus_n = np.reshape(bonus_n, (100,1))
                bonus_n = th.Tensor(bonus_n)
                new_reward_n = replay_data.rewards + bonus_n
                
                #calculate y_i
                with th.no_grad():
                    # Select action according to policy and add clipped noise
                    # noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                    # noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    # next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                    
                    next_actions_n = []
                    for i in range(self.original_env.n):
                        noise = replay_data_n[i].actions.clone().data.normal_(0, policy_n[i].target_policy_noise)
                        noise = noise.clamp(-policy_n[i].target_noise_clip, policy_n[i].target_noise_clip)
                        n_a = (policy_n[i].actor_target(replay_data_n[i].next_observations) + noise).clamp(-1, 1)
                        next_actions_n.append(n_a)
                    # print("algo next_actions_n====: ", next_actions_n)
                    # Concatenates actions and observations of all the agents
                    next_act_n = next_actions_n[0]
                    next_obs_n = replay_data_n[0].next_observations
                    for i in range(1, len(next_actions_n)):
                        next_act_n = th.cat((next_act_n, next_actions_n[i]), dim=1)
                        next_obs_n = th.cat((next_obs_n, replay_data_n[i].next_observations), dim=1)
                    # print("algo next_act_n====: ", next_act_n)
                    # print("algo next_obs_n====",next_obs_n)

                    # Compute the next Q-values: min over all critics targets
                    # next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values = th.cat(self.critic_target(next_obs_n, next_act_n), dim=1)
                    # print("algo next_q_values===", next_q_values)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # print("algo next_q_values===", next_q_values)
                    # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_value
                    target_q_values = new_reward_n + (1 - replay_data.dones) * self.gamma * next_q_values
                    # print("algo target_q_values===", target_q_values)


                # Get current Q-values estimates for each critic network
                # current_q_values = self.critic(replay_data.observations, replay_data.actions)
                current_q_values = self.critic(obs_n, act_n)
                # print("algo current_q_values===", current_q_values)
                # Compute critic loss L
                critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
                critic_losses.append(critic_loss.item())
                # print("algo critic_losses===", critic_losses)

                # Optimize the critics
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # Delayed policy updates
                if gradient_step % self.policy_delay == 0:
                    # Compute actor loss
                    # actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                    
                    # action of the current agent is calculate by its actor, other action is come from the replay buffer.
                    a_i = self.actor(replay_data.observations)
                    agent_actions_n = []
                    for i in range(self.original_env.n):
                        if i == self.agent_id:
                            agent_actions_n.append(a_i)
                        else:
                            agent_actions_n.append(replay_data_n[i].actions)

                    act_n_q1 = agent_actions_n[0]
                    for i in range(1, self.original_env.n):
                        act_n_q1 = th.cat((act_n_q1, agent_actions_n[i]), dim=1)
                    # print("algo act_n_q1===", act_n_q1)
                    actor_loss = -self.critic.q1_forward(obs_n, act_n_q1).mean()
                    actor_losses.append(actor_loss.item())
                    # print("algo actor_losses===", actor_losses)

                    # Optimize the actor
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

            self._n_updates += gradient_steps
            logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            logger.record("train/actor_loss", np.mean(actor_losses))
            logger.record("train/critic_loss", np.mean(critic_losses))
            

    # def learn(
    #     self,
    #     total_timesteps: int,
    #     callback: MaybeCallback = None,
    #     log_interval: int = 4,
    #     eval_env: Optional[GymEnv] = None,
    #     eval_freq: int = -1,
    #     n_eval_episodes: int = 5,
    #     tb_log_name: str = "MADDPG",
    #     eval_log_path: Optional[str] = None,
    #     reset_num_timesteps: bool = True,
    # ) -> OffPolicyAlgorithm:

    #     return super(MADDPGAlgo, self).learn(
    #         total_timesteps=total_timesteps,
    #         callback=callback,
    #         log_interval=log_interval,
    #         eval_env=eval_env,
    #         eval_freq=eval_freq,
    #         n_eval_episodes=n_eval_episodes,
    #         tb_log_name=tb_log_name,
    #         eval_log_path=eval_log_path,
    #         reset_num_timesteps=reset_num_timesteps,
    #     )

    def _excluded_save_params(self) -> List[str]:
        return super(MADDPGAlgo, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
