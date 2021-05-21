from collections import defaultdict
import numpy as np
from typing import List
from stable_baselines3.exploration.simhash import Simhash

class CountBased():
    """

    """
    def __init__(
        self,
        beta: float = 1.0,
        strategy: str = 'None',
        simhash_k: int = 32,
        d_size: int = 0,
    ):
        self.table = defaultdict(int)
        self.beta = beta
        self.strategy = strategy
        if self.strategy == 'simhash':
            self.sim_hash = Simhash(k=simhash_k, d_size=d_size)


    def obs_function(
        self,
        obs: np.ndarray,
    )-> str:

        if self.strategy == 'decimals':
            obs = obs.round(decimals = 2)
        elif self.strategy == 'simhash':
            obs = self.sim_hash.conti_to_discre(obs)
        obs = obs.tobytes()
        return obs


    def update(
        self,
        obs: np.ndarray,
    )-> None:
        # Copy to avoid modification by reference
        obs = np.array(obs).copy()
        obs = self.obs_function(obs)
        # When a key is first encountered, creates a default count of zero.
        self.table[obs]+=1


    def reward(
        self,
        obs: np.ndarray,
    )-> float:
        # Copy to avoid modification by reference
        obs = np.array(obs).copy()
        obs = self.obs_function(obs)
        obs_count = self.table[obs]
        return self.beta / np.sqrt(obs_count)
        