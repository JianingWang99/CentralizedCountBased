import numpy as np

class Simhash():
    """

    """
    def __init__(
        self,
        k: int,
        d_size: int, 
    ):  
        # k controls the granularity
        self.k = k
        self.D = d_size
        # A is a k*D matrix with i.i.d. entries drawn from a standard Gaussian distribution N(0,1)
        self.A = np.random.normal(loc=0.0, scale=1.0, size =(self.k, self.D))

    def conti_to_discre(
        self,
        obs: np.ndarray,
    )-> np.ndarray:
        #no preprocessing
        conti_obs = np.array(obs).copy()
        # sign: if array value is greater than 0 it returns 1, less than 0 returns -1, 0 returns 0. 
        # return {-1, 1}^k
        discre_obs = np.sign(np.dot(self.A, conti_obs))
        return discre_obs
       