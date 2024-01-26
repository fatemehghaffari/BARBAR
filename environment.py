import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    """
    def __init__(self, p_dist, r_dist):
        if len(p_dist) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True
        if np.random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                # corr = np.random.uniform(0, C)
                reward = np.random.binomial(1, self.r_dist[action])
        reward = np.random.binomial(1, self.r_dist[action])
        if self.corr_ver == 1:
            if np.random.binomial(1, self.corr_rate):
                reward = int(not(reward))
                # reward = 1 - reward
        if self.corr_ver == 2:
            if np.random.binomial(1, self.corr_rate_list_v2[action]):
                reward = int(not(reward))
        if self.corr_ver == 3:
            # if np.argsort(self.r_dist)[action] == (len(self.r_dist) - 1):
                # reward  = np.random.binomial(1, 1)
            sr = np.argsort(self.r_dist)
            reward = np.random.binomial(1, self.r_dist[np.where(sr == (len(sr) -1 -sr[action]))])
        return 0, reward, done, {}

    def _reset(self):
        return 0

    def _render(self, mode='human', close=False):
        pass

class BanditNArmedBernoulli(BanditEnv):
    """
    n armed bandit mentioned on page 30 of Sutton and Barto's
    [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0)

    Actions always pay out
    Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
    Actual reward is drawn from a normal distribution (q*(a), 1)
    """
    def __init__(self, bandits, means, corr_ver = None, corr_rate = None):
        p_dist = np.full(bandits, 1)
        r_dist = means
        self.corr_ver = corr_ver
        self.corr_rate = corr_rate
        if self.corr_ver == 2:
            self.corr_rate_list_v2 = (((np.array(range(bandits))+1)*10)[::-1]/(10 * bandits)) * self.corr_rate
            self.corr_rate_list_v2[-1] = 0
            self.corr_rate_list_v2 = self.corr_rate_list_v2[np.argsort(r_dist[::-1])]


        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

# class BanditNArmedBernoulliCorrupt1(BanditEnv):
#     """
#     Actions always pay out
#     Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
#     Actual reward is drawn from a normal distribution (q*(a), 1)

#     Corruption special case 1: fixed unkown rate
#     """
#     def __init__(self, bandits, means):
#         p_dist = np.full(bandits, 1)
#         r_dist = []
#         corr_rate = 0.4

#         for i in range(bandits):
#             if means[i] >= 0.5:
#                 mu = means[i] - corr_rate
#             else:
#                 mu = means[i] + corr_rate
            
#             if mu > 1:
#                 mu = 1
#             elif mu < 0:
#                 mu = 0

#             r_dist.append([np.random.binomial(1, mu), 1])

#         BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)
        
class BanditNArmedBernoulliCorrupt1(BanditEnv):
    """
    Actions always pay out
    Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
    Actual reward is drawn from a normal distribution (q*(a), 1)

    Corruption special case 1: fixed unkown rate
    """
    def __init__(self, bandits, means):
        p_dist = np.full(bandits, 1)
        corr_rate = 1

        for i in range(bandits):
            if means[i] >= 0.5:
                means[i] = means[i] - corr_rate
            else:
                means[i] = means[i] + corr_rate
            
            if means[i] > 1:
                means[i] = 1
            elif means[i] < 0:
                means[i] = 0

        r_dist = list(means)
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

