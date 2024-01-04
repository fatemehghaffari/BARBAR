import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from environment import BanditNArmedBernoulli, BanditNArmedBernoulliCorrupt1
import time 
from copy import deepcopy
from barbar import BARBAR, BARBAR_dist_het
from ucb import run_simulation







def main():
    '''Compares the epsilon-greedy approach to the upper confidence bounds
    approach for solving the multi-armed bandit problem.
    n: Number of arms
    L: Number of agents
    T: Number of rounds
    means_real: Real means of arms
    env_corr: corrupted environment
    env: Uncorrupted environment
    '''

    n = 10
    L = 5
    T = 20000
    # means_real = np.random.uniform(0, 1, 10)
    means_real = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8])
    env_corr = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate_v1 = 1)
    env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate_v1 = 0.2)
    # ev_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 2, corr_total_v2 = 5000)
    env = BanditNArmedBernoulli(n, deepcopy(means_real))
    env.reset()


    # results_eps = run_simulation(epsilon_greedy_action)
    results_ucb = run_simulation(env, means_real, T)
    results_ucb_corr1 = run_simulation(env_corr, means_real, T)
    results_barbar = BARBAR(env_corr, means_real, n, T, delta = 0.2)
    # K = np.random.randint(0, 2, size=[1, 10])

    K = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    results_barbar_het = BARBAR_dist_het(env_corr2, means_real, L, K, T, delta = 0.2)

    plt.plot(results_ucb_corr1, color='red')
    plt.plot(results_ucb, color='blue')
    plt.plot(np.array(results_barbar), color='green')
    plt.plot(results_barbar_het, color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Pseudo Regret")
    plt.title("Bernoulli 10-armed-bandit regret vs time")
    ucb_corr1_patch = mpatches.Patch(color='red', label='upper confidence bounds - Corruption v1')
    ucb_patch = mpatches.Patch(color='blue', label='upper confidence bounds')
    barbar_patch = mpatches.Patch(color='green', label='BARBAR - Corruption v1')
    barbar_het_patch = mpatches.Patch(color='orange', label='BARBAR Het - 5 agents - Corruption v1')
    plt.legend(handles=[ucb_corr1_patch, ucb_patch, barbar_patch, barbar_het_patch])
    plt.show()


if __name__ == "__main__":
    main()