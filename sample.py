import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from environment import BanditNArmedBernoulli, BanditNArmedBernoulliCorrupt1
import time 
from copy import deepcopy
n = 10
# means_real = np.random.uniform(0, 1, 10)
means_real = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8])
env_corr = BanditNArmedBernoulliCorrupt1(n, deepcopy(means_real))
env = BanditNArmedBernoulli(n, deepcopy(means_real))
env.reset()


def run_simulation(get_action, env, num_trials=5000):
    '''Runs a 10-armed bandit simulation for multiple trials.
    
    Args:
    - get_action: a function which has input (t, means, count) and returns action
    - num_trials: number of iterations

    Returns:
    - A list of regret versus time 
    '''
    
    means = np.zeros(10)
    count = np.zeros(10)
    choices = []

    for t in range(num_trials):

        action = get_action(t, means, count)
        observation, reward, done, info = env.step(action)

        # Keep track of sample means for exploitation, and choices for regret calculation 
        count[action] += 1
        means[action] = (1 - 1/count[action]) * means[action] + (1/count[action]) * reward
        choices.append(action)
    def regret(t):
        best = np.argmax(means_real)
        return means_real[best] * t - sum([means_real[choices[i]] for i in range(t)])

    return [regret(t) for t in range(num_trials)]

def epsilon_greedy_action(t, means, count, epsilon=0.5):
    '''epsilon percent of the time, choose a random action.
    (1-epsilon) percent of the time, exploit by choosing the action
    with the highest mean reward.
    
    This is a suboptimal solution, achieving linear regret. 
    '''

    explore = np.random.uniform() < epsilon

    if explore:
        return env.action_space.sample()
    else:
        return np.argmax(means)


def upper_confidence_bounds_action(t, means, count, epsilon=0.0):
    '''Play each arm once, then choose according to the equation given
    by Auer, Cesa-Bianchi & Fisher (2002).
    
    This is said to achieve the most optimal solution, with logarithmic regret.
    '''

    if t < 10:
        return t
    else:
        return np.argmax(means + np.sqrt(2 * np.log(t) / count))

def BARBAR(K, T, delta = 0.2):
    # lmbda = 1024 * np.log((8 * K / delta) * np.log2(T))
    lmbda = np.log((8 * K / delta) * np.log2(T))
    t = 0
    m = 1
    n = np.zeros([1, K])
    Delta = np.ones([1, K])
    actions_sum = np.zeros(K)
    actions = []
    while t < T:
        n = lmbda / (Delta) ** 2
        print("sing", n)
        N = np.sum(n)
        pr = n / N
        ereward = np.zeros(K)
        for i in range(int(N)):
            action = np.random.choice(np.array(range(K)), p=pr[0])
            observation, reward, done, info = env_corr.step(action)
            ereward[action] += reward
            actions_sum[action] += 1
            actions.append(action)
            t += 1
        r = ereward / n
        rs = np.max(r - (Delta / 16))
        Delta = np.maximum(2 ** (-1 * m), rs - r)
        m += 1

    def regret(t):
        best = np.argmax(means_real)
        return means_real[best] * t - sum([means_real[actions[i]] for i in range(t)])

    return [regret(t) for t in range(T)]
        
def BARBAR_dist_het_agent(env, K, N, pr):
    '''
    K_j: a 1xK array for agent j, if K_j[i] is 1, agent j has access to arm i, 0 otherwise
    L: a 1xK array showing the accessability of arms, arm i is accessed by L[i] agents
    T: # of rounds
    '''
    ereward = np.zeros(K)
    actions_sum = np.zeros(K)
    actions = []
    for i in range(int(N)):
        action = np.random.choice(np.array(range(K)), p=pr)
        observation, reward, done, info = env.step(action)
        ereward[action] += reward
        actions_sum[action] += 1
        actions.append(action)
    #     rs = np.max(r - (Delta / 16))
    #     Delta = np.maximum(2 ** (-1 * m), rs - r)
    #     m += 1
    # def regret(t):
    #     best = np.argmax(means_real)
    #     return means_real[best] * t - sum([means_real[actions[i]] for i in range(t)])

    # return [regret(t) for t in range(T)]
    return ereward, actions


def BARBAR_dist_het(L, K, T, delta = 0.2):
    L_list = np.sum(K, axis = 0)
    r = np.zeros([L, len(L_list)])
    rs = np.zeros(L)
    Delta = deepcopy(K)
    m = 1
    t = 0
    all_actions = [[]] * L
    k = K.shape[1]
    lmbda = np.log((8 * k / delta) * np.log2(T))
    n = np.zeros([L, k])
    while t < T:
        N_max = 0
        for j in range(L):
            n[j] = (lmbda / (Delta[j]) ** 2) * K[j]
            print("het", n[j])
            n[np.isnan(n)] = 0
            N = np.sum(n[j]/L_list)
            N_max = max(N_max, N)
            pr = n[j] / (L_list * N)
            print(pr)
            if j == 0:
                envv = env_corr
            else:
                envv = env
            er, actions = BARBAR_dist_het_agent(envv, k, N, pr)
            er = (er * L_list)/n[j]
            er[np.isnan(er)] = 0
            r[j] = er
            all_actions[j] += actions
        t += N_max
        r_mean = np.mean(r, axis=0)
        rs = np.max(K * r_mean - (1/16) * Delta, axis=1)
        Delta = np.maximum(2**(-1*m), (np.reshape(rs, [len(rs), 1]) - r_mean)*K)
        m += 1
        print(m, t)
    def regret(t):
        agg_reg = 0
        best = np.argmax(means_real * K, axis = 1)
        for ind in range(L):
            if means_real[best[ind]] * t - sum([means_real[all_actions[ind][i]] for i in range(t)]) < 0:
                print("WHATTTTTTT", means_real[best[ind]] * t - sum([means_real[all_actions[ind][i]] for i in range(t)]))
            agg_reg += means_real[best[ind]] * t - sum([means_real[all_actions[ind][i]] for i in range(t)])
        return agg_reg

    return [regret(t) for t in range(T)]



def main():
    '''Compares the epsilon-greedy approach to the upper confidence bounds
    approach for solving the multi-armed bandit problem.'''

    # results_eps = run_simulation(epsilon_greedy_action)
    results_ucb = run_simulation(upper_confidence_bounds_action, env)
    results_ucb_corr1 = run_simulation(upper_confidence_bounds_action, env_corr)
    results_barbar = BARBAR(10, 5000, delta = 0.2)
    # K = np.random.randint(0, 2, size=[1, 10])

    K = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    results_barbar_het = BARBAR_dist_het(6, K, 5000, delta = 0.2)
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
    barbar_het_patch = mpatches.Patch(color='orange', label='BARBAR Het - 20 agents - Corruption v1')
    plt.legend(handles=[ucb_corr1_patch, ucb_patch, barbar_patch])
    plt.show()


if __name__ == "__main__":
    main()