import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from environment import BanditNArmedBernoulli, BanditNArmedBernoulliCorrupt1
import time 
from copy import deepcopy
from barbar import BARBAR, BARBAR_dist_het, BARBAR_lf_het
from ucb import run_simulation, run_simulation_multi_agent

# class Labeloffset():
    # def __init__(self,  ax, label="", axis="y"):
    #     self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
    #     self.label=label
    #     ax.callbacks.connect(axis+'lim_changed', self.update)
    #     ax.figure.canvas.draw()
    #     self.update(None)

    # def update(self, lim):
    #     fmt = self.axis.get_major_formatter()
    #     self.axis.offsetText.set_visible(False)
    #     self.axis.set_label_text(self.label + " "+ fmt.get_offset() )
        
def exp1(n, T, num_rounds):
    '''
    Cumulative regret @Round 20k for diff num of agents (5 - 105)
    DistHet
    LFHet
    UCB muli agent
    '''
    L_list = list(range(5, 51))[::5]
    Cr = 1
    results_ucb_corr1_multi_agent = np.zeros([num_rounds, len(L_list)])
    results_barbar_het = np.zeros([num_rounds, len(L_list)])
    results_barbar_het_lf = np.zeros([num_rounds, len(L_list)])
    
    for nr in range(num_rounds):
        print("Round ", nr, " of ", num_rounds)

        means_real = np.random.uniform(0, 1, n)
        
        for ind in range(len(L_list)):
            env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = (L_list[0] / L_list[ind]))
            env_corr2.reset()

            print("     Num of agents: ", L_list[ind])
            # K = np.random.binomial(1, 5 / L_list[ind], size=[L_list[ind], n])
            K = np.random.binomial(1, 0.5, size=[L_list[ind], n])
            while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
                # K = np.random.binomial(1, 5 / L_list[ind], size=[L_list[ind], n])
                K = np.random.binomial(1, 0.5, size=[L_list[ind], n])
            results_ucb_corr1_multi_agent[nr, ind] = run_simulation_multi_agent(K, env_corr2, L_list[ind], means_real, T, regret_mode = "final")
            results_barbar_het[nr, ind] = BARBAR_dist_het(env_corr2, means_real, L_list[ind], K, T, regret_mode = "final", delta = 0.2)
            results_barbar_het_lf[nr, ind] = BARBAR_lf_het(env_corr2, means_real, L_list[ind], K, T, regret_mode = "final", delta = 0.2)


    np.save('exp1_results_ucb_corr1_multi_agent_rate_try3.npy', results_ucb_corr1_multi_agent)
    np.save('exp1_results_barbar_het_rate_try3.npy', results_barbar_het)
    np.save('exp1_rresults_barbar_het_lf_rate_try3.npy', results_barbar_het_lf)

    fig, ax = plt.subplots() 
    print(results_ucb_corr1_multi_agent.mean(axis=0))
    print(results_ucb_corr1_multi_agent.std(axis=0))
    ax.plot(L_list, results_ucb_corr1_multi_agent.mean(axis=0), color='red', marker="o", label = "CoopMAUCB")
    ax.fill_between(L_list, results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0),
                    results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0), color='#888888', alpha=0.4)

    ax.plot(L_list, results_barbar_het.mean(axis=0), color='blue', marker="d", label = "DistHetBarbar")
    ax.fill_between(L_list, results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0),
                    results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0), color='#888888', alpha=0.4)
    
    ax.plot(L_list, results_barbar_het_lf.mean(axis=0), color='green', marker="s", label = "LFHetBarbar")
    ax.fill_between(L_list, results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0),
                    results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0), color='#888888', alpha=0.4)

    plt.xlabel("Number of Agents")
    plt.ylabel("Cumulative Regret @Round 20K")
    plt.xlim([L_list[0], L_list[-1]])
    plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)

    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")
    ax.legend()


    plt.show()

    fig, ax = plt.subplots() 
    
    ax.plot(L_list, results_ucb_corr1_multi_agent.mean(axis=0)/L_list, color='red', marker="o", label = "CentralizedMAUCB")
    ax.fill_between(L_list, (results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0))/L_list,
                    (results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0))/L_list, color='#e27979', alpha=0.4)
    
    ax.plot(L_list, results_barbar_het.mean(axis=0)/L_list, color='blue', marker="d", label = "DistHetBarbar")
    ax.fill_between(L_list, (results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0))/L_list,
                    (results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0))/L_list, color='#a4b9db', alpha=0.4)
    
    ax.plot(L_list, results_barbar_het_lf.mean(axis=0)/L_list, color='green', marker="s", label = "LFHetBarbar")
    ax.fill_between(L_list, results_barbar_het_lf.mean(axis=0)/L_list - results_barbar_het_lf.std(axis=0)/L_list,
                    results_barbar_het_lf.mean(axis=0)/L_list + results_barbar_het_lf.std(axis=0)/L_list, color='#bcd7ae', alpha=0.4)


    plt.xlabel("Number of Agents")
    plt.ylabel("Per-agent Avg Regret @Round 20K")
    plt.xlim([L_list[0], L_list[-1]])
    plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)

    # lo = Labeloffset(ax, label="Per-agent Avg Regret @Round 20K", axis="y")
    ax.legend()


    plt.show()
    # plt.title("Bernoulli 10-armed-bandit regret vs time")
  
def exp2(n, L, T, num_rounds, Num_corr_agents = None):
    '''
    Cumulative regret @ every round up to Round 20k
    Everything bagel
    '''
    results_ucb_corr1 = np.zeros([num_rounds, T])
    results_ucb_corr1_multi_agent = np.zeros([num_rounds, T])
    results_barbar = np.zeros([num_rounds, T])
    results_barbar_het = np.zeros([num_rounds, T])
    results_barbar_het_lf = np.zeros([num_rounds, T])

    for nr in range(num_rounds):
        print("Round ", nr, " of ", num_rounds)

        means_real = np.random.uniform(0, 1, n)
        env_corr = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
        env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
        env = BanditNArmedBernoulli(n, deepcopy(means_real))
        env.reset()
        env_corr.reset()
        env_corr2.reset()

        # # results_ucb += run_simulation(env, means_real, T, step = 1)
        # results_ucb_corr1[nr] += run_simulation(env_corr, means_real, T * L, step = L)
        # print("     UCB")
        # # results_ucb_multi_agent += run_simulation_multi_agent(env, L ,means_real, T)
        

        # results_barbar[nr] += BARBAR(env_corr, means_real, n, T * L, delta = 0.5, step = L)
        # # K = np.random.binomial(1, 5 / L, size=[L, n])
        # print("     BARBAR")

        K = np.random.binomial(1, 0.5, size=[L, n])
        while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
            # K = np.random.binomial(1, 5 / L, size=[L, n])
            K = np.random.binomial(1, 0.5, size=[L, n])
            
        if Num_corr_agents != None:
            # results_ucb_corr1_multi_agent[nr] = run_simulation_multi_agent(K, env, L, means_real, T, Num_corr_agents = 1, env_corr = env_corr2)
            results_barbar_het[nr] = BARBAR_dist_het(env, means_real, L, K, T, delta = 0.2, Num_corr_agents = 1, env_corr = env_corr2)
            results_barbar_het_lf[nr] = BARBAR_lf_het(env, means_real, L, K, T, delta = 0.2, Num_corr_agents = 1, env_corr = env_corr2)
        else:
            # results_ucb_corr1_multi_agent[nr] = run_simulation_multi_agent(K, env_corr2, L, means_real, T)
            # print("MA UCB")
            results_barbar_het[nr] = BARBAR_dist_het(env_corr2, means_real, L, K, T, delta = 0.2)
            print("     Dist HET")
            results_barbar_het_lf[nr] = BARBAR_lf_het(env_corr2, means_real, L, K, T, delta = 0.2)
            print("     LF HET")



    # np.save('exp2_results_ucb_corr1.npy', results_ucb_corr1)
    # np.save('exp2_results_barbar.npy', results_barbar)
    # np.save('exp2_results_ucb_corr1_multi_agent.npy', results_ucb_corr1_multi_agent)
    results_ucb_corr1 = np.load('exp2_results_ucb_corr1.npy')
    results_barbar = np.load('exp2_results_barbar.npy')
    results_ucb_corr1_multi_agent = np.load('exp2_results_ucb_corr1_multi_agent.npy')
    np.save('exp2_results_barbar_het.npy', results_barbar_het)
    np.save('exp2_results_barbar_het_lf.npy', results_barbar_het_lf)

    fig, ax = plt.subplots() 
    ax.plot(list(range(T)), results_ucb_corr1.mean(axis=0), marker='x',markevery=2500, label = "UCB - w/ Corruption", color = 'orange')

    ax.fill_between(list(range(T)), (results_ucb_corr1.mean(axis=0) - results_ucb_corr1.std(axis=0)),
                    (results_ucb_corr1.mean(axis=0) + results_ucb_corr1.std(axis=0)), color='#f4c28e', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar.mean(axis=0), marker='+',markevery=2500, label = "BARBAR - w/ Corruption", color='pink')
    ax.fill_between(list(range(T)), (results_barbar.mean(axis=0) - results_barbar.std(axis=0)),
                    (results_barbar.mean(axis=0) + results_barbar.std(axis=0)), color='#eca8d0', alpha=0.4)
    
    ax.plot(list(range(T)), results_ucb_corr1_multi_agent.mean(axis=0), marker='o',markevery=2500, label = "CentralizedMAUCB - w/ Corruption", color='red')
    ax.fill_between(list(range(T)), (results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0)),
                    (results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0)), color='#e27979', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het.mean(axis=0), marker='d',markevery=2500, label = "DistHetBarbar - w/ Corruption", color='blue')
    ax.fill_between(list(range(T)), (results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0)),
                    (results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0)), color='#a4b9db', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het_lf.mean(axis=0), marker='s',markevery=2500, label = "LFHetBarbar - w/ Corruption", color = "green")
    ax.fill_between(list(range(T)), (results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0)),
                    (results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0)), color='#bcd7ae', alpha=0.4)

    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.xlim([0, T])
    plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")
    ax.legend()


    plt.show()

def exp3(n, L, T, num_rounds):
    '''
    Cumulative regret @Round 20k for diff num of agents (5 - 105)
    DistHet
    LFHet
    UCB muli agent
    '''
    r_list = np.array(range(3, 11))/10
    results_ucb_corr1_multi_agent = np.zeros([num_rounds, len(r_list)])
    results_barbar_het = np.zeros([num_rounds, len(r_list)])
    results_barbar_het_lf = np.zeros([num_rounds, len(r_list)])
    for nr in range(num_rounds):
        print("Round ", nr, " of ", num_rounds)

        means_real = np.random.uniform(0, 1, n)
        env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
        env_corr2.reset()

        for ind in range(len(r_list)):
            print("     Rate of agents: ", r_list[ind])
            # K = np.random.binomial(1, 5 / L, size=[L, n])
            K = np.random.binomial(1, r_list[ind], size=[L, n])
            while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
                # K = np.random.binomial(1, 5 / L, size=[L, n])
                K = np.random.binomial(1, r_list[ind], size=[L, n])
            results_ucb_corr1_multi_agent[nr, ind] = run_simulation_multi_agent(K, env_corr2, L, means_real, T, regret_mode = "final")
            results_barbar_het[nr, ind] = BARBAR_dist_het(env_corr2, means_real, L, K, T, regret_mode = "final", delta = 0.2)
            results_barbar_het_lf[nr, ind] = BARBAR_lf_het(env_corr2, means_real, L, K, T, regret_mode = "final", delta = 0.2)

    # np.save('exp3_results_ucb_corr1_multi_agent_rate_try3.npy', results_ucb_corr1_multi_agent)
    # np.save('exp3_results_barbar_het_rate_try3.npy', results_barbar_het)
    # np.save('exp3_rresults_barbar_het_lf_rate_try3.npy', results_barbar_het_lf)

    fig, ax = plt.subplots() 
    
    ax.plot(r_list, results_ucb_corr1_multi_agent.mean(axis=0), color='red', marker="o", label = "CoopMAUCB")
    ax.fill_between(r_list, (results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0)),
                    (results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0)), color='#e27979', alpha=0.4)
    
    ax.plot(r_list, results_barbar_het.mean(axis=0), color='blue', marker="d", label = "DistHetBarbar")
    ax.fill_between(r_list, results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0),
                    results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0), color='#a4b9db', alpha=0.4)
    
    ax.plot(r_list, results_barbar_het_lf.mean(axis=0), color='green', marker="s", label = "LFHetBarbar")
    ax.fill_between(r_list, results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0),
                    results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0), color='#bcd7ae', alpha=0.4)
    
    plt.xlabel("Prob. of Assigning an Arm to an Agent")
    plt.ylabel("Cumulative Regret @Round 10K")
    plt.xlim([r_list[0], r_list[-1]])
    plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)

    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")
    ax.legend()


    plt.show()

    fig, ax = plt.subplots() 
    
    ax.plot(r_list, results_ucb_corr1_multi_agent.mean(axis=0)/L, color='red', marker="o", label = "CentralizedMAUCB")
    ax.fill_between(r_list, (results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0))/L,
                    (results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0))/L, color='#888888', alpha=0.4)
    
    ax.plot(r_list, results_barbar_het.mean(axis=0)/L, color='blue', marker="d", label = "DistHetBarbar")
    ax.fill_between(r_list, (results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0))/L,
                    (results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0))/L, color='#888888', alpha=0.4)
    
    ax.plot(r_list, results_barbar_het_lf.mean(axis=0)/L, color='green', marker="s", label = "LFHetBarbar")
    ax.fill_between(r_list, (results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0))/L,
                    (results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0))/L, color='#888888', alpha=0.4)

    plt.xlabel("Prob. of Assigning an Arm to an Agent")
    plt.ylabel("Per-agent Avg Regret @Round 20K")
    plt.xlim([r_list[0], r_list[-1]])
    plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)

    # lo = Labeloffset(ax, label="Per-agent Avg Regret @Round 20K", axis="y")
    ax.legend()


    plt.show()
    # plt.title("Bernoulli 10-armed-bandit regret vs time")

def exp4(n, L, T, num_rounds):
    exp2(n, L, T, num_rounds, Num_corr_agents = 1)

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

    n = 50
    L = 10
    T = 2000
    num_rounds = 2
    # exp 1: 
        # X_axis: Cumulative regret at the end of round 20k
        # Y_axis: Number of agents 5 - 100
        # dist_het lf_het ucb_multi
    # exp1(n, T, num_rounds)
    # exp2(n, L, T, num_rounds)
    exp3(n, L, T, num_rounds)
    # exp4(n, L, T, num_rounds)
#     results_ucb = np.zeros(T)
#     results_ucb_corr1 = np.zeros(T)
#     results_ucb_corr1_multi_agent = np.zeros(T)
#     results_ucb_multi_agent = np.zeros(T)
#     results_barbar = np.zeros(T)
#     results_barbar_het = np.zeros(T)
#     results_barbar_het_lf = np.zeros(T)
#     for num in range(10):
#         print(num)
#         # means_real = np.random.uniform(0, 1, 10)
#         means_real = np.array([0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
#         env_corr = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 2, corr_rate = 1)
#         env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 2, corr_rate = 0.2)
#         # ev_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 2, corr_total_v2 = 5000)
#         env = BanditNArmedBernoulli(n, deepcopy(means_real))
#         env.reset()

#         # results_eps = run_simulation(epsilon_greedy_action)
#         results_ucb += run_simulation(env, means_real, T * L, step = L)
#         results_ucb_corr1 += run_simulation(env_corr, means_real, T * L, step = L)

#         results_ucb_multi_agent += run_simulation_multi_agent(env, L ,means_real, T)
#         results_ucb_corr1_multi_agent += run_simulation_multi_agent(env_corr2, L, means_real, T)

#         results_barbar += BARBAR(env_corr, means_real, n, T * L, delta = 0.5, step = L)
#         # K = np.random.randint(0, 2, size=[1, 10])
#         # K = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        
#         K = np.random.binomial(1, 0.8, size=[L, n])
#         results_barbar_het += BARBAR_dist_het(env_corr2, means_real, L, K, T, delta = 0.2)
#         results_barbar_het_lf += BARBAR_lf_het(env_corr2, means_real, L, K, T, delta = 0.2)

#     results_ucb /= 10
#     results_ucb_corr1 /= 10
#     results_ucb_corr1_multi_agent /= 10
#     results_ucb_multi_agent /= 10
#     results_barbar /= 10
#     results_barbar_het /= 10
#     results_barbar_het_lf /= 10
#     plt.plot(results_ucb_corr1, color='red')
#     plt.plot(results_ucb, color='blue')

#     plt.plot(results_ucb_corr1_multi_agent, color='pink')
#     plt.plot(results_ucb_multi_agent, color='purple')

#     plt.plot(np.array(results_barbar), color='green')
#     plt.plot(results_barbar_het, color='orange')
#     plt.plot(results_barbar_het_lf, color='cyan')

#     plt.xlabel("Timestep")
#     plt.ylabel("Pseudo Regret")
#     plt.title("Bernoulli 10-armed-bandit regret vs time")

#     ucb_corr1_patch = mpatches.Patch(color='red', label='upper confidence bounds - Corruption v1')
#     ucb_patch = mpatches.Patch(color='blue', label='upper confidence bounds')

#     ucb_corr1_multi_agent_patch = mpatches.Patch(color='pink', label='upper confidence bounds - Multi-agent - Corruption v1')
#     ucb_multi_agent_patch = mpatches.Patch(color='purple', label='upper confidence bounds - Multi-agent - Multi-agent')

#     barbar_patch = mpatches.Patch(color='green', label='BARBAR - Corruption v1')
#     barbar_het_patch = mpatches.Patch(color='orange', label='BARBAR Het - 5 agents - Corruption v1')
#     barbar_lf_het_patch = mpatches.Patch(color='cyan', label='BARBAR Het LF - 5 agents - Corruption v1')
#     plt.legend(handles=[ucb_corr1_patch, ucb_patch, ucb_corr1_multi_agent_patch, ucb_multi_agent_patch, barbar_patch, barbar_het_patch, barbar_lf_het_patch])
#     plt.show()


#     # K = np.random.binomial(1, 0.4, size=[L, n])
#     # while 0 in np.sum(K, axis=0):
#     #     K = np.random.binomial(1, 0.4, size=[L, n])
#     # print(K)
#     # results_barbar_het_2 = BARBAR_dist_het(env_corr2, means_real, L, K, T, delta = 0.2)
#     # results_barbar_het_lf_2 = BARBAR_lf_het(env_corr2, means_real, L, K, T, delta = 0.2)

#     # K = np.random.binomial(1, 0.6, size=[L, n])
#     # while 0 in np.sum(K, axis=0):
#     #     K = np.random.binomial(1, 0.4, size=[L, n])
#     # print(K)
#     # results_barbar_het_3 = BARBAR_dist_het(env_corr2, means_real, L, K, T, delta = 0.2)
#     # results_barbar_het_lf_3 = BARBAR_lf_het(env_corr2, means_real, L, K, T, delta = 0.2)

#     # K = np.random.binomial(1, 0.8, size=[L, n])
#     # while 0 in np.sum(K, axis=0):
#     #     K = np.random.binomial(1, 0.4, size=[L, n])
#     # print(K)
#     # results_barbar_het_4 = BARBAR_dist_het(env_corr2, means_real, L, K, T, delta = 0.2)
#     # results_barbar_het_lf_4 = BARBAR_lf_het(env_corr2, means_real, L, K, T, delta = 0.2)

#     # K = np.random.binomial(1, 1, size=[L, n])
#    # while 0 in np.sum(K, axis=0):
#   #     K = np.random.binomial(1, 0.4, size=[L, n])
#     # print(K)
#     # results_barbar_het_5 = BARBAR_dist_het(env_corr2, means_real, L, K, T, delta = 0.2)
#     # results_barbar_het_lf_5 = BARBAR_lf_het(env_corr2, means_real, L, K, T, delta = 0.2)

#     # plt.plot(results_barbar_het_2, label='dist 0.4')
#     # plt.plot(results_barbar_het_3, label='dist 0.6')
#     # plt.plot(results_barbar_het_4, label='dist 0.8')
#     # plt.plot(results_barbar_het_5, label='dist 1')

#     # plt.plot(results_barbar_het_lf_2, label='lf 0.4')
#     # plt.plot(results_barbar_het_lf_3, label='lf 0.6')
#     # plt.plot(results_barbar_het_lf_4, label='lf 0.8')
#     # plt.plot(results_barbar_het_lf_5, label='lf 1')

#     # plt.legend()
#     # plt.show()
if __name__ == "__main__":
    main()