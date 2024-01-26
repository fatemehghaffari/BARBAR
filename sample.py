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
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
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
    L_list = list(range(10, 51))[::5]
        
    # # # Cr = 1
    # results_ucb_corr1_multi_agent = np.zeros([num_rounds, len(L_list)])
    # results_barbar_het = np.zeros([num_rounds, len(L_list)])
    # results_barbar_het_lf = np.zeros([num_rounds, len(L_list)])
    # results_barbar = np.zeros([num_rounds, len(L_list)])
    # for nr in range(num_rounds):
    #     print("Round ", nr, " of ", num_rounds)

    #     means_real = np.random.uniform(0, 1, n)
    #     # means_real = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    #     for ind in range(len(L_list)):
    #         # env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = (L_list[ind]/L_list[-1]))
    #         env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
    #         env_corr2.reset()
    #         env = BanditNArmedBernoulli(n, deepcopy(means_real))
    #         env.reset()
    #         print("     Num of agents: ", L_list[ind])
    #         # K = np.random.binomial(1, 5 / L_list[ind], size=[L_list[ind], n])
    #         K = np.random.binomial(1, 0.4, size=[L_list[ind], n])
    #         while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
    #             # K = np.random.binomial(1, 5 / L_list[ind], size=[L_list[ind], n])
    #             K = np.random.binomial(1, 0.4, size=[L_list[ind], n])
            # for j in range(K.shape[0]):   
            #     # run_simulation_multi_agent(K, env_corr2, L, means_real, T, 2)
            #     env_corr = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]), corr_ver = 1, corr_rate = 1)
            #     env_corr.reset()

            #     env2 = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]))
            #     env2.reset()
            #         # # results_ucb += run_simulation(env, means_real, T, step = 1)
            #     if j <= 9:
            #         x = BARBAR(env_corr, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1, regret_mode='final')
            #         results_barbar[nr, ind] += x
            #     else:
            #         results_barbar[nr, ind] += BARBAR(env2, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1, regret_mode='final')
    #         # results_ucb_corr1_multi_agent[nr, ind] = run_simulation_multi_agent(K , env_corr2, L_list[ind], means_real, T,
    #         #                                                                      regret_mode = "final")
    #         # results_barbar_het[nr, ind] = BARBAR_dist_het(env_corr2, means_real, L_list[ind], K , T, 
    #         #                                               regret_mode = "final", delta = 0.01)
    #         # results_barbar_het_lf[nr, ind] = BARBAR_lf_het(env_corr2, means_real, L_list[ind], K , T, 
    #         #                                                regret_mode = "final", delta = 0.01)
            # results_ucb_corr1_multi_agent[nr, ind] = run_simulation_multi_agent(K[:L_list[0], :], env_corr2, L_list[0], means_real, T,
            #                                                                      regret_mode = "final")
            # results_barbar_het[nr, ind] = BARBAR_dist_het(env_corr2, means_real, L_list[0], K[:L_list[0], :], T, 
            #                                               regret_mode = "final", delta = 0.01)
            # results_barbar_het_lf[nr, ind] = BARBAR_lf_het(env_corr2, means_real, L_list[0], K[:L_list[0], :], T, 
            #                                                regret_mode = "final", delta = 0.01)
            # if L_list[ind] - L_list[0] > 0:
            #     results_ucb_corr1_multi_agent[nr, ind] += run_simulation_multi_agent(K[L_list[0]:, :], env, L_list[ind] - L_list[0], means_real, T,
            #                                                                      regret_mode = "final")
            #     results_barbar_het[nr, ind] += BARBAR_dist_het(env, means_real, L_list[ind] - L_list[0], K[L_list[0]:, :], T, 
            #                                                 regret_mode = "final", delta = 0.01)
            #     results_barbar_het_lf[nr, ind] += BARBAR_lf_het(env, means_real, L_list[ind] - L_list[0], K[L_list[0]:, :], T, 
                                                        #    regret_mode = "final", delta = 0.01)


    results_barbar = np.load('exp1_results_barbar_L10K50T10KCv2.npy')
    results_ucb_corr1_multi_agent = np.load('exp1_results_ucb_corr1_multi_agent_rate_L10K50T10KCv2.npy')
    results_barbar_het = np.load('exp1_results_barbar_het_rate_L10K50T10KCv2.npy')
    results_barbar_het_lf = np.load('exp1_rresults_barbar_het_lf_rate_L10K50T10KCv2.npy')
                
    # results_ucb_corr1_multi_agent = np.load('exp1_results_ucb_corr1_multi_agent_rate_L10K50T10KCv2.npy')
    # results_barbar_het = np.load('exp1_results_barbar_het_rate_L10K50T10KCv2.npy')
    # results_barbar_het_lf = np.load('exp1_rresults_barbar_het_lf_rate_L10K50T10KCv2.npy')

    # results_barbar_het = np.load('exp1_results_barbar_het_rate_try3.npy')
    # results_barbar_het_lf = np.load('exp1_rresults_barbar_het_lf_rate_try3.npy')

    fig, ax = plt.subplots() 

    ax.plot(L_list, results_barbar.mean(axis=0), color='orange', marker="*", label = "IndBARBAR")
    ax.fill_between(L_list, results_barbar.mean(axis=0) - results_barbar.std(axis=0),
                    results_barbar.mean(axis=0) + results_barbar.std(axis=0), color='#f4c28e', alpha=0.4)

    ax.plot(L_list, results_ucb_corr1_multi_agent.mean(axis=0), color='red', marker="o", label = "CO-UCB")
    ax.fill_between(L_list, results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0),
                    results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0), color='#e27979', alpha=0.4)

    ax.plot(L_list, results_barbar_het.mean(axis=0), color='blue', marker="d", label = "DistBARBAR")
    ax.fill_between(L_list, results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0),
                    results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0), color='#a4b9db', alpha=0.4)
    
    ax.plot(L_list, results_barbar_het_lf.mean(axis=0), color='green', marker="s", label = "LFBARBAR")
    ax.fill_between(L_list, results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0),
                    results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0), color='#bcd7ae', alpha=0.4)
    plt.grid()
    plt.xlabel("Number of Agents")
    plt.ylabel("Cumulative Regret @10K")
    plt.xlim([L_list[0], L_list[-1]])
    plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")
    ax.legend(prop={'size': 15})
    plt.tight_layout()
    plt.savefig('exp1_K50L10T10kCv2_w_barbar.png', dpi=300)
    
    plt.show()


def exp5(n, L, T, num_rounds):
    '''
    Cumulative regret @Round 20k for diff num of agents (5 - 105)
    DistHet
    LFHet
    UCB muli agent
    '''
    C_list = np.array(range(11))/10
    # # results_ucb_corr1_multi_agent = np.zeros([num_rounds, len(C_list)])
    # # results_barbar_het = np.zeros([num_rounds, len(C_list)])
    # # results_barbar_het_lf = np.zeros([num_rounds, len(C_list)])
    # results_barbar = np.zeros([num_rounds, len(C_list)])
    # for nr in range(num_rounds):
    #     print("Round ", nr, " of ", num_rounds)

    #     means_real = np.random.uniform(0, 1, n)
        
    #     for ind in range(len(C_list)):
    #         # env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = (C_list[ind]))
    #         env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
    #         env = BanditNArmedBernoulli(n, deepcopy(means_real))
    #         env.reset()
    #         env_corr2.reset()

    #         print("     Corr Rate: ",  C_list[ind])
    #         # K = np.random.binomial(1, 5 / C_list[ind], size=[C_list[ind], n])
    #         K = np.random.binomial(1, 0.4, size=[L, n])
            
    #         while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
    #             # K = np.random.binomial(1, 5 / C_list[ind], size=[C_list[ind], n])
    #             K = np.random.binomial(1, 0.4, size=[L, n])
    #         for j in range(K.shape[0]):   
    #             # run_simulation_multi_agent(K, env_corr2, L, means_real, T, 2)
    #             env_corr = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]), corr_ver = 1, corr_rate = 1)
    #             env_corr.reset()

    #             env2 = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]))
    #             env2.reset()
    #                 # # results_ucb += run_simulation(env, means_real, T, step = 1)
    #             if j < int(L * C_list[ind]):
    #                 x = BARBAR(env_corr, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1, regret_mode='final')
    #                 results_barbar[nr, ind] += x
    #             else:
    #                 results_barbar[nr, ind] += BARBAR(env2, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1, regret_mode='final')
    #         # if C_list[ind] != 0:
    #         #     results_ucb_corr1_multi_agent[nr, ind] = run_simulation_multi_agent(K[:int(L * C_list[ind]), :], env_corr2, int(L * C_list[ind]), means_real, T, regret_mode = "final")
    #         #     results_barbar_het[nr, ind] = BARBAR_dist_het(env_corr2, means_real, int(L * C_list[ind]), K[:int(L * C_list[ind]), :], T, regret_mode = "final", delta = 0.2)
    #         #     results_barbar_het_lf[nr, ind] = BARBAR_lf_het(env_corr2, means_real, int(L * C_list[ind]), K[:int(L * C_list[ind]), :], T, regret_mode = "final", delta = 0.2)
    #         # if C_list[ind] != 1:
    #         #     results_ucb_corr1_multi_agent[nr, ind] += run_simulation_multi_agent(K[int(L * C_list[ind]):, :], env, int(L -  L * C_list[ind]), means_real, T, regret_mode = "final")
    #         #     results_barbar_het[nr, ind] += BARBAR_dist_het(env, means_real, int(L -  L * C_list[ind]), K[int(L * C_list[ind]):, :], T, regret_mode = "final", delta = 0.2)
    #         #     results_barbar_het_lf[nr, ind] += BARBAR_lf_het(env, means_real, int(L -  L * C_list[ind]), K[int(L * C_list[ind]):, :], T, regret_mode = "final", delta = 0.2)
    #         results_ucb_corr1_multi_agent[nr, ind] += run_simulation_multi_agent(K, env, L, means_real, T, regret_mode = "final", Num_corr_agents = int(L * C_list[ind]), env_corr = env_corr2)
    #         results_barbar_het[nr, ind] += BARBAR_dist_het(env, means_real, L, K, T, regret_mode = "final", delta = 0.01, Num_corr_agents = int(L * C_list[ind]), env_corr = env_corr2)
    #         results_barbar_het_lf[nr, ind] += BARBAR_lf_het(env, means_real, L, K, T, regret_mode = "final", delta = 0.01, Num_corr_agents = int(L * C_list[ind]), env_corr = env_corr2)


    # np.save('exp5_results_ucb_corr1_multi_agent_rate_L10K50T10_50.npy', results_ucb_corr1_multi_agent)
    # np.save('exp5_results_barbar_het_rate_L10K50T10_50.npy', results_barbar_het)
    # np.save('exp5_rresults_barbar_het_lf_rate_L10K50T10_50.npy', results_barbar_het_lf)
    # np.save('exp5_barbar.npy', results_barbar)
    results_barbar = np.load('exp5_barbar.npy')
    results_ucb_corr1_multi_agent = np.load('exp5_results_ucb_corr1_multi_agent_rate_L10K50T10_50.npy')
    results_barbar_het = np.load('exp5_results_barbar_het_rate_L10K50T10_50.npy')
    results_barbar_het_lf = np.load('exp5_rresults_barbar_het_lf_rate_L10K50T10_50.npy')
    print(results_barbar.mean(axis=0)[-3])
    print(results_ucb_corr1_multi_agent.mean(axis=0)[-3])
    print(results_barbar_het.mean(axis=0)[-3])
    print("lf to co ucb:", (results_ucb_corr1_multi_agent.mean(axis=0)[-3] - results_barbar_het_lf.mean(axis=0)[-3]) / results_ucb_corr1_multi_agent.mean(axis=0)[-3])
    print("lf to co barbar:", (results_barbar.mean(axis=0)[-3] - results_barbar_het_lf.mean(axis=0)[-3]) / results_barbar.mean(axis=0)[-3])
    print("dist to co ucb:", (results_ucb_corr1_multi_agent.mean(axis=0)[-3] - results_barbar_het.mean(axis=0)[-3]) / results_ucb_corr1_multi_agent.mean(axis=0)[-3])
    print("dist to co barbar:", (results_barbar.mean(axis=0)[-3] - results_barbar_het.mean(axis=0)[-3]) / results_barbar.mean(axis=0)[-3])
    
    print()
    fig, ax = plt.subplots() 
    print(results_ucb_corr1_multi_agent.mean(axis=0))
    print(results_ucb_corr1_multi_agent.std(axis=0))

    ax.plot(C_list, results_barbar.mean(axis=0), color='orange', marker="*", label = "IndBARBAR")
    ax.fill_between(C_list, results_barbar.mean(axis=0) - results_barbar.std(axis=0),
                    results_barbar.mean(axis=0) + results_barbar.std(axis=0), color='#f4c28e', alpha=0.4)
    
    ax.plot(C_list, results_ucb_corr1_multi_agent.mean(axis=0), color='red', marker="o", label = "CO-UCB")
    ax.fill_between(C_list, results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0),
                    results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0), color='#e27979', alpha=0.4)

    ax.plot(C_list, results_barbar_het.mean(axis=0), color='blue', marker="d", label = "DistBARBAR")
    ax.fill_between(C_list, results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0),
                    results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0), color='#a4b9db', alpha=0.4)
    
    ax.plot(C_list, results_barbar_het_lf.mean(axis=0), color='green', marker="s", label = "LFBARBAR")
    ax.fill_between(C_list, results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0),
                    results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0), color='#bcd7ae', alpha=0.4)
    

    plt.xlabel("Corruption Rate")
    plt.ylabel("Cumulative Regret @10K")
    plt.xlim([C_list[0], C_list[-1]])
    plt.ylim(bottom = 0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(prop={'size': 15})
    plt.tight_layout()
    plt.savefig('exp5s_K50L10T10kCv2_tight_2.png', dpi=300)
    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")
    


    plt.show()

def exp2(n, L, T, num_rounds, Num_corr_agents = None):
    '''
    Cumulative regret @ every round up to Round 20k
    Everything bagel
    '''
    # results_ucb_corr1 = np.zeros([num_rounds, T])
    # results_ucb_corr1_multi_agent = np.zeros([num_rounds, T])
    # results_barbar = np.zeros([num_rounds, T])
    # results_barbar_het = np.zeros([num_rounds, T])
    # results_barbar_het_lf = np.zeros([num_rounds, T])

    # for nr in range(num_rounds):
    #     print("Round ", nr, " of ", num_rounds)

    #     means_real = np.random.uniform(0, 1, n)
    #     # means_real = np.array([0.1 for _ in range(9)] + [0.9])
    #     env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
    #     env = BanditNArmedBernoulli(n, deepcopy(means_real))
    #     env.reset()
        
    #     env_corr2.reset()

    #     K = np.random.binomial(1, 0.4, size=[L, n])
    #     while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
    #         # K = np.random.binomial(1, 5 / L, size=[L, n])
    #         K = np.random.binomial(1, 0.4, size=[L, n])
    #     for j in range(K.shape[0]):   
    #         # run_simulation_multi_agent(K, env_corr2, L, means_real, T, 2)
    #         env_corr = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]), corr_ver = 1, corr_rate = 1)
    #         env_corr.reset()
    #         env2 = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]))
    #         env2.reset()
    #             # # results_ucb += run_simulation(env, means_real, T, step = 1)
    #         if j < int(L * 1):
    #             results_barbar[nr] += BARBAR(env_corr, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1)
    #             print("     BARBAR")
    #         else: 
    #             results_barbar[nr] += BARBAR(env, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1)
    #             print("     BARBAR")

    #     # if Num_corr_agents != None:
    #     results_ucb_corr1_multi_agent[nr] = run_simulation_multi_agent(K, env, L, means_real, T, Num_corr_agents = L, env_corr = env_corr2)
    #     results_barbar_het[nr] = BARBAR_dist_het(env, means_real, L, K, T, delta = 0.01, Num_corr_agents = L, env_corr = env_corr2)
    #     results_barbar_het_lf[nr] = BARBAR_lf_het(env, means_real, L, K, T, delta = 0.01, Num_corr_agents = L, env_corr = env_corr2)
        # else:
        #     results_ucb_corr1_multi_agent[nr] = run_simulation_multi_agent(K[:int(L*0.4), :], env, int(L*0.4), means_real, T)
        #     print("MA UCB")
        #     results_barbar_het[nr] = BARBAR_dist_het(env, means_real, int(L*0.4), K[:int(L*0.4), :], T, delta = 0.01)
        #     print("     Dist HET")
        #     results_barbar_het_lf[nr] = BARBAR_lf_het(env, means_real, int(L*0.4), K[:int(L*0.4), :], T, delta = 0.01)
        #     print("     LF HET")

        #     results_ucb_corr1_multi_agent[nr] += run_simulation_multi_agent(K[int(L*0.4):, :], env_corr2, int(L*0.6), means_real, T)
        #     print("MA UCB")
        #     results_barbar_het[nr] += BARBAR_dist_het(env_corr2, means_real, int(L*0.6), K[int(L*0.4):, :], T, delta = 0.01)
        #     print("     Dist HET")
        #     results_barbar_het_lf[nr] += BARBAR_lf_het(env_corr2, means_real, int(L*0.6), K[int(L*0.4):, :], T, delta = 0.01)
        #     print("     LF HET")



    # np.save('exp2_results_barbar.npy', results_barbar)
    # np.save('exp2_results_ucb_corr1_multi_agent.npy', results_ucb_corr1_multi_agent)
    # np.save('exp2_results_barbar_het.npy', results_barbar_het)
    # np.save('exp2_results_barbar_het_lf.npy', results_barbar_het_lf)

    results_barbar = np.load('exp2_results_barbar.npy')
    results_ucb_corr1_multi_agent = np.load('exp2_results_ucb_corr1_multi_agent.npy')
    results_barbar_het = np.load('exp2_results_barbar_het.npy')
    results_barbar_het_lf = np.load('exp2_results_barbar_het_lf.npy')

    # results_ucb_corr1_multi_agent = np.load('exp2_results_ucb_corr1_multi_agent.npy')
    # results_ucb_corr1 = np.load('exp2_results_ucb_corr1.npy')
    # results_barbar = np.load('exp2_results_barbar.npy')
    # results_ucb_corr1_multi_agent = np.load('exp2_results_ucb_corr1_multi_agent.npy')
    # results_barbar_het = np.load('exp2_results_barbar_het.npy')
    # results_barbar_het_lf = np.load('exp2_results_barbar_het_lf.npy')

    fig, ax = plt.subplots()
    
    ax.plot(list(range(T)), results_barbar.mean(axis=0), marker='+',markevery=2000, label = "IndBARBAR", color='orange')
    ax.fill_between(list(range(T)), (results_barbar.mean(axis=0) - results_barbar.std(axis=0)),
                    (results_barbar.mean(axis=0) + results_barbar.std(axis=0)), color='#f4c28e', alpha=0.4)
    
    ax.plot(list(range(T)), results_ucb_corr1_multi_agent.mean(axis=0), marker='o',markevery=2000, label = "CO-UCB", color='red')
    ax.fill_between(list(range(T)), (results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0)),
                    (results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0)), color='#e27979', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het.mean(axis=0), marker='d',markevery=2000, label = "DistBARBAR", color='blue')
    ax.fill_between(list(range(T)), (results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0)),
                    (results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0)), color='#a4b9db', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het_lf.mean(axis=0), marker='s',markevery=2000, label = "LFBARBAR", color = "green")
    ax.fill_between(list(range(T)), (results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0)),
                    (results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0)), color='#bcd7ae', alpha=0.4)

    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.xlim([0, T])
    plt.ylim(bottom = 0)
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    # ax.xaxis.set_major_formatter(formatter)
    ax.legend(prop={'size': 16})
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')

    plt.tight_layout()
    plt.savefig('exp2_K50L10T10kC1_tight_2.png', dpi=300)
    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")



    plt.show()

def exp3(n, L, T, num_rounds):
    '''
    Cumulative regret @Round 20k for diff num of agents (5 - 105)
    DistHet
    LFHet
    UCB muli agent
    '''
    r_list = np.array(range(3, 11))/10
    # results_ucb_corr1_multi_agent = np.zeros([num_rounds, len(r_list)])
    # results_barbar_het = np.zeros([num_rounds, len(r_list)])
    # results_barbar_het_lf = np.zeros([num_rounds, len(r_list)])
    # results_barbar = np.zeros([num_rounds, len(r_list)])
    # for nr in range(num_rounds):
    #     print("Round ", nr, " of ", num_rounds)
    #     means_real = np.random.uniform(0, 1, n)
        
    #     env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
    #     env_corr2.reset()
    #     env = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 0)
    #     env.reset()

    #     for ind in range(len(r_list)):
    #     # for ind in range(5):
    #         print("     Rate of agents: ", r_list[ind])
    #         # K = np.random.binomial(1, 5 / L, size=[L, n])
    #         K = np.random.binomial(1, r_list[ind], size=[L, n])
    #         # K[:, means_real.argmax()] = 1
    #         # K[:, means_real.argmin()] = 1
    #         while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
    #             # K = np.random.binomial(1, 5 / L, size=[L, n])
    #             K = np.random.binomial(1, r_list[ind], size=[L, n])
    #             # K[:, means_real.argmax()] = 1
    #             # K[:, means_real.argmin()] = 1
    #         for j in range(K.shape[0]):   
    #             # run_simulation_multi_agent(K, env_corr2, L, means_real, T, 2)
    #             env_corr = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]), corr_ver = 1, corr_rate = 1)
    #             env_corr.reset()

    #             env2 = BanditNArmedBernoulli(len(means_real[K[j] == 1]), deepcopy(means_real[K[j] == 1]))
    #             env2.reset()
    #                 # # results_ucb += run_simulation(env, means_real, T, step = 1)
    #             if j <= 7:
    #                 x = BARBAR(env_corr, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1, regret_mode='final')
    #                 results_barbar[nr, ind] += x
    #             else:
    #                 results_barbar[nr, ind] += BARBAR(env2, means_real[K[j] == 1], len(means_real[K[j] == 1]), T, delta = 0.1, regret_mode='final')
    #         print(K.sum(axis = 0).min())
            
    #         # results_ucb_corr1_multi_agent[nr, ind] = run_simulation_multi_agent(K[:int(L*0.4), :], env, int(L * 0.4), means_real, T, regret_mode = "final")
    #         # results_barbar_het[nr, ind] = BARBAR_dist_het(env, means_real, int(L * 0.4), K[:int(L*0.4), :], T, regret_mode = "final", delta = 0.01)
    #         # results_barbar_het_lf[nr, ind] = BARBAR_lf_het(env, means_real, int(L * 0.4), K[:int(L*0.4), :], T, regret_mode = "final", delta = 0.01)

    #         # results_ucb_corr1_multi_agent[nr, ind] += run_simulation_multi_agent(K[int(L*0.4):, :], env_corr2, int(L * 0.6), means_real, T, regret_mode = "final")
    #         # results_barbar_het[nr, ind] += BARBAR_dist_het(env_corr2, means_real, int(L * 0.6), K[int(L*0.4):, :], T, regret_mode = "final", delta = 0.01)
    #         # results_barbar_het_lf[nr, ind] += BARBAR_lf_het(env_corr2, means_real, int(L * 0.6), K[int(L*0.4):, :], T, regret_mode = "final", delta = 0.01)
    #         # print(results_ucb_corr1_multi_agent[nr, ind], results_barbar_het[nr, ind], results_barbar_het_lf[nr, ind])
    #         results_ucb_corr1_multi_agent[nr, ind] += run_simulation_multi_agent(K, env, int(L), means_real, T, regret_mode = "final", Num_corr_agents = 8, env_corr = env_corr2)
    #         results_barbar_het[nr, ind] += BARBAR_dist_het(env, means_real, int(L), K, T, regret_mode = "final", delta = 0.01, Num_corr_agents = 8, env_corr = env_corr2)
    #         results_barbar_het_lf[nr, ind] += BARBAR_lf_het(env, means_real, int(L), K, T, regret_mode = "final", delta = 0.01, Num_corr_agents = 8, env_corr = env_corr2)
    #         print(results_ucb_corr1_multi_agent[nr, ind], results_barbar_het[nr, ind], results_barbar_het_lf[nr, ind])

    # np.save('exp3_results_barbar_rate_try3.npy', results_barbar)
    results_barbar = np.load('exp3_results_barbar_rate_try3.npy')
    results_ucb_corr1_multi_agent = np.load('exp3_results_ucb_corr1_multi_agent_rate_try3.npy')
    results_barbar_het = np.load('exp3_results_barbar_het_rate_try3.npy')
    results_barbar_het_lf = np.load('exp3_rresults_barbar_het_lf_rate_try3.npy')

    fig, ax = plt.subplots() 
    
    ax.plot(r_list, results_barbar.mean(axis=0), color='orange', marker="*", label = "IndBARBAR")
    ax.fill_between(r_list, (results_barbar.mean(axis=0) - results_barbar.std(axis=0)),
                    (results_barbar.mean(axis=0) + results_barbar.std(axis=0)), color='#f4c28e', alpha=0.4)
    

    ax.plot(r_list, results_ucb_corr1_multi_agent.mean(axis=0), color='red', marker="o", label = "CO-UCB")
    ax.fill_between(r_list, (results_ucb_corr1_multi_agent.mean(axis=0) - results_ucb_corr1_multi_agent.std(axis=0)),
                    (results_ucb_corr1_multi_agent.mean(axis=0) + results_ucb_corr1_multi_agent.std(axis=0)), color='#e27979', alpha=0.4)
    
    ax.plot(r_list, results_barbar_het.mean(axis=0), color='blue', marker="d", label = "DistBARBAR")
    ax.fill_between(r_list, results_barbar_het.mean(axis=0) - results_barbar_het.std(axis=0),
                    results_barbar_het.mean(axis=0) + results_barbar_het.std(axis=0), color='#a4b9db', alpha=0.4)
    
    ax.plot(r_list, results_barbar_het_lf.mean(axis=0), color='green', marker="s", label = "LFBARBAR")
    ax.fill_between(r_list, results_barbar_het_lf.mean(axis=0) - results_barbar_het_lf.std(axis=0),
                    results_barbar_het_lf.mean(axis=0) + results_barbar_het_lf.std(axis=0), color='#bcd7ae', alpha=0.4)
    
    plt.xlabel("Prob. of Assigning an Arm to an Agent")
    plt.ylabel("Cumulative Regret @10K")
    plt.xlim([r_list[0], r_list[-1]])
    # plt.ylim(bottom = 0)
    plt.grid()
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(prop={'size': 15})
    plt.tight_layout()
    plt.savefig('exp3_K50L10T10kCv2_tight.png', dpi=300)
    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")
    


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
    # plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)

    # lo = Labeloffset(ax, label="Per-agent Avg Regret @Round 20K", axis="y")
    ax.legend()


    plt.show()
    # plt.title("Bernoulli 10-armed-bandit regret vs time")

def exp4(n, L, T, num_rounds):
    exp2(n, L, T, num_rounds, Num_corr_agents = 1)

def exp6(n, L, T, num_rounds, Num_corr_agents = None):
    '''
    Cumulative regret @ every round up to Round 20k
    Everything bagel
    '''
    # results_ucb_corr1_multi_agent_corr = np.zeros([num_rounds, T])
    # results_barbar_het_corr = np.zeros([num_rounds, T])
    # results_barbar_het_lf_corr = np.zeros([num_rounds, T])

    # results_ucb_corr1_multi_agent_nocorr = np.zeros([num_rounds, T])
    # results_barbar_het_nocorr = np.zeros([num_rounds, T])
    # results_barbar_het_lf_nocorr = np.zeros([num_rounds, T])

    # for nr in range(num_rounds):
    #     print("Round ", nr, " of ", num_rounds)
    #     # means_real = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1])
    #     means_real = np.random.uniform(0, 1, n)
    #     env_corr = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
    #     env_corr2 = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 1)
    #     env = BanditNArmedBernoulli(n, deepcopy(means_real), corr_ver = 1, corr_rate = 0)
    #     env.reset()
    #     env_corr.reset()
    #     env_corr2.reset()

    #     # # results_ucb += run_simulation(env, means_real, T, step = 1)
    #     # results_ucb_corr1[nr] += run_simulation(env_corr, means_real, T * L, step = L)
    #     # print("     UCB")
    #     # # results_ucb_multi_agent += run_simulation_multi_agent(env, L ,means_real, T)
        

    #     # results_barbar[nr] += BARBAR(env_corr, means_real, n, T * L, delta = 0.5, step = L)
    #     # # K = np.random.binomial(1, 5 / L, size=[L, n])
    #     # print("     BARBAR")

    #     K = np.random.binomial(1, 0.4, size=[L, n])
    #     while (0 in np.sum(K, axis=0)) or (0 in np.sum(K, axis=1)):
    #         # K = np.random.binomial(1, 5 / L, size=[L, n])
    #         K = np.random.binomial(1, 0.4, size=[L, n])
            
    #     if Num_corr_agents != None:
    #         results_ucb_corr1_multi_agent_corr[nr] = run_simulation_multi_agent(K, env, L, means_real, T, Num_corr_agents = 1, env_corr = env_corr2)
    #         results_barbar_het_corr[nr] = BARBAR_dist_het(env, means_real, L, K, T, delta = 0.01, Num_corr_agents = 1, env_corr = env_corr2)
    #         results_barbar_het_lf_corr[nr] = BARBAR_lf_het(env, means_real, L, K, T, delta = 0.01, Num_corr_agents = 1, env_corr = env_corr2)
    #     else:
    #         results_ucb_corr1_multi_agent_corr[nr] = run_simulation_multi_agent(K, env, L, means_real, T, Num_corr_agents = 8, env_corr = env_corr2)
    #         print("     MA UCB C")
    #         results_barbar_het_corr[nr] = BARBAR_dist_het(env, means_real, L, K, T, delta = 0.01, Num_corr_agents = 8, env_corr = env_corr2)
    #         print("     Dist HET C")
    #         results_barbar_het_lf_corr[nr] = BARBAR_lf_het(env, means_real, L, K, T, delta = 0.01, Num_corr_agents = 8, env_corr = env_corr2)
    #         print("     LF HET C")


    #         results_ucb_corr1_multi_agent_nocorr[nr] = run_simulation_multi_agent(K, env, L, means_real, T)
    #         print("     MA UCB NC")
    #         results_barbar_het_nocorr[nr] = BARBAR_dist_het(env, means_real, L, K, T, delta = 0.01)
    #         print("     Dist HET NC")
    #         results_barbar_het_lf_nocorr[nr] = BARBAR_lf_het(env, means_real, L, K, T, delta = 0.01)

    #         print("     LF HET NC")



    # np.save('exp2_results_ucb_corr1.npy', results_ucb_corr1)
    # np.save('exp2_results_barbar.npy', results_barbar)
    results_ucb_corr1_multi_agent_corr = np.load('exp6_results_ucb_corr1_multi_agent_corr_L10K50T10.npy')
    results_ucb_corr1_multi_agent_nocorr = np.load('exp6_results_ucb_corr1_multi_agent_nocorr_L10K50T10.npy')

    results_barbar_het_corr = np.load('exp6_results_barbar_het_corr_L10K50T10.npy')
    results_barbar_het_nocorr = np.load('exp6_results_barbar_het_nocorr_L10K50T10.npy')

    results_barbar_het_lf_corr = np.load('exp6_results_barbar_het_lf_corr_L10K50T10.npy')
    results_barbar_het_lf_nocorr = np.load('exp6_results_barbar_het_lf_nocorr_L10K50T10.npy')

    fig, ax = plt.subplots() 
    ax.plot(list(range(T)), results_ucb_corr1_multi_agent_corr.mean(axis=0), marker='o',markevery=2000, label = "CO-UCB - w/ Corruption", color='red')
    ax.fill_between(list(range(T)), (results_ucb_corr1_multi_agent_corr.mean(axis=0) - results_ucb_corr1_multi_agent_corr.std(axis=0)),
                    (results_ucb_corr1_multi_agent_corr.mean(axis=0) + results_ucb_corr1_multi_agent_corr.std(axis=0)), color='#e27979', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het_corr.mean(axis=0), marker='d',markevery=2000, label = "DistBARBAR - w/ Corruption", color='blue')
    ax.fill_between(list(range(T)), (results_barbar_het_corr.mean(axis=0) - results_barbar_het_corr.std(axis=0)),
                    (results_barbar_het_corr.mean(axis=0) + results_barbar_het_corr.std(axis=0)), color='#a4b9db', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het_lf_corr.mean(axis=0), marker='s',markevery=2000, label = "LFBARBAR - w/ Corruption", color = "green")
    ax.fill_between(list(range(T)), (results_barbar_het_lf_corr.mean(axis=0) - results_barbar_het_lf_corr.std(axis=0)),
                    (results_barbar_het_lf_corr.mean(axis=0) + results_barbar_het_lf_corr.std(axis=0)), color='#bcd7ae', alpha=0.4)
    
    ax.plot(list(range(T)), results_ucb_corr1_multi_agent_nocorr.mean(axis=0), marker='+',linestyle = 'dotted', markevery=2000, label = "CO-UCB - w/o Corruption", color='red')
    ax.fill_between(list(range(T)), (results_ucb_corr1_multi_agent_nocorr.mean(axis=0) - results_ucb_corr1_multi_agent_nocorr.std(axis=0)),
                    (results_ucb_corr1_multi_agent_nocorr.mean(axis=0) + results_ucb_corr1_multi_agent_nocorr.std(axis=0)), color='#e27979', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het_nocorr.mean(axis=0), marker='x',linestyle = 'dotted',markevery=2000, label = "DistBARBAR - w/o Corruption", color='blue')
    ax.fill_between(list(range(T)), (results_barbar_het_nocorr.mean(axis=0) - results_barbar_het_nocorr.std(axis=0)),
                    (results_barbar_het_nocorr.mean(axis=0) + results_barbar_het_nocorr.std(axis=0)), color='#a4b9db', alpha=0.4)
    
    ax.plot(list(range(T)), results_barbar_het_lf_nocorr.mean(axis=0), marker='*',linestyle = 'dotted',markevery=2000, label = "LFBARBAR - w/o Corruption", color = "green")
    ax.fill_between(list(range(T)), (results_barbar_het_lf_nocorr.mean(axis=0) - results_barbar_het_lf_nocorr.std(axis=0)),
                    (results_barbar_het_lf_nocorr.mean(axis=0) + results_barbar_het_lf_nocorr.std(axis=0)), color='#bcd7ae', alpha=0.4)

    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.xlim([0, T])
    plt.ylim(bottom = 0)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    plt.grid()
    # lo = Labeloffset(ax, label="Cumulative Regret @Round 20K", axis="y")
    ax.legend(prop={'size': 15})
    plt.savefig('exp6_K50L10T10k_500.png', dpi = 500)
    plt.show()

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
    T = 10000
    num_rounds = 10
    # exp 1: 
        # X_axis: Cumulative regret at the end of round 20k
        # Y_axis: Number of agents 5 - 100
        # dist_het lf_het ucb_multi
    # exp1(n, T, num_rounds)
    # exp2(n, L, T, num_rounds)
    exp3(n, L, T, num_rounds)
    # exp4(n, L, T, num_rounds)
    exp5(n, L, T, num_rounds)
    # exp6(n, L, T, num_rounds)
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