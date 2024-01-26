import numpy as np 
from copy import deepcopy
from time import sleep
from collections import Counter
def BARBAR(env, means_real, K, T, delta = 0.2, step = 1, regret_mode = 'round'):
    # lmbda = 1024 * np.log((8 * K / delta) * np.log2(T))
    lmbda = (1)*np.log((8 * K / delta) * np.log2(T))
    t = 0
    m = 1
    n = np.zeros([1, K])
    Delta = np.ones([1, K])
    actions_sum = np.zeros(K)
    actions = []
    while t < T:
        n = lmbda / (Delta) ** 2
        N = np.sum(n)
        pr = n / N
        ereward = np.zeros(K)
        for i in range(int(N)):
            action = np.random.choice(np.array(range(K)), p=pr[0])
            observation, reward, done, info = env.step(action)
            ereward[action] += reward
            actions_sum[action] += 1
            actions.append(action)
            t += 1
        r = ereward / n
        rs = np.max(r - (Delta / 16))
        Delta = np.maximum(2 ** (-1 * m), rs - r)
        m += 1

    def regret(t, regret_mode):
        if regret_mode == 'final':
            best = np.argmax(means_real)
            reg_t = means_real[best] * t - sum([means_real[actions[i]] for i in range(t)])
            return reg_t
        else:
            best = np.argmax(means_real)
            reg_t = [means_real[best] * t - sum([means_real[actions[i]] for i in range(t)]) for t in list(range(T)[0 : T : step])]
            return reg_t
    return regret(T, regret_mode)
        
def BARBAR_dist_het_agent(env, K, L, N, pr, T, t):
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


def BARBAR_dist_het(env, means_real, L, K, T, regret_mode = "round", delta = 0.2, step = 1, Num_corr_agents = None, env_corr = None):
    L_list = np.sum(K, axis = 0)
    r = np.zeros([L, len(L_list)])
    rs = np.zeros(L)
    Delta = deepcopy(K)
    m = 1
    t = 0
    all_actions = [[] for _ in range(L)]
    k = K.shape[1]
    lmbda = (1)*np.log((8 * k / delta) * np.log2(T))
    n = np.zeros([L, k])
    # print(Delta)
    while t < T:
        r = np.zeros([L, k])
        n = (lmbda / (Delta) ** (2)) * K
        n[np.isnan(n)] = 0
        x = n/L_list
        x[np.isnan(x)] = 0
        N = np.sum(n, axis = 1)
        pr = n / np.reshape(N, [L, 1])
        # pr = n / np.matmul(np.reshape(N, [L, 1]), np.reshape(L_list, [1, k]))
        pr[np.isnan(pr)] = 0
        for j in range(L):
            if N[j] == 0:
                continue
            if Num_corr_agents != None:
                if j in range(Num_corr_agents):
                    er, actions = BARBAR_dist_het_agent(env_corr, k, L, N[j], pr[j], T, t)
                    er[np.isnan(er)] = 0
                    r[j] = er / n[j]
                    r[np.isnan(r)] = 0
                    all_actions[j] += actions
                else:
                    er, actions = BARBAR_dist_het_agent(env, k, L, N[j], pr[j], T, t)
                    er[np.isnan(er)] = 0
                    r[j] = er/ n[j]
                    r[np.isnan(r)] = 0
                    all_actions[j] += actions
            else:
                er, actions = BARBAR_dist_het_agent(env, k, L, N[j], pr[j], T, t)
                r[j] = er / n[j]
                r[np.isnan(r)] = 0
                all_actions[j] += actions
        t += N.max()
        r_mean = np.sum(r, axis=0)/L_list
        # r_mean *= K
        r_mean[np.isnan(r_mean)] = 0
        rs = np.max(K * (r_mean - (1/16) * Delta.max(axis=0)), axis=1)
        # print(rs)
        # sleep(10)
        Delta = np.maximum(2**(-1*m), (np.reshape(rs, [len(rs), 1]) - r_mean)*K)*K

        m += 1    
    # for ind in range(L):
    #     print(Counter(all_actions[ind]))
    #     print(len(all_actions[ind]))
    #     print()
        # sleep(3)
    def regret(t, regret_mode):
        best = np.argmax(means_real * K, axis = 1)
        if regret_mode == "final":
            agg_reg = 0
            for ind in range(L):
                if t > len(all_actions[ind]):
                    l = len(all_actions[ind])
                    agg_reg += means_real[best[ind]] * l - sum([means_real[all_actions[ind][i]] for i in range(l)])
                else:
                    agg_reg += float(format(means_real[best[ind]] * t, '.3f')) - float(format(sum([means_real[all_actions[ind][i]] for i in range(t)]), '.3f'))
            return agg_reg
        else:
            obs_arr = np.zeros(t)
            best_arr = np.zeros(t)
            for ind in range(L):
                if t > len(all_actions[ind]):
                    l = len(all_actions[ind])
                else:
                    l = t
                # obs_arr = (np.around(np.add.accumulate(np.array([means_real[all_actions[ind][i]] for i in range(l)])), decimals=3))
                # best_arr = (np.around(best[ind] * (np.array(range(l)) + 1), decimals=3))
                # print(means_real[best[ind]], Counter(means_real[all_actions[ind]]))
                obs_arr += np.add.accumulate(np.pad(np.array([means_real[all_actions[ind][i]] for i in range(l)]), (0, t - l)))
                c = means_real[best[ind]] * l
                best_arr += np.pad(means_real[best[ind]] * (np.array(range(l)) + 1), (0, t - l), 'constant', constant_values=(0, c))
            return best_arr - obs_arr
    
    return regret(T, regret_mode)

def BARBAR_lf_het(env, means_real, L, K, T, regret_mode = "round", delta = 0.2, step = 1, Num_corr_agents = None, env_corr = None):
    L_list = np.sum(K, axis = 0)
    k = K.shape[1]
    r = np.zeros(k)
    rs = 0
    Delta = np.ones(k)
    m = 1
    t = 0
    all_actions = [[] for _ in range(L)]
    
    lmbda = (1)*np.log((8 * k / delta) * np.log2(T))
    n = np.zeros(k)
    while t < T:
        N_min = T
        nf = (lmbda / (Delta) ** 2)
        n = nf * K
        n[np.isnan(n)] = 0
        x = (n)/L_list
        x[np.isnan(x)] = 0
        N = np.sum(x, axis = 1)
        pr = x / np.reshape(N, [L, 1])
        # pr = n / np.matmul(np.reshape(N, [L, 1]), np.reshape(L_list, [1, k]))
        pr[np.isnan(pr)] = 0
        r = np.zeros(k)
        for j in range(L):
            if N[j] == 0:
                continue
            action = np.random.choice(np.array(range(k)), p=pr[j])
            pra = np.zeros(k)
            pra[action] = 1
            if Num_corr_agents != None:
                if j in range(Num_corr_agents):
                    er, actions = BARBAR_dist_het_agent(env_corr, k, L, N[j], pra, T, t)
                    r += er
                    all_actions[j] += actions
                else:
                    er, actions = BARBAR_dist_het_agent(env, k, L, N[j], pra, T, t)
                    r += er
                    all_actions[j] += actions
            else:
                er, actions = BARBAR_dist_het_agent(env, k, L, N[j], pra, T, t)
                r += er
                all_actions[j] += actions
        t += N.max()
        r = r/nf
        r[np.isnan(r)] = 0
        rs = np.max(r - (1/16) * Delta)
        Delta = np.maximum(2**(-1*m), (rs - r))
        m += 1  
        
    # from collections import Counter
    # for ind in range(L):
    #     print(Counter(all_actions[ind]))
    #     print(len(all_actions[ind]))
    #     print()
    def regret(t, regret_mode):
        best = np.argmax(means_real * K, axis = 1)
        if regret_mode == "final":
            agg_reg = 0
            for ind in range(L):
                if t > len(all_actions[ind]):
                    l = len(all_actions[ind])
                    agg_reg += means_real[best[ind]] * l - sum([means_real[all_actions[ind][i]] for i in range(l)])
                else:
                    agg_reg += float(format(means_real[best[ind]] * t, '.3f')) - float(format(sum([means_real[all_actions[ind][i]] for i in range(t)]), '.3f'))
            return agg_reg
        else:
            obs_arr = np.zeros(t)
            best_arr = np.zeros(t)
            for ind in range(L):
                if (t > len(all_actions[ind])):
                    l = len(all_actions[ind])
                else:
                    l = t
                
                obs_arr += np.add.accumulate(np.pad((np.array([means_real[all_actions[ind][i]] for i in range(l)])), (0, t - l)))
                c = means_real[best[ind]] * l
                best_arr += np.pad(means_real[best[ind]] * (np.array(range(l)) + 1), (0, t - l), 'constant', constant_values=(0, c))
                
            return best_arr - obs_arr

    return regret(T, regret_mode)