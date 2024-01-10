import numpy as np 
from copy import deepcopy
from time import sleep
from collections import Counter
def BARBAR(env, means_real, K, T, delta = 0.2, step = 1):
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

    def regret(t):
        best = np.argmax(means_real)
        return means_real[best] * t - sum([means_real[actions[i]] for i in range(t)])

    return [regret(t) for t in list(range(T)[0 : T : step])]
        
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


def BARBAR_dist_het(env, means_real, L, K, T, regret_mode = "round", delta = 0.2, step = 1):
    L_list = np.sum(K, axis = 0)
    r = np.zeros([L, len(L_list)])
    rs = np.zeros(L)
    Delta = deepcopy(K)
    m = 1
    t = 0
    all_actions = [[] for _ in range(L)]
    k = K.shape[1]
    lmbda = np.log((8 * k / delta) * np.log2(T))
    n = np.zeros([L, k])
    while t < T:
        N_max = 0
        r = np.zeros([L, len(L_list)])
        for j in range(L):
            n[j] = (lmbda / (Delta[j]) ** 2) * K[j]
            n[np.isnan(n)] = 0
            N = np.sum(n[j]/L_list)
            N_max = max(N_max, N)
            pr = n[j] / (L_list * N)
            er, actions = BARBAR_dist_het_agent(env, k, N, pr)
            er = (er * L_list)/n[j]
            er[np.isnan(er)] = 0
            r[j] = er
            all_actions[j] += actions
        t += N_max
        r_mean = np.mean(r, axis=0)
        rs = np.max(K * r_mean - (1/16) * Delta, axis=1)
        Delta = np.maximum(2**(-1*m), (np.reshape(rs, [len(rs), 1]) - r_mean)*K)
        m += 1    
    # from collections import Counter
    # for ind in range(L):
    #     print(Counter(all_actions[ind]))
    #     print(len(all_actions[ind]))
    #     print()
    def regret(t):
        agg_reg = 0
        best = np.argmax(means_real * K, axis = 1)
        for ind in range(L):
            if t > len(all_actions[ind]):
                l = len(all_actions[ind])
                agg_reg += means_real[best[ind]] * l - sum([means_real[all_actions[ind][i]] for i in range(l)])
            else:
                agg_reg += float(format(means_real[best[ind]] * t, '.3f')) - float(format(sum([means_real[all_actions[ind][i]] for i in range(t)]), '.3f'))
        return agg_reg
    
    if regret_mode == "round":
        return [regret(t) for t in list(range(T)[0 : T : step])]
    elif regret_mode == "final":
        return regret(T)

def BARBAR_lf_het(env, means_real, L, K, T, regret_mode = "round", delta = 0.2, step = 1):
    L_list = np.sum(K, axis = 0)
    k = K.shape[1]
    r = np.zeros(k)
    rs = 0
    Delta = np.ones(k)
    m = 1
    t = 0
    all_actions = [[] for _ in range(L)]
    
    lmbda = np.log((8 * k / delta) * np.log2(T))
    n = np.zeros(k)
    while t < T:
        N_max = 0
        n = (lmbda / ((Delta) ** 2))
        r = np.zeros(k)
        for j in range(L):
            N = np.sum(n * K[j]/L_list) 
            N_max = max(N_max, N)
            pr = (n * K[j]) / (L_list * N)
            action = np.random.choice(np.array(range(k)), p=pr)
            pra = np.zeros(k)
            pra[action] = 1
            er, actions = BARBAR_dist_het_agent(env, k, N, pra)
            r += er
            all_actions[j] += actions
        t += N_max
        r = r/n
        rs = np.max(r - (1/16) * Delta)
        Delta = np.maximum(2**(-1*m), (rs - r))
        m += 1    
    # from collections import Counter
    # for ind in range(L):
    #     print(Counter(all_actions[ind]))
    #     print(len(all_actions[ind]))
    #     print()
    def regret(t):
        agg_reg = 0
        best = np.argmax(means_real * K, axis = 1)
        for ind in range(L):
            if t > len(all_actions[ind]):
                l = len(all_actions[ind])
                agg_reg += means_real[best[ind]] * l - sum([means_real[all_actions[ind][i]] for i in range(l)])
            else:
                agg_reg += float(format(means_real[best[ind]] * t, '.3f')) - float(format(sum([means_real[all_actions[ind][i]] for i in range(t)]), '.3f'))
        return agg_reg
    if regret_mode == "round":
        return [regret(t) for t in list(range(T)[0 : T : step])]
    elif regret_mode == "final":
        return regret(T)