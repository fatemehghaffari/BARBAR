import numpy as np
from time import sleep
def upper_confidence_bounds_action(t, means, count, epsilon=0.0):
    '''Play each arm once, then choose according to the equation given
    by Auer, Cesa-Bianchi & Fisher (2002).

    This is said to achieve the most optimal solution, with logarithmic regret.
    '''

    if t < len(means):
        return t
    else:
        return np.argmax(means + np.sqrt(2 * np.log(t) / count))

def run_simulation(env, means_real, num_trials, step = 1):
    '''Runs a 10-armed bandit simulation for multiple trials.

    Args:
    - get_action: a function which has input (t, means, count) and returns action
    - num_trials: number of iterations

    Returns:
    - A list of regret versus time
    '''

    means = np.zeros(len(means_real))
    count = np.zeros(len(means_real))
    choices = []

    for t in range(num_trials):

        action = upper_confidence_bounds_action(t, means, count)
        observation, reward, done, info = env.step(action)

        # Keep track of sample means for exploitation, and choices for regret calculation
        count[action] += 1
        means[action] = (1 - 1/count[action]) * means[action] + (1/count[action]) * reward
        choices.append(action)
    from collections import Counter
    # print(Counter(choices))

    def regret(t):
        best = np.argmax(means_real)
        return means_real[best] * t - sum([means_real[choices[i]] for i in range(t)])

    return [regret(t) for t in list(range(num_trials)[0 : num_trials : step])]


def run_simulation_multi_agent(K, env, L, means_real, num_trials, type = "Centralized", step = 1, regret_mode = "round", Num_corr_agents = None, env_corr = None):
    '''Runs a 10-armed bandit simulation for multiple trials.

    Args:
    - get_action: a function which has input (t, means, count) and returns action
    - num_trials: number of iterations

    Returns:
    - A list of regret versus time
    '''
    n = len(means_real)
    means = np.zeros([L, n])
    count = np.zeros([L, n])
    choices = [[] for _ in range(L)]
    L_list = np.sum(K, axis = 0)
    m = 1
    if type == "Centralized":
        for t in range(num_trials):
            for ag in range(L):
                action = upper_confidence_bounds_action(t, means[ag][K[ag]==1], count[ag][K[ag]==1])
                action = np.array(range(n))[K[ag]==1][action]
                if Num_corr_agents != None:
                    if ag in range(Num_corr_agents):
                        observation, reward, done, info = env_corr.step(action)

                        #####
                        # Keep track of sample means for exploitation, and choices for regret calculation
                        count[ag, action] += 1
                        means[ag, action] = (1 - 1/count[ag, action]) * means[ag, action] + (1/count[ag, action]) * reward
                        choices[ag].append(action)
                    else:
                        observation, reward, done, info = env.step(action)
                        # Keep track of sample means for exploitation, and choices for regret calculation
                        count[ag, action] += 1
                        means[ag, action] = (1 - 1/count[ag, action]) * means[ag, action] + (1/count[ag, action]) * reward
                        choices[ag].append(action)
                else:
                    observation, reward, done, info = env.step(action)
                    # Keep track of sample means for exploitation, and choices for regret calculation
                    count[ag, action] += 1
                    means[ag, action] = (1 - 1/count[ag, action]) * means[ag, action] + (1/count[ag, action]) * reward
                    choices[ag].append(action)
                    if t == 2**m:
                        for og in range(L):
                            if K[og, action] == 1:
                                count[og, action] += 1
                                means[og, action] = (1 - 1/count[og, action]) * means[og, action] + (1/count[og, action]) * reward
                                m += 1
            # if t == 2**m:
            #     means = (means.sum(axis=0) * K) / L_list
            #     count = count.sum(axis=0) * K
            #     m += 1
    # from collections import Counter
    # for j in range(L):
    #     print(Counter(choices[j]))
    def regret(t, regret_mode):
        best = np.argmax(means_real * K, axis = 1)
        if regret_mode == "final":
            agg_reg = 0
            for ind in range(L):
                if t > len(choices[ind]):
                    l = len(choices[ind])
                    agg_reg += means_real[best[ind]] * l - sum([means_real[choices[ind][i]] for i in range(l)])
                else:
                    agg_reg += (means_real[best[ind]] * t) - sum([means_real[choices[ind][i]] for i in range(t)])
            return agg_reg
        else:
            best_arr = np.zeros(t)
            obs_arr = np.zeros(t)
            for ind in range(L):
                best_arr += (means_real[best[ind]] * (np.array(range(t)) + 1))
                if t > len(choices[ind]):
                    l = len(choices[ind])
                else:
                    l = t
                obs_arr += (np.add.accumulate(np.array([means_real[choices[ind][i]] for i in range(t)])))
            return best_arr - obs_arr

    return regret(num_trials, regret_mode)



