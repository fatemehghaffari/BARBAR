import numpy as np

def upper_confidence_bounds_action(t, means, count, epsilon=0.0):
    '''Play each arm once, then choose according to the equation given
    by Auer, Cesa-Bianchi & Fisher (2002).
    
    This is said to achieve the most optimal solution, with logarithmic regret.
    '''

    if t < 10:
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
    
    means = np.zeros(10)
    count = np.zeros(10)
    choices = []

    for t in range(num_trials):

        action = upper_confidence_bounds_action(t, means, count)
        observation, reward, done, info = env.step(action)

        # Keep track of sample means for exploitation, and choices for regret calculation 
        count[action] += 1
        means[action] = (1 - 1/count[action]) * means[action] + (1/count[action]) * reward
        choices.append(action)
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

    if type == "Centralized":
        for t in range(num_trials):
            for ag in range(L):
                action = upper_confidence_bounds_action(t, np.mean(means, axis = 0) * K[ag], np.mean(count, axis = 0) * K[ag])
                if Num_corr_agents != None:
                    if ag == 0:
                        observation, reward, done, info = env_corr.step(action)
                        # Keep track of sample means for exploitation, and choices for regret calculation 
                        count[ag, action] += 1
                        means[ag, action] = (1 - 1/count[ag, action]) * means[ag, action] + (1/count[ag, action]) * reward
                        choices[ag].append(action)
                        continue
                observation, reward, done, info = env.step(action)
                # Keep track of sample means for exploitation, and choices for regret calculation 
                count[ag, action] += 1
                means[ag, action] = (1 - 1/count[ag, action]) * means[ag, action] + (1/count[ag, action]) * reward
                choices[ag].append(action)
    def regret(t):
        reg = 0
        for ag in range(L):
            best = np.argmax(means_real * K[ag])
            reg += means_real[best] * t - sum([means_real[choices[ag][i]] for i in range(t)])
        return reg
    if regret_mode == "round":
        return [regret(t) for t in list(range(num_trials)[0 : num_trials : step])]
    elif regret_mode == "final":
        return regret(num_trials)



