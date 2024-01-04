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
    
def run_simulation(env, means_real, num_trials):
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

    return [regret(t) for t in range(num_trials)]



