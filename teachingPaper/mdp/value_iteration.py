import numpy as np

def value_iteration(p, reward, discount, eps=1e-2):

    n_states, n_actions, _ = p.shape

    v = np.zeros(n_states)
    q = np.zeros((n_states, n_actions))

    # p = [np.matrix(p[:, a, :]) for a in range(n_actions)]

    # exit()

    delta = np.inf

    while delta>eps:
        v_old = v.copy()
        for s in range(n_states):
            for a in range(n_actions):
                q[s,a] = np.dot(p[s,a,:], reward+discount*v_old)
            v[s] = np.max(q[s,:])
        delta = np.max(np.abs(v_old - v))
    return v

def stochastic_value_iteration(p, reward, discount, eps=1e-3):

    n_states, n_actions, _ = p.shape

    v = np.zeros(n_states)

    # p = [np.matrix(p[:, a, :]) for a in range(n_actions)]

    delta = np.inf

    while delta>eps:
        v_old = v

        for s in range(n_states):
            print(np.multiply(p[s,:,:], reward+discount*v))
            q = discount*np.array([p[a] @ v for a in range(n_actions)])

            v = reward + np.average(q, axis=0)[0]

        delta = np.max(np.abs(v_old - v))

    return v

def find_policy(p, reward, discount, eps=1e-3):

    n_states, n_actions, _ = p.shape

    v = value_iteration(p, reward, discount)
    # print(v.reshape((5,5)))
    # v = optimal_value(25,4,p, reward, discount)
    # print(v.reshape((5,5)))
    # exit()

    Q = np.zeros((n_states, n_actions))

    y = [np.matrix(p[:,a,:]) for a in range(n_actions)]
    for a in range(n_actions):
        Q[:,a] = y[a].dot(reward + discount*v)
    Q -= Q.max(axis=1).reshape(n_states, 1)
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape(n_states, 1)

    return Q

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-3):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v

def find_policy1(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy