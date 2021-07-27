"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Billy Lyons, 2021
billy.lyons@ed.ac.uk

Adapted from Matthew Alger: https://github.com/MatthewJA/Inverse-Reinforcement-Learning
"""

from itertools import product
import time
import numpy as np
import value_iteration as V

def normalize(vals):
  """
  normalize to (0, max_val)
  input:
	vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)

def compute_state_visitation_frequency(world, gamma, trajectories, policy):
	"""
	Compute expected state visitation frequency by dynamic programming

	Given the policy generated by the perceived value function, it iterates over
	the initial starting states to calculate the state visitation frequency probabilistically

	inputs:
		world: for features and state and transition information
		gamma: discount factor
		trajectories: list of expert behaviours
		policy: NxA policy

	output:
		freq: Nx1 vector of state visitation frequencies
	"""

	n_states, _, n_actions =  world.transition_prob.shape

	T = len(trajectories[0].transitions())
	mu = np.zeros((n_states, T))
	for traj in trajectories:
		init = traj.transitions()[0][0]
		mu[init, 0] +=1.0
	mu[:,0] = mu[:,0] / len(trajectories)

	d = mu.copy()
	# Test
	# start = time.time()
	for t in range(T-1):
		tmp = np.array([np.multiply(policy[:,a], world.transition_prob[:,:,a]) for a in range(n_actions)]).sum(axis=0)
		d[:,t+1]= np.dot(d[:,t],tmp)
	# print(d.sum(axis=1))
	# end = time.time()
	# new = end-start
	# print("New: ", new)
	# start = time.time()
	for s in range(n_states):
		for t in range(T-1):
			mu[s, t+1] = sum([sum([mu[pre_s, t]*world.transition_prob[pre_s, s, a]*policy[pre_s, a] for a in range(n_actions)]) for pre_s in range(n_states)])
	print("P", sum(mu))
	p = np.sum(mu, 1)
	# end = time.time()
	# old = end-start
	# print("Old: ", old)
	# print("X faster: ", old/new)
	

	# c = 0
	# print(mu[:,c]==d[:,c])
	# print(mu[:,c])
	# print(d[:,c])
	# print(np.sum(mu[:,c]))
	# print(np.sum(d[:,c]))

	# print(np.sum(p))
	# print(np.sum(d))
	# exit()
	# print(d.sum(axis=1))
	# exit()
	print(sum(d))
	exit()
	return d.sum(axis=1)

def irl(world, gamma, trajectories, epochs, learning_rate):
	"""
	Find the reward function for the given trajectories

	This is based on algorithm 9.1 and equation 9.2 from the Ziebart thesis

	input:
		features: the feature matrix of the state space. In this case the Nth
			row of the identity matrix represents the features at state N
		n_actions: number of actions, 4 in cardinal directions
		gamma: discount factor standard to RL
		p_transition: state transition probabilities given action
		trajectories: uncertain, list?
		epochs: number of steps for grad descent
		learning rate: gradient descent learning rate

	output:
		reward: vector of shape Nx1
	"""
	features = world.features
	p_transition = world.transition_prob
	n_states, _, = features.shape
	# Initialise weights
	alpha = np.random.uniform(size=(n_states,))

	# Calculate feature expectations
	feature_expectations = find_feature_expectations(world, trajectories)

	# print(feature_expectations.reshape((5,5)))
	# print(np.sum(feature_expectations))
	# exit()
	# delta = np.inf
	# eps = 1e-4
	# while delta > eps:
	for epoch in range(epochs):
		a_old = alpha.copy()

		rewards = np.dot(features, alpha)
		# print(rewards)

		# start = time.time()
		# Compute policy
		_, policy = V.value_iteration(world, p_transition, rewards, gamma)
		# print("POLICY", policy)
		# time.sleep(0.1)
		# end= time.time()
		# print("Duration: ", end-start)
		# exit()
		# Compute state visitation frequency
		esvf = compute_state_visitation_frequency(world, gamma, trajectories, policy)
		print(esvf.reshape((11,11)))
		# print(feature_expectations)
		# print("ESVF", esvf)
		# Compute gradients
		grad = feature_expectations - features.T.dot(esvf)

		# print("GRAD", grad)
		# Gradient descent
		alpha += learning_rate*grad

		delta = np.max(np.abs(alpha - a_old))
		# print(alpha)
		# print(a_old)
		print("DELTA", delta)

	rewards = np.dot(features, alpha)

	return normalize(rewards)
	# return normalize(rewards)

def find_feature_expectations(world, trajectories):
	n_states, n_features = world.features.shape
	fe = np.zeros(n_states)

	for t in trajectories:
		for s in t.states():
			fe += world.features[s,:]
	return fe/len(trajectories)


def find_policy(n_states, r, n_actions, discount,
						   transition_probability):
	"""
	Find a policy with linear value iteration. Based on the code accompanying
	the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

	n_states: Number of states N. int.
	r: Reward. NumPy array with shape (N,).
	n_actions: Number of actions A. int.
	discount: Discount factor of the MDP. float.
	transition_probability: NumPy array mapping (state_i, action, state_k) to
		the probability of transitioning from state_i to state_k under action.
		Shape (N, A, N).
	-> NumPy array of states and the probability of taking each action in that
		state, with shape (N, A).
	"""

	# V = value_iteration.value(n_states, transition_probability, r, discount)

	# NumPy's dot really dislikes using inf, so I'm making everything finite
	# using nan_to_num.
	V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

	diff = np.ones((n_states,))
	while (diff > 1e-4).all():  # Iterate until convergence.
		new_V = r.copy()
		for j in range(n_actions):
			for i in range(n_states):
				new_V[i] = softmax(new_V[i], r[i] + 
					np.sum(transition_probability[i, j, k] * V[k]
						   for k in range(n_states)))

		# # This seems to diverge, so we z-score it (engineering hack).
		new_V = (new_V - new_V.mean())/new_V.std()

		diff = abs(V - new_V)
		V = new_V

	# We really want Q, not V, so grab that using equation 9.2 from the thesis.
	Q = np.zeros((n_states, n_actions))
	for i in range(n_states):
		for j in range(n_actions):
			p = np.array([transition_probability[i, j, k]
						  for k in range(n_states)])
			Q[i, j] = p.dot(r + discount*V)

	# Softmax by row to interpret these values as probabilities.
	Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
	Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
	return Q
