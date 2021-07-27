"""
Contains multiple methods of performing inverse reinforcement learning for analysis
"""

import numpy as np
from time import sleep

# Max Ent IRL ###########################################################################

def normalize(vals):
	"""
	normalize to (0, max_val)
	input:
	vals: 1d array
	"""
	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)

# Main call - Not working yet - why?
def m_irl(world, trajectories, epochs, lr):

	n_states = world.n_states
	n_actions = world.n_actions
	_, n_features = world.features.shape
	p_transition = world.transition_prob

	delta = np.inf
	eps=1e-4

	theta = np.random.rand(n_states)
	# theta = np.ones(n_states)
	# theta = np.zeros(n_states) + 0.001
	e_features = feature_expectations(world, trajectories)
	p_initial = inital_probabilities(world, trajectories)
	# print(trajectories[-1])
	# for i in range(400):
	# optim.reset(theta)
	# print(e_features.reshape((5,5)))
	t=0
	# print(len(trajectories),"\n")
	while (delta > eps and t<epochs):
		theta_old = theta.copy()

		# Per state reward
		r  = world.features.dot(theta) # Essentially just alpha but could have different features

		# Backwards pass
		e_svf = expected_svf(world, p_initial, r)

		grad = e_features - world.features.T.dot(e_svf)
		# print(grad)
		theta*=np.exp(lr*grad, dtype=np.float64)
		# theta += lr*grad
		# optim.step(grad)
		delta = np.max(np.abs(theta_old - theta))
		t+=1

	return normalize(world.features.dot(theta))

# Expected state visitatoin

def expected_svf(world, p_initial, rewards):
	p_action = local_action_probability(world, rewards)
	return expected_svf_from_policy(world, p_initial, p_action)

def local_action_probability(world, rewards):
	n_states, n_actions = world.n_states, world.n_actions
	z_states = np.zeros((n_states))
	z_action = np.zeros((n_states, n_actions))
	p_transition = world.transition_prob

	p = [np.array(world.transition_prob[:, :, a]) for a in range(n_actions)]
	er = np.exp(rewards)*np.eye((n_states))

	zs = np.zeros(n_states)
	za = np.zeros((n_states, n_actions))
	zs[world.terminals] = 1.0

	for _ in range(2 * n_states):
		for a in range(n_actions):
			za[:,a] = np.matmul(er, np.matmul(p_transition[:,:,a], zs.T, dtype=np.float64), dtype=np.float64)
		zs = za.sum(axis=1)
	return za / zs[:, None]

def expected_svf_from_policy(world, p_initial, p_action, eps = 1e-5):
	n_states, n_actions = world.n_states, world.n_actions
	p_transition = np.copy(world.transition_prob)
	p_transition[world.terminals, :, :] = 0.0

	p_transition = [np.array(p_transition[:,:,a]) for a in range(n_actions)]

	# print(p_action)
	# forward computation of state expectations
	d = np.zeros(n_states)

	delta = np.inf

	while delta > eps:
		# print([p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)])
		d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
		d_ = p_initial + np.array(d_).sum(axis=0)
		delta, d = np.max(np.abs(d_ - d)), d_
		# print(delta)
	return d

# Functions for everyone ###############################################################

def feature_expectations(world, trajectories):
	n_states, n_features = world.features.shape

	fe = np.zeros(n_states)

	for t in trajectories:
		for s in t.states():
			fe += world.features[s,:]
	return fe/len(trajectories)

def inital_probabilities(world, trajectories):
	# Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
	# And returns initial probability and count
	n_states, n_features = world.features.shape
	p = np.zeros(n_states)

	for t in trajectories:
		p[t.transitions()[0][0]] += 1.0
	return p/len(trajectories)