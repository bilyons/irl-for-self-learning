import numpy as np
from itertools import product
from .utility import *
from .optimizer import *

def maxent_irl(world, terminal, trajectories, eps=1e-5, eps_esvf=1e-5):
	n_states, n_actions, _ = world.transition_prob.shape
	_, n_features = world.state_features.shape
	features = world.state_features

	expected_features = feature_expectations_from_trajectories(features, trajectories)
	initial_probabilities = initial_probabilities_from_trajectories(features, trajectories)

	theta = np.random.rand(n_states)
	delta = np.inf

	optimiser = ExpSga(lr=linear_decay(lr0=0.1))

	optimiser.reset(theta)
	while delta > eps:
		theta_old = theta.copy()

		reward = features.dot(theta)

		e_svf = expected_svf(world.transition_prob, initial_probabilities, reward, terminal)

		grad = expected_features - features.T.dot(e_svf)

		optimiser.step(grad)

		delta = np.max(np.abs(theta_old-theta))

	return features.dot(theta)

def expected_svf(p_transition, p_initial, rewards, terminals):
	p_action = local_action_probabilities(p_transition, terminals, rewards)
	return expected_svf_from_policy(p_transition, p_initial, p_action, terminals)

def local_action_probabilities(p_transition, terminals, reward):

	n_states, n_actions, _ = p_transition.shape

	er = np.exp(reward, dtype=np.float64)*np.eye((n_states), dtype=np.float64)

	zs = np.zeros(n_states, dtype=np.float64)
	za = np.zeros((n_states, n_actions), dtype=np.float64)
	zs[terminals] = 1.0

	for _ in range(2 * n_states):
		for a in range(n_actions):
			za[:,a] = np.matmul(er, np.matmul(p_transition[:,a,:], zs.T))
		zs = za.sum(axis=1)
	return za / zs[:, None]

def expected_svf_from_policy(p_transition, p_initial, p_action, terminals, eps = 1e-5):
	n_states, n_actions, _ = p_transition.shape
	p_transition = np.copy(p_transition)
	p_transition[terminals, :, :] = 0.0

	p_transition = [np.array(p_transition[:,a,:], dtype=np.float64) for a in range(n_actions)]
	# print(p_transition)
	# exit()
	d = np.zeros(n_states, dtype=np.float64)

	delta = np.inf

	while delta > eps:
		d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
		d_ = p_initial + np.array(d_).sum(axis=0)
		delta, d = np.max(np.abs(d_ - d)), d_
	return d
