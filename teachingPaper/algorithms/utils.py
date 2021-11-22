"""
Common functions across IRL algorithms
"""

import numpy as np

def feature_expectations_from_trajectories(feature_matrix, trajectories):

	n_states, n_features = feature_matrix.shape

	feature_expectations = np.zeros(n_states)

	for t in trajectories:
		for s in t.states():
			feature_expectations += feature_matrix[s,:]

	return fe/len(trajectories)


def initial_probabilities_from_trajectories(feature_matrix, trajectories):

	n_states, n_features = feature_matrix.shape

	initial_probabilities = np.zeros(n_states)

	for t in trajectories:
		initial_probabilities[t.transitions()[0][0]] += 1.0

	return p/len(trajectories)