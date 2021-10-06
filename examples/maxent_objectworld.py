"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.maxent as maxent
import irl.mdp.objectworld as objectworld
from irl.value_iteration import *

def main(grid_size, discount, n_objects, n_colours, n_trajectories, epochs,
		 learning_rate):
	"""
	Run maximum entropy inverse reinforcement learning on the objectworld MDP.

	Plots the reward function.

	grid_size: Grid size. int.
	discount: MDP discount factor. float.
	n_objects: Number of objects. int.
	n_colours: Number of colours. int.
	n_trajectories: Number of sampled trajectories. int.
	epochs: Gradient descent iterations. int.
	learning_rate: Gradient descent learning rate. float.
	"""

	wind = 0.3
	trajectory_length = grid_size * 3

	ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,
								 discount)
	ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
	ground_policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
						 ground_r, ow.discount, stochastic=False)

	trajectories = ow.generate_trajectories(n_trajectories,
											trajectory_length,
											lambda s: ground_policy[s])
	feature_matrix = ow.feature_matrix(discrete=False)
	# r = maxent.irl(feature_matrix, ow.n_actions, discount,
	#     ow.transition_probability, trajectories, epochs, learning_rate)
	r = maxent.deep_maxent_irl(feature_matrix, ow, discount, trajectories, epochs, learning_rate)

	# generate value function values for both ground_r and r
	stochastic_ground_policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
						 ground_r, ow.discount, stochastic=True)
	ground_value = policy_eval(stochastic_ground_policy, ground_r, ow.transition_probability, ow.n_states, ow.n_actions, discount_factor=ow.discount)

	stochastic_new_policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
						 r, ow.discount, stochastic=True)
	new_value = policy_eval(stochastic_new_policy, r, ow.transition_probability, ow.n_states, ow.n_actions, discount_factor=ow.discount)
	# policy_eval(policy, reward, transition_probabilities, nS, nA, discount_factor=1.0, theta=0.00001):

	plt.subplot(2, 2, 1)
	plt.pcolor(ground_r.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Groundtruth reward")
	plt.subplot(2, 2, 2)
	plt.pcolor(r.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Recovered reward")

	plt.subplot(2, 2, 3)
	plt.pcolor(ground_value.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Groundtruth value function")
	plt.subplot(2, 2, 4)
	plt.pcolor(new_value.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Recovered value function")

	plt.show()

if __name__ == '__main__':
	main(10, 0.95, 15, 2, 50, 500, 0.001)
	# grid_size, discount, n_objects, n_colours, n_trajectories, epochs, learning_rate
