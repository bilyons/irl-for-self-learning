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
from irl.utils import *

import os

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
	
	# Put both the groundtruth reward and the IRL reward into the same scale [0,1].
	ground_r = maxent.normalize(ground_r)

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
	new_policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
						 r, ow.discount, stochastic=False)
	new_value = policy_eval(stochastic_new_policy, r, ow.transition_probability, ow.n_states, ow.n_actions, discount_factor=ow.discount)
	# policy_eval(policy, reward, transition_probabilities, nS, nA, discount_factor=1.0, theta=0.00001):

	# plot and save the value function values in a table
	random_seed = np.random.randint(1,100000)
	acc = calc_accuracy(new_policy, ground_policy, ow.n_states)
	print("acc= "+ str(np.round(acc, decimals=2)))

	foldername = "results/"+ "traj"+ str(n_trajectories)+"_e"+ str(epochs)+ "_r"+ str(random_seed) + "_acc"+ str(np.round(acc, decimals=2)) + "/"
	os.mkdir(foldername)
	fig, ax = plt.subplots()
	ax = plot_table(ax, np.round(ground_value, decimals=3).reshape(grid_size, grid_size), 'iter = ' +'optimal')
	fig.savefig(foldername+ "True_value.png")
	
	fig, ax = plt.subplots()
	# new_value = np.arange(100)
	ax = plot_table(ax, np.round(new_value, decimals=3).reshape(grid_size, grid_size), 'iter = ' + str(epochs))
	fig.savefig(foldername+ "IRL_value.png")

	policy_map = [0 for i in range(100)]
	ground_policy_plot = optimal_policy_map(policy_map, ground_policy)
	fig, ax = plt.subplots()
	title = 'Optimal Policy Map for Groundtruth Reward Function'
	ax = plot_table(ax, np.array(ground_policy_plot).reshape(10, 10), title)
	fig.savefig(foldername+ "Ground_Policy_Map.png")

	policy_map = [0 for i in range(100)]
	new_policy_plot = optimal_policy_map(policy_map, new_policy)
	fig, ax = plt.subplots()
	title = "Optimal Policy Map for IRL Reward Function, acc= "+ str(np.round(acc, decimals=2))
	ax = plot_table(ax, np.array(new_policy_plot).reshape(10, 10), title)
	fig.savefig(foldername+ "IRL_Policy_Map.png")

	# policy comparison
	fig, ax = plt.subplots()
	title = "Optimal Policy Map Comparison for extracted Reward Function, acc= "+ str(np.round(acc, decimals=2))
	policy_diff = []
	for i in range(ow.n_states):
		if ground_policy_plot[i] == new_policy_plot[i]:
			policy_diff.append(ground_policy_plot[i])
		else:
			policy_diff.append(ground_policy_plot[i]+"/"+new_policy_plot[i])
	ax = plot_table(ax, np.array(policy_diff).reshape(10, 10), title)
	fig.savefig(foldername+ "Policy_Map_Comparison.png")


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

	plt.savefig(foldername+ "ObjectWorld.png")
	# plt.show()
	print("results saved to folder: ", foldername)

	return acc
	
	

if __name__ == '__main__':
	accs = []
	
	# for i in range(2):
	# 	acc = main(10, 0.95, 15, 2, 32, 200, 0.1)
	# 	accs.append(acc)
	for i in range(1):
		acc = main(10, 0.95, 15, 2, 64, 200, 0.01)
		accs.append(acc)
	# for i in range(2):
	# 	acc = main(10, 0.95, 15, 2, 128, 200, 0.1)
	# 	accs.append(acc)
	# for i in range(2):
	# 	acc = main(10, 0.95, 15, 2, 256, 200, 0.1)
	# 	accs.append(acc)
	# for i in range(2):
	# 	acc = main(10, 0.95, 15, 2, 512, 200, 0.1)
	# 	accs.append(acc)
	print(accs)
	# grid_size, discount, n_objects, n_colours, n_trajectories, epochs, learning_rate
