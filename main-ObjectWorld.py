from mimetypes import init
from envs.discrete.objectworld import ObjectWorld
# from irl.standard.irl_set_traj_length import *
from irl.deep.deep_maxent_irl import *
import matplotlib.pyplot as plt

import utils.img.img_utils as img_utils

# import torch
# torch.autograd.set_detect_anomaly(True)


def main():
	# Generate samples from original script
	"""
	grid_size: Grid size. int.
	discount: MDP discount factor. float.
	n_trajectories: Number of sampled trajectories. int.
	epochs: Gradient descent iterations. int.
	learning_rate: Gradient descent learning rate. float.
	"""
	grid_size = 10
	wind = 0.3
	trajectory_length = 8
	discount = 0.9
	n_objects = 15
	n_colours = 2
	n_trajectories = 200
	learning_rate = 0.01
	epochs = 100
  

	ow = ObjectWorld(grid_size, n_objects, n_colours, wind, discount)
	ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])

	policy = find_policy(ow.n_states, ow.n_actions, ow.transition_prob,
												ground_r, ow.discount, stochastic=False)

	trajectories = ow.generate_trajectories(n_trajectories, trajectory_length, lambda s: policy[s])
	
	feature_matrix = ow.feature_matrix(discrete=False)

	rewards = deep_maxent_irl(ow, discount, trajectories, epochs, learning_rate)
	print("final rewards is: ", rewards)

	
	# plots
	plt.figure(figsize=(10,4))
	plt.subplot(1, 2, 1)
	img_utils.heatmap2d(np.reshape(ground_r, (grid_size, grid_size)), 'Ground Truth Reward', block=False)
	plt.subplot(1, 2, 2)
	img_utils.heatmap2d(np.reshape(rewards, (grid_size, grid_size)), 'Recovered reward Map', block=False)
	
	# plt.savefig('reward-deep-ObjectWorld.png')
	plt.show()
	




if __name__ == "__main__":
	main()


