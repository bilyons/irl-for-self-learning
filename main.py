
from mimetypes import init
from envs.discrete.objectworld import ObjectWorld
# from irl.standard.irl_set_traj_length import *
from irl.deep.deep_maxent_irl import *
import envs.discrete.originalGridWorld.objectworld as original
import matplotlib.pyplot as plt

import utils.img.img_utils as img_utils
import envs.discrete.originalGridWorld.value_iteration as value_iteration

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
    grid_size = 5
    wind = 0.3
    trajectory_length = 3* grid_size
    discount = 0.01
    n_trajectories = 20
    learning_rate = 0.01
    epochs = 100
  

    # grid_size: Grid size. int.
    # n_objects: Number of objects in the world. int.
    # n_colours: Number of colours to colour objects with. int.
    # wind: Chance of moving randomly. float.
    # discount: MDP discount. float.
    
    # ow = original.Objectworld(grid_size, 2, 2, wind, discount)
    #print("transition_probabilities: ",gw.transition_probability)

    # n_trajectories: Number of trajectories. int.
    #     trajectory_length: Length of an episode. int.
    #     policy: Map from state integers to action integers.
    #     -> [[(state int, action int, reward float)]]
    # trajectories = ow.generate_trajectories(n_trajectories,
    #                                         trajectory_length,
    #                                         ow.optimal_policy)
    #------------------------------------------------------------------------------------------------



    # Make the objectworld and associated data.
    ow = original.Objectworld(grid_size, 2, 2, wind, discount)
    feature_matrix = ow.feature_matrix()
    ground_reward = np.array([ow.reward(i) for i in range(ow.n_states)])
    optimal_policy = value_iteration.find_policy(ow.n_states,
                                                 ow.n_actions,
                                                 ow.transition_probability,
                                                 ground_reward,
                                                 discount).argmax(axis=1)
    trajectories = ow.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            optimal_policy.take)


    trajs = trajectories[:,:,0:2]
    # print(trajs)
    # print(ground_reward)


    # Args::
	# 		full_size: grid size of a side, envs are square, resulting NxN, integer
	# 		p_slip: traditionally "wind", change of slipping during transition, float
	# 		n_objects: number of objects in the world, integer
	# 		n_colours: number of possible colours, integer
	# 	Returns:
	# 		Class object of type ObjectWorld

    env = ObjectWorld(grid_size, wind, 2 , 2)        # terminals = [grid_size**2-1]) 
    
    # 
    # trajectory should be (T,L,2)--> (sample_number, trajectory_length, state_sequence, action_sequence)
    


    rewards = deep_maxent_irl(env, discount, trajs, epochs, learning_rate)
    print("final rewards is: ", rewards)


    # Plot/ Visualize
    # ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    ground_r = ground_reward


    # plt.subplot(1, 2, 1)
    # plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Groundtruth reward")
    # plt.subplot(1, 2, 2)
    # plt.pcolor(rewards.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Recovered reward")
    # plt.show()

     # plots
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    img_utils.heatmap2d(np.reshape(ground_r, (grid_size, grid_size)), 'Ground Truth Reward', block=False)
    plt.subplot(1, 2, 2)
    img_utils.heatmap2d(np.reshape(rewards, (grid_size, grid_size)), 'Recovered reward Map', block=False)
    
    # plt.savefig('reward-deep-19Aug.png')
    
    plt.savefig('reward.png')
    plt.show()
    




if __name__ == "__main__":
    main()


