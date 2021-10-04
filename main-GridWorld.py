
from mimetypes import init
from envs.discrete.gridworld_test import GridWorld
# from irl.standard.irl_set_traj_length import *
from irl.deep.deep_maxent_irl import *
import envs.discrete.originalGridWorld.gridworld as original
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
    grid_size = 5
    wind = 0.3
    trajectory_length = 3* grid_size
    discount = 0.01
    n_trajectories = 20
    learning_rate = 0.01
    epochs = 100
  

    gw = original.Gridworld(grid_size, wind, discount)
    #print("transition_probabilities: ",gw.transition_probability)
    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy)
    trajs = trajectories[:,:,0:2]
    # print(trajs)


  
    env = GridWorld(grid_size, wind, terminals = [grid_size**2-1]) 
    print("initial rewards: ", env.rewards)  
    print("number_initial_states", env.n_states)
    # 
    # trajectory should be (T,L,2)--> (sample_number, trajectory_length, state_sequence, action_sequence)
    


    rewards = deep_maxent_irl(env, discount, trajs, epochs, learning_rate)
    print("final rewards is: ", rewards)


    # Plot/ Visualize
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
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
    # plt.savefig('reward.png')
    plt.show()
    




if __name__ == "__main__":
    main()


