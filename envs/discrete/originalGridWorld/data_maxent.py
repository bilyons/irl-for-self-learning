#Python Script and the GridWord Python API used to get data,
#Start Position is always at left-down cornor
#||Ground Truth|| goal is at right-upper cornor, only here has reward 1 otherwisr 0
import numpy as np
import matplotlib.pyplot as plt
import gridworld

def main(grid_size, discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    wind = 0.3
    trajectory_length = 3*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)
    
    #print("transition_probabilities: ",gw.transition_probability)
    
    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy)

    print("----before----\ntrajectories: ", trajectories)
    print("\n")
    # print("-----after---\ntrajectories0: ", trajectories[:,:,0:2])
    print("-----trajectory-state_sequence: ", trajectories[:,:,0]) 
    print("-----trajectory-action_sequence: ", trajectories[:,:,1])
    
    # #reward = self.reward(next_state_int)
    #trajectory.append((state_int, action_int, reward))

    feature_matrix = gw.feature_matrix()

    print("feature_matrix: ", feature_matrix)

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    
    r = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, trajectories, epochs, learning_rate)
    
    print("Ground-Trueth: ", ground_r)
    
    print("transition_probabilities: ",gw.transition_probability)

    print(r)
    
    #modifying the transition probability to mach (action ,state, state)
    prob_a = gw.transition_probability.reshape(100, -1)
    prob = np.zeros_like(prob_a)
    for i in range(100):    #4 actions, 25 states
        which_position = i // 4
        which_action = i%4
        prob[ which_action*25 + which_position ,: ] = prob_a[ i ,: ]
    

    prob = prob.reshape(1,100, -1)
    a_file = open("transition.txt", "w")
    for row in prob:
        np.savetxt(a_file, row, delimiter=',', fmt='%1.3f')
    a_file.close()


    ao_file = open("original_transition.txt", "w")
    for row in gw.transition_probability:
        np.savetxt(ao_file, row, delimiter=',', fmt='%1.3f')
    ao_file.close()

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(5, 0.01, 20, 200, 0.01)
