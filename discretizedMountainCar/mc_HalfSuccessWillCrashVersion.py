from curses import A_ALTCHARSET
import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from RLlearner import QLearning

from irl.deep.deep_maxent_irl_mc import *
import utils.img.img_utils as img_utils

#import imitation repo
from typing import Any, Mapping, Type
import pytest
import torch as th
from stable_baselines3.common import vec_env
from imitation.algorithms import base
from imitation.algorithms.mce_irl import (
    MCEIRL,
    TabularPolicy,
    mce_occupancy_measures,
    mce_partition_fh,
)
from imitation.data import rollout
from imitation.envs import resettable_env
from imitation.envs.examples import model_envs
from imitation.rewards import reward_nets
from imitation.util.util import tensor_iter_norm


plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# Create an environment and set random seed
env = gym.make('MountainCar-v0')
# print(env.action_space)    Discrete(3)
# print(env.observation_space)
# Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
# n1_states = (env.observation_space.high-env.observation_space.low)*\
# 					np.array([10,100]) # by changing these we can change the granularity
# print(n1_states)
# n1_states = np.round(n1_states, 0).astype(int) + 1
# print(n1_states)

raws, rolling_mean, Q = QLearning(env, 0.01, 0.9, 0.9, 0.1, 200)

plt.figure()
plt.plot(raws); plt.title("Scores");
plt.savefig("raw_score.png")
plt.figure()
plt.plot(rolling_mean); plt.title("Mean Scores");
plt.savefig("rolling_mean.png")

#  look at the final Q-table that is learned by the agent.
def plot_q_table(q_table):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(q_image, cmap='jet');
    cbar = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig("Qtable.png")

plot_q_table(Q)    # Q: d_state[0], d_state[1], action

#----------------trajectory------------------------------------------------------
def action_selection(state, epsilon, stochastic = False, tau=None):
        # Determine action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            if stochastic:
                prob = softmax(Q[state[0], state[1],:]/tau)
                action = np.random.choice(env.action_space.n, p=prob)
            else:
                action = np.argmax(Q[state[0], state[1], :])
        return action

traj_len = 20
traj_num = 10

def int_to_point(state, Q):
		# Converts a state from s in size**2 to (y,x) coordinate
    return state % Q.shape[0], state // Q.shape[0]    # (y,x)

def point_to_int(coord, Q):
    # Converts a coordinate represented state to full index
    return coord[1]*Q.shape[0] + coord[0]

def generate_trajs(traj_num,traj_len, Q):
    all_trajs = []
    for i in range(traj_num):
        traj = []

        # Initialising params
        done = False
        # episode_reward, reward = 0,0
        state = env.reset() # Non discretised

        # Discretise the state
        # We do this as above then it is really simple to follow it through
        d_state = (state-env.observation_space.low)*\
                    np.array([10,100])
        d_state = np.round(d_state, 0).astype(int)
        # traj.append(point_to_int(d_state, Q))
        # trajectory.append((state_int, action_int, reward))

        for i in range(traj_len):
            if(not done):
                # action = action_selection(d_state, 0.01)
                action = np.argmax(Q[d_state[0], d_state[1], :])
                # Get new state
                new_state, reward, done, _ = env.step(action)

                # Discretise new state
                d_new_state = (new_state-env.observation_space.low)*\
                        np.array([10,100])
                d_new_state = np.round(d_new_state, 0).astype(int)
                state_int = point_to_int(d_state, Q)

            traj.append((state_int, action, reward))
            d_state = d_new_state # Becomes new start state    
        
        all_trajs.append(traj)
    all_trajs = np.array(all_trajs)
    return all_trajs

all_trajs = generate_trajs(10, 50, Q)
all_trajs = np.array(all_trajs)
print(all_trajs, all_trajs.shape)    # all_trajs: NUM_TRAJ,SEQ_LEN, STATE_DIM
trajectories = all_trajs[:,:,0:2]    # (10, 50, 2)

irl_env = gym.make('MountainCar-v0')

# def trim_trajs(all_trajectories):
   #----------------------------------------------------------------------         




# wind = 0.3
# trajectory_length = 25
discount = 0.95
# n_trajectories = 200
learning_rate = 0.001
epochs = 3

state_dim = Q.shape[0]*Q.shape[1]

ground_r = np.array([1. for s in range(state_dim)])

# policy = find_policy(state_dim, 3, np.ones((state_dim,3,state_dim)), ground_r, 0.95, stochastic=False)#ow.transition_prob,

# print(policy)
# feature_matrix = np.eye(state_dim)  

rewards = deep_maxent_irl(irl_env, discount, trajectories, epochs, learning_rate)
print("final rewards is: ", rewards)


# plots
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
img_utils.heatmap2d(np.reshape(ground_r, (Q.shape[0], Q.shape[1])), 'Ground Truth Reward', block=False)
plt.subplot(1, 2, 2)
img_utils.heatmap2d(np.reshape(rewards, (Q.shape[0], Q.shape[1])), 'Recovered reward Map', block=False)

plt.savefig('IRL.png')

#----------------------------------------------------------------------
# # If the scores are noisy, plot a rolling mean of the scores.
# def plot_scores(scores, rolling_window=100, figname = "default.png"):
#     """Plot scores and optional rolling mean using specified window."""
    
#     plt.figure()
#     plt.plot(scores); plt.title("Scores");
#     rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
#     plt.plot(rolling_mean);
#     plt.savefig(figname)
#     return rolling_mean

# # rolling_mean = plot_scores(raws, figname= "mean_scores.png")
# # Run in test mode and analyze scores obtained
# test_scores = run(q_agent, env, num_episodes=100, mode='test')
# print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
# _ = plot_scores(test_scores, rolling_window=10, figname="test_score.png")

#------------------------------Process----------------------------------
# # Modify the Grid
# # TODO: Create a new agent with a different state space grid
# state_grid_new = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
# q_agent_new = QLearningAgent(env, state_grid_new)
# q_agent_new.scores = []  # initialize a list to store scores for this agent

# # Train it over a desired number of episodes and analyze scores
# # Note: This cell can be run multiple times, and scores will get accumulated
# q_agent_new.scores += run(q_agent_new, env, num_episodes=50000)  # accumulate scores
# rolling_mean_new = plot_scores(q_agent_new.scores)

# # Run in test mode and analyze scores obtained
# test_scores = run(q_agent_new, env, num_episodes=100, mode='test')
# print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
# _ = plot_scores(test_scores, "custom_scores.png")

# # Visualize the learned Q-table
# plot_q_table(q_agent_new.q_table)
#----------------------------------------------------------------------

# # Watch a Smart Agent
# state = env.reset()
# score = 0
# for t in range(200):
#     action = q_agent_new.act(state, mode='test')
#     env.render()
#     state, reward, done, _ = env.step(action)
#     score += reward
#     if done:
#         break 
# print('Final score:', score)
# env.close()