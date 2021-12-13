from discretizedMountainCar import create_uniform_grid, QLearningAgent, run, plot_scores, plot_q_table
import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from RLlearner import QLearningAgent

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# Create an environment and set random seed
env = gym.make('MountainCar-v0')
env.seed(505)

# Modify the Grid
# TODO: Create a new agent with a different state space grid
state_grid_new = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
q_agent_new = QLearningAgent(env, state_grid_new)
q_agent_new.scores = []  # initialize a list to store scores for this agent

# Train it over a desired number of episodes and analyze scores
# Note: This cell can be run multiple times, and scores will get accumulated
q_agent_new.scores += run(q_agent_new, env, num_episodes=50000)  # accumulate scores
rolling_mean_new = plot_scores(q_agent_new.scores)

# Run in test mode and analyze scores obtained
test_scores = run(q_agent_new, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = plot_scores(test_scores, "custom_scores.png")

# Visualize the learned Q-table
plot_q_table(q_agent_new.q_table)