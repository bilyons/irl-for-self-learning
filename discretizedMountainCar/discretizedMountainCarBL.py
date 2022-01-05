import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from RLlearner import QLearning

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# Create an environment and set random seed
env = gym.make('MountainCar-v0')

raws, rolling_mean = QLearning(env, 0.1, 0.9, 0.9, 0.1, 5000)

plt.figure()
# Plot scores obtained per episode
plt.plot(raws); plt.title("Scores");
plt.savefig("raw_score.png")

# If the scores are noisy, plot a rolling mean of the scores.
def plot_scores(scores, rolling_window=100, figname = "default.png"):
    """Plot scores and optional rolling mean using specified window."""
    
    plt.figure()
    plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    plt.savefig(figname)
    return rolling_mean

rolling_mean = plot_scores(raws, figname= "mean_scores.png")

# Run in test mode and analyze scores obtained
test_scores = run(q_agent, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = plot_scores(test_scores, rolling_window=10, figname="test_score.png")

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
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    plt.savefig("Qtable.png")


plot_q_table(q_agent.q_table)

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