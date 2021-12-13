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
env.seed(505);

def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.
    
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    x_grid_len = (high[0] - low[0])/(bins[0]*1.0)
    y_grid_len = (high[1] - low[1])/(bins[1]*1.0)
    x_grid = [low[0]+i*x_grid_len for i in range(1, bins[0])]
    y_grid = [low[1]+i*y_grid_len for i in range(1, bins[1])]
    
    return [np.array(x_grid), np.array(y_grid)]


low = [-1.0, -5.0]
high = [1.0, 5.0]
create_uniform_grid(low, high)  # [test]


def discretize(sample, grid):
    """Discretize a sample as per given grid.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    
    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    x = np.digitize(np.array([sample[0]]), grid[0], right=False)
    y = np.digitize(np.array([sample[1]]), grid[1], right=False)
    return [x[0], y[0]]


# Test with a simple grid and some samples
grid = create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
print('grid: {}'.format(grid))
samples = np.array(
    [[-1.0 , -5.0],
     [-0.81, -4.1],
     [-0.8 , -4.0],
     [-0.5 ,  0.0],
     [ 0.2 , -1.9],
     [ 0.8 ,  4.0],
     [ 0.81,  4.1],
     [ 1.0 ,  5.0]])
discretized_samples = np.array([discretize(sample, grid) for sample in samples])
print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")

import matplotlib.collections as mc

def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    """Visualize original and discretized samples on a given 2-dimensional grid."""

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)
    
    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    print('grid_extended: {}'.format(grid_extended))
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    print('grid_centers: {}'.format(grid_centers))
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples
    print('locs: {}'.format(locs))
    
    ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
    ax.plot(locs[:, 0], locs[:, 1], 's')  # plot discretized samples in mapped locations
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))  # add a line connecting each original-discretized sample
    ax.legend(['original', 'discretized'])
    plt.savefig("visualizeGrid.png")

    
visualize_samples(samples, discretized_samples, grid, low, high)


# Apply to RL env
# Create a grid to discretize the state space
state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
print("state_grid: ", state_grid)
# Obtain some samples from the space, discretize them, and then visualize them
state_samples = np.array([env.observation_space.sample() for i in range(10)])
discretized_state_samples = np.array([discretize(sample, state_grid) for sample in state_samples])
visualize_samples(state_samples, discretized_state_samples, state_grid,
                  env.observation_space.low, env.observation_space.high)
plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space


def run(agent, env, num_episodes=500000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        # Save final score
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores

q_agent = QLearningAgent(env, state_grid)
scores = run(q_agent, env)

plt.figure()
# Plot scores obtained per episode
plt.plot(scores); plt.title("Scores");
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

rolling_mean = plot_scores(scores, figname= "mean_scores.png")

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