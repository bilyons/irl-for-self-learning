import gym
import numpy as np
import pandas as pd
from scipy.special import softmax

# You don't need quite so much of what you have above, I will try to explain here
# First, you don't need to make the env again, you can pass it from the main file
def QLearning(env, lr, discount, epsilon, min_eps, episodes, stochastic=False, tau=None):
	"""Define a simple Q learning agent for a discretised environment. - BL
	
	Parameters
	----------
	env       : environment, OpenAI Gym
	lr        : learning rate of Q Learning
	discount  : discount factor
	epsilon   : initial exploration rate
	min_eps   : lowest possible exploration rate
	episodes  : number of episodes to train on
	stochastic: if stochastic, boltzmann dist, if false, epsilon greedy
	tau       : temperature of boltzmann

	Returns
	-------
	A list of the average rewards as a rolling window of 100
	"""	

	# For Jerry:
	# There are a lot of built in features of the open ai gym environment which means
	# you don't have to do as much work with the discretisation code.
	n_states = (env.observation_space.high-env.observation_space.low)*\
					np.array([10,100]) # by changing these we can change the granularity
	n_states = np.round(n_states, 0).astype(int) + 1

	Q = np.random.uniform(low=-1, high=1,
		size = (n_states[0], n_states[1], env.action_space.n))

	if stochastic == True and tau == None:
		print("Must have value for tau if stochastic policy used")
		exit()

	def action_selection(state, epsilon, stochastic, tau):
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

	# Reward tracking 
	rewards = []
	avg_rewards = []

	# Linear reduction in epsilon (you might want to change this)
	delta_eps = (epsilon-min_eps)/episodes

	for i in range(episodes):

		# Initialising params
		done = False
		episode_reward, reward = 0,0
		state = env.reset() # Non discretised

		# Discretise the state
		# We do this as above then it is really simple to follow it through
		d_state = (state-env.observation_space.low)*\
					np.array([10,100])
		d_state = np.round(d_state, 0).astype(int)

		while not done:
			# Render environment for last five episodes
			# if i >= (episodes - 20):
			# 	env.render()
			action = action_selection(d_state, epsilon, stochastic, tau)

			# Get new state
			new_state, reward, done, _ = env.step(action)

			# Discretise new state
			d_new_state = (new_state-env.observation_space.low)*\
					np.array([10,100])
			d_new_state = np.round(d_new_state, 0).astype(int)

			if done and new_state[0]>=0.5:
				Q[d_state[0], d_state[1], action] = reward # Updating terminal Q value
			else:
				# Gotta learn
				delta = lr*(reward + discount*np.max(Q[d_new_state[0], d_new_state[1],:]-
														Q[d_state[0], d_state[1], action]))
				Q[d_state[0], d_state[1], action] += delta
			episode_reward += reward

			d_state = d_new_state # Becomes new start state

		if epsilon > min_eps:
			epsilon -= delta_eps

		rewards.append(episode_reward)

		if i%100==0:
			print(i)

	reward_series = pd.Series(rewards)
	windows = reward_series.rolling(100)
	moving_averages = windows.mean()
	moving_averages_list = moving_averages.tolist()
	without_nans = moving_averages_list[100-1:]

	return rewards, without_nans, Q

# Q[d_state[0], d_state[1], action]