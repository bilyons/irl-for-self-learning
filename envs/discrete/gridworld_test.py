"""
Implements gridworld MDP

Billy Lyons, 2021
billy.lyons@ed.ac.uk

Adapted from Matthew Alger: https://github.com/MatthewJA/Inverse-Reinforcement-Learning
"""

import numpy as np
from itertools import product

class GridWorld(object):
	"""
	Gridworld environment
	"""

	def __init__(self, size, wind, terminals = [20, 24], initial_rewards = None):
		"""
		input:
			size: grid size of a side, envs are square, resulting NxN
			terminals: list of terminating states
			rewards: array of rewards in the state space
			wind: traditionally "wind", change of slipping during transition (MDP Uncertainty)
		"""
		self.actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # Right, Up, Left, Down
		# [(1,0), (-1,0), (0, 1), (0, -1)]#, (0, 0)]
		self.n_actions = len(self.actions)
		self.n_states = size**2
		self.full_size = size
		self.wind = wind
		self.initial_rewards = initial_rewards
		self.terminals = terminals     #[20, 24]

		self.objects = {}
		
		
		self.offset = 0               # offset means where the robot is initially spawn at (offset to 0 state)
		self.min_ = int(self.offset)  # self.n_states - self.full_size #int(self.offset)
		self.max_ = self.n_states - 1 # int((self.offset+self.spawn_size))

		self.features = state_features(self)

		# Construct probability array
		self.transition_prob = self._transition_prob_table()

		self.rewards = self.create_rewards()

	# Jerry, in case you see this, I think I am going to revisit these to be more efficient
	# But they are suitable for now

	def int_to_point(self, state):
		# Converts a state from s in size**2 to (y,x) coordinate
		return state % self.full_size, state // self.full_size

	def point_to_int(self, coord):
		# Converts a coordinate represented state to full index
		return coord[1]*self.full_size + coord[0]

	def _transition_prob_table(self):
		"""
		Builds the internal probability transition table.
		Returns:
			The probability transition table of the form
				NOTE: MAJOR CHANGE: [state_from, action,  state_to]
			containing all transition probabilities. The individual
			transition probabilities are defined by `self._transition_prob'.
		"""
		# table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))
		table = np.zeros(shape=(self.n_states, self.n_actions, self.n_states))

		# s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
		s1, a, s2 = range(self.n_states), range(self.n_actions), range(self.n_states)
		# for s_from, s_to, a in product(s1, s2, a):
		# 	table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)
		for s_from,  a, s_to in product(s1, a, s2):
			table[s_from, a, s_to] = self._transition_prob(s_from, a, s_to)

		return table

	def _transition_prob(self, s_from, a, s_to):
		"""
		Compute the transition probability for a single transition.
		Args:
			s_from: The state in which the transition originates.
			s_to: The target-state of the transition.
			a: The action via which the target state should be reached.
		Returns:
			The transition probability from `s_from` to `s_to` when taking
			action `a`.
		"""
		fx, fy = self.int_to_point(s_from)
		tx, ty = self.int_to_point(s_to)
		ax, ay = self.actions[a]

		# deterministic transition defined by action
		# intended transition defined by action
		if fx + ax == tx and fy + ay == ty:
			return 1.0 - self.wind + self.wind / self.n_actions

		# we can slip to all neighboring states
		if abs(fx - tx) + abs(fy - ty) == 1:
			return self.wind / self.n_actions

		# we can stay at the same state if we would move over an edge
		if fx == tx and fy == ty:
			# intended move over an edge
			if not 0 <= fx + ax < self.full_size or not 0 <= fy + ay < self.full_size:
				# double slip chance at corners
				if not 0 < fx < self.full_size - 1 and not 0 < fy < self.full_size - 1:
					return 1.0 - self.wind + 2.0 * self.wind / self.n_actions

				# regular probability at normal edges
				return 1.0 - self.wind + self.wind / self.n_actions

			# double slip chance at corners
			if not 0 < fx < self.full_size - 1 and not 0 < fy < self.full_size - 1:
				return 2.0 * self.wind / self.n_actions

			# single slip chance at edge
			if not 0 < fx < self.full_size - 1 or not 0 < fy < self.full_size - 1:
				return self.wind / self.n_actions

			# otherwise we cannot stay at the same state
			return 0.0

		# otherwise this transition is impossible
		return 0.0

	def state_features(self):
		return np.identity(self.n_states)

	def create_rewards(self):
		"""
		On startup, creates the reward array for two different rewards
		Returns:
			The transition probability from `s_from` to `s_to` when taking
			action `a`.
		"""
		rewards = np.zeros((self.n_states))
		#rewards[self.min_] = 1.0 - self.r_dif   #rewards[self.max_] = 1.0

		#See if the initial reward has been manually designed as input
		if self.initial_rewards is None:                 # if not defined randomly initialize reward
			rewards = np.random.random((self.n_states))  # random floats in [0.0, 1.0), no negative reward
		else:
			rewards = self.initial_rewards
		
		return rewards

	def get_reward(self, state):
		"""
		Reward collection function
		Args:
			state: current state of the agent
		Returns:
			reward: the reward at the given state in the MDP
		"""
		return self.rewards[state]

	def is_goal(self, state):
		"""
		Determines if the episode has finished where there is no set length
		Args:
			state: current state of the agent
		Returns:
			boolean: returns True or False depending on if this is a terminating state
				?? terminating states are where the task reward > 1 (not reflexive reward)
		"""
		# if self.rewards[state] > 0:   return True
		# else:  return False

		if state in self.terminals:
			return True
		else:
			return False

	def movement(self, state, action):
		"""
		Determines the next state an agent will be in
		Args:
			state: current state of the agent
			action: the action selected by the agent
		Returns:
			integer: the new state of the agent, drawn from the distribution of 
			possible future states given an action in the initial state
		"""
		return np.random.choice(self.n_states, p=self.transition_prob[state,:,action])

def state_features(world):
	# Returns a feature matrix where each state is an individual feature
	# Identity matrix the size of the state space
	return np.identity(world.n_states)



# test Gridworld env
if __name__ == "__main__":
	env = GridWorld(3, 0, initial_rewards=[0,0,0,1,0,0,0,0,0])
	print(env.rewards)  # random (or not) initial rewards
    
	a = 0   # test action
	print(env.movement(0, a))
	# NOTE: action <-> (right, left, up, down) // Coordinate system <--> (x(right/left), y(up/down))
	# self.actions = [(1,0), (-1,0), (0, 1), (0, -1)]
	print(env.int_to_point(env.movement(0, a))) # Turn to coordinate system

	