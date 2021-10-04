"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn
from itertools import product

class GridWorld(object):
	"""
	Gridworld MDP.
	"""

	def __init__(self, grid_size, wind, discount):
		"""
		grid_size: Grid size. int.
		wind: Chance of moving randomly. float.
		discount: MDP discount. float.
		-> Gridworld
		"""

		self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
		self.n_actions = len(self.actions)
		self.n_states = grid_size**2
		self.grid_size = grid_size
		self.wind = wind
		self.discount = discount

		# Preconstruct the transition probability array.
		self.transition_prob = self._transition_prob_table()

	def __str__(self):
		return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
											  self.discount)

	def feature_vector(self, i, feature_map="ident"):
		"""
		Get the feature vector associated with a state integer.

		i: State int.
		feature_map: Which feature map to use (default ident). String in {ident,
			coord, proxi}.
		-> Feature vector.
		"""

		if feature_map == "coord":
			f = np.zeros(self.grid_size)
			x, y = i % self.grid_size, i // self.grid_size
			f[x] += 1
			f[y] += 1
			return f
		if feature_map == "proxi":
			f = np.zeros(self.n_states)
			x, y = i % self.grid_size, i // self.grid_size
			for b in range(self.grid_size):
				for a in range(self.grid_size):
					dist = abs(x - a) + abs(y - b)
					f[self.point_to_int((a, b))] = dist
			return f
		# Assume identity map.
		f = np.zeros(self.n_states)
		f[i] = 1
		return f

	def feature_matrix(self, feature_map="ident"):
		"""
		Get the feature matrix for this gridworld.

		feature_map: Which feature map to use (default ident). String in {ident,
			coord, proxi}.
		-> NumPy array with shape (n_states, d_states).
		"""

		features = []
		for n in range(self.n_states):
			f = self.feature_vector(n, feature_map)
			features.append(f)
		# print(np.array(features))
		# exit()
		return np.array(features)

	def state_to_coordinate(self, state):
		# Converts a state from s in size**2 to (y,x) coordinate
		return state % self.grid_size, state // self.grid_size

	def coordinate_to_state(self, coord):
		# Converts a coordinate represented state to full index
		return coord[1]*self.grid_size + coord[0]

	def neighbouring(self, i, k):
		"""
		Get whether two points neighbour each other. Also returns true if they
		are the same point.

		i: (x, y) int tuple.
		k: (x, y) int tuple.
		-> bool.
		"""

		return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1
		
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
		fx, fy = self.state_to_coordinate(s_from)
		tx, ty = self.state_to_coordinate(s_to)
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
			if not 0 <= fx + ax < self.grid_size or not 0 <= fy + ay < self.grid_size:
				# double slip chance at corners
				if not 0 < fx < self.grid_size - 1 and not 0 < fy < self.grid_size - 1:
					return 1.0 - self.wind + 2.0 * self.wind / self.n_actions

				# regular probability at normal edges
				return 1.0 - self.wind + self.wind / self.n_actions

			# double slip chance at corners
			if not 0 < fx < self.grid_size - 1 and not 0 < fy < self.grid_size - 1:
				return 2.0 * self.wind / self.n_actions

			# single slip chance at edge
			if not 0 < fx < self.grid_size - 1 or not 0 < fy < self.grid_size - 1:
				return self.wind / self.n_actions

			# otherwise we cannot stay at the same state
			return 0.0

		# otherwise this transition is impossible
		return 0.0

	def reward(self, state_int):
		"""
		Reward for being in state state_int.

		state_int: State integer. int.
		-> Reward.
		"""

		if state_int == self.n_states - 1:
			return 1
		return 0

	def average_reward(self, n_trajectories, trajectory_length, policy):
		"""
		Calculate the average total reward obtained by following a given policy
		over n_paths paths.

		policy: Map from state integers to action integers.
		n_trajectories: Number of trajectories. int.
		trajectory_length: Length of an episode. int.
		-> Average reward, standard deviation.
		"""

		trajectories = self.generate_trajectories(n_trajectories,
											 trajectory_length, policy)
		rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
		rewards = np.array(rewards)

		# Add up all the rewards to find the total reward.
		total_reward = rewards.sum(axis=1)

		# Return the average reward and standard deviation.
		return total_reward.mean(), total_reward.std()

	def optimal_policy(self, state_int):
		"""
		The optimal policy for this gridworld.

		state_int: What state we are in. int.
		-> Action int.
		"""

		sx, sy = self.int_to_point(state_int)

		if sx < self.grid_size and sy < self.grid_size:
			return rn.randint(0, 2)
		if sx < self.grid_size-1:
			return 0
		if sy < self.grid_size-1:
			return 1
		raise ValueError("Unexpected state.")

	def optimal_policy_deterministic(self, state_int):
		"""
		Deterministic version of the optimal policy for this gridworld.

		state_int: What state we are in. int.
		-> Action int.
		"""

		sx, sy = self.int_to_point(state_int)
		if sx < sy:
			return 0
		return 1

	def generate_trajectories(self, n_trajectories, trajectory_length, policy,
									random_start=False):
		"""
		Generate n_trajectories trajectories with length trajectory_length,
		following the given policy.

		n_trajectories: Number of trajectories. int.
		trajectory_length: Length of an episode. int.
		policy: Map from state integers to action integers.
		random_start: Whether to start randomly (default False). bool.
		-> [[(state int, action int, reward float)]]
		"""

		trajectories = []
		for _ in range(n_trajectories):
			if random_start:
				sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
			else:
				sx, sy = 0, 0

			trajectory = []
			for _ in range(trajectory_length):
				if rn.random() < self.wind:
					action = self.actions[rn.randint(0, 4)]
				else:
					# Follow the given policy.
					action = self.actions[policy(self.point_to_int((sx, sy)))]

				if (0 <= sx + action[0] < self.grid_size and
						0 <= sy + action[1] < self.grid_size):
					next_sx = sx + action[0]
					next_sy = sy + action[1]
				else:
					next_sx = sx
					next_sy = sy

				state_int = self.point_to_int((sx, sy))
				action_int = self.actions.index(action)
				next_state_int = self.point_to_int((next_sx, next_sy))
				reward = self.reward(next_state_int)
				trajectory.append((state_int, action_int, reward))

				sx = next_sx
				sy = next_sy

			trajectories.append(trajectory)

		return np.array(trajectories)
