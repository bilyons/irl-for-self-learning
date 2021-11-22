"""
Grid world Markov Decision Processes (MDPs)
"""

import numpy as np
from itertools import product

class GridWorld:
	"""
	Basic deterministic gridworld MDP.

	Args:
		size: Width and height of the world

	Attributes:
		n_states: number of states
		n_actions: number of actions
		transition_prob: array of transition probabilities.
			[state_from, action, state_to]
		size: width and height of the world
		actions: the actions of this world
	"""

	def __init__(self, size):
		self.size = size

		self.actions = [(1,0), (-1,0), (0,1), (0,-1)]

		self.n_states = size**2
		self.n_actions = len(self.actions)

		self.transition_prob = self._transition_prob_table()
		self.state_features = self._state_features()

	def state_to_coordinate(self, state):
		return state%self.size, state//self.size

	def coordinate_to_state(self, coord):
		return (max(0, min(coord[1],self.size-1))*self.size + max(0, min(self.size-1, coord[0])))

	def _state_features(self):
		return np.identity(self.n_states)

	def _transition_prob_table(self):
		"""
		Create transition probability array. Static.

		Returns:
			array: [state_from, action, state_to]
		"""

		table = np.zeros(shape=(self.n_states, self.n_actions, self.n_states))

		s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)

		for s_from, a, s_to in product(s1, a, s2):
			table[s_from, a, s_to] = self._transition_prob(s_from, a, s_to)
		return table

	def _transition_prob(self, s_from, a, s_to):
		"""
		Compute transition property for a given transition
		"""
		fx, fy = self.state_to_coordinate(s_from)
		tx, ty = self.state_to_coordinate(s_to)

		ax, ay = self.actions[a]

		# deterministic transition defined by action
		if fx + ax == tx and fy + ay == ty:
			return 1.0

		# we can stay at the same state if we would move over an edge
		if fx == tx and fy == ty:
			if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
				return 1.0

		# otherwise this transition is impossible
		return 0.0

	def __repr__(self):
		return "GridWorld(size={})".format(self.size)

class StochasticGridWorld(GridWorld):

	def __init__(self, size, p_slip):
		self.p_slip = p_slip

		super().__init__(size)

	def _transition_prob(self,s_from, a, s_to):

		fx, fy = self.state_to_coordinate(s_from)
		tx, ty = self.state_to_coordinate(s_to)

		ax, ay = self.actions[a]

		# intended transition defined by action
		if fx + ax == tx and fy + ay == ty:
			return 1.0 - self.p_slip + self.p_slip / self.n_actions

		# we can slip to all neighboring states
		if abs(fx - tx) + abs(fy - ty) == 1:
			return self.p_slip / self.n_actions

		# we can stay at the same state if we would move over an edge
		if fx == tx and fy == ty:
			# intended move over an edge
			if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
				# double slip chance at corners
				if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
					return 1.0 - self.p_slip + 2.0 * self.p_slip / self.n_actions

				# regular probability at normal edges
				return 1.0 - self.p_slip + self.p_slip / self.n_actions

			# double slip chance at corners
			if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
				return 2.0 * self.p_slip / self.n_actions

			# single slip chance at edge
			if not 0 < fx < self.size - 1 or not 0 < fy < self.size - 1:
				return self.p_slip / self.n_actions

			# otherwise we cannot stay at the same state
			return 0.0

		# otherwise this transition is impossible
		return 0.0

	def __repr__(self):
		return "StochasticGridWorld(size={}, p_slip={})".format(self.size, self.p_slip)
