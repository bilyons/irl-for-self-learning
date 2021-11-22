"""
Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
from itertools import chain


class Trajectory:
    """
    A trajectory consisting of states, corresponding actions, and outcomes.
    Args:
        transitions: The transitions of this trajectory as an array of
            tuples `(state_from, action, state_to)`. Note that `state_to` of
            an entry should always be equal to `state_from` of the next
            entry.
    """
    def __init__(self, transitions):
        self._t = transitions

    def transitions(self):
        """
        The transitions of this trajectory.
        Returns:
            All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
        """
        return self._t

    def states(self):
        """
        The states visited in this trajectory.
        Returns:
            All states visited in this trajectory as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def states_actions(self):
        """
        The states visited in this trajectory.
        Returns:
            All states visited in this trajectory as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0:2], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)

    def __len__(self):
        length = 0
        for s in self.states():
            length+=1
        return length


def generate_trajectory(world, policy, start, terminal=None):

    state = start

    trajectory = []

    if isinstance(terminal, list):
        while state not in terminal:
            action = np.random.choice(range(world.n_actions), p=policy[state,:])

            next_s = range(world.n_states)
            next_p = world.transition_prob[state,action, :]

            next_state = np.random.choice(next_s, p=next_p)

            trajectory += [(state, action, next_state)]

            state = next_state

    else:
        for t in range(terminal-1):
            action = np.random.choice(range(world.n_actions), p=policy[state,:])

            next_s = range(world.n_states)
            next_p = world.transition_prob[state,action, :]

            next_state = np.random.choice(next_s, p=next_p)

            trajectory += [(state, action, next_state)]

            state = next_state

    return Trajectory(trajectory)

def generate_trajectories(n, world, policy, start, terminal=None):

    def _generate_one():
        s = start

        return generate_trajectory(world, policy, s, terminal)

    return (_generate_one() for _ in range(n))

