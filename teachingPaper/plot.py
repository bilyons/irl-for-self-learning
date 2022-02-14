"""
Utilities for plotting.
"""
import matplotlib 
from itertools import product

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.tri as tri

def is_square(i: int) -> bool:
	return i == math.isqrt(i) ** 2

def plot_transition_probabilities(ax, world, border=None, **kwargs):
	"""
	Plot the transition probabilities of a GridWorld instance.
	Args:
		ax: The matplotlib Axes instance used for plotting.
		world: The GridWorld for which the transition probabilities should
			be plotted.
		border: A map containing styling information regarding the
			state-action borders. All key-value pairs are directly forwarded
			to `pyplot.triplot`.
		All further key-value arguments will be forwarded to
		`pyplot.tripcolor`.
	"""
	xy = [(x - 0.5, y - 0.5) for y, x in product(range(world.grid_size + 1), range(world.grid_size + 1))]
	xy += [(x, y) for y, x in product(range(world.grid_size), range(world.grid_size))]

	t, v = [], []
	for sy, sx in product(range(world.grid_size), range(world.grid_size)):
		state = world.state_point_to_index((sx, sy))
		state_r = world.state_point_to_index_clipped((sx + 1, sy))
		state_l = world.state_point_to_index_clipped((sx - 1, sy))
		state_t = world.state_point_to_index_clipped((sx, sy + 1))
		state_b = world.state_point_to_index_clipped((sx, sy - 1))

		# compute cell points
		bl, br = sy * (world.grid_size + 1) + sx, sy * (world.grid_size + 1) + sx + 1
		tl, tr = (sy + 1) * (world.grid_size + 1) + sx, (sy + 1) * (world.grid_size + 1) + sx + 1
		cc = (world.grid_size + 1)**2 + sy * world.grid_size + sx

		# compute triangles
		t += [(tr, cc, br)]                             # action = (1, 0)
		t += [(tl, bl, cc)]                             # action = (-1, 0)
		t += [(tl, cc, tr)]                             # action = (0, 1)
		t += [(bl, br, cc)]                             # action = (0, -1)

		# stack triangle values
		v += [world.p_transition[state, state_r, 0]]    # action = (1, 0)
		v += [world.p_transition[state, state_l, 1]]    # action = (-1, 0)
		v += [world.p_transition[state, state_t, 2]]    # action = (0, 1)
		v += [world.p_transition[state, state_b, 3]]    # action = (0, -1)

	x, y = zip(*xy)
	x, y = np.array(x), np.array(y)
	t, v = np.array(t), np.array(v)

	ax.set_aspect('equal')
	ax.set_xticks(range(world.grid_size))
	ax.set_yticks(range(world.grid_size))
	ax.set_xlim(-0.5, world.grid_size - 0.5)
	ax.set_ylim(-0.5, world.grid_size - 0.5)

	p = ax.tripcolor(x, y, t, facecolors=v, vmin=0.0, vmax=1.0, **kwargs)

	if border is not None:
		ax.triplot(x, y, t, **border)

	return p


def plot_state_values(ax, world, values, border, **kwargs):
	"""
	Plot the given state values of a GridWorld instance.
	Args:
		ax: The matplotlib Axes instance used for plotting.
		world: The GridWorld for which the state-values should be plotted.
		values: The state-values to be plotted as table
			`[state: Integer] -> value: Float`.
		border: A map containing styling information regarding the state
			borders. All key-value pairs are directly forwarded to
			`pyplot.triplot`.
		All further key-value arguments will be forwarded to
		`pyplot.imshow`.
	"""
	if is_square(world.n_states):
		p = ax.imshow(np.reshape(values, (world.grid_size, world.grid_size)), origin='lower', **kwargs)
	else:
		p = ax.imshow(np.reshape(values, (1, world.grid_size)), origin='lower', **kwargs)

	if border is not None:
		if is_square(world.n_states):
			for i in range(0, world.grid_size + 1):
				ax.plot([i - 0.5, i - 0.5], [-0.5, world.grid_size - 0.5], **border, label=None)
				ax.plot([-0.5, world.grid_size - 0.5], [i - 0.5, i - 0.5], **border, label=None)
				for x in range(world.grid_size):
					for y in range(world.grid_size):
						plt.text(x,y, '%.2f' % values[x+5*y],
							 horizontalalignment='center',
							 verticalalignment='center',
							 )
		else:
			for i in range(0, world.grid_size + 1):
				ax.set_yticks([])
				ax.plot([-0.5, world.grid_size - 0.5], [1 - 0.5, 1 - 0.5], **border, label=None)
				for x in range(world.grid_size):
					plt.text(x,0, '%.2f' % values[x],
							 horizontalalignment='center',
							 verticalalignment='center',
							 )

	return p


def plot_deterministic_policy(ax, world, policy, **kwargs):
	"""
	Plot a deterministic policy as arrows.
	Args:
		ax: The matplotlib Axes instance used for plotting.
		world: The GridWorld for which the policy should be plotted.
		policy: The policy to be plotted as table
			`[state: Index] -> action: Index`.
		All further key-value arguments will be forwarded to
		`pyplot.arrow`.
	"""
	arrow_direction = [(0.33, 0), (-0.33, 0), (0, 0.33), (0, -0.33)]

	for state in range(world.n_states):
		cx, cy = world.state_to_coordinate(state)
		dx, dy = arrow_direction[policy[state]]
		ax.arrow(cx - 0.5 * dx, cy - 0.5 * dy, dx, dy, head_width=0.1, **kwargs)


def plot_stochastic_policy(ax, world, policy, border=None, **kwargs):
	"""
	Plot a stochastic policy.
	Args:
		ax: The matplotlib Axes instance used for plotting.
		world: The GridWorld for which the policy should be plotted.
		policy: The stochastic policy to be plotted as table
			`[state: Index, action: Index] -> probability: Float`
			representing the probability p(action | state) of an action
			given a state.
		border: A map containing styling information regarding the
			state-action borders. All key-value pairs are directly forwarded
			to `pyplot.triplot`.
		All further key-value arguments will be forwarded to
		`pyplot.tripcolor`.
	"""

	if is_square(world.n_states):
		xy = [(x - 0.5, y - 0.5) for y, x in product(range(world.grid_size + 1), range(world.grid_size + 1))]
		# print(xy)
		xy += [(x, y) for y, x in product(range(world.grid_size), range(world.grid_size))]
		# print("second ", xy)

		t, v = [], []
		for sy, sx in product(range(world.grid_size), range(world.grid_size)):
			# print(sy, sx)
			state = world.point_to_int((sx, sy))

			# compute cell points
			bl, br = sy * (world.grid_size + 1) + sx, sy * (world.grid_size + 1) + sx + 1
			tl, tr = (sy + 1) * (world.grid_size + 1) + sx, (sy + 1) * (world.grid_size + 1) + sx + 1
			cc = (world.grid_size + 1)**2 + sy * world.grid_size + sx

			# compute triangles
			t += [(tr, cc, br)]                 # action = (1, 0)
			t += [(tl, bl, cc)]                 # action = (-1, 0)
			t += [(tl, cc, tr)]                 # action = (0, 1)
			t += [(bl, br, cc)]                 # action = (0, -1)

			# stack triangle values
			v += [policy[state, 0]]             # action = (1, 0)
			v += [policy[state, 2]]             # action = (-1, 0)
			v += [policy[state, 1]]             # action = (0, 1)
			v += [policy[state, 3]]             # action = (0, -1)



		# print(xy)
		x, y = zip(*xy)
		# print(x,y)
		x, y = np.array(x), np.array(y)
		t, v = np.array(t), np.array(v)

		ax.set_aspect('equal')
		ax.set_xticks(range(world.grid_size))
		ax.set_yticks(range(world.grid_size))
		ax.set_xlim(-0.5, world.grid_size - 0.5)
		ax.set_ylim(-0.5, world.grid_size - 0.5)

		p = ax.tripcolor(x, y, t, facecolors=v, vmin=0.0, vmax=1.0, **kwargs)

		if border is not None:
			ax.triplot(x, y, t, **border)

		return p

	else:

		xy = [(x - 0.5, 1 - 0.5) for x in range(world.grid_size + 1)]
		# print(xy)
		xy += [(x, 1) for x in range(world.grid_size)]
		# print(xy)

		t, v = [], []
		for sx in range(world.grid_size):
			# print(sy, sx)
			state = world.coordinate_to_state(sx)

			# compute cell points
			bl, br = 1 * (world.grid_size + 1) + sx, 1 * (world.grid_size + 1) + sx + 1
			tl, tr = (1 + 1) * (world.grid_size + 1) + sx, (1 + 1) * (world.grid_size + 1) + sx + 1
			cc = (world.grid_size + 1)**2 + 1 * world.grid_size + sx

			# compute triangles
			t += [(tr, cc, br)]                 # action = (1, 0)
			t += [(tl, bl, cc)]                 # action = (-1, 0)
			t += [(tl, cc, tr)]                 # action = (0, 1)
			t += [(bl, br, cc)]                 # action = (0, -1)

			# stack triangle values
			v += [0]          # action = (1, 0)
			v += [0]          # action = (-1, 0)
			v += [policy[state, 0]]             # action = (0, 1)
			v += [policy[state, 1]]             # action = (0, -1)

		# print(xy)
		x, y = zip(*xy)
		# print(x,y)
		x, y = np.array(x), np.array(y)
		t, v = np.array(t), np.array(v)

		# ax.set_aspect('equal')
		ax.set_xticks(range(world.grid_size))
		ax.set_yticks(range(1))
		ax.set_xlim(-0.5, world.grid_size - 0.5)
		ax.set_ylim(-0.5, 1 - 0.5)

		p = ax.tripcolor(x, y, t, facecolors=v, vmin=0.0, vmax=1.0, **kwargs)

		if border is not None:
			ax.triplot(x, y, t, **border)

		return p

def plot_trajectory(ax, world, trajectory, **kwargs):
	"""
	Plot a trajectory as line.
	Args:
		ax: The matplotlib Axes instance used for plotting.
		world: The GridWorld for which the trajectory should be plotted.
		trajectory: The `Trajectory` object to be plotted.
		All further key-value arguments will be forwarded to
		`pyplot.tripcolor`.
	"""
	xy = [world.state_to_coordinate(s) for s in trajectory.states()]
	x, y = zip(*xy)

	return ax.plot(x, y, **kwargs)