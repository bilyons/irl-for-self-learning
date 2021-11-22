import sys, os
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from mdp.gridworld import *
from mdp.value_iteration import *
from mdp.trajectory import *
from algorithms import maxent as M
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm, trange
def normalize(vals):
	"""
	normalize to (0, max_val)
	input:
	vals: 1d array
	"""
	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)

def main(size, n_rewards, r_dif, reflexive, er):

	print(f"N_rewards: {n_rewards} R_dif: {r_dif} and Reflexive: {reflexive} error: {er}")

	# Create gridworld
	gw = StochasticGridWorld(5,0.0)

	# Create reward function
	reward = np.zeros(gw.n_states)

	if n_rewards == 2:
		initial = 2
		terminal = [20,24]
		reward[24] = 1.0
		reward[20] = 1.0 - r_dif
	elif n_rewards == 4:
		initial = 12
		terminal = [0, 4, 20,24]
		reward[24] = 1.0
		reward[20] = 0.8
		reward[4] = 0.6
		reward[0] = 0.4
	else:
		print("Incorrect number of rewards for current design")
		exit()

	if reflexive == True:
		folder = "rrl"
	else:
		folder = "primal"

	for tests in tqdm(range(0, 300)):

		policy = find_policy(gw.transition_prob, reward, 0.9)
		
		trajectories = list(generate_trajectories(1, gw, policy, initial, terminal))

		solved = False

		teach_r = np.ones(gw.n_states)*-1.0
		teach_r[terminal] = 0.5

		demo = 1

		saved_policies = []
		saved_errors = []

		while not solved:
			saved_policies.append(policy)

			r = M.maxent_irl(gw, terminal, trajectories)
			r = normalize(r)

			error = mean_squared_error(reward, r, squared=False)

			saved_errors.append(error)

			if error <= er:
				solved = True
				print(f"Agent {tests} successful at demo {demo}")
				break

			if reflexive == True:
				policy = find_policy(gw.transition_prob, (reward - r + teach_r), 0.9)

			trajectories = trajectories + list(generate_trajectories(1, gw, policy, initial, terminal))

			demo+=1

			if demo == 100:
				print(f"Agent {tests} failed at demo {demo}")
				break

		saving = saved_errors + saved_policies + [r]

		if n_rewards == 2:
			path = os.getcwd()+"/data/r_dif/"+str(r_dif)+"/"+folder+"/"
		else:
			path = os.getcwd()+"/data/r_dif/multi/"+str(er)+"/"+folder+"/"

		with open(path+'run_{}.pkl'.format(tests), 'wb') as filehandle:
			pickle.dump(saving, filehandle)

if __name__ == "__main__":
	main(5, 2, 0.1, False, 0.02)
