import numpy as np
import argparse
from collections import namedtuple
from tqdm import tqdm
from parfor import pmap
from mdp.value_iteration import find_policy
import mdp.trajectory as T
import mdp.gridworld as G
import plot as P

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

import pickle
import os, os.path
import matplotlib.pyplot as plt

r_dif = 0.5


successes = 0
num_eps = []
evd = []
style = {
	'border': {'color': 'red', 'linewidth': 0.5},
}
preliminary_demos = 5

max_pol = 100 - preliminary_demos
style = {
	'border': {'color': 'red', 'linewidth': 0.5},
}


# # Get ground truth rewards
# ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

# # # print(gw.transition_probability[23,0,:])
# # print(np.random.choice(gw.transition_probability[23,0,:]))
# # exit()
# def policy_eval(policy, reward, transition_probabilities, nS, nA, discount_factor=1.0, theta=0.00001):
# 	"""
# 	Policy Evaluation.
# 	"""
# 	V = np.zeros(nS)
# 	while True:
# 		delta = 0
# 		for s in range(nS):
# 			v = 0
# 			for a, a_prob in enumerate(policy[s]):
# 				if a_prob == 0.0:
# 					continue
# 				ns_prob = transition_probabilities[s, a]
# 				next_v = V[np.arange(nS)]
# 				r = reward[s]
# 				v += np.sum(ns_prob * a_prob * (r + discount_factor * next_v))
# 			delta = max(delta, np.abs(v - V[s]))
# 			V[s] = v
# 		# print(delta)
# 		if delta < theta:
# 			break
# 	return np.array(V)


avg_rew = np.zeros(25)

# ground_truth_policy = find_policy(gw.n_states, gw.n_actions, gw.transition_probability,
# 									ground_r, 0.95, stochastic=True)
# gt_policy_exec = T.stochastic_policy_adapter(ground_truth_policy)

# v_true = policy_eval(ground_truth_policy, ground_r, gw.transition_probability, 
# 									gw.n_states, gw.n_actions, discount_factor=0.95, theta=0.001) 
err_rrl = np.zeros((99, 300))
err_rrl0 = np.zeros((99, 300))
err_rrl1 = np.zeros((99, 300))
err_rrl2 = np.zeros((99, 300))
err_rrl3 = np.zeros((99, 300))
err_rrl4 = np.zeros((99, 300))
err_rrl5 = np.zeros((99, 300))
err_primal = np.zeros((99,300))
max_dem = 0
# print(v_true.reshape((5,5)))
# for i in range(300):

# 	# path = os.getcwd()+"/data/r_dif/"+str(r_dif)+"/"+"rrl/"
# 	path = os.getcwd()+"/data/r_dif/multi/0.03/rrl/"
# 			# open(path+'run.pkl', 'w')
# 	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
# 		rrl = pickle.load(filehandle)

# 	rng = int((len(rrl)-1)/2)

# 	for q in range(rng):
# 		err_rrl[q,i] += rrl[q]

# 	if rng < max_pol:
# 		successes += 1
# 		num_eps.append(rng)
# 		avg_rew+=rrl[-1]

# 	# if rng>=max_pol-1:
# 		# print(rrl[-1].reshape((5,5)))

# 	if rng> max_dem:
# 		max_dem = rng
# 	# print(i)
# 	# print(rng)
# 	# for j in range(rng, len(rrl)):
# 	# 	ax = plt.figure(num='Policy changes', figsize=[20,5]).add_subplot(1, 5, 1)
		
# 	# 	plt.title(f"Policy at episode {i,j}")
# 	# 	P.plot_stochastic_policy(ax, gw, rrl[j], **style)
# 	# 	plt.draw()
# 	# 	plt.show()


# 	# if i ==8:
# 	# 	# print(j-rng)
# 	# 	print(rng)
# 	# 	ax = plt.figure(num='Policy changes', figsize=[20,5]).add_subplot(1, 5, 1)
		
# 	# 	plt.title(f"Policy at episode 0")
# 	# 	P.plot_stochastic_policy(ax, gw, rrl[10], **style)
# 	# 	# ax.get_xaxis().set_visible(False)
# 	# 	# ax.get_yaxis().set_visible(False)
# 	# 	ax = plt.figure(num='Policy changes').add_subplot(1, 5, 2)
# 	# 	plt.title(f"Policy at episode 1")
# 	# 	P.plot_stochastic_policy(ax, gw, rrl[11], **style)
# 	# 	# ax.get_xaxis().set_visible(False)
# 	# 	# ax.get_yaxis().set_visible(False)
# 	# 	ax = plt.figure(num='Policy changes').add_subplot(1, 5, 3)
# 	# 	# ax.get_xaxis().set_visible(False)
# 	# 	# ax.get_yaxis().set_visible(False)
# 	# 	# plt.title(f"Policy at episode 2")
# 	# 	# P.plot_stochastic_policy(ax, gw, rrl[11], **style)
# 	# 	# ax = plt.figure(num='Policy changes').add_subplot(2, 4, 4)
# 	# 	plt.title(f"Policy at episode 3")
# 	# 	P.plot_stochastic_policy(ax, gw, rrl[13], **style)
# 	# 	ax = plt.figure(num='Policy changes').add_subplot(1, 5, 4)
# 	# 	# ax.get_xaxis().set_visible(False)
# 	# 	# ax.get_yaxis().set_visible(False)
# 	# 	# plt.title(f"Policy at episode 4")
# 	# 	# P.plot_stochastic_policy(ax, gw, rrl[13], **style)
# 	# 	# ax = plt.figure(num='Policy changes').add_subplot(2, 4, 6)
# 	# 	plt.title(f"Policy at episode 5")
# 	# 	P.plot_stochastic_policy(ax, gw, rrl[15], **style)
# 	# 	ax = plt.figure(num='Policy changes').add_subplot(1, 5, 5)
# 	# 	# ax.get_xaxis().set_visible(False)
# 	# 	# ax.get_yaxis().set_visible(False)
# 	# 	# plt.title(f"Policy at episode 6")
# 	# 	# P.plot_stochastic_policy(ax, gw, rrl[15], **style)
# 	# 	# ax = plt.figure(num='Policy changes').add_subplot(2, 4, 8)
# 	# 	plt.title(f"Policy at episode 7")
# 	# 	P.plot_stochastic_policy(ax, gw, rrl[17], **style)
# 	# 	plt.subplots_adjust(left=0,
# 	# 				bottom=0, 
# 	# 				right=1.2, 
# 	# 				top=1, 
# 	# 				wspace=0.1, 
# 	# 				hspace=0.0)
# 	# 	# plt.margins(0,0)
# 	# 	# plt.figure(figsize=(10,3))
# 	# 	# plt.set_size_inches(10, 4)
# 	# 	# ax.get_xaxis().set_visible(False)
# 	# 	# ax.get_yaxis().set_visible(False)
# 	# 	plt.savefig('Policy_changes.png', bbox_inches='tight', dpi=100)
# 	# 	plt.draw()
# 	# 	plt.show()
	


# 	# path = os.getcwd()+"/data/r_dif/"+str(r_dif)+"/"+"primal/"
# 	path = os.getcwd()+"/data/r_dif/multi/0.02/primal/"
# 	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
# 		primal = pickle.load(filehandle)

# 	rng = int((len(primal)-1)/2)
# 	print(rng)
# 	for q in range(rng):
# 		# print(primal[q])
# 		# print(primal[q])
# 		err_primal[q,i] += primal[q]


# x_m = range(0, 99)
# y_rrl = np.nanmean(err_rrl, axis=1)
# e_rrl = np.nanstd(err_rrl, axis=1)
# y_primal = np.nanmean(err_primal, axis=1)
# e_primal = np.nanstd(err_primal, axis=1)
# # print(y_primal)
# f, ax = plt.subplots(1)
# # ax.plot(x_m, y_m, 'or')
# ax.plot(x_m, y_rrl, '-', color='blue', label='RRL')
# ax.plot(x_m, y_primal, '-', color='green', label='Forward RL')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1])
# ax.fill_between(x_m, y_rrl - e_rrl, y_rrl + e_rrl, color='blue', alpha=0.2)
# ax.fill_between(x_m, y_primal - e_primal, y_primal + e_primal, color='green', alpha=0.2)
# ax.set_ylim(ymin=0)
# ax.set_xlim([0,98])
# ax.margins(x=0)
# plt.title(f"Demonstration error for four rewards, threshold: 0.03")
# plt.xlabel("Demonstration Number")
# plt.ylabel("RMSE")
# # ax.set_xlim(0,3500)
# # plt.xscale("log")
# # plt.yscale("log")
# plt.show()

for i in range(300):

	# path = os.getcwd()+"/data/r_dif/"+str(r_dif)+"/"+"rrl/"
	path = os.getcwd()+"/data/r_dif/0.5/rrl/"
			# open(path+'run.pkl', 'w')
	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
		rrl = pickle.load(filehandle)

	rng = int((len(rrl)-1)/2)

	for q in range(rng):
		err_rrl0[q,i] += rrl[q]

	path = os.getcwd()+"/data/r_dif/0.4/rrl/"
	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
		primal = pickle.load(filehandle)

	rng = int((len(primal)-1)/2)
	print(rng)
	for q in range(rng):
		# print(primal[q])
		# print(primal[q])
		err_rrl1[q,i] += primal[q]

	# path = os.getcwd()+"/data/r_dif/"+str(r_dif)+"/"+"primal/"
	path = os.getcwd()+"/data/r_dif/0.3/rrl/"
	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
		primal = pickle.load(filehandle)

	rng = int((len(primal)-1)/2)
	print(rng)
	for q in range(rng):
		# print(primal[q])
		# print(primal[q])
		err_rrl2[q,i] += primal[q]

	path = os.getcwd()+"/data/r_dif/0.2/rrl/"
	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
		primal = pickle.load(filehandle)

	rng = int((len(primal)-1)/2)
	print(rng)
	for q in range(rng):
		# print(primal[q])
		# print(primal[q])
		err_rrl3[q,i] += primal[q]

	path = os.getcwd()+"/data/r_dif/0.1/rrl/"
	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
		primal = pickle.load(filehandle)

	rng = int((len(primal)-1)/2)
	print(rng)
	for q in range(rng):
		# print(primal[q])
		# print(primal[q])
		err_rrl4[q,i] += primal[q]

	path = os.getcwd()+"/data/r_dif/0.0/rrl/"
	with open(path+'run_{}.pkl'.format(i), 'rb') as filehandle:
		primal = pickle.load(filehandle)

	rng = int((len(primal)-1)/2)
	print(rng)
	for q in range(rng):
		# print(primal[q])
		# print(primal[q])
		err_rrl5[q,i] += primal[q]
# print(max_dem)

x_m = range(0, 99)
y_rrl0 = np.nanmean(err_rrl0, axis=1)
e_rrl0 = np.nanstd(err_rrl0, axis=1)
y_rrl1 = np.nanmean(err_rrl1, axis=1)
e_rrl1 = np.nanstd(err_rrl1, axis=1)
y_rrl2 = np.nanmean(err_rrl2, axis=1)
e_rrl2 = np.nanstd(err_rrl2, axis=1)
y_rrl3 = np.nanmean(err_rrl3, axis=1)
e_rrl3 = np.nanstd(err_rrl3, axis=1)
y_rrl4 = np.nanmean(err_rrl4, axis=1)
e_rrl4 = np.nanstd(err_rrl4, axis=1)
y_rrl5 = np.nanmean(err_rrl5, axis=1)
e_rrl5 = np.nanstd(err_rrl5, axis=1)
# print(y_primal)
f, ax = plt.subplots(1)
# ax.plot(x_m, y_m, 'or')
ax.plot(x_m, y_rrl0, '-', label='0.5')
ax.plot(x_m, y_rrl1, '-', label='0.4')
ax.plot(x_m, y_rrl2, '-', label='0.3')
ax.plot(x_m, y_rrl3, '-', label='0.2')
ax.plot(x_m, y_rrl4, '-', label='0.1')
ax.plot(x_m, y_rrl5, '-', label='0.0')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
ax.fill_between(x_m, y_rrl0 - e_rrl0, y_rrl0 + e_rrl0,alpha=0.05)
ax.fill_between(x_m, y_rrl1 - e_rrl1, y_rrl1 + e_rrl1,alpha=0.05)
ax.fill_between(x_m, y_rrl2 - e_rrl2, y_rrl2 + e_rrl2,alpha=0.05)
ax.fill_between(x_m, y_rrl3 - e_rrl3, y_rrl3 + e_rrl3,alpha=0.05)
ax.fill_between(x_m, y_rrl4 - e_rrl4, y_rrl4 + e_rrl4,alpha=0.05)
ax.fill_between(x_m, y_rrl5 - e_rrl5, y_rrl5 + e_rrl5,alpha=0.05)
ax.set_ylim(ymin=0)
ax.set_xlim([0,98])
ax.margins(x=0)
plt.title(f"Effects of precision on demonstration number")
plt.xlabel("Demonstration Number")
plt.ylabel("RMSE")
plt.show()


pol_from_reward = find_policy(gw.n_states, gw.n_actions, gw.transition_probability,
								rrl[-1], 0.95, stochastic=True)

v_demo = policy_eval(pol_from_reward, rrl[-1], gw.transition_probability, 
								gw.n_states, gw.n_actions, discount_factor=0.95, theta=0.001)

evd.append(np.square(v_true - v_demo))	
	
	# for j in range(len(listed)):
	# 	print(j+2, listed[j])

	# print(listed[-1].reshape((5,5)))

# print((avg_rew/200).reshape((5,5)))
# # exit()
# pol_from_avg_reward = find_policy(gw.n_states, gw.n_actions, gw.transition_probability,
# 								avg_rew/200, 0.95, stochastic=True)

# avg_demo = policy_eval(pol_from_reward, avg_rew/200, gw.transition_probability, 
# 								gw.n_states, gw.n_actions, discount_factor=0.95, theta=0.001) 
# style = {
# 	'border': {'color': 'red', 'linewidth': 0.5},
# }

print(successes/3)
print(np.mean(num_eps))
print(np.std(num_eps))

print(np.mean(evd))
print(np.std(evd))

# ax = plt.figure(num='Ground Truth').add_subplot(111)
# plt.title("Ground Truth")
# P.plot_state_values(ax, gw, v_true, **style)
# plt.draw()

# ax = plt.figure(num='RRL').add_subplot(111)
# plt.title("RRL")
# P.plot_state_values(ax, gw, avg_demo, **style)
# plt.draw()

# plt.show()