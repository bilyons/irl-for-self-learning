"""
Implements deep maximum entropy inverse reinforcement leanring based on the work of
Ziebart et al. 2008 and Wulfmeier et al. 2015 using Pytorch, as a basis for extension
to the self-learning framework.

https://arxiv.org/pdf/1507.04888.pdf - Wulfmeier paper

B.I. Lyons, Jerry Zhao, 2015
Billy.Lyons@ed.ac.uk, X.Zhao-44@sms.ed.ac.uk
"""

from itertools import product

import numpy as np
import numpy.random as rn
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from irl.standard.irl_set_traj_length import compute_state_visitation_frequency


def normalize(vals):

	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)

# from irl.standard.irl_no_set_length import *
# Import functions: softmax, normalize, compute_state_visitation_frequnecy, irl, 
#                   find_feature_expectations, find_policy

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Alg1:

Initialise weights
for each weight update solve

r^n = nn_forward(f, theta^n)

Solve MDP
value approximation - alg 2 - Zeibart's Thesis
make policy - alg 3 - Zeibart's Thesis

Determine MaxEnt Loss and Grads

Backprop and Update

We will begin by assuming a set length of the trajectory
"""

# Functions for everyone ###############################################################
# From previous work

def find_feature_expectations(world, trajectories):
	n_states, n_features = world.features.shape
	fe = np.zeros(n_states)

	for t in trajectories:
		for s, a in t:  #.states():
			fe += world.features[s,:]
	return fe/len(trajectories)


def inital_probabilities(world, trajectories):
	# Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
	# And returns initial probability and count
	n_states, n_features = world.features.shape
	p = np.zeros(n_states)

	for t in trajectories:
		p[t[0][0]] += 1.0
	return p/len(trajectories)

# trajectory should be (T,L,2)--> (sample_number, trajectory_length, state_sequence, action_sequence)
def demo_svf(trajs, n_states):

	p = np.zeros(n_states)
	
	for traj in trajs:
		for s,a in traj:
			p[s] += 1

	p = p/len(trajs)
	return p

class DeepMEIRL(nn.Module):

	def __init__(self, feature_space, reward_space, lr=0.01,  hl1=3, hl2=3):

		super(DeepMEIRL, self).__init__()

		self.feature_space = feature_space
		self.reward_space = reward_space
		self.fc1 = nn.Linear(self.feature_space, 3)
		self.fc2 = nn.Linear(3, 3)
		self.fc3 = nn.Linear(3, self.reward_space)
		# Remember to add in convolutions later maybe Billy? if relevant

	def forward(self, feature_space):
		output = F.relu(self.fc1(feature_expectations))
		output = F.relu(self.fc2(output))
		output = F.relu(self.fc3(output))
		return output # Probably need to bring the normalizing function over to here



# FIXME: haven't debugged yet
def value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True):
 
	N_STATES, _, N_ACTIONS = np.shape(P_a)

	values = np.zeros([N_STATES])

	# estimate values
	while True:
		values_tmp = values.copy()

		for s in range(N_STATES):
			v_s = []
			values[s] = max([sum([P_a[s, s1, a]*(rewards[s] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

		if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
			break


	if deterministic:
	# generate deterministic policy
		policy = np.zeros([N_STATES])
		for s in range(N_STATES):
			policy[s] = np.argmax([sum([P_a[s, s1, a]*(rewards[s]+gamma*values[s1]) 
										for s1 in range(N_STATES)]) 
										for a in range(N_ACTIONS)])

		return values, policy

	else:
		# generate stochastic policy
		policy = np.zeros([N_STATES, N_ACTIONS])
		for s in range(N_STATES):
			v_s = np.array([sum([P_a[s, s1, a]*(rewards[s] + gamma*values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
			policy[s,:] = np.transpose(v_s/np.sum(v_s))
		return values, policy




def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
	"""
	feat_map    NxD matrix - the features for each state
	P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob
	gamma       float - RL discount factor
	trajs       a list of demonstrations
	lr          float - learning rate
	n_iters     int - number of optimization steps

	OUTPUT: rewards     Nx1 vector - recoverred state rewards
	
	"""

	torch.manual_seed(0)

	N_STATES, _, N_ACTIONS = np.shape(P_a)

	N_STATES1, N_FEATURES = feat_map.shape

	assert(N_STATES==N_STATES1)

	# init nn model
	nn_r = DeepMEIRL(N_FEATURES, N_STATES)

	# find state visitation frequencies using demonstrations
	mu_D = demo_svf(trajs, N_STATES)

	# training 
	for iteration in range(n_iters):
		if iteration % (n_iters/10) == 0:
			print("iteration: {}".format(iteration))
		
		# compute the reward matrix
		rewards = nn_r.forward()
		
		# compute policy 
		_, policy = value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
		
		# compute expected svf
		mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
		
		# compute gradients on rewards:
		grad_r = mu_D - mu_exp

		# apply gradients to the neural network
		grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)
		

	rewards = nn_r.forward()  #get_rewards(feat_map)
	# return sigmoid(normalize(rewards))
	return normalize(rewards)

