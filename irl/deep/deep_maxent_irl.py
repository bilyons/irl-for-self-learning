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

def feature_expectations(world, trajectories):
	n_states, n_features = world.features.shape

	fe = np.zeros(n_states)

	for t in trajectories:
		for s in t.states():
			fe += world.features[s,:]
	return fe/len(trajectories)

def inital_probabilities(world, trajectories):
	# Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
	# And returns initial probability and count
	n_states, n_features = world.features.shape
	p = np.zeros(n_states)

	for t in trajectories:
		p[t.transitions()[0][0]] += 1.0
	return p/len(trajectories)

class DeepMEIRL(nn.Module):

	def __init__(self, feature_space, hl1, hl2, reward_space, lr):

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
