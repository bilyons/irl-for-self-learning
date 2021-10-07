"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product

import numpy as np
import numpy.random as rn

from . import value_iteration

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.optim import Adam

import time
from torch.utils.tensorboard import SummaryWriter


def normalize(vals):

	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("starting on ", device, ".......")


def irl(feature_matrix, n_actions, discount, transition_probability,
		trajectories, epochs, learning_rate):

	n_states, d_states = feature_matrix.shape

	# Initialise weights.
	alpha = rn.uniform(size=(d_states,))

	# Calculate the feature expectations \tilde{phi}.
	feature_expectations = find_feature_expectations(feature_matrix,
													 trajectories)

	# Gradient descent on alpha.
	for i in range(epochs):
		# print("i: {}".format(i))
		r = feature_matrix.dot(alpha)
		expected_svf = find_expected_svf(n_states, r, n_actions, discount,
										 transition_probability, trajectories)
		grad = feature_expectations - feature_matrix.T.dot(expected_svf)

		alpha += learning_rate * grad

	return feature_matrix.dot(alpha).reshape((n_states,))

def find_svf(n_states, trajectories):
	"""
	Find the state visitation frequency from trajectories.

	n_states: Number of states. int.
	trajectories: 3D array of state/action pairs. States are ints, actions
		are ints. NumPy array with shape (T, L, 2) where T is the number of
		trajectories and L is the trajectory length.
	-> State visitation frequencies vector with shape (N,).
	"""

	svf = np.zeros(n_states)

	for trajectory in trajectories:
		for state, _, _ in trajectory:
			svf[state] += 1

	svf /= trajectories.shape[0]

	return svf

def find_feature_expectations(feature_matrix, trajectories):
	"""
	Find the feature expectations for the given trajectories. This is the
	average path feature vector.

	feature_matrix: Matrix with the nth row representing the nth state. NumPy
		array with shape (N, D) where N is the number of states and D is the
		dimensionality of the state.
	trajectories: 3D array of state/action pairs. States are ints, actions
		are ints. NumPy array with shape (T, L, 2) where T is the number of
		trajectories and L is the trajectory length.
	-> Feature expectations vector with shape (D,).
	"""

	feature_expectations = np.zeros(feature_matrix.shape[1])

	for trajectory in trajectories:
		for state, _, _ in trajectory:
			feature_expectations += feature_matrix[state]

	feature_expectations /= trajectories.shape[0]

	return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
					  transition_probability, trajectories):
	"""
	Find the expected state visitation frequencies using algorithm 1 from
	Ziebart et al. 2008.

	n_states: Number of states N. int.
	alpha: Reward. NumPy array with shape (N,).
	n_actions: Number of actions A. int.
	discount: Discount factor of the MDP. float.
	transition_probability: NumPy array mapping (state_i, action, state_k) to
		the probability of transitioning from state_i to state_k under action.
		Shape (N, A, N).
	trajectories: 3D array of state/action pairs. States are ints, actions
		are ints. NumPy array with shape (T, L, 2) where T is the number of
		trajectories and L is the trajectory length.
	-> Expected state visitation frequencies vector with shape (N,).
	"""

	n_trajectories = trajectories.shape[0]
	trajectory_length = trajectories.shape[1]

	# policy = find_policy(n_states, r, n_actions, discount,
	#                                 transition_probability)
	policy = value_iteration.find_policy(n_states, n_actions,
										 transition_probability, r, discount)

	start_state_count = np.zeros(n_states)
	for trajectory in trajectories:
		start_state_count[trajectory[0, 0]] += 1
	p_start_state = start_state_count/n_trajectories

	expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
	for t in range(1, trajectory_length):
		expected_svf[:, t] = 0
		for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
			expected_svf[k, t] += (expected_svf[i, t-1] *
								  policy[i, j] * # Stochastic policy
								  transition_probability[i, j, k])

	return expected_svf.sum(axis=1)

def softmax(x1, x2):
	"""
	Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

	x1: float.
	x2: float.
	-> softmax(x1, x2)
	"""

	max_x = max(x1, x2)
	min_x = min(x1, x2)
	return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
						   transition_probability):
	"""
	Find a policy with linear value iteration. Based on the code accompanying
	the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

	n_states: Number of states N. int.
	r: Reward. NumPy array with shape (N,).
	n_actions: Number of actions A. int.
	discount: Discount factor of the MDP. float.
	transition_probability: NumPy array mapping (state_i, action, state_k) to
		the probability of transitioning from state_i to state_k under action.
		Shape (N, A, N).
	-> NumPy array of states and the probability of taking each action in that
		state, with shape (N, A).
	"""

	# V = value_iteration.value(n_states, transition_probability, r, discount)

	# NumPy's dot really dislikes using inf, so I'm making everything finite
	# using nan_to_num.
	V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

	diff = np.ones((n_states,))
	while (diff > 1e-4).all():  # Iterate until convergence.
		new_V = r.copy()
		for j in range(n_actions):
			for i in range(n_states):
				new_V[i] = softmax(new_V[i], r[i] + discount*
					np.sum(transition_probability[i, j, k] * V[k]
						   for k in range(n_states)))

		# # This seems to diverge, so we z-score it (engineering hack).
		new_V = (new_V - new_V.mean())/new_V.std()

		diff = abs(V - new_V)
		V = new_V

	# We really want Q, not V, so grab that using equation 9.2 from the thesis.
	Q = np.zeros((n_states, n_actions))
	for i in range(n_states):
		for j in range(n_actions):
			p = np.array([transition_probability[i, j, k]
						  for k in range(n_states)])
			Q[i, j] = p.dot(r + discount*V)

	# Softmax by row to interpret these values as probabilities.
	Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
	Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
	return Q

def expected_value_difference(n_states, n_actions, transition_probability,
	reward, discount, p_start_state, optimal_value, true_reward):
	"""
	Calculate the expected value difference, which is a proxy to how good a
	recovered reward function is.

	n_states: Number of states. int.
	n_actions: Number of actions. int.
	transition_probability: NumPy array mapping (state_i, action, state_k) to
		the probability of transitioning from state_i to state_k under action.
		Shape (N, A, N).
	reward: Reward vector mapping state int to reward. Shape (N,).
	discount: Discount factor. float.
	p_start_state: Probability vector with the ith component as the probability
		that the ith state is the start state. Shape (N,).
	optimal_value: Value vector for the ground reward with optimal policy.
		The ith component is the value of the ith state. Shape (N,).
	true_reward: True reward vector. Shape (N,).
	-> Expected value difference. float.
	"""

	policy = value_iteration.find_policy(n_states, n_actions,
		transition_probability, reward, discount)
	value = value_iteration.value(policy.argmax(axis=1), n_states,
		transition_probability, true_reward, discount)

	evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
	return evd





# ________________________________________________________________________________________
class DeepIRL(nn.Module):
	
	def __init__(self, feature_space, hidden_space, lr=0.001):

		super(DeepIRL, self).__init__()

		self.feature_space = feature_space
		# self.reward_space = reward_space
		self.hidden_space = hidden_space
		self.fc1 = nn.Linear(self.feature_space, self.hidden_space)     # HIDDEN_SPACE = 64
		self.fc2 = nn.Linear(self.hidden_space, self.hidden_space)
		self.fc3 = nn.Linear(self.hidden_space, 1)
		
		self.params = list(self.parameters())
		# optim_alpha = 0.99 # RMSProp alpha optim_eps = 0.00001 # RMSProp epsilon
		# self.optimiser = RMSprop(params=self.params, lr=lr, alpha=optim_alpha, eps=optim_eps)
		epsilon=1e-8
		self.optimiser = Adam(params=self.params, lr=lr, betas=(0.9, 0.999), eps=epsilon)
		
	def forward(self, x):

		output = F.relu(self.fc1(x))
		output = F.relu(self.fc2(output))
		output = self.fc3(output)  # Notice: here only has 1 output, you cannot do RELU anymore!!
		return output 

#  With convolutional layers to better the performance and add model complexity
class ConvIRL(nn.Module):
	
	def __init__(self, feature_space, hidden_space, lr=0.001):

		super(ConvIRL, self).__init__()

		self.feature_space = feature_space
		# self.reward_space = reward_space
		self.hidden_space = hidden_space

		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 2, kernel_size=5, stride=1),     #---- (100-5)+1 = 96
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))        #---- (96-2)/2 +1 = 96/2 = 48
		self.layer2 = nn.Sequential(
			nn.Conv2d(2, 4, kernel_size=5, stride=1),     #---- 48-5+1 = 44
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))        #---- (44-2)/2 +1 = 44/2 = 22
		self.layer3 = nn.Sequential(
			nn.Conv2d(4, 8, kernel_size=5, stride=1),     #---- 22-5+1 = 18
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))        #---- (18-2)/2 +1 = 18/2 = 9
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(9 * 9 * 8, 500)
		self.fc2 = nn.Linear(500, 100)
		self.fc3 = nn.Linear(100, 1)

		# self.fc1 = nn.Linear(self.feature_space, self.hidden_space)     # HIDDEN_SPACE = 64
		# self.fc2 = nn.Linear(self.hidden_space, self.hidden_space)
		# self.fc3 = nn.Linear(self.hidden_space, 1)
		
		self.params = list(self.parameters())
		# optim_alpha = 0.99 # RMSProp alpha optim_eps = 0.00001 # RMSProp epsilon
		# self.optimiser = RMSprop(params=self.params, lr=lr, alpha=optim_alpha, eps=optim_eps)
		epsilon=1e-8
		self.optimiser = Adam(params=self.params, lr=lr, betas=(0.9, 0.999), eps=epsilon)
		
	def forward(self, x):
		out = x.view(1,1, x.size(0), x.size(1))
		print(out.shape)
		out = self.layer1(out)
		print(out.shape)
		out = self.layer2(out)
		print(out.shape)
		out = self.layer3(out)
		print(out.shape)
		out = out.reshape(out.size(0), -1)
		print(out.shape)
		out = self.drop_out(out)
		print(out.shape)
		out = self.fc1(out)
		out = self.fc2(out)
		# out = self.fc3(out)
		print(out.shape)
		
		return out

		# out = F.relu(self.fc1(out))
		# out = F.relu(self.fc2(out))
		# out = self.fc3(out)  # Notice: here only has 1 output, you cannot do RELU anymore!!
		# return out 


def compute_feature_matrix(n_states):
	return torch.eye(n_states)


def deep_maxent_irl(feature_matrix, env, gamma, trajs, n_iters, lr):
	start_time = time.time()
	
	# feature_matrix, n_actions, discount, transition_probability,
	# 	trajectories, epochs, learning_rate
 
	P_a = env.transition_probability

	N_STATES,N_ACTIONS, _  = P_a.shape
	
	# FIXME: the following line shouldn't have! Should be a predifined featurre matrix with any dimensions.
	feature_matrix = np.eye(N_STATES)

	torch.manual_seed(0)

	N_STATES1, N_FEATURES = feature_matrix.shape
 
	feature_exp = find_feature_expectations(feature_matrix, trajs)

	tb = SummaryWriter()
	# N_FEATURES = feat_map.shape[0]
	N_HIDDEN = 128
	# nn_r = DeepIRL(N_FEATURES, N_HIDDEN, lr=lr).to(device)   # network output is of dimension (N_STATES, 1)
	nn_r = ConvIRL(N_FEATURES, N_HIDDEN, lr=lr).to(device)
 
	feature_matrix_copy = feature_matrix
	
	feature_matrix = torch.tensor(feature_matrix, dtype = torch.float).to(device)
	
	for iteration in range(n_iters):
	 
		rewards = nn_r.forward(feature_matrix).squeeze()
		print("rewards: ", rewards)
		
		# compute policy 
		# _, policy = value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
		
		# compute expected svf
		new_rewards = rewards.cpu().detach().clone().numpy()
		exp_svf = find_expected_svf(N_STATES, new_rewards, N_ACTIONS, gamma,
					  P_a, trajs)
		
		# compute gradients on rewards:
		# grad_r = mu_D - mu_exp
		# apply gradients to the neural network
		# grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)
		
		# grad_r = feature_exp - exp_svf
		grad_r = feature_exp - feature_matrix_copy.T.dot(exp_svf)

		print("iteration: {}".format(iteration), "grad: ", ((grad_r**2).sum()).item())
		tb.add_scalar("loss", ((grad_r**2).sum()).item(), iteration)

		# Optimise
		nn_r.optimiser.zero_grad()
		# rewards = torch.from_numpy(rewards).to(device)
		grad_r = torch.from_numpy(grad_r).type(torch.float).to(device)
		# grad_r.requires_grad = True
		
		# print(rewards.requires_grad)
  
		flat_rewards = rewards.view(-1,1)
		grad = -grad_r.view(-1,1)
		flat_rewards.backward(grad)
	 	
		# rewards.backward(grad_r)
		grad_norm = torch.nn.utils.clip_grad_norm_(nn_r.params, 10)
		nn_r.optimiser.step()

	rewards = nn_r.forward(feature_matrix).squeeze()  #get_rewards(feat_map)
	rewards = rewards.cpu().detach().numpy()
 
	end_time = time.time()
	print("Total time for training on ", device, ": ", end_time-start_time)
	tb.close()
 
	return normalize(rewards)               #.reshape((N_STATES,))

# With spec: main(5, 0.01, 20, 2000, 0.001), CPU: 52.92 sec.
# For object world: Total time for training on  cuda :  15.732850790023804
# Object world with larger (100) epochs: main(10, 0.95, 15, 2, 50, 500, 0.01)
# Cuda Time: 
