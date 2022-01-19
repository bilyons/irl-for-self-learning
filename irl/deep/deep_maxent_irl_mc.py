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
from torch.optim import RMSprop
from torch.optim import Adam

from irl.standard.irl_set_traj_length import compute_state_visitation_frequency

# !! FIx all vriables from np.array() to torch.tensor()


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

def find_feature_expectations(feature_matrix, trajectories):
    n_states, n_features = feature_matrix.shape
    fe = torch.zeros(n_features)
    print(fe.shape)
    print(feature_matrix.shape)
    print(trajectories.shape)

    for t in trajectories:            #trajectories shape--> (T,L,2)
        for step in t:  #.states():
            print(step)
            fe += feature_matrix[int(step[0]),:]
    return fe/len(trajectories)


def inital_probabilities(world, trajectories):
    # Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
    # And returns initial probability and count
    n_states, n_features = world.features.shape
    p = torch.zeros(n_states)

    for t in trajectories:
        p[t[0][0]] += 1.0
    return p/len(trajectories)

# trajectory should be (T,L,2)--> (sample_number, trajectory_length, state_sequence, action_sequence)
def demo_svf(trajs, n_states):

    p = torch.zeros(n_states)
    
    for traj in trajs:
        for s,a in traj:
            p[s] += 1

    p = p/len(trajs)
    return p

class DeepMEIRL(nn.Module):
    """
    Here, the network and loss backpropagate design is got from:
    Network last output should be 1 element, but consider the N_STATES as batchsize and then resize the tensor:
    https://github.com/neka-nat/inv_rl/blob/master/max_ent_deep_irl.py#L47
    And some inspirations get from my another RL project in using self.optimiser = RMSprop()
    # Normal L2 loss, take mean over actual data//
    and also the loss function can be masked or padded for non-fixed trajectory length:
    https://github.com/jerryzhao173985/Competition_3v3snakes/blob/master/pymarl/src/learners/q_learner.py#L88-L103
    Useful for future adaptation for this code!

    """

    def __init__(self, feature_space, hidden_space, lr=0.001):

        super(DeepMEIRL, self).__init__()

        self.feature_space = feature_space
        # self.reward_space = reward_space
        self.hidden_space = hidden_space
        self.fc1 = nn.Linear(self.feature_space, self.hidden_space)     # HIDDEN_SPACE = 64
        self.fc2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.fc3 = nn.Linear(self.hidden_space, 1)
        # Remember to add in convolutions later maybe Billy? if relevant
        
        
        self.params = list(self.parameters())
        optim_alpha = 0.99 # RMSProp alpha
        optim_eps = 0.00001 # RMSProp epsilon
        # self.optimiser = RMSprop(params=self.params, lr=lr, alpha=optim_alpha, eps=optim_eps)
        epsilon=1e-8
        self.optimiser = Adam(params=self.params, lr=lr, betas=(0.9, 0.999), eps=epsilon)
        # Good default settings : learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8

    def forward(self, x):

        output = F.relu(self.fc1(x))
        # print("============Network======out1: ", output)
        output = F.relu(self.fc2(output))
        # print("============Network======out2: ", output)
        output = self.fc3(output)  # Notice: here only has 1 output, you cannot do RELU anymore!!
        # print("============Network======out3: ", output)
        return output # Probably need to bring the normalizing function over to here



# FIXME: haven't debugged yet
def value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True):
    
    P_a = P_a.permute(0,2,1)                  # original transition_prob table[(s_from, a, s_to)]  ##original: tp  = np.ones((n_states, n_actions, n_states))
    N_STATES, _, N_ACTIONS = P_a.size()     

    values = torch.zeros((N_STATES,))
    # print(values.shape)
    # print(rewards.shape)
    # print("NUM: ", N_STATES, N_ACTIONS)

    # estimate values
    k=0
    while k<2:
        k+=1
        print(k)
        
        values_tmp = values.clone().detach()         # Returns a copy of input.

        for s in range(N_STATES):
            # print(s)
            v_s = []
            values[s] = max([sum([P_a[s, s1, a]*(rewards[s] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

        if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
            break


    if deterministic:
    # generate deterministic policy
        policy = torch.zeros([N_STATES],  dtype = torch.float32)
        for s in range(N_STATES):
            
            policy[s] = max([sum([P_a[s, s1, a]*(rewards[s]+gamma*values[s1]) 
                                        for s1 in range(N_STATES)]) 
                                        for a in range(N_ACTIONS)])

        return values, policy

    else:
        # generate stochastic policy
        policy = torch.zeros([N_STATES, N_ACTIONS], dtype = torch.float32)
        for s in range(N_STATES):
            v_s = torch.tensor([sum([P_a[s, s1, a]*(rewards[s] + gamma*values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
            policy[s,:] = np.transpose(v_s/np.sum(v_s))
        return values, policy



def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = torch.zeros(n_states, dtype = torch.float32)

    # reward = torch.from_numpy(reward)
    # transition_probabilities = torch.from_numpy(transition_probabilities)
    reward = torch.tensor(reward, dtype = torch.float32)
    transition_probabilities = torch.tensor(transition_probabilities, dtype = torch.float32)
    


    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = torch.tensor([float("-inf")])
            # print(max_v.shape, max_v)

            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                # print(torch.dot(tp, reward + discount*v))
                max_v = torch.max(max_v, torch.dot(tp, reward + discount*v))

            new_diff = torch.abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v


def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """
    Find the optimal policy.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = torch.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1)[0].reshape((n_states, 1))  # For numerical stability.
        Q = torch.exp(Q)/torch.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = torch.tensor([_policy(s) for s in range(n_states)])
    return policy


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
    policy = find_policy(n_states, n_actions,
                        transition_probability, r, discount)

    start_state_count = torch.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[int(trajectory[0, 0])] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = torch.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1].clone() *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)


def compute_feature_matrix(n_states):
    return torch.eye(n_states)


def deep_maxent_irl(env, gamma, trajs, n_iters, lr, n_states = 285, n_actions = 3):

    tp  = np.ones((n_states, n_actions, n_states))
    P_a = torch.tensor(tp, dtype = torch.float32)

    N_STATES,N_ACTIONS, _  = P_a.shape
    
    # feature_matrix = compute_feature_matrix(N_STATES)
    # feature_matrix = torch.tensor(feature_matrix, dtype = torch.float32, requires_grad = True)
    # print("feature_matrix: ", feature_matrix)
    feature_matrix = torch.eye(N_STATES, 10)

    # print(trajs.shape)     # (20, 15, 2)
    
    """
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

    OUTPUT: rewards     Nx1 vector - recoverred state rewards
    
    """

    
    
    # N_FEATURES = 2
    # feat_map = torch.tensor(torch.ones(N_STATES, N_FEATURES))
    # feat_map = torch.tensor(env.features)
    # feat_map = torch.ones(env.transition_prob.shape[0])

    # feat_map = torch.ones((N_STATES,))

    P_a = torch.tensor(tp, dtype = torch.float32)

    torch.manual_seed(0)

    N_STATES,N_ACTIONS, _ = P_a.shape

    N_STATES1, N_FEATURES = feature_matrix.shape
    assert(N_STATES1==N_STATES)
    feature_exp = find_feature_expectations(feature_matrix, trajs)
    
    print("feature_exp", feature_exp)

    # N_FEATURES = feat_map.shape[0]

    # init nn model
    N_HIDDEN = 2
    nn_r = DeepMEIRL(N_FEATURES, N_HIDDEN)
    # network output is of dimension (N_STATES, 1)
    

    # find state visitation frequencies using demonstrations
    # mu_D = demo_svf(trajs, N_STATES)
    #!!! Noitice: not using demo_svf here, but instead use feature_exp in deep IRL

    # training 
    for iteration in range(n_iters):
        # if iteration % (n_iters/10) == 0:
        print("iteration: {}".format(iteration))
        
        # compute the reward matrix
# """feat_map[0].flatten()"""
        rewards = nn_r.forward(feature_matrix).squeeze()

        # print(torch.ones((N_STATES,)), torch.ones((N_STATES,)).size())
        print("rewards: ", rewards.shape, rewards)
        
        # compute policy 
        _, policy = value_iteration(P_a, rewards, gamma, error=0.1, deterministic=True)
        print("policy", policy)
        
        # compute expected svf
        # mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
        # mu_exp = find_expected_svf(P_a, gamma, trajs, policy, deterministic=True)
        exp_svf = find_expected_svf(N_STATES, rewards, N_ACTIONS, gamma,
                      P_a, trajs)
        
        print("exp_svf", exp_svf)
        
        # compute gradients on rewards:
        # grad_r = mu_D - mu_exp
        # apply gradients to the neural network
        # grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)
        # --error--
        # error = mu_D - mu_exp
        error = feature_exp - exp_svf
        
        print("error", error)

        # Normal L2 loss, take mean over actual data
        loss = (error ** 2).sum() 
        print("loss: ", loss.item())

        # Optimise
        nn_r.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(nn_r.params, 10)
        nn_r.optimiser.step()


    rewards = nn_r.forward(feature_matrix).squeeze()  #get_rewards(feat_map)
    # rewards = torch.squeeze(rewards)
    # return sigmoid(normalize(rewards))
    rewards = rewards.detach().numpy()
    return normalize(rewards)               #.reshape((N_STATES,))

