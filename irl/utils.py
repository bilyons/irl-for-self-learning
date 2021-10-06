import numpy as np
import matplotlib.pyplot as plt

# To plot arrows from optimal policy
def optimal_policy_map(policy_map, opt_policy):
    # arrows = ['\u2191','\u2193','\u2190', '\u2192', ] #  up, down, left, right
    arrows = ['\u2192', '\u2191','\u2190', '\u2193' ]   #  Right, Up, Left, Down, self.actions = [(1, 0), (0, 1), (-1, 0), (0, -1)] 
    for state in range(100):
        policy_map[state] = arrows[int(opt_policy[state])]
    return(policy_map)


# plot tables of state values
def plot_table(ax, data, title = ""):
    data = data[::-1]
    the_table = ax.table(cellText= data, loc='center')
    the_table.scale(1, 2.22)
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
    ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    ax.set_title(title)
    return ax


# calculate accuracy
def calc_accuracy(Oagent,Oexpert,state_num):
    ms = 0
    Oagent = np.reshape(Oagent,state_num)
    Oexpert = np.reshape(Oexpert,state_num)
    for i in range(state_num):        
        if Oagent[i] == Oexpert[i]:
            ms = ms + 1
    accuracy = ms/state_num
    return accuracy

# --------not used in the scripts but generally useful functions------------
# 1. optimal value and optimal policy can be estimated together given reward
# Estimate optimal values and coresponding actions
def estimation(eps,delta,V,x,y,R,gamma,w,actions):
    pi = np.zeros((x,y))
    while delta > eps:
        delta = 0
        for i in range(x):
            for j in range(y):
                previous = V[i,j]
                action_lst = [update_value(action,i,j,x,y,R,V,gamma) for action in actions]
                V[i,j] = max(action_lst)
                pi[i,j] = np.argmax(action_lst)
                delta = max(delta,abs(previous-V[i,j]))
    return V,pi