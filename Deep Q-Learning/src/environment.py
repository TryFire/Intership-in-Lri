import random
import numpy as np

# varibales and functions


##################
## part Q-network
##################
# length of [layers_nodes] is the number of full connected layers of the network
# element in [layers_nodes] is the number of nodes in each layer
layers_nodes = [16,8]

# learning_rate for network
learning_rate = 0.0002

# factor discounted
beta = 0.9



##################
## epsilon-greedy
##################
# origin epsilon 
epsilon = 0.8

# the minimun epsilon
epsilon_min = 0.04

# pertage of spsilon reduced by each time
epsilon_reduce_percent = 0.05

'''
function will be called each time the network trained
to reduce spsilon
'''
def decrease_epsilon():
	global epsilon
	if epsilon > epsilon_min:
		epsilon -= epsilon*epsilon_reduce_percent


##################
## part MDP
##################

'''
caculate the probability of Bernouill

input: 
	theta: parameter of Bernouill
	y: 1 or 0

output: 
	the probability of being [y] in the case of [theta]
'''
def prob_bernouill(theta, y):
	return np.power(theta, y)*np.power((1-theta), (1-y))

# the theta possible in the belief for each arm selected
theta_possible = [0.98,0.95,0.9,0.85,0.8,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.3,0.2,0.1,0.01]


'''
a example for getting the theta possible

input:
	up: the max probability that occurs in theta possible
	down: the min probability that occurs in theta possible

output:
	theta possible
'''
def get_theta_possible(up,down,n=14):
    
	interval = (up-down)/(n-2)
	theta_possible = []
	theta_possible.append(up+(1-up)/2)
	for i in range(n-2):
	    theta_possible.append(up-i*interval)
	theta_possible.append(down)
	theta_possible.append(down/2)
	return theta_possible



##################
## part algorithm
##################
# il will execute [epochs] times for each algorithm
epochs = 1

# time(iteration)
T = 6000

# converge is detected when algo choose [converge_number] same action continuous
converge_number = 10


# after the algo finishs, the variables will be assigned.
# list of Time when algo converged of each epoch
list_T_when_converged = None
# list of cost when algo converged of each epoch
list_cost_when_converged = None
# list of cost then algo finishs iterations of each epoch
list_cost_total = None
# list of resource that is optimal found by DQN(when converged)
list_action_optimal_found_by_DQN = None
# true optimal resource
list_action_optimal = None


################
## part batch
################
# number of observations in batch
batchs = 10
# each [period_extract_batch] time, algo will extract [batchs] obvervations
# from [observations], then train the network
period_extract_batch = 10
# memory for saving <s_t, a_t, r_t. s_t+1>
observations = []
# size of memory
observations_size = 1000


