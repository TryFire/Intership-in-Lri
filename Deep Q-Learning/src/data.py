import numpy as np
import random

##################
## part simulation 
## the variables should be defined for personalize 
##################
# number of ressources
K = 100
# index of [K] resources
actions = [i for i in range(K)]

# when execute algo, the number of actions want to be selected from all resources to be tested
number_arms_tobe_tested_one_time = 10

#  for the artificial data  :  theta true is a list of parameters of Bernoulli duch that theta_true[i] is 
#  the bernouilli parameter of arm i
# length of [theta_true] should be [K]
theta_true = None#[0.8834447552661182, 0.6058323343652732, 0.4963200902839761, 0.31970456684978665, 0.22868166574795704] 

# coefficient of cost resource
c_a = 10
# fee should paied by NO if QoS less than Threshold
gamma = 55


'''
algo will use this function to get the cost ressource

input:
	a : the number of ressources sellected

output: cost corresponding to use a ressource
'''
def ressource_cost(a):
	return c_a*(a+1)  #  this cost is an example for artificial data 


'''
output: artificial list of expected cost according to the number of ressources selected
'''
def expected_costs():
	return [ressource_cost(a) + theta_true[a]*gamma for a in actions]

'''
algo will call this function to get result(true cost, QoS less than Threshold or not) of simulation
you can modify this function to get 

this function is in charge of simulation
it will use [theta_true] and has a probability of theta_true[a] 
that NO should pay the fee [gamma]

input:
	a : the number of ressources sellected

output: 
	(y_a, r) where 
	y_a : 1 or 0
	r : the true cost 
'''
def simulate(a):
	y_a = None
    # get the real theta of action a
	theta = theta_true[a]
    
    # simulate
	if random.uniform(0,1) <= theta:
		y_a = 1
	else:
		y_a = 0
    # calculate the true cost of the choosed arm a
	r =  ressource_cost(a) + gamma*y_a
	return y_a, r

'''
an example of getting artificial data

when [theta_true] isn't defined, 
[theta_true],[c_a],[gamma] will be initialized by this function

input:
	K : number of ressources
	gamma : fee should paied by NO if QoS less than Threshold 
	min_index : the resource which has the lowest expected cost 

output: 
	(c, gamma, thetas) where 
	c : c_a, coefficient of cost resource
	gamma : fee should paied by NO if QoS less than Threshold
	theta: for the artificial data  :  theta true is a list of parameters of Bernoulli duch that theta_true[i] is 
#		   the bernouilli parameter of arm i
'''
def get_paras(K, gamma=10, min_index=20):
	alpha = 1/np.log(K+1)-0.00001
	thetas = [1-alpha*np.log(k+1) for k in range(K)]

    #gamma = 10
	c = alpha*gamma/min_index

	return c, gamma, thetas