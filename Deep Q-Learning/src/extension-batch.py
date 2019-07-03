

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import sys
import random

import environment as env
import data
from mdp import MDP
from dqn import DQN

'''
extract [n] arms from [names] respecting that the interval between two arms is same
if length of [names] is less than 1.3*n, it will return [names]

input:
    names: arms origin
    n: number of arms that we want to selected

output:
    arms selected
'''
def split(names, n=10):
    N = len(names)
    if N <= n:
        return names
    else:
        selected = []
        # calculate the interval
        interval = (int)(N / (n-1))
        for i in range(n-1):
            selected.append(names[interval*i])
        selected.append(names[-1])
        return selected



'''
main algo

input:
    Test_N: number of epochs(number of times the algo will execute)
    T: Time(iteration)
    converge_number: converge is detected when algo choose [converge_number] same action continuous

output:
    (list_T_converge,list_cost_converge,list_cost,list_a_converge,list_action_opt) where
    list_T_converge:    list of Time when algo converged of each epoch
    list_cost_converge: list of cost when algo converged of each epoch
    list_cost:          list of cost then algo finishs iterations of each epoch
    list_a_converge:    list of resource that is optimal found by DQN(when converged)
    list_action_opt:    true optimal resource

'''
def dqn_extension_batch(resource, Test_N=20, T=3000, converge_number=20, BATCH_NUMBER=10):
    list_T_converge = []
    list_cost_converge = []
    list_a_converge = []
    list_cost = []
    list_action_found = []
    list_action_opt = []

    for g in range(Test_N):
            
        print('============== one epoch starts =================')
        
        # total arms 
        arms = resource
        # the number of arms I want to select to test
        N_arms_wanted = data.number_arms_tobe_tested_one_time
        # the arms selected that will be tested
        selected_arms_name = split(arms, N_arms_wanted)
        # number of arms selected
        N_arms = len(selected_arms_name)

        # inspectation of iterations that algo executes
        T_total = 0
        # inspectation of costs
        cost_total = 0

        # the iteratons in each test
        T_iteration = T
        # list of cost when converged in each test
        cost_converges = []
        # list of time when converged in each test
        T_converges = []
        
        while 1 :
            ################
            # FOR ONE TEST:#
            ################

            # new a MDP with number of arms selected, and arms selected
            mdp = MDP(N_arms, selected_arms_name)
            # the shape of x that will be input to Q-network
            x_size = len(mdp.theta_joint)+mdp.K
            # the shape of target that will be input to Q-network
            y_size = 1
            # new a Q-network
            dqn = DQN(mdp,x_size,y_size)

            # the cost in current test
            cost_itertion = 0
            # the time converged in each test, initialized by iterations, once algo converged, il will be assigned by new value
            T_converge = T_iteration
            # boolean variable which indique algo converged or not after iteration in current test
            converge = False
            # the cost when converged in current test
            cost_converge = 0
            # action that is just choosed by algo 
            a_continuous = 0
            # number of times that action is choosed by algo
            a_continuous_number = 0
            # action when converged
            a_converge = 0
            # for batch

            # initialize observations
            env.observations = []
            
            for i in range(int(T_iteration/env.period_extract_batch)):
                for j in range(env.period_extract_batch):

                    # total iteration
                    T_total += 1
                    # get action, and Q of the action
                    a_t, action = mdp.get_action(dqn)

                    # part test algo converge or not
                    if a_continuous == a_t:
                        a_continuous_number += 1
                        if a_continuous_number == converge_number:
                            converge = True
                            T_converge = i
                            cost_converge = cost_itertion
                            a_converge = action
                    else :
                        a_continuous = a_t
                        a_continuous_number = 1
                    
                    # simulate resource
                    y_a, true_cost = data.simulate(action)
                    # calculate next state
                    next_state, proba_y_a = mdp.get_next_state(a_t, y_a)
                    # calculate expected estimate cost
                    expected_cost_estimated = mdp.get_estimated_expected_cost(mdp.state, a_t)

                    observ = (mdp.state,a_t,expected_cost_estimated, next_state)
                    #print(next_state)
                    # store transition
                    env.observations.append(observ)
                    if len(env.observations) > env.observations_size:
                        env.observations = env.observations[1:]

                    mdp.transition(next_state)

                # sample minibatch
                batchs = random.choices(env.observations, k=BATCH_NUMBER)
                states = list(map(lambda x:x[0], batchs))
                actions = list(map(lambda x:x[1], batchs))
                rewards = list(map(lambda x:x[2], batchs))
                next_states = list(map(lambda x:x[3], batchs))


                # get targets
                targets = [[dqn.get_target(r,ns)] for r,ns in zip(rewards, next_states)]
                
                # train the network
                dqn.train(states, actions, targets)

            # update the varibles
            converge = False
            cost_total += cost_itertion
            T_converges.append(T_converge)
            cost_converges.append(cost_converge)
            
            # if this test is the final test then finish algo
            if len(arms) == N_arms:
                break

            # get optimal interval
            arm_opt_found,arm_left, arm_right = mdp.get_optimal_interval(dqn)
            
            # get new interval of resources where the optimal resource shoule be
            arms =data.actions[arm_left : arm_right+1]
            # select new arms that will be tested in next test
            selected_arms_name = split(arms, N_arms_wanted)
            # number of arms selected that will be tested in next test
            N_arms = len(selected_arms_name)
        

        #################
        # algo finished #
        #################
        print('============== one epoch finished =================')

        # evaluate for this epoch
        c_all = data.expected_costs()
        arm_optimal_true = np.argmin(c_all)
        print('opt arm found by DQN                : ', a_converge)
        print('opt arm true                        : ', arm_optimal_true)
        print('regret  / T (when converged)        : ', sum(cost_converges)/sum(T_converges)-min(c_all), '<', max(c_all)-min(c_all))
        print('regret all / T                      : ', cost_total/T_total-min(c_all), '<', max(c_all)-min(c_all))
        
        list_action_opt.append(arm_optimal_true)
        list_a_converge.append(a_converge)
        list_T_converge.append(sum(T_converges))
        list_cost_converge.append(sum(cost_converges)/sum(T_converges) - min(c_all))
        list_cost.append(cost_total/T_total - min(c_all))
                         
    return list_T_converge,list_cost_converge,list_cost,list_action_found,list_action_opt

if __name__ == '__main__':
    # if theta_true isn't defined, then use function get_paras to initialize the variables
    if not data.theta_true:
        data.c_a, data.gamma, data.theta_true = data.get_paras(data.K)
    
    # execute algo
    results = dqn_extension_batch(data.actions, env.epochs,env.T,env.converge_number,env.batchs)

    # save the results
    env.list_T_when_converged = results[0] 
    env.list_cost_when_converged = results[1] 
    env.list_cost_total = results[2] 
    env.list_action_optimal_found_by_DQN = results[3] 
    env.list_action_optimal =  results[4] 
            

    print('T converged               :', env.list_T_when_converged)
    print('cost/Time when converged  :', env.list_cost_when_converged)
    print('cost/Time                 :', env.list_cost_total)
    print('arm optimal found by DQN  :', env.list_action_optimal_found_by_DQN)
    print('arm optimal true          :', env.list_action_optimal)