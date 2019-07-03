# for get n-uplet de theta
from itertools import combinations
import random
import environment as env
import numpy as np
import data

class MDP:
    # the belief
    state = None
    # discret theta possible
    theta_joint=[]
    # theta discrete(possible, finite) estimated of each action
    actions = None
    # number of arms tested
    K = None


    '''
    input:
        K : number of arms tested
        actions: list of names of actions
    '''
    def __init__(self, K, actions):
        # number of actions
        self.K = K
        # actions
        self.actions = actions
        # get all theta joint by <permutations>
        # which respect the contraint of theta_1 > theta_2 > ... > theta_K
        self.theta_joint = list(combinations(env.theta_possible, self.K))
        # initialize the belief by probibility joint uniform
        self.initialize_state()

    

    '''
    this function isn't used, perhaps il will be used in the future

    this function is in charge of getting theta possible that theta_i > theta_j, for i < j

    input:
        up: the max theta in theta possible
        down: the min theta in theta possible
        n: the number of theta possible

    output:
        list of theta possible that theta_i > theta_j, for i < j

    '''
    def get_theta_possible(self,up,down,n=14):
    
        interval = (up-down)/(n-2)
        theta_possible = []
        theta_possible.append(up+(1-up)/2)
        for i in range(n-2):
            theta_possible.append(up-i*interval)
        theta_possible.append(down)
        theta_possible.append(down/2)
        return theta_possible
        


    '''
    initialize the belief with uniform probability
    '''
    def initialize_state(self):
        self.state = {}
        for i in range(len(self.theta_joint)):
            self.state[self.theta_joint[i]] = 1/(len(self.theta_joint))
    


    '''
    calculate the new belief of theta for each action

    input:    
        a : action performed
        y_a : 1 or 0, the result return by client(simulation)

    output : the state(belief) of next time
        '''
    def get_next_state(self, a, y_a):
        
        # calculate probability of belief of theta for action a
        prob_Y_y_a = 0.0
        next_state = {}
        for joint in self.theta_joint:
            prob_Y_y_a += env.prob_bernouill(joint[a], y_a)*self.state[joint]

        # calculate the new probability
        for joint in self.theta_joint:

            theta = joint[a]
            p1 = env.prob_bernouill(theta, y_a)
            next_state[joint] = p1*self.state[joint]/prob_Y_y_a

        return next_state, prob_Y_y_a
    


    '''
    update current state
    '''
    def transition(self, next_state):
        self.state = next_state



    '''
    get action based on current state and weight of network by epsilon-greedy
    epsilon reduce by iteration increasing
    
    get action randomly with a probability of epsilon
    otherwize get action which can minimizer Q calculated by current network
    
    input: 
        dqn: the class dqn

    output : (a, self.actions[a]) where
        a: the index of arm tested
        self.actions[a] : the name of the arm
    '''
    def get_action(self, dqn):
        a = None
        # get all Q by the current network for each possible action
        Q = dqn.get_Q_for_all_actions(self.state)
        
        '''a = np.argmin(Q)
        '''
        #self.epsilon = 0
        if random.uniform(0,1)<env.epsilon:
            # get action randomly
            a = np.random.randint(0,self.K)
        else :
            # get action which can minimizer the Q
            a = np.argmin(Q)
        
        return a, self.actions[a]
    


    '''
    get the optimal interval where the optimal resource is

    input :
        dqn: the class dqn

    output:
        the name of current optimal resource found by Q-network,
        the name of left resource current optimal resource found by Q-network,
        the name of right resource current optimal resource found by Q-network,

    '''
    def get_optimal_interval(self, dqn):
        action_optimal_index = None
        # get all Q by the current network for each possible action
        Q = dqn.get_Q_for_all_actions(self.state)
        
        action_optimal_index = np.argmin(Q)
        
        left_action_index = 0 if action_optimal_index == 0 else action_optimal_index - 1
        right_action_index = self.K-1 if action_optimal_index == self.K-1 else action_optimal_index + 1 
        
        return self.actions[action_optimal_index],self.actions[left_action_index], self.actions[right_action_index]
    


    '''
    calculate the estimated theta of arm [a], in state input

    input:
        state: the belief
        a: index of arm

    output:
        the estimated theta

    '''
    def get_estimated_theta(self, state, a):
        prob_Y_y_a = 0.0
        y_a = 1
        for joint in self.theta_joint:
            prob_Y_y_a += env.prob_bernouill(joint[a], y_a)*state[joint]
            
        return prob_Y_y_a


    '''
    calculate the estimated expected cost of arm in state input

    input:
        state: the belief
        a: index of arm

    output:
        the estimated expected cost of arm a 

    '''
    def get_estimated_expected_cost(self, state, a):
        return data.ressource_cost(self.actions[a])+self.get_estimated_theta(state,a)*data.gamma
