import tensorflow as tf
import numpy as np
import environment as env

class DQN:
    # the input of the first layer of network
    x_input = None
    # the input of target of network
    target_input = None
    
    # all value of loss of each iteration(training)
    history_loss = None
    # loss function
    loss = None
    # learning rate 
    learning_rate = None
    # GradientDescentOptimizer of training
    train_op = None
    # the layer of prediction of network
    prediction = None
    
    # number of iteration
    iteration = None
        
    # session of network which manage the run of the network
    session = None
    
    def __init__(self,mdp,xsize,ysize):
        self.mdp = mdp
        # initialize the variables
        self.iteration = 0

        self.history_loss = []

        # build the network
        self.create_network(xsize, ysize)
    
    def create_network(self, x_size, y_size):
        # the input layer of data
        self.x_input = tf.placeholder(shape=[None,x_size], dtype=tf.float32)
        # the input layer of targer
        self.target_input = tf.placeholder(shape=[None,y_size], dtype=tf.float32)
        
        last_layer = self.x_input
        last_shape = x_size
        
        for nodes in env.layers_nodes:
            last_layer = self.add_layer(last_layer, last_shape, nodes, activation_function=tf.nn.relu)
            last_shape = nodes
        
        # the output layer 
        # the shape of output (None, 1)
        self.prediction = self.add_layer(last_layer, last_shape, y_size, activation_function=None)        
        
        # define the loss function
        # if input pieces of data, then take the mean
        self.loss = tf.reduce_mean(tf.square(self.target_input-self.prediction))
        # us GradientDescentOptimizer and minimizer the loss
        self.train_op = tf.train.GradientDescentOptimizer(env.learning_rate).minimize(self.loss)
        
        # initialize the weights randomly of the network 
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
    

    
    def get_Q(self, state, action):
        '''
        calculate all Q for state and arm

        input:
            state: the belief
            action: arm

        output:
            Q calculated by network

        '''
        return self.predict([state], [action])[0][0]
    

    def get_Q_for_all_actions(self, state):
        '''
        calculate all Q for each arm possible tested by state  input

        input:
            state: the belief

        output:
            list of Q calculated by network
        '''
        return [self.get_Q(state, a) for a in range(self.mdp.K)]
   
    def get_target(self, r, next_state):
        '''
        calculate the target by r + minimun Q of state_t+1
        
        r : current reword by simulating action
        next_state : the state of t+1 
        
        return : target
        '''
        Q = self.get_Q_for_all_actions(next_state)
        return env.beta * np.min(Q) + r
            
        
    def train(self,s,a,y):
        '''
        train the network
        
        s : current state
        a : action performed
        y : the target
        
        return : the loss of current train
        '''
        # iteration increases
        self.iteration += len(y)
        x = self.get_x(s,a)
        _,loss = self.session.run([self.train_op, self.loss], feed_dict={self.x_input:x,self.target_input:y})
        # save the loss of current train
        self.history_loss.append(loss)
        
        if self.iteration%40 == 0:
            env.decrease_epsilon()
        
        return loss
        
    
    def predict(self,s,a):
        '''
        get the output of network by input
        
        s : current stata
        a : action performed
        
        return : Q predicted by network
        '''
        # calculate the data can be inputed by state and action
        x = self.get_x(s,a)
        return self.session.run(self.prediction, feed_dict={self.x_input:x})
    
    def get_x(self,states,actions):
        '''
        get data can be input to the network
        '''
        res = []
        for state, action in zip(states,actions):
            res.append(np.append(self.F(state), self.G(action)))
        return np.array(res)
        
    def F(self,state):
        '''
        function maps state to vector
        '''
        return np.array([state[joint] for joint in self.mdp.theta_joint])
    
    def G(self,k):
        '''
        function maps state to vector
        '''
        r = np.zeros(self.mdp.K)
        r[k]=1
        return r
    
    
    '''
    create one layer of network
    
    input:
        inputs : the input of the layer
        in_size : input size
        out_size : output size (number of nodes)
        activation_function : activation function of the layer

    output:
        one layer of network
    '''
    def add_layer(self,inputs,in_size,out_size,activation_function=None):
        
        w = tf.Variable(tf.random_normal([in_size,out_size], seed=0))
        b = tf.Variable(tf.zeros([1,out_size])+0.1)
        f = tf.matmul(inputs,w) + b
        if activation_function is None:
            outputs = f
        else:
            outputs = activation_function(f)
        return outputs
