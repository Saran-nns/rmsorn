# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import os
# from sorn.utils import Helper
from rmsorn.utils import Initializer
from configparser import ConfigParser
import random
import tqdm
import pickle

parser = ConfigParser()
cwd = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(cwd, 'configuration.ini')
parser.read(config_file)

class RMSorn(object):

    # RM-SORN network Initialization
    def __init__(self):
        pass

    """Get network variables from configuration file as class variables of SORN"""
    
    nu = int(parser.get('Network_Config', 'Nu'))  # Number of input units
    ne = int(parser.get('Network_Config', 'Ne'))  # Number of excitatory units
    ni = int(0.2 * ne)  # Number of inhibitory units in the network
    no = int(parser.get('Network_Config', 'No')) # Number of output layer units
    
    eta_stdp = float(parser.get('Network_Config', 'eta_stdp'))
    eta_inhib = float(parser.get('Network_Config', 'eta_inhib'))
    eta_ip = float(parser.get('Network_Config', 'eta_ip'))
    te_max = float(parser.get('Network_Config', 'te_max'))
    ti_max = float(parser.get('Network_Config', 'ti_max'))
    ti_min = float(parser.get('Network_Config', 'ti_min'))
    te_min = float(parser.get('Network_Config', 'te_min'))
    mu_ip = float(parser.get('Network_Config', 'mu_ip'))
    sigma_ip = float(parser.get('Network_Config', 'sigma_ip'))

    network_type_ee = str(parser.get('Network_Config','network_type_ee'))
    network_type_ei = str(parser.get('Network_Config','network_type_ei'))
    network_type_ie = str(parser.get('Network_Config','network_type_ie'))
    network_type_oe = str(parser.get('Network_Config','network_type_oe'))
    lambda_ee = int(parser.get('Network_Config','lambda_ee'))
    lambda_ei = int(parser.get('Network_Config','lambda_ei'))
    lambda_ie = int(parser.get('Network_Config','lambda_ie'))
    lambda_oe = int(parser.get('Network_Config','lambda_oe'))

    mr = float(parser.get('Network_Config', 'mr')) 
    mo = float(parser.get('Network_Config', 'mo')) 
    
    
    # Initialize weight matrices
    @staticmethod
    def initialize_weight_matrix(self, network_type, synaptic_connection, self_connection, lambd_w): 
        """
        Args:
        network_type(str) - Spare or Dense
        synaptic_connection(str) - EE,EI,IE: Note that Spare connection is defined only for EE connections
        self_connection(str) - True or False: i-->i ; Network is tested only using j-->i
        lambd_w(int) - Average number of incoming and outgoing connections per neuron
        
        Returns:
        weight_matrix(array) -  Array of connection strengths 
        """
        
        if (network_type == "Sparse") and (self_connection == "False"):

            """Generate weight matrix for E-E/ E-I connections with mean lamda incoming and outgiong connections per neuron"""
            
            weight_matrix = Initializer.generate_lambd_connections(synaptic_connection, RMSorn.ne, RMSorn.ni, lambd_w, lambd_std = 1)
        
        # Dense matrix for W_ie

        elif (network_type == 'Dense') and (self_connection == 'False'):

            # Gaussian distribution of weights
            # weight_matrix = np.random.randn(Sorn.ne, Sorn.ni) + 2 # Small random values from gaussian distribution
            # Centered around 1 
            # weight_matrix.reshape(Sorn.ne, Sorn.ni) 
            # weight_matrix *= 0.01 # Setting spectral radius 
            
            # Uniform distribution of weights
            weight_matrix = np.random.uniform(0.0,0.1,(RMSorn.ne, RMSorn.ni))
            weight_matrix.reshape((RMSorn.ne,RMSorn.ni))
            
        elif (network_type == 'Dense_output') and (self_connection == 'False'):

            # Gaussian distribution of weights
            # weight_matrix = np.random.randn(Sorn.ne, Sorn.ni) + 2 # Small random values from gaussian distribution
            # Centered around 1 
            # weight_matrix.reshape(Sorn.ne, Sorn.ni) 
            # weight_matrix *= 0.01 # Setting spectral radius
            
            # Uniform distribution of weights
            weight_matrix = np.random.uniform(0.0,0.1,(RMSorn.no, RMSorn.ne))
            weight_matrix.reshape((RMSorn.no,RMSorn.ne))

        return weight_matrix

    @staticmethod
    def initialize_threshold_matrix(self, te_min,te_max, ti_min,ti_max):
    
        # Initialize the threshold for excitatory and inhibitory neurons
        
        """Args:
            te_min(float) -- Min threshold value for excitatory units
            ti_min(float) -- Min threshold value for inhibitory units
            te_max(float) -- Max threshold value for excitatory units
            ti_max(float) -- Max threshold value for inhibitory units
        Returns:
            te(vector) -- Threshold values for excitatory units
            ti(vector) -- Threshold values for inhibitory units"""

        te = np.random.uniform(0., te_max, (RMSorn.ne, 1))
        ti = np.random.uniform(0., ti_max, (RMSorn.ni, 1))
        
        # For patter recognition task: Heavyside step function with fixed threshold
        to = 0.5

        return te, ti,to

    def initialize_activity_vector(self,ne, ni, no):
        
        # Initialize the activity vectors X and Y for excitatory and inhibitory neurons
        
        """Args:
            ne(int) -- Number of excitatory neurons
            ni(int) -- Number of inhibitory neurons
        Returns:
             x(array) -- Array of activity vectors of excitatory population
             y(array) -- Array of activity vectors of inhibitory population"""

        x = np.zeros((ne, 2))
        y = np.zeros((ni, 2))
        o = np.zeros((no, 2))
        return x, y, o
    
class Plasticity(RMSorn):
    """
    Instance of class Sorn. Inherits the variables and functions defined in class Sorn
    Encapsulates all plasticity mechanisms mentioned in the article """

    # Initialize the global variables for the class //Class attributes

    def __init__(self):
        
        super().__init__()
        self.nu = RMSorn.nu                  # Number of input units
        self.ne = RMSorn.ne                  # Number of excitatory units
        self.no = RMSorn.no
        self.eta_stdp = RMSorn.eta_stdp      # STDP plasticity Learning rate constant; SORN1 and SORN2
        self.eta_ip = RMSorn.eta_ip          # Intrinsic plasticity learning rate constant; SORN1 and SORN2
        self.eta_inhib = RMSorn.eta_inhib    # Intrinsic plasticity learning rate constant; SORN2 only
        self.h_ip = 2 * RMSorn.nu / RMSorn.ne  # Target firing rate
        self.mu_ip = RMSorn.mu_ip            # Mean target firing rate  
        self.ni = RMSorn.ni                  # Number of inhibitory units in the network
        self.time_steps = RMSorn.time_steps  # Total time steps of simulation
        self.te_min = RMSorn.te_min          # Excitatory minimum Threshold
        self.te_max = RMSorn.te_max          # Excitatory maximum Threshold
        

    def stdp(self, wee, x, mr, cutoff_weights):
        
        """ Apply STDP rule : Regulates synaptic strength between the pre(Xj) and post(Xi) synaptic neurons"""

        x = np.asarray(x)
        xt_1 = x[:,0]
        xt = x[:,1]
        wee_t = wee.copy()
        
        # STDP applies only on the neurons which are connected.
        
        for i in range(len(wee_t[0])): # Each neuron i, Post-synaptic neuron
            
            for j in range(len(wee_t[0:])): # Incoming connection from jth pre-synaptic neuron to ith neuron
                
                if wee_t[j][i] != 0. : # Check connectivity
                    
                    # Get the change in weight
                    delta_wee_t = mr*self.eta_stdp * (xt[i] * xt_1[j] - xt_1[i]*xt[j])

                    # Update the weight between jth neuron to i ""Different from notation in article 

                    wee_t[j][i] = wee[j][i] + delta_wee_t
        
        """ Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight"""
        wee_t = Initializer.prune_small_weights(wee_t,cutoff_weights[0])
        
        """Check and set all weights < upper cutoff weight """
        wee_t = Initializer.set_max_cutoff_weight(wee_t,cutoff_weights[1])

        return wee_t
    
    def ostdp(self,woe, x, mo):
        
        """ Apply STDP rule : Regulates synaptic strength between the pre(Xj) and post(Xi) synaptic neurons"""
        x = np.asarray(x)
        xt_1 = x[:, 0]
        xt = x[:, 1]
        woe_t = woe.copy()
        # STDP applies only on the neurons which are connected.
        for i in range(len(woe_t[0])):  # Each neuron i, Post-synaptic neuron
            for j in range(len(woe_t[0:])):  # Incoming connection from jth pre-synaptic neuron to ith neuron
                if woe_t[j][i] != 0.:  # Check connectivity
                    # Get the change in weight
                    delta_woe_t = mo*self.eta_stdp * (xt[i] * xt_1[j] - xt_1[i] * xt[j])
                    # Update the weight between jth neuron to i ""Different from notation in article
                    woe_t[j][i] = woe[j][i] + delta_woe_t
        return woe_t
    
    def ip(self, te, x):
        
        # IP rule: Active unit increases its threshold and inactive decreases its threshold.

        xt = x[:, 1]

        te_update = te + self.eta_ip * (xt.reshape(self.ne, 1) - self.h_ip)
        
        """ Check whether all te are in range [0.0,1.0] and update acordingly"""
        
        # Update te < 0.0 ---> 0.0
        # te_update = prune_small_weights(te_update,self.te_min)
        
        # Set all te > 1.0 --> 1.0
        # te_update = set_max_cutoff_weight(te_update,self.te_max)

        return te_update
    
    def ss(self, wee_t):
        """Synaptic Scaling or Synaptic Normalization"""
        wee_t = wee_t / np.sum(wee_t,axis=0)
        return wee_t
    
    @staticmethod
    def modulation_factor(reward_history, current_reward ,window_sizes):
        """ Grid search for Modulation factor. Returns the maximum moving average over history of rewards with corresponding window
        Args:
            reward_history (list): List with the history of rewards 
            window_sizes (list): List of window sizes for gridsearch
        Returns:
            [int]: Modulation factor
        """
        
        def running_mean(x, K):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[K:] - cumsum[:-K]) / float(K)

        reward_avgs = [] # Holds the mean of all rolling averages for each window
        for window_size in window_sizes:
            if window_size<=len(reward_history):
                reward_avgs.append(np.mean(running_mean(reward_history, window_size)))
        
        best_reward= np.max(reward_avgs) 
        best_reward_window = window_sizes[np.argmax(best_reward)]
        print("current_reward %s | Best rolling avergage reward %s | Best Rolling average window %s"%(current_reward, best_reward, best_reward_window  ))
        mo = current_reward - best_reward
        mr = mo.copy()
        # TODO: What if mo != mr ?
        return mo, mr, best_reward, best_reward_window
            

    @staticmethod
    def initialize_plasticity():
        
        """NOTE: DO NOT TRANSPOSE THE WEIGHT MATRIX WEI FOR SORN 2 MODEL"""
        # Create and initialize sorn object and variables
        rmsorn_init = RMSorn()
        WEE_init = rmsorn_init.initialize_weight_matrix(network_type="Sparse", synaptic_connection='EE',
                                                      self_connection='False',
                                                      lambd_w=20)
        WEI_init = rmsorn_init.initialize_weight_matrix(network_type="Dense", synaptic_connection='EI',
                                                      self_connection='False',
                                                      lambd_w=100)
        WIE_init = rmsorn_init.initialize_weight_matrix(network_type="Dense", synaptic_connection='IE',
                                                      self_connection='False',
                                                      lambd_w=100)
        WOE_init = rmsorn_init.initialize_weight_matrix(network_type="Dense_output", synaptic_connection='OE',
                                                      self_connection='False',
                                                      lambd_w=100)

        Wee_init = Initializer.zero_sum_incoming_check(WEE_init)
        Wei_init = Initializer.zero_sum_incoming_check(WEI_init.T)  # For SORN 1
#         Wei_init = Initializer.zero_sum_incoming_check(WEI_init)
        Wie_init = Initializer.zero_sum_incoming_check(WIE_init)
        Woe_init = Initializer.zero_sum_incoming_check(WOE_init.T)
        print('Network Initialized')
        print('Number of connections in Wee %s , Wei %s, Wie %s Woe %s' %(np.count_nonzero(Wee_init), np.count_nonzero(Wei_init),
                                                                           np.count_nonzero(Wie_init), np.count_nonzero(Woe_init)))
        print('Shapes Wee %s Wei %s Wie %s Woe %s' % (Wee_init.shape, Wei_init.shape, Wie_init.shape, Woe_init.shape))

        # Normalize the incoming weights
        Wee_init = Initializer.normalize_weight_matrix(Wee_init)
        Wei_init = Initializer.normalize_weight_matrix(Wei_init)
        Wie_init = Initializer.normalize_weight_matrix(Wie_init)

        te_init, ti_init, to_init = sorn_init.initialize_threshold_matrix(RMSorn.te_min, RMSorn.te_max, RMSorn.ti_min, RMSorn.ti_max)
        x_init, y_init, o_init = sorn_init.initialize_activity_vector(RMSorn.ne, RMSorn.ni,RMSorn.no)
        
        return Wee_init, Wei_init, Wie_init,Woe_init, te_init, ti_init, to_init,x_init, y_init, o_init

    @staticmethod
    def reorganize_network():
        return NotImplementedError

class MatrixCollection(RMSorn):
    def __init__(self,phase, matrices = None):
        super().__init__()
        
        self.phase = phase
        self.matrices = matrices
        if self.phase == 'Plasticity' and self.matrices == None :

            self.time_steps = RMSorn.time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie,self.Woe, self.Te, self.Ti, self.To, self.X, self.Y, self.O = [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps
            
            wee, wei, wie, woe, te, ti, to, x, y, o = Plasticity.initialize_plasticity()

            # Assign initial matrix to the master matrices
            self.Wee[0] = wee
            self.Wei[0] = wei
            self.Wie[0] = wie
            self.Woe[0] = woe
            self.Te[0] = te
            self.Ti[0] = ti
            self.To[0] = to
            self.X[0] = x
            self.Y[0] = y
            self.O[0] = o

        elif self.phase == 'Plasticity' and self.matrices != None:
            
            self.time_steps = RMSorn.time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie,self.Woe, self.Te, self.Ti,self.To, self.X, self.Y,self.O = [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps
            # Assign matrices from plasticity phase to the new master matrices for training phase
            self.Wee[0] = matrices['Wee']
            self.Wei[0] = matrices['Wei']
            self.Wie[0] = matrices['Wie']
            self.Woe[0] = matrices['Woe']
            self.Te[0] = matrices['Te']
            self.Ti[0] = matrices['Ti']
            self.To[0] = matrices['To']
            self.X[0] = matrices['X']
            self.Y[0] = matrices['Y']
            self.O[0] = matrices['O']
            
        elif self.phase == 'Training':

            """NOTE:
            time_steps here is diferent for plasticity or trianing phase"""
            self.time_steps = RMSorn.time_steps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie,self.Woe, self.Te, self.Ti,self.To, self.X, self.Y,self.O = [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                                                             [0] * self.time_steps
            # Assign matrices from plasticity phase to new respective matrices for training phase
            self.Wee[0] = matrices['Wee']
            self.Wei[0] = matrices['Wei']
            self.Wie[0] = matrices['Wie']
            self.Woe[0] = matrices['Woe']
            self.Te[0] = matrices['Te']
            self.Ti[0] = matrices['Ti']
            self.To[0] = matrices['To']
            self.X[0] = matrices['X']
            self.Y[0] = matrices['Y']
            self.O[0] = matrices['O']

    # @staticmethod
    def weight_matrix(self, wee, wei, wie, woe, i):
        # Get delta_weight from Plasticity.stdp 
        # i - training step
        self.Wee[i + 1] = wee
        self.Wei[i + 1] = wei
        self.Wie[i + 1] = wie
        self.Woe[i + 1] = woe
        return self.Wee, self.Wei, self.Wie, self.Woe

    # @staticmethod
    def threshold_matrix(self, te, ti,to, i):
        self.Te[i + 1] = te
        self.Ti[i + 1] = ti
        self.To[i + 1] = to
        return self.Te, self.Ti, self.To

    # @staticmethod
    def network_activity_t(self, excitatory_net, inhibitory_net, output_net, i):
        self.X[i + 1] = excitatory_net
        self.Y[i + 1] = inhibitory_net
        self.O[i + 1] = output_net
        return self.X, self.Y, self.O

    # @staticmethod
    def network_activity_t_1(self, x, y,o, i):
        x_1, y_1, o_1 = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps
        x_1[i] = x
        y_1[i] = y
        o_1[i] = o

        return x_1, y_1, o_1

class NetworkState(Plasticity):
    
    """The evolution of network states"""

    def __init__(self, v_t):
        super().__init__()
        self.v_t = v_t
    
    def incoming_drive(self,weights,activity_vector):
            
        # Broadcasting weight*acivity vectors 
        
        incoming = weights* activity_vector
        incoming = np.array(incoming.sum(axis=0))
        return incoming
        
    def excitatory_network_state(self, wee, wei, te, x, y,white_noise_e):
        
        """ Activity of Excitatory neurons in the network"""
        xt = x[:, 1]  
        xt = xt.reshape(self.ne, 1)
        yt = y[:, 1]
        yt = yt.reshape(self.ni, 1)

        incoming_drive_e = np.expand_dims(self.incoming_drive(weights = wee,activity_vector=xt),1)
        incoming_drive_i = np.expand_dims(self.incoming_drive(weights = wei,activity_vector=yt),1)
        
        if self.v_t.shape[0] < self.ne:
            
            inp = [0]*self.ne
            inp[:len(self.v_t)] = self.v_t
            self.v_t = inp.copy()
            
        tot_incoming_drive = incoming_drive_e -  incoming_drive_i + white_noise_e + np.expand_dims(np.asarray(self.v_t),1) - te
        
        """Heaviside step function"""

        """Implement Heaviside step function"""
        heaviside_step = np.expand_dims([0.] * len(tot_incoming_drive),1)
        heaviside_step[tot_incoming_drive > 0] = 1.

        xt_next = np.asarray(heaviside_step.copy())

        return xt_next

    def inhibitory_network_state(self, wie, ti, x,white_noise_i):
    
        # Activity of inhibitory neurons
        wie = np.asarray(wie)
        xt = x[:, 1]
        xt = xt.reshape(RMSorn.ne, 1)    
        incoming_drive_e = np.expand_dims(self.incoming_drive(weights = wie, activity_vector=xt),1) 
        tot_incoming_drive = incoming_drive_e + white_noise_i - ti

        """Implement Heaviside step function"""
        heaviside_step = np.expand_dims([0.] * len(tot_incoming_drive),1)
        heaviside_step[tot_incoming_drive > 0] = 1.

        yt_next = np.asarray(heaviside_step.copy())  

        return yt_next


    def recurrent_drive(self, wee, wei, te, x, y,white_noise_e):
        
        """Network state due to recurrent drive received by the each unit at time t+1"""
        
    
        xt = x[:, 1]  
        xt = xt.reshape(self.ne, 1)
        yt = y[:, 1]
        yt = yt.reshape(self.ni, 1)
        
        incoming_drive_e = np.expand_dims(self.incoming_drive(weights = wee,activity_vector=xt),1)
        incoming_drive_i = np.expand_dims(self.incoming_drive(weights = wei,activity_vector=yt),1)
        
        tot_incoming_drive = incoming_drive_e -  incoming_drive_i + white_noise_e - te
 
        """Implement Heaviside step function"""
        heaviside_step = np.expand_dims([0.] * len(tot_incoming_drive),1)
        heaviside_step[tot_incoming_drive > 0] = 1.

        xt_next = np.asarray(heaviside_step.copy())

        return xt_next
    
    def output_network_state(self,woe, to, x):
        """ Output layer states
        Args:
            woe (array): Connection weights between Reurrent network and Output layer
            to (array): Threshold of Ouput layer neurons
            x (array): Excitatory recurrent network states
        """
        woe = np.asarray(woe)
        xt = x[:, 1]
        xt = xt.reshape(RMSorn.ne, 1)
        
        incoming_drive_o = np.expand_dims(self.incoming_drive(weights=woe, activity_vector=xt), 1)
        tot_incoming_drive = incoming_drive_o - to
        
        # TODO: If output neuron is 1, the use Heavyside step function
        if type(to) == list:
        
            """Winner takes all"""
            ot_next = np.where(tot_incoming_drive == tot_incoming_drive.max(), tot_incoming_drive, 0.)
            return ot_next
        else:
            """Implement Heaviside step function"""
            heaviside_step = np.expand_dims([0.] * len(tot_incoming_drive),1)
            heaviside_step[tot_incoming_drive > 0] = 1.
            return heaviside_step


# Simulate / Train SORN
class SimulateRMSorn(RMSorn):
    """
        Args:
        inputs - one hot vector of inputs
        Returns:
        matrix_collection - collection of all weight matrices in dictionaries
        """
    def __init__(self,phase,matrices,inputs,sequence_length, targets, reward_window_sizes, epochs):
        super().__init__()
        self.time_steps = np.shape(inputs)[0]*sequence_length * epochs
        RMSorn.time_steps = np.shape(inputs)[0]*sequence_length* epochs
#         self.inputs = np.asarray(np.tile(inputs,(1,epochs)))
        self.inputs = inputs
        self.phase = phase
        self.matrices = matrices
        self.epochs = epochs
        self.reward_window_sizes = reward_window_sizes
        self.sequence_length = sequence_length
        
    def train_sorn(self):  
        # Collect the network activity at all time steps
        X_all = [0]*self.time_steps
        Y_all = [0]*self.time_steps
        R_all = [0]*self.time_steps
        O_all = [0]*self.time_steps
        Rewards,Mo,Mr = [],[],[]
        frac_pos_active_conn = []
        
        """ DONOT INITIALIZE WEIGHTS"""
        matrix_collection = MatrixCollection(phase = self.phase, matrices = self.matrices)  
        time_steps_counter= 0
        """ Generate white noise"""
        white_noise_e = Initializer.white_gaussian_noise(mu= 0., sigma = 0.04,t = RMSorn.ne)
        white_noise_i = Initializer.white_gaussian_noise(mu= 0., sigma = 0.04,t = RMSorn.ni)

        # Buffers to get the resulting x, y and o vectors at the current time step and update the master matrix
        x_buffer, y_buffer, o_buffer = np.zeros(( RMSorn.ne, 2)), np.zeros((RMSorn.ni, 2)), np.zeros(( RMSorn.no, 2))

        te_buffer, ti_buffer, to_buffer = np.zeros((RMSorn.ne, 1)), np.zeros((RMSorn.ni, 1)), np.zeros(( RMSorn.no, 1))

        # Get the matrices and rename them for ease of reading
        Wee, Wei, Wie,Woe = matrix_collection.Wee, matrix_collection.Wei, matrix_collection.Wie, matrix_collection.Woe
        Te, Ti,To = matrix_collection.Te, matrix_collection.Ti,matrix_collection.To
        X, Y, O = matrix_collection.X, matrix_collection.Y, matrix_collection.O
        i = 0   
        for k in tqdm.tqdm(range(self.inputs.shape[0])):
            
            for j in range(self.sequence_length):
                """ Fraction of active connections between E-E network"""
                frac_pos_active_conn.append((Wee[i] > 0.0).sum())
                network_state = NetworkState(self.inputs[k][j])  # Feed Input as an argument to the class
                # Recurrent drive,excitatory, inhibitory and output network states 
                r = network_state.recurrent_drive(Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e = 0.)
                excitatory_state_xt_buffer = network_state.excitatory_network_state(Wee[i], Wei[i], Te[i], X[i], Y[i],white_noise_e = 0.)
                inhibitory_state_yt_buffer = network_state.inhibitory_network_state(Wie[i], Ti[i], X[i],white_noise_i = 0.)
                output_state_ot_buffer = network_state.output_network_state(Woe[i], To[i], X[i])
                """ Update X and Y """
                x_buffer[:, 0] = X[i][:, 1]  # xt -->(becomes) xt_1
                x_buffer[:, 1] = excitatory_state_xt_buffer.T  # New_activation; x_buffer --> xt
                y_buffer[:, 0] = Y[i][:, 1]
                y_buffer[:, 1] = inhibitory_state_yt_buffer.T
                o_buffer[:, 0] = O[i][:, 1]
                o_buffer[:, 1] = output_state_ot_buffer.T

                """Plasticity phase"""
                plasticity = Plasticity()
                # Reward and mo, mr 
                current_reward = output_state_ot_buffer*targets[k][j]
                Rewards.extend(current_reward)
                mo, mr, best_reward, best_reward_window = plasticity.modulation_factor(Rewards, current_reward, self.reward_window_sizes)  
                print('Input %s | Target %s | predicted %s | mr %s, mo %s'%(self.inputs[k].tolist(), targets[k][j],output_state_ot_buffer, mr, mo))
                Mo.append(mo)
                Mr.append(mr)
                
                # STDP, Intrinsic plasticity and Synaptic scaling
                Wee_t = plasticity.stdp(Wee[i],x_buffer,mr, cutoff_weights = (0.0,1.0))
                Woe_t = plasticity.ostdp(Woe[i],x_buffer,mo)
                Te_t = plasticity.ip(Te[i],x_buffer)
                Wee_t = Plasticity().ss(Wee_t)
                Woe_t = Plasticity().ss(Woe_t)

                """Assign the matrices to the matrix collections"""
                matrix_collection.weight_matrix(Wee_t, Wei[i], Wie[i],Woe_t, i)
                matrix_collection.threshold_matrix(Te_t, Ti[i],To[i], i)
                matrix_collection.network_activity_t(x_buffer, y_buffer,o_buffer, i)

                X_all[i] = x_buffer[:,1]
                Y_all[i] = y_buffer[:,1]
                R_all[i] = r
                O_all[i] = o_buffer[:,1]
            i+=1                                                        
        plastic_matrices = {'Wee':matrix_collection.Wee[-1], 
                            'Wei': matrix_collection.Wei[-1], 
                            'Wie':matrix_collection.Wie[-1],
                            'Woe':matrix_collection.Woe[-1],
                            'Te': matrix_collection.Te[-1], 'Ti': matrix_collection.Ti[-1],
                            'X': X[-1], 'Y': Y[-1]}
        
        return plastic_matrices,X_all,Y_all,R_all,frac_pos_active_conn
    
    
