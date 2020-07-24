import numpy as np 
from rmsorn import *

class PatternRecognition(object):
    """1234, 4321, 4213, 2431 and has to recognize the word 1234"""
    
    def __init__(self):
        super().__init__()
        # Note: Last element in each sequence is the target   
        self.training_sequence = np.repeat(np.array([[[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]],
                                          [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], 
                                          [[1,0,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0]], 
                                          [[0,0,1,0], [0,0,0,0], [0,1,0,0], [0,0,0,1]]]), 
                                          repeats=1000, axis=0)

        self.sequence_targets = np.repeat(np.array([1,0,0,0]),repeats=1000,axis=0)

    def generate_sequence():
         
        input_str = ['1234','4321', '4213', '2431']
        training_input = []
        targets = []
        for  i in range(100):
            idx = random.randint(0,3)
            inp = input_str[idx]
            if inp == '1234':
                input_seq = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
                target = [1,1,1,1]
            elif inp == '4321':
                input_seq = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
                target = [0,0,0,0]
            elif inp == '4213':
                input_seq = [[1,0,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0]]
                target = [0,0,0,0]
            else:
                input_seq = [[0,0,1,0], [0,0,0,0], [0,1,0,0], [0,0,0,1]]
                target = [0,0,0,0]
            training_input.append(input_seq)
            targets.append(target)
            
        return training_input, targets
        
train_plast_inp_mat,X_all_inp,Y_all_inp,R_all, frac_pos_active_conn = SimulateRMSorn(phase = 'Plasticity', 
                                                                               matrices = None,
                                                                               inputs = np.asarray(training_input),sequence_length = 4, targets = targets,
                                                                                reward_window_sizes = [1,5,10,20],
                                                                               epochs = 1).train_sorn()