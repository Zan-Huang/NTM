import tensorflow as tf
import pandas as pd
import numpy as np

def __init__(self, output, seq_length, batch_size, output_dim, vector_dim):
    self.x = tf.placeholder(name='x',dtype=tf.float32,shape=[batch_size, seq_length, vector_dim])
    self.y = output

    cell = NTMCell #fill here
    controller_state, read_vector, w = cell.initial_state(batch_size, tf.float32)
    for i in range(seq_length):
        output, controller_state, read_vector, w = cell.call(self.x[:,i,:],controller_state,w,read_vector)
        
