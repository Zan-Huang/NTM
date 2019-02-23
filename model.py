import tensorflow as tf
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'NTMCode')

import controller

g = tf.Graph()

class StockPredictor(object):
    def __init__(self, seq_length, batch_size, vector_dim):
        self.x = tf.placeholder(name='x',dtype=tf.float32,shape=[batch_size, seq_length, vector_dim])
        self.y = label

        memory_vector = tf.get_variable("memory_vector_test", initializer = tf.get_variable(np.random.rand(80, 100)))
        unit_size = 512
        cell = controller.NTM(unit_size, memory_vector, x.shape[2], g)
        controller_state, read_vector, w = cell.initial_state(batch_size, tf.float32)
        for i in range(seq_length):
            output, controller_state, read_vector, w = cell.call(self.x[:,i,:],controller_state,w,read_vector)

        loss = tf.losses.mean_squared_error(output, label)
        self.output = output

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(lr, b1, b2, e)
            grads = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(grads)

            self.loss_summary = tf.summary.scalar(loss)

    def return_parameters():
        return self.y, self.output