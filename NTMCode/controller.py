from __future__ import print_function
import tensorflow as tf
import numpy as np

import convshift
import content_focus
import write
import focus_location
import read

class NTM(object):
    def __init__(self, unit_size, memory, graph_argument):
        self.unit_size = unit_size
        self.memory = memory
        self.controller = tf.nn.rnn_cell.BasicRNNCell(unit_size)
        self.step = 0
        self.graph_argument = graph_argument

    def __call__(self, x, previous_controller, prev_read):
        with self.graph_argument.name_scope("concat"):
            NTM_Input = tf.concat([x], prev_read, axis=1)
            
        ###Much code to fill in here on Sunday

        self.step += 1

        return NTM_output, controller_state, read_vector_batch, weight_batch, memory

    def initial_state(self, batch_size, dtype):
        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

        controller_state = expand(tf.tanh(tf.get_variable('init_state', self.unit_size, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25))), dim=0, N=batch_size)
        read_vector_batch = [expand(tf.nn.softmax(tf.get_variable('init_read', [self.memory.shape[1]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), dim=0, N=batch_size)]
        weight_batch = [expand(tf.nn.softmax(tf.get_variable('init_write', [self.memory.shape[0]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), dim=0, N=batch_size)]
        memory = expand(tf.tanh(tf.get_variable('init_memory', [self.memory.shape[0], self.memory.shape[1]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), dim=0, N=batch_size)
        return controller_state, read_vector_batch, weight_batch, memory
