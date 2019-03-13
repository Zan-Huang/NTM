from __future__ import print_function
import tensorflow as tf
import numpy as np
import keras

import convshift
import content_focus
import write
import focus_location
import read
import addressing

class NTM(object):
    def __init__(self, unit_size, memory, output_dim, graph_argument):
        self.unit_size = unit_size
        self.memory = memory
        self.controller = tf.contrib.rnn.GRUCell(unit_size)
        self.step = 0
        self.output_dim = output_dim
        self.graph_argument = graph_argument

    def __call__(self, x, previous_controller, previous_weights, prev_read):
        print(self.memory.get_shape(),'0')
        
        with tf.variable_scope("concat", reuse = (self.step > 0)):
            x = tf.layers.Flatten()(x)
            NTM_Input = tf.concat([x, prev_read], axis=1)

        with tf.variable_scope("controller", reuse = (self.step > 0)):
            controller_output, controller_state = self.controller(NTM_Input, previous_controller)

        with tf.variable_scope("parameter_feedforward", reuse = (self.step > 0)):
            parameter_weight = tf.get_variable('parameter_weight', [controller_output.get_shape()[1], 5 + 3 * (self.memory.get_shape()[1])], initializer = tf.contrib.layers.xavier_initializer())
            #parameter weight is determined by controller outputs and the memory M dimension M is multiplied by three because
            #of the key vector and the two add and erase vectors
            parameter_bias = tf.get_variable('parameter_bias', [5 + 3 * (self.memory.get_shape()[1])], initializer = tf.contrib.layers.xavier_initializer())

            parameters = tf.nn.xw_plus_b(controller_output, parameter_weight, parameter_bias)

        #head_parameter = parameters
        #erase_add = tf.split(parameters[self.memory.get_shape()[1]+1+1+3+1], 2, axis=1)
        head_parameter = parameters[:self.memory.get_shape()[1]+1+1+3+1+1]
        erase_add = parameters[self.memory.get_shape()[1]+1+1+3+1+1:]
        #Form focus vectors




        k = tf.tanh(head_parameter[0][0:self.memory.get_shape()[1]])
        beta = tf.sigmoid(head_parameter[0][self.memory.get_shape()[1]]) * 10
        g = tf.sigmoid(head_parameter[0][self.memory.get_shape()[1] + 1])
        s = tf.nn.softmax(head_parameter[0][self.memory.get_shape()[1] + 2:self.memory.get_shape()[1] + 2 + 3])
        gamma = tf.log(tf.exp(head_parameter[0][-1]) + 1) + 1
        with tf.variable_scope('adressing_head'):
            ad = addressing.addressing(k, beta, g, s,gamma, self.memory, previous_weights)
            w = ad.address()
        #Reading
        with tf.variable_scope("read_vector"):
            read_vector = read.reading_function(w, self.memory)


        #Writing
        erase_vector = tf.sigmoid(erase_add[0], axis=1)
        add_vector = tf.tanh(erase_add[1], axis=1)
        with tf.variable_scope("write_vector"):
            comp_mem = write.write(w,erase_vector,add_vector,self.memory)
        self.memory = comp_mem

        #Going from controller to NTM

        with tf.graph_argument.variable_scope("output_feedforward", reuse=(self.step>0)):
            output_weight = tf.get_variable('output_weight', [controller_output.shape()[1]])
            output_bias = tf.get_variable('output_bias', [output_dim])
            output = tf.nn.xw_plus_b(controller_output, output_weight, output_bias)


        self.step += 1

        return output, controller_state, read_vector, w

    def initial_state(self, batch_size, dtype):
        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

        controller_state = expand(tf.tanh(tf.get_variable('init_state', self.unit_size, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25))), dim=0, N=batch_size)
        read_vector_batch = expand(tf.nn.softmax(tf.get_variable('init_read', [self.memory.shape[1]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), dim=0, N=batch_size)
        weight_batch = expand(tf.nn.softmax(tf.get_variable('init_write', [self.memory.shape[0]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), dim=0, N=batch_size)
        self.memory = expand(tf.tanh(tf.get_variable('init_memory', [self.memory.shape[0], self.memory.shape[1]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), dim=0, N=batch_size)
        return controller_state, read_vector_batch, weight_batch
