from __future__ import print_function
import tensorflow as tf
import numpy as np
import math

#Below is Testing

init_op = tf.global_variables_initializer()

init = tf.constant(np.random.rand(20))
init2 = tf.constant(np.random.rand(),dtype=tf.float64)
init3 = tf.constant(np.random.rand(5))

a = tf.get_variable("conved_weight_vector_test", initializer = init)
b = tf.get_variable("gamma_test", initializer = init2)
c = tf.get_variable("shift_vector_test", initializer = init3)

def conv_function(v, k):
    size = int(v.get_shape()[0])
    kernel_size = int(k.get_shape()[0])
    kernel_shift = int(math.floor(kernel_size/2.0))

    def loop(idx):
        if idx < 0: return size + idx
        if idx >= size : return idx - size
        else: return idx

    kernels = []
    for i in xrange(size):
        indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
        v_ = tf.gather(v, indices)
        kernels.append(tf.reduce_sum(v_ * k, 0))

    return tf.dynamic_stitch([i for i in xrange(size)], kernels)

def sharpness_function(conved_weight_vector, gamma):
    weight = []
    for i in range(0, conved_weight_vector.shape.as_list()[0]):
        element_gama = tf.pow(conved_weight_vector[i],gamma)
        summed_element_gama = tf.reduce_sum(tf.pow(conved_weight_vector,gamma))
        weight.append(tf.divide(element_gama,summed_element_gama))
    return tf.stack([w for w in weight])


with tf.Session() as sess:
    print(conv_function(a,c))
