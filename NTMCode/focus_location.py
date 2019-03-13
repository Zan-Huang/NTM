from __future__ import print_function
import tensorflow as tf
import numpy as np

#Below is testing
'''
init_op = tf.global_variables_initializer()

init = tf.constant(np.random.rand(1,20))
init2 = tf.constant(np.random.rand(1, 20))
init3 = tf.constant(np.random.rand(),dtype=tf.float64)

a = tf.get_variable("weight_vector_test", initializer = init)
b = tf.get_variable("content_weight_vector_test", initializer = init2)
c = tf.get_variable("iterpolation_scalar_test", initializer = init3)
'''
def location_lookup(weight_vector, content_weight_vector, interpolation_scalar, one):
    print(content_weight_vector.get_shape(), 'cont')
    content_interpolation = tf.multiply(interpolation_scalar, content_weight_vector)
    location_interpolation = tf.multiply(tf.subtract(one,interpolation_scalar),weight_vector)
    interpolation_weight = tf.add(content_interpolation, location_interpolation)
    return interpolation_weight
'''
with tf.Session() as sess:
    print(location_lookup(a,b,c))
'''
