from __future__ import print_function
import tensorflow as tf
import numpy as np

#Below is Testing
'''
init_op = tf.global_variables_initializer()

init = tf.constant(np.random.rand(1,20))
init2 = tf.constant(np.random.rand(),dtype=tf.float64)

a = tf.get_variable("conved_weight_vector_test", initializer = init)
b = tf.get_variable("gamma_test", initializer = init2)
'''
def sharpness_function(conved_weight_vector, gamma):
    weight = []
    for i in range(0,conved_weight_vector.shape.as_list()[0]):
        element_gama = tf.pow(conved_weight_vector[i],gamma)
        summed_element_gama = tf.reduce_sum(tf.pow(conved_weight_vector,gamma))
        weight.append(tf.divide(element_gama,summed_element_gama))
    return tf.stack([w for w in weight])
'''
with tf.Session() as sess:
    print(sharpness_function(a,b))
'''
