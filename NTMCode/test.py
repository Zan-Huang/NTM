from __future__ import print_function
import tensorflow as tf
import numpy as np

#Below is Testing

init_op = tf.global_variables_initializer()

init = tf.constant(np.random.rand(5, 1))
init2 = tf.constant(np.random.rand(1, 5))

a = tf.get_variable("weight_vector_test", initializer = init)
b = tf.get_variable("memory_vector_test", initializer = init2)

with tf.Session() as sess:
	print(a[:])