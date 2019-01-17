from __future__ import print_function
import tensorflow as tf
import numpy as np
#Below is Testing

init_op = tf.global_variables_initializer()

init = tf.constant(np.random.rand(1, 20))
init2 = tf.constant(np.random.rand(20, 10))
init3 = tf.constant(np.random.rand(1, 10))

a = tf.get_variable("weight_vector_test", initializer = init)
b = tf.get_variable("memory_vector_test", initializer = init2)
c = tf.get_variable("erase_vector_test", initializer = init3)

def erase_function(weight_vector, erase_vector, past_timestep_memory_matrix):
	if weight_vector.shape.as_list()[1] != past_timestep_memory_matrix.shape.as_list()[0]:
		raise Exception('The size of the memory matrix does not match the memory vector. Check size of W and make sure it is N')
	if erase_vector.shape.as_list()[1] != past_timestep_memory_matrix.shape.as_list()[1]:
		raise Exception('The size of the weight vector does not match the erase vector. Check size of W and make sure it is E')
	ones = tf.ones(erase_vector.shape, tf.float64)
	erased = []
	for i in range(0,past_timestep_memory_matrix.shape.as_list()[0]):
		inners = tf.subtract(ones, tf.multiply(weight_vector[0][i],erase_vector))
		outers = tf.multiply(past_timestep_memory_matrix[i], inners)
		erased.append(outers)
	erased_memory = tf.stack(erased)
	return erased_memory

with tf.Session() as sess:
	print(erase_function(a,c,b))