from __future__ import print_function
import tensorflow as tf
import numpy as np

#Below is Testing
'''
init_op = tf.global_variables_initializer()
init = tf.constant(np.random.rand(10))
init2 = tf.constant(np.random.rand(10, 20))
a = tf.get_variable("weight_vector_test", initializer = init)
b = tf.get_variable("memory_matrix_test", initializer = init2)
'''
def reading_function(weight_vector, memory_matrix):
    if weight_vector.shape.as_list()[0] != memory_matrix.shape.as_list()[0]:
        raise Exception('The size of the memory matrix does not match the memory vector. Check size of W and make sure it is N')
    read_vector = tf.tensordot(weight_vector, memory_matrix,1)
    return read_vector
'''
with tf.Session() as sess:
    print(reading_function(a, b))
    sess.run(init_op)
'''
