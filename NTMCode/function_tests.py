<<<<<<< HEAD
from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import convshift
import content_focus
import write
import focus_location
import read

 #Below is testing

 #N = 30
 #M = 20

init_op = tf.global_variables_initializer()

init = tf.constant(np.random.rand(30))
init2 = tf.constant(np.random.rand(30,20))
init3 = tf.constant(np.random.rand(20))
init4 = tf.constant(np.random.rand(),dtype=tf.float64)
init5 = tf.constant(np.random.rand(),dtype=tf.float64)
init6 = tf.constant(np.array([-1,0,1]),dtype=tf.float64)
init7 = tf.constant(np.random.rand(),dtype=tf.float64)
init8 = tf.constant(np.array([1]),dtype=tf.float64)

w_prev = tf.get_variable("previous_weight", initializer = init)
M_prev = tf.get_variable("previous_Memory", initializer = init2)
key_vec = tf.get_variable("memory_key_vector", initializer = init3)
beta = tf.get_variable("Key_strength", initializer = init4)
inter_gate = tf.get_variable("interpolation", initializer = init5)
conv_vec = tf.get_variable("conv_vec", initializer = init6)
gamma = tf.get_variable("gama_scal", initializer = init7)
one = tf.get_variable("one", initializer = init8)

def main():
    content_weight = content_focus.content_address(beta, key_vec, M_prev)
    interpolation_weight = focus_location.location_lookup(w_prev, content_weight, inter_gate, one)
    blurred_weight = convshift.conv_function(interpolation_weight, conv_vec)
    weight = convshift.sharpness_function(blurred_weight,gamma)
    return weight

with tf.Session() as sess:
    print(main())
    sess.run(init_op)
=======
import tensorflow
import content_focus
import convshift
import function_tests
import read
import weight
import write

def function_tests():
    initTESTMemory_Matrix = tf.constant(np.random.rand(30, 20))
    initWeightMatrix = tf.constant(np.random.rand(20, 1))

    ubutTESTMemory_Matrix = tf.get_variable("memory_vector_fortesting", initializer = initTESTMemory_Matrix)
    ubutWeightMatrix = tf.get_variable("memory_weight_test", initializer = initWeightMatrix)

    return tests

if __name__ == '__function_tests__':
    function_tests()
>>>>>>> b882755514d1d8041d6128a57cb5c10a78d30c33
