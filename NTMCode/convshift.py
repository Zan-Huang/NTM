from __future__ import print_function
import tensorflow as tf
import numpy as np

#Below is Testing

init_op = tf.global_variables_initializer()

init = tf.constant(np.random.rand(20))
init2 = tf.constant(np.random.rand(),dtype=tf.float64)
init3 = tf.constant(np.random.rand(5))

a = tf.get_variable("conved_weight_vector_test", initializer = init)
b = tf.get_variable("gamma_test", initializer = init2)
c = tf.get_variable("shift_vector_test", initializer = init3)


def conv_function(interpolation_weight_vector, shift_vector):
    weight_elements = []
    print(shift_vector[-1],'herro')
    for i in range(0,interpolation_weight_vector.shape.as_list()[0]):
        inner_sum = []
        for j in range(0, interpolation_weight_vector.shape.as_list()[0]):
            conv_index = i - j
            print(conv_index)
            element_conv = tf.multiply(interpolation_weight_vector[j], shift_vector[conv_index])
            inner_sum.append(element_conv)
        inner_sum = tf.reduce_sum(tf.stack([sum for sum in inner_sum]))
        weight_elements.append(inner_sum)
    weights = tf.stack(weight for weight in weight_elements)
    return weights


def sharpness_function(conved_weight_vector, gamma):
    weight = []
    for i in range(0, conved_weight_vector.shape.as_list()[0]):
        element_gama = tf.pow(conved_weight_vector[i],gamma)
        summed_element_gama = tf.reduce_sum(tf.pow(conved_weight_vector,gamma))
        weight.append(tf.divide(element_gama,summed_element_gama))
    return tf.stack([w for w in weight])


with tf.Session() as sess:
    print(conv_function(a,c))
