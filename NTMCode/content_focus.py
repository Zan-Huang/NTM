from __future__ import print_function
import tensorflow as tf
import numpy as np

def similarity_measure(key_vector, memory_vector):
    if key_vector.shape.as_list()[0] != memory_vector().shape.as_list()[1]:
        raise Exception('The length of key vector is not equal to memory vector row length.')
    dot_product_term = tf.matmul(key_vector, memory_vector)
    key_vector_norm = tf.norm(key_vector, ord='euclidean')
    memory_vector_norm = tf.norm(memory_vector)
    norm_products = tf.multiply(key_vector_norm, memory_vector_norm)
    similarity_measure_result = tf.divide(dot_product_term, norm_products)
    return similarity_measure_result


#def content_address(beta_strength):
