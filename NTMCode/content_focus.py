from __future__ import print_function
import tensorflow as tf
import numpy as np


def similarity_measure(key_vector, memory_vector_slice):
    print("test")
    print(key_vector.shape.as_list()[0])
    print(memory_vector_slice.shape)
    if key_vector.shape.as_list()[0] != memory_vector_slice.shape.as_list()[0]:
        raise Exception('The length of key vector is not equal to memory vector row length.')
    dot_product_term = tf.tensordot(key_vector, memory_vector_slice, 1)
    key_vector_norm = tf.norm(key_vector, ord='euclidean')
    memory_vector_norm = tf.norm(memory_vector_slice)
    norm_products = tf.multiply(key_vector_norm, memory_vector_norm)
    similarity_measure_result = tf.divide(dot_product_term, norm_products)
    return similarity_measure_result


def content_address(beta_strength, key_vector, memory_vector):
    content_filler = tf.constant(np.zeros(memory_vector.shape.as_list()[0]))
    weight_vector = tf.get_variable("content_weights", initializer = content_filler)
    composite_top = []

    content_bottom = 0
    for j in range(0, memory_vector.shape.as_list()[1]):
        content_bot_temp = tf.math.exp(tf.multiply(beta_strength, similarity_measure(key_vector, tf.gather_nd(memory_vector,[1,j]))))
        content_bottom = tf.add(content_bot_temp, content_bottom)

    for i in range(0, memory_vector.shape.as_list()[1]):
        content_vector_top = tf.math.exp(tf.multiply(beta_strength, similarity_measure(key_vector, tf.gather_nd(memory_vector,[1,i]))))
        final_top = content_vector_top / content_bottom
        composite_top.append(final_top)
    final_focus_vector = tf.stack([piece for piece in composite_top])
    return final_focus_vector
