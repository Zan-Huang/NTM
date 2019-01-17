from __future__ import print_function
import tensorflow as tf
import numpy as np

init_op = tf.global_variables_initializer()
init = tf.constant(np.random.rand(10,), dtype=float)
init2 = tf.constant(np.random.rand(20, 10), dtype=float)
key_vector = tf.get_variable("weight_vector_test", initializer = init)
memory_vector = tf.get_variable("memory_vector_test", initializer = init2)

def similarity_measure(key_vector, memory_vector_slice):
    if key_vector.shape.as_list()[0] != memory_vector.shape.as_list()[1]:
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
    for j in range(0, memory_vector.shape.as_list()[0]):
        content_bot_temp = tf.math.exp(tf.multiply(beta_strength, similarity_measure(key_vector, memory_vector[j])))
        content_bottom = tf.add(content_bot_temp, content_bottom)

    for i in range(0, memory_vector.shape.as_list()[0]):
        content_vector_top = tf.math.exp(tf.multiply(beta_strength, similarity_measure(key_vector, memory_vector[i])))
        final_top = content_vector_top / content_bottom
        composite_top.append(final_top)

    final_focus_vector = tf.stack([piece for piece in composite_top])
    return final_focus_vector

with tf.Session() as sess:
    print(content_address(tf.cast(5.0, tf.float32), tf.cast(key_vector, tf.float32), memory_vector))
    sess.run(init_op)
