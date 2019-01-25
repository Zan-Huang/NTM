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
