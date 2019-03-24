import tensorflow as tf
import numpy as np

import content_focus
import convshift
import focus_location

class addressing(object):

    def __init__(self, k, beta, g, s, gamma, M_prev, w_prev, sess):
        self.k = k
        self.beta = beta
        self.g = g
        self.s = s
        self.gamma = gamma
        print("*")
        print(M_prev.shape.as_list())
        print("*")

        self.M_prev = M_prev

        print("self")
        print(self.M_prev.shape.as_list())
        print("self")

        self.w_prev = w_prev
        #self.ones = tf.ones(M_prev.shape()[1])
        self.ones = 1.0
        self.sess = sess

    def address(self):
        print(self.M_prev.shape.as_list())
        print("+++++++++++++")
        content_weight = content_focus.content_address(self.beta, self.k, self.M_prev, self.sess)

        #interpolation_weight = focus_location.location_lookup(self.w_prev, content_weight, self.g, self.ones)
        interpolation_weight = focus_location.location_lookup(self.w_prev, content_weight, self.g, self.ones)
        blurred_weight = convshift.conv_function(self.g, self.s)
        focus_weight = convshift.sharpness_function(blurred_weight,self.gamma)
        return focus_weight
