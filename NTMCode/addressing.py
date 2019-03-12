import tensorflow as tf
import numpy as np

import content_focus
import convshift
import focus_location

class adressing(object):

    def __init__(self, k, beta, g, s, gamma, M_prev, w_prev):
        self.k = k
        self.beta = beta
        self.g = g
        self.s = s
        self.gamma = gamma
        self.M_prev = M_prev
        self.w_prev = w_prev
        #self.ones = tf.Variable(np.array([M_prev.shape()][1]))

    def address(self):
        content_weight = content_focus.content_address(self.beta, self.k, self.M_prev)
        #interpolation_weight = focus_location.location_lookup(self.w_prev, content_weight, self.g, self.ones)
        interpolation_weight = focus_location.location_lookup(self.w_prev, content_weight, self.g, tf.Variable(np.array([M_prev.shape()][1])))
        blurred_weight = convshift.conv_function(self.g, self.s)
        focus_weight = convshift.sharpness_function(blurred_weight,self.gamma)
        return focus_weight
