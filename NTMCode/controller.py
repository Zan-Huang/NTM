from __future__ import print_function
import tensorflow as tf
import numpy as np

import convshift
import content_focus
import write
import focus_location
import read

class NTM(object):
    def __init__(self, unit_size, memory, graph_argument):
        self.unit_size = unit_size
        self.memory = memory
        self.controller = tf.nn.rnn_cell.BasicRNNCell(unit_size)
        self.step = 0
        self.graph_argument = graph_argument

    def __call__(self, x, previous_controller, prev_read):
        with self.graph_argument.name_scope("concat"):
            NTM_Input = tf.concat([x], prev_read, axis=1)
