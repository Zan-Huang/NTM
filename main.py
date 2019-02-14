import tensorflow as tf
import numpy as np
import sys
import model

sys.path.insert(0, 'NTMCode')

def train():



	with tf.Session() as sees:
		saver = tf.train.Saver(tf.global_variables())
		tf.global_variables_initializer().run()
		train_writer = tf.summary.FileWriter(args.tensorboard, sens.graph)
		for i in range(args.num_epocs):
			feed_dict = {input:__need numpy array here__}

		sess.run


def main():

    model1 = model.StockPredictor()

if __name__ == '__main__':
    main()
