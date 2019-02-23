import tensorflow as tf
import numpy as np
import sys
import model

data = np.load('data.npy')
sys.path.insert(0, 'NTMCode')

flags = tf.app.flags
FLAGS = flags.FLAGS


def main(argv):
	print(data.shape)
	train()

def train(self):
	model = model.StockPredictor(self.args.output,self.args.seq_length,self.args.batch_size,self.args.output_dim,self.args.vector_dim)

	with tf.Session() as sess:
		saver = tf.train.Saver(tf.global_variables())
		tf.global_variables_initializer().run()
		train_writer = tf.summary.FileWriter(self.args.tensorboard_dir, self.sess.graph)
		for i in range(args.num_epocs):
			feed_dict = {input: data}

		#sess.run(model,feed_dict)
		sess.run(feed_dict)



if __name__ == '__main__':
    main()
