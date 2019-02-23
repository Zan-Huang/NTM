import tensorflow as tf
import numpy as np
import sys
import model

data = np.load('data.npy')
sys.path.insert(0, 'NTMCode')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('--seq_length', 100, 'length of stock days')
flags.DEFINE_float('--learning_rate', 1e-2, 'learning rate of the model')
flags.DEFINE_string('--tensorboard_dir', './summary', 'where to save the tensorboard')
flags.DEFINE_integer('--batch_size', 1, 'Length of a training batch')
flags.DEFINE_integer('--vector_dim', 5, 'Number of features')
flags.DEFINE_integer('--num_epochs', 1000, 'How many epochs to run the model for')


def main():
	print(data.shape)
	train()

def build_model(self):
	self.train_inputs = tf.placeholder(tf.float32, [FLAGS.numEl, FLAGS.seq_length, FLAGS.vector_dim])
	self.train_labels = tf.placeholder(tf.float32, [FALGS.numEl])

def train():
	modelf = model.StockPredictor(FLAGS.seq_length,FLAGS.batch_size,FLAGS.vector_dim)

	with tf.Session() as sess:
		saver = tf.train.Saver(tf.global_variables())
		tf.global_variables_initializer().run()
		train_writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)
		train_dict = {self.train_inputs:self.train_x, self.train_labels:self.train_y}
		for i in range(args.num_epocs):
			feed_dict = train_dict

			#sess.run(model,feed_dict)
			sess.run(feed_dict)



if __name__ == '__main__':
    main()
