import tensorflow as tf
import numpy as np
import sys
import model

#data = np.load('data.npy')
sys.path.insert(0, 'NTMCode')
train_x = np.random.rand(1,500,40)
train_y = np.random.rand(40)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('seq_length', 500, 'length of stock days')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'learning rate of the model')
tf.app.flags.DEFINE_string('tensorboard_dir', './summary', 'where to save the tensorboard')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Length of a training batch')
tf.app.flags.DEFINE_integer('vector_dim', 40, 'Number of features')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'How many epochs to run the model for')


def main(argv = None):
	#print(data.shape)
	train()
'''
def build_model():
	self.train_inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.seq_length, FLAGS.vector_dim])
	self.train_labels = tf.placeholder(tf.float32, [FALGS.batch_size])
'''
def train():
#	print(seq_length)
	train_inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.seq_length, FLAGS.vector_dim])
	train_labels = tf.placeholder(tf.float32, [FLAGS.batch_size])

	#with tf.Session() as sess:
	with tf.Session() as sess:
		model1 = model.StockPredictor(FLAGS.seq_length,FLAGS.batch_size,FLAGS.vector_dim, sess)
		saver = tf.train.Saver(tf.global_variables())
		tf.global_variables_initializer().run()
		train_writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)
		train_dict = {train_inputs:train_x, train_labels:train_y}
		for i in range(args.num_epocs):
			feed_dict = train_dict

			#sess.run(model,feed_dict)
			sess.run(feed_dict)



if __name__ == '__main__':
    tf.app.run()
