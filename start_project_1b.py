#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 7

learning_rate = 0.001
beta = 0.001
epochs = 10000

batch_size = 8
num_neuron = 10
seed = 10
np.random.seed(seed)

def main():
	#read and divide data into test and train sets 
	admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
	X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
	Y_data = Y_data.reshape(Y_data.shape[0], 1)

	X_data = (X_data - np.mean(X_data, axis=0))/ np.std(X_data, axis=0)

	# divide to training set and test set (70:30)
	testX = X_data[280:]
	testY = Y_data[280:]
	trainX = X_data[:280]
	trainY = Y_data[0:280]

	# Create the model
	x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [None, 1])
	
	# Build the graph for the deep net
	w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(float(NUM_FEATURES))), name='weights')
	b1 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b2 = tf.Variable(tf.zeros([1]), name='biases')
	y = tf.matmul(h, w2) + b2

	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
	cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), axis = 1))
	loss = tf.reduce_mean(cost + beta*regularization)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)

	# train
	N = len(trainX)
	idx = np.arange(N)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		tr_err, te_err = [], []
		for i in range(epochs):
			np.random.shuffle(idx)
			trainX = trainX[idx]
			trainY = trainY[idx]

			for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
				train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

			tr_err.append(loss.eval(feed_dict={x:trainX, y_:trainY}))
			te_err.append(loss.eval(feed_dict={x:testX, y_:testY}))
			if i%10 == 0:
				print('batch %d: iter %d, train error %g, test error %g'%(batch_size, i, tr_err[i], te_err[i]))

	# plot learning curves
	plt.figure(1)
	plt.plot(range(epochs), tr_err, label='train error')
	plt.plot(range(epochs), te_err, label='test error')
	plt.xlabel(str(epochs) + ' iterations')
	plt.ylabel('mean square error')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
