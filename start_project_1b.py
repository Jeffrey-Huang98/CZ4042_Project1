#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 7

learning_rate = 0.001
epochs = 1000

batch_size = 8
num_neuron = 10
seed = 10
np.random.seed(seed)

# build a feedforward network
def ffn(x, hidden_units):
    
  with tf.name_scope('hidden'):
    weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_units], stddev=1.0 / np.sqrt(float(NUM_FEATURES))), name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),name='biases')
    h = tf.nn.relu(tf.matmul(x, weights) + biases)
    
  with tf.name_scope('output'):
    weights = tf.Variable(tf.truncated_normal([hidden_units, 1], stddev=1.0 / np.sqrt(float(hidden_units))), name='weights')
    biases = tf.Variable(tf.zeros([1]), name='biases')
    u = tf.matmul(h, weights) + biases
    
  return u

# train the network and find errors
def train_exp(batch_size):
	#read and divide data into test and train sets 
	admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
	X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
	Y_data = Y_data.reshape(Y_data.shape[0], 1)

	# divide to training set and test set (70:30)
	testX = X_data[:120]
	testY = Y_data[:120]
	trainX = X_data[120:]
	trainY = Y_data[120:]

	trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

	# Create the model
	x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [None, 1])
	
	# Build the graph for the deep net
	y = ffn(x, num_neuron)

	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	loss = tf.reduce_mean(tf.square(y_ - y))
	train_op = optimizer.minimize(loss)

	# train
	N = len(trainX)
	idx = np.arange(N)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		err = []
		for i in range(epochs):
			np.random.shuffle(idx)
			trainX = trainX[idx]
			trainY = trainY[idx]

			for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
				train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

			test_err = loss.eval(feed_dict={x:testX, y_:testY})
			if i%10 == 0:
				print('batch %d: iter %d, test error %g'%(batch_size, i, test_err))
			err.append(test_err)

	return err

def main():
	train_err = train_exp(8)

	# plot learning curves
	plt.figure(1)
	plt.plot(range(epochs), train_err)
	plt.xlabel(str(epochs) + ' iterations')
	plt.ylabel('Train Error')
	plt.show()

if __name__ == '__main__':
	main()
