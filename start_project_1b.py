#
# Project 1, starter code part b
#

import tensorflow as tf
import pandas as pd
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 7

learning_rate = 0.001
beta = 0.001
epochs = 6750
keep_prob = 0.8

batch_size = 8
num_neuron = 10
seed = 10
np.random.seed(seed)

feature_removed = []

def ffn(x, num_neuron, num_feature):
	# Build the graph for the deep net
	w1 = tf.Variable(tf.truncated_normal([num_feature, num_neuron], stddev=1.0 / np.sqrt(float(num_feature))), name='weights')
	b1 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b2 = tf.Variable(tf.zeros([1]), name='biases')
	y = tf.matmul(h, w2) + b2

	return y, w1, w2

def ffn4(x, num_neuron, num_feature):
	# Build the graph for the deep net
	w1 = tf.Variable(tf.truncated_normal([num_feature, num_neuron], stddev=1.0 / np.sqrt(float(num_feature))), name='weights')
	b1 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b2 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

	w3 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b3 = tf.Variable(tf.zeros([1]), name='biases')
	y = tf.matmul(h2, w3) + b3

	return y, w1, w2, w3

def ffn4_dropout(x, num_neuron, num_feature):
	# Build the graph for the deep net
	w1 = tf.Variable(tf.truncated_normal([num_feature, num_neuron], stddev=1.0 / np.sqrt(float(num_feature))), name='weights')
	b1 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
	h1_dropout = tf.nn.dropout(h1, keep_prob)

	w2 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b2 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
	h2_dropout = tf.nn.dropout(h2, keep_prob)

	w3 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b3 = tf.Variable(tf.zeros([1]), name='biases')
	y = tf.matmul(h2_dropout, w3) + b3

	return y, w1, w2, w3

def ffn5_dropout(x, num_neuron, num_feature):
	# Build the graph for the deep net
	w1 = tf.Variable(tf.truncated_normal([num_feature, num_neuron], stddev=1.0 / np.sqrt(float(num_feature))), name='weights')
	b1 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
	h1_dropout = tf.nn.dropout(h1, keep_prob)

	w2 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b2 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
	h2_dropout = tf.nn.dropout(h2, keep_prob)

	w3 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b3 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h3 = tf.nn.relu(tf.matmul(h2_dropout, w3) + b3)
	h3_dropout = tf.nn.dropout(h3, keep_prob)

	w4 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b4 = tf.Variable(tf.zeros([1]), name='biases')
	y = tf.matmul(h3_dropout, w4) + b4

	return y, w1, w2, w3, w4

def ffn5(x, num_neuron, num_feature):
	# Build the graph for the deep net
	w1 = tf.Variable(tf.truncated_normal([num_feature, num_neuron], stddev=1.0 / np.sqrt(float(num_feature))), name='weights')
	b1 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b2 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

	w3 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b3 = tf.Variable(tf.zeros([num_neuron]), name='biases')
	h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

	w4 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(float(num_neuron))), name='weights')
	b4 = tf.Variable(tf.zeros([1]), name='biases')
	y = tf.matmul(h3, w4) + b4

	return y, w1, w2, w3, w4	

def train_model(qn_num, trainX, trainY, testX, testY):
	# Create the model
	x = tf.placeholder(tf.float32, [None, len(trainX[0])])
	y_ = tf.placeholder(tf.float32, [None, 1])

	y, w1, w2 = ffn(x, num_neuron,  len(trainX[0]))
	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
	error = tf.reduce_mean(tf.square(y - y_))
	loss = tf.reduce_mean(error + beta*regularization)
	train_op = optimizer.minimize(loss)

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

			tr_err.append(error.eval(feed_dict={x:trainX, y_:trainY}))
			te_err.append(error.eval(feed_dict={x:testX, y_:testY}))
		
			if(qn_num == 1):
				if i%10 == 0:
					print('batch %d: iter %d, train error %g, test error %g'%(batch_size, i, tr_err[i], te_err[i]))

		if(qn_num == 3):
			return te_err
		else:
			N = len(testX)
			idx = np.arange(N)
			np.random.shuffle(idx)
			testX, testY = testX[idx], testY[idx]
			target = testY[:50]
			testX = testX[:50]
			pred = y.eval(feed_dict={x:testX})

			return tr_err, te_err, pred, target

# recursive feature elimination function
def rfe(num_removed_features, X_data, Y_data):
	err_list = []
	for i in range (len(X_data[0])):
		print(i)
		data = prepare_data_with_remove_feature(i,X_data)
		testX = data[:120]
		testY = Y_data[:120]
		trainX = data[120:]
		trainY = Y_data[120:]
		te_err = train_model(3, trainX, trainY, testX, testY)
		te_err = np.mean(te_err)
		err_list.append(te_err)
	print(err_list)
	index = np.argmin(err_list)
	print('REMOVE INDEX')
	print(index)
	feature_removed.append(index)
	data = prepare_data_with_remove_feature(index, X_data)
	if(num_removed_features > 1):
		return rfe(num_removed_features-1, data, Y_data)
	else:
		return data

# function to prepare data with removed feature
def prepare_data_with_remove_feature(index, data):
	length = len(data[0])	
	data1 = data[0:,0:index]
	data2 = data[0:,index+1:length]
	data3 = np.zeros((len(data),length-1))
	data3[:,:len(data1[0])] = data1
	data3[:,len(data1[0]):] = data2
	return data3

# train model for 4-layer neural networks
def train_model4(trainX, trainY, testX, testY, dropOut):
	# Create the model
	x = tf.placeholder(tf.float32, [None, len(trainX[0])])
	y_ = tf.placeholder(tf.float32, [None, 1])

	if(dropOut):
		y, w1, w2, w3 = ffn4_dropout(x, 50,  len(trainX[0]))
	else:
		y, w1, w2, w3 = ffn4(x, 50,  len(trainX[0]))
	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) 
	error = tf.reduce_mean(tf.square(y - y_))
	loss = tf.reduce_mean(error + beta*regularization)
	train_op = optimizer.minimize(loss)

	# train
	N = len(trainX)
	idx = np.arange(N)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		te_err = []
		for i in range(epochs):
			np.random.shuffle(idx)
			trainX = trainX[idx]
			trainY = trainY[idx]

			for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
				train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

			te_err.append(error.eval(feed_dict={x:testX, y_:testY}))
		
			if i%1000 == 0:
				print('batch %d: iter %d, test error %g'%(batch_size, i, te_err[i]))

		return te_err

# train model for 5-layer neural networks
def train_model5(trainX, trainY, testX, testY, dropOut):
	# Create the model
	x = tf.placeholder(tf.float32, [None, len(trainX[0])])
	y_ = tf.placeholder(tf.float32, [None, 1])

	if(dropOut):
		y, w1, w2, w3, w4 = ffn5_dropout(x, 50,  len(trainX[0]))
	else:
		y, w1, w2, w3, w4 = ffn5(x, 50,  len(trainX[0]))
	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
	error = tf.reduce_mean(tf.square(y - y_))
	loss = tf.reduce_mean(error + beta*regularization)
	train_op = optimizer.minimize(loss)
	# train
	N = len(trainX)
	idx = np.arange(N)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		te_err = []
		for i in range(epochs):
			np.random.shuffle(idx)
			trainX = trainX[idx]
			trainY = trainY[idx]

			for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
				train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

			te_err.append(error.eval(feed_dict={x:testX, y_:testY}))
		
			if i%1000 == 0:
				print('batch %d: iter %d, test error %g'%(batch_size, i,  te_err[i]))

		return te_err

def main():
	print('Question number 1')
	#read and divide data into test and train sets 
	admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
	X, Y = admit_data[1:,1:8], admit_data[1:,-1]
	Y_data = Y.reshape(Y.shape[0], 1)

	X_data = (X - np.mean(X, axis=0))/ np.std(X, axis=0)

	# divide to training set and test set (70:30)
	testX = X_data[:120]
	testY = Y_data[:120]
	trainX = X_data[120:]
	trainY = Y_data[120:]

	tr_err, te_err, pred, target = train_model(1, trainX, trainY, testX, testY)

	print('Question Number 2')
	data = admit_data[1:,1:9]
	header = ['GRE', 'TOEFL', 'UniRating', 'SOP', 'LOR', 'CGPA', 'Research', 'Admit']
	df = pd.DataFrame(data)
	corr = df.corr()
	print(corr)

	print('Question Number 3')
	fivef_data = rfe(2, X_data, Y_data)
	print(feature_removed)
	sixf_data = prepare_data_with_remove_feature(feature_removed[0],X_data)
	trainX6, testX6 = sixf_data[120:], sixf_data[:120]
	trainX5, testX5 = fivef_data[120:], fivef_data[:120]

	#compare the accuracy of the model
	te_err7 = train_model(3,trainX, trainY, testX, testY)
	te_err6 = train_model(3, trainX6, trainY, testX6, testY)
	te_err5 = train_model(3, trainX5, trainY, testX5, testY)
	err7_mean = np.mean(te_err7)
	err6_mean = np.mean(te_err6)
	err5_mean = np.mean(te_err5)
	print(err7_mean)
	print(err6_mean)
	print(err5_mean)
	err_mean = [err7_mean, err6_mean, err5_mean]

	print('Question Number 4')
	if(np.argmin(err_mean) == 1):
		data = sixf_data
	elif(np.argmin(err_mean) == 2):
		data = fivef_data
	else:
		data = X_data
	trainXnn, testXnn = data[120:], data[:120]

	te_err3nn = train_model(3, trainXnn, trainY, testXnn, testY)
	te_err4nn = train_model4(trainXnn, trainY, testXnn, testY, False)
	te_err5nn = train_model5(trainXnn, trainY, testXnn, testY, False)
	te_err4nn_do = train_model4(trainXnn, trainY, testXnn, testY, True)
	te_err5nn_do = train_model5(trainXnn, trainY, testXnn, testY, True)
	print("mean error")
	print(np.mean(te_err3nn))
	print(np.mean(te_err4nn))
	print(np.mean(te_err5nn))
	print(np.mean(te_err4nn_do))
	print(np.mean(te_err5nn_do))

	# plot learning curves
	plt.figure(1)
	plt.plot(range(epochs), tr_err, label='Train error')
	plt.plot(range(epochs), te_err, label='Test error')
	plt.xlabel(str(epochs) + ' epochs')
	plt.ylabel('Mean square error')
	plt.title('Train and Test Errors against Epochs')
	plt.legend()

	plt.figure(2)
	plt.plot(target, 'b^', label='targets')
	plt.plot(pred, 'ro', label='predicted')
	plt.xlabel('Test data index')
	plt.ylabel('Y')
	plt.title('Target and Predicted Values')
	plt.legend()

	plt.figure(3)
	f = plt.figure(figsize=(19,15))
	plt.matshow(corr, fignum=f.number)
	plt.xticks(range(df.shape[1]),header, fontsize=10)
	plt.yticks(range(df.shape[1]),header, fontsize=12)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=12)
	plt.title('Correlation Matrix')

	plt.figure(4)
	plt.plot(range(epochs), te_err7, label='All features')
	plt.plot(range(epochs), te_err6, label='Six features')
	plt.plot(range(epochs), te_err5, label='Five features')
	plt.xlabel(str(epochs) + ' epochs')
	plt.ylabel('Mean square error')
	plt.title('Accuracy of models using different data features')
	plt.legend()

	plt.figure(5)
	plt.plot(range(epochs), te_err4nn, label='4-layer')
	plt.plot(range(epochs), te_err5nn, label='5-layer')
	plt.xlabel(str(epochs) + ' epochs')
	plt.ylabel('Mean square error')
	plt.title('Accuracy of models with different layers')
	plt.legend()

	plt.figure(6)
	plt.plot(range(epochs), te_err3nn, label='3-layer')
	plt.plot(range(epochs), te_err4nn, label='4-layer')
	plt.plot(range(epochs), te_err5nn, label='5-layer')
	plt.plot(range(epochs), te_err4nn_do, label='4-layer dropout')
	plt.plot(range(epochs), te_err5nn_do, label='5-layer dropout')
	plt.xlabel(str(epochs) + ' epochs')
	plt.ylabel('Mean square error')
	plt.title('Accuracy of models with different layers and dropouts')
	plt.legend()
	
	plt.show()
	

if __name__ == '__main__':
	main()
