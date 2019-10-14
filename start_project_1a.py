#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 5000
batch_size = 32
num_neurons = 10
num_folds = 5
seed = 10
np.random.seed(seed)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# build a feedforward network
def ffn(x, hidden_units):
    
  with tf.name_scope('hidden'):
    weights = tf.Variable(
      tf.truncated_normal([NUM_FEATURES, hidden_units], stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),name='biases')
    h = tf.nn.relu(tf.matmul(x, weights) + biases)
    
  with tf.name_scope('output'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units, NUM_CLASSES], stddev=1.0 / np.sqrt(float(hidden_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    u = tf.nn.softmax(tf.matmul(h, weights) + biases)
    
  return u

# train the network and find errors
def train_exp():
  #read train data

  train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
  #divide 2D array to training set
  trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
  #normalize the X to [0,1]
  trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
  #create a matrix of dimension train_y row x 3 filled with zeros 
  trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
  #create classification matrix
  trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

  # experiment with small datasets
  # divide to training set and test set (70:30)
  testX = trainX[:638]
  testY = trainY[:638]
  trainX = trainX[638:]
  trainY = trainY[638:]

  err = []
  for fold in range(num_folds):
      start, end = fold*20, (fold+1)*20
      x_test, y_test = trainX[start:end], trainY[start:end]
      x_train  = np.append(trainX[:start], trainX[end:], axis=0)
      y_train = np.append(trainY[:start], trainY[end:], axis=0) 

      err_ = []
      for no_hidden in num_neurons:
          x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
          y_ = tf.placeholder(tf.float32, [None, 1])

          y = ffn(x, no_hidden)

          # Create the model

          loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))
      
          train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
      
          # train
          with tf.Session() as sess:
              tf.global_variables_initializer().run()

              for i in range(epochs):
                  train.run(feed_dict={x:x_train, y_: y_train})
                  
              err_.append(loss.eval(feed_dict={x:x_test, y_:y_test}))
      
      err.append(err_)

  
  cv_err = np.mean(np.array(err), axis = 0)
  print('cv errors {}'.format(cv_err))

  return cv_err




# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Build the graph for the deep net
	
weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
logits  = tf.matmul(x, weights) + biases

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    for i in range(epochs):
        train_op.run(feed_dict={x: trainX, y_: trainY})
        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g'%(i, train_acc[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

