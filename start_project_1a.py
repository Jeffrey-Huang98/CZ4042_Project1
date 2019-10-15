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

  # hidden layer  
  w1 = tf.Variable(
    tf.truncated_normal([NUM_FEATURES, hidden_units], stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
      name='weights')
  b1 = tf.Variable(tf.zeros([hidden_units]),name='biases')
  h = tf.nn.relu(tf.matmul(x, w1) + b1)
    
  # output layer
  w2 = tf.Variable(
      tf.truncated_normal([hidden_units, NUM_CLASSES], stddev=1.0 / np.sqrt(float(hidden_units))),
      name='weights')
  b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
  u = tf.matmul(h, w2) + b2
  y = tf.nn.softmax(u)
    
  return u, y, w1, w2

# train the network and find errors
def train_exp(batch_size):
  # read train data

  train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
  # divide 2D array to training set
  trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
  # normalize the X to [0,1]
  trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
  # create a matrix of dimension train_y row x 3 filled with zeros 
  trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
  # create classification matrix
  trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

  # divide the data into training set and test set (70:30)
  testX = trainX[:638]
  testY = trainY[:638]
  trainingX = trainX[638:]
  trainingY = trainY[638:]
  print(trainingY)


  # calculate the general fold size 
  fold_size = len(trainingX) // num_folds
  # calculate the number of surplus records
  surplus_size = len(trainingX) % num_folds

  # Create the model
  x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  # Build the graph for the deep net
  u, y, w1, w2 = ffn(x, num_neurons)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u)
  loss = tf.reduce_mean(cross_entropy)

  # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))

  train = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = train.minimize(loss)

  correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  start_ = 0
  acc = []


  for fold in range(num_folds):
    if(surplus_size != 0):
      start, end = start_, (fold+1)*(fold_size + 1)
      start_ = end
      surplus_size -= 1
    else:
      start, end = start_, start_+fold_size
      start_ = end

    x_test, y_test = trainingX[start:end], trainingY[start:end]
    x_train  = np.append(trainingX[:start], trainingX[end:], axis=0)
    y_train = np.append(trainingY[:start], trainingY[end:], axis=0) 
    print(y_train)

    
    
    err_ = []
    # train
    N = len(x_train)
    idx = np.arange(N)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      train_acc = []
      for i in range(epochs):
        np.random.shuffle(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]

        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
          
        train_acc.append(accuracy.eval(feed_dict={x:x_test, y_:y_test}))
        if i % 100 == 0:
          print('batch %d: iter %d,  accuracy %g'%(batch_size, i, train_acc[i]))
        # err.append(train_acc)
      
      acc.append(train_acc)
       
  # cv_err = np.mean(np.array(err), axis = 0)
  # print('cv errors {}'.format(cv_err))

  return acc


def main():

  # perform experiments
  train_acc = train_exp(32)

  # compute mean error
  # mean_err = np.mean(np.array(err), axis = 0)
  # print(mean_err)
  
  # plot learning curves
  plt.figure(1)
  for i in range(num_folds):
    plt.plot(range(epochs), train_acc[i], label='folds %g'% i)
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('Train accuracy')
  plt.legend()
  plt.show()



if __name__ == '__main__':
    main()

