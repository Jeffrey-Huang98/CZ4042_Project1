#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
beta = 0.000001
epochs = 3000
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
def train_exp(batch_size, num_neurons, beta):
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

  # calculate the general fold size 
  fold_size = len(trainingX) // num_folds
  # calculate the number of surplus records
  surplus_size = len(trainingX) % num_folds

  # Create the model
  x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  # Build the graph for the deep net
  u, y, w1, w2 = ffn(x, num_neurons)

  regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)  
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u)
  loss = tf.reduce_mean(cross_entropy + beta*regularization)

  train = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = train.minimize(loss)

  correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(y_, 1)), tf.float32)
  error_prediction = tf.cast(tf.not_equal(tf.argmax(u, 1), tf.argmax(y_, 1)), tf.float32)
  error = tf.reduce_mean(error_prediction)
  accuracy = tf.reduce_mean(correct_prediction)

  start_ = 0
  train_err = []
  test_err = []
  validation_err = []
  train_acc = []
  test_acc = []
  validation_acc = []
  time_taken = 0

  for fold in range(num_folds):
    # divide the fold into 5 equal parts 
    if(surplus_size != 0):
      start, end = start_, (fold+1)*(fold_size + 1)
      start_ = end
      surplus_size -= 1
    else:
      start, end = start_, start_+fold_size
      start_ = end

    x_valid, y_valid = trainingX[start:end], trainingY[start:end]
    x_train  = np.append(trainingX[:start], trainingX[end:], axis=0)
    y_train = np.append(trainingY[:start], trainingY[end:], axis=0) 

    err_ = []
    # train
    N = len(x_train)
    idx = np.arange(N)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      
      time_to_update = 0
      for i in range(epochs):
        np.random.shuffle(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]

        t = time.time()
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
        # handle the data that is not used after batching  
        if(N % batch_size != 0):
          start = N//batch_size
          train_op.run(feed_dict={x:x_train[start:], y_: y_train[start:]})
        time_to_update += time.time() - t
        
        cur_train_err = error.eval(feed_dict={x:x_train, y_:y_train})        
        cur_test_err = error.eval(feed_dict={x:testX, y_:testY})
        cur_validation_err = error.eval(feed_dict={x:x_valid, y_:y_valid})
        cur_train_acc = accuracy.eval(feed_dict={x:x_train, y_:y_train})
        cur_test_acc = accuracy.eval(feed_dict={x:testX, y_:testY})
        cur_validation_acc = accuracy.eval(feed_dict={x:x_valid, y_:y_valid})
        
        if fold == 0:
          train_err.append(cur_train_err)
          test_err.append(cur_test_err)
          validation_err.append(cur_validation_err)  
          train_acc.append(cur_train_acc)
          test_acc.append(cur_test_acc)
          validation_acc.append(cur_validation_acc)
        else:
          train_err[i] += cur_train_err
          test_err[i] += cur_test_err
          validation_err[i] += cur_validation_err
          train_acc[i] += cur_train_acc
          test_acc[i] += cur_test_acc
          validation_acc[i] += cur_validation_acc
        if i % 100 == 0:
          print('fold %g batch %d: iter %d,  train_accuracy %g, validation_accuracy %g, test_accuracy %g'%(fold+1, batch_size, i, cur_train_acc, cur_validation_acc, cur_test_acc))
      
      time_taken += time_to_update/epochs
      
  # divide the accuracy element wise with the number of folds
  train_err = np.divide(train_err, num_folds)
  test_err = np.divide(test_err, num_folds)
  validation_err = np.divide(validation_err, num_folds)
  train_acc = np.divide(train_acc, num_folds)
  test_acc = np.divide(test_acc, num_folds)
  validation_acc = np.divide(validation_acc, num_folds)

  return train_acc, validation_acc, test_acc, train_err, validation_err, test_err, time_taken

def train_exp_ffn_4(batch_size, hidden1, hidden2, beta):

  # Create the model
  x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  # hidden layer 1
  w1 = tf.Variable(
    tf.truncated_normal([NUM_FEATURES, hidden1], stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
      name='weights')
  b1 = tf.Variable(tf.zeros([hidden1]),name='biases')
  h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

  # hidden layer 2
  w2 = tf.Variable(
    tf.truncated_normal([hidden1, hidden2], stddev=1.0 / np.sqrt(float(hidden1))),
      name='weights')
  b2 = tf.Variable(tf.zeros([hidden2]),name='biases')
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    
  # output layer
  w3 = tf.Variable(
      tf.truncated_normal([hidden2, NUM_CLASSES], stddev=1.0 / np.sqrt(float(hidden2))),
      name='weights')
  b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
  u = tf.matmul(h2, w3) + b3
  y = tf.nn.softmax(u)

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

  # calculate the general fold size 
  fold_size = len(trainingX) // num_folds
  # calculate the number of surplus records
  surplus_size = len(trainingX) % num_folds 

  regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)  + tf.nn.l2_loss(w3)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u)
  loss = tf.reduce_mean(cross_entropy + beta*regularization)

  train = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = train.minimize(loss)

  correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(y_, 1)), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  start_ = 0
  train_acc = []
  test_acc = []
  validation_acc = []
  time_taken = 0

  for fold in range(num_folds):
    if(surplus_size != 0):
      start, end = start_, (fold+1)*(fold_size + 1)
      start_ = end
      surplus_size -= 1
    else:
      start, end = start_, start_+fold_size
      start_ = end

    x_valid, y_valid = trainingX[start:end], trainingY[start:end]
    x_train  = np.append(trainingX[:start], trainingX[end:], axis=0)
    y_train = np.append(trainingY[:start], trainingY[end:], axis=0) 

    err_ = []
    # train
    N = len(x_train)
    idx = np.arange(N)


    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      
      time_to_update = 0
      for i in range(epochs):
        np.random.shuffle(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]

        t = time.time()
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
        # handle the data that is not used after batching
        if(N % batch_size != 0):
          start = N//batch_size
          train_op.run(feed_dict={x:x_train[start:], y_: y_train[start:]})
        time_to_update += time.time() - t

        cur_train_acc = accuracy.eval(feed_dict={x:x_train, y_:y_train})
        cur_test_acc = accuracy.eval(feed_dict={x:testX, y_:testY})
        cur_validation_acc = accuracy.eval(feed_dict={x:x_valid, y_:y_valid})
        
        if fold == 0:
          train_acc.append(cur_train_acc)
          test_acc.append(cur_test_acc)
          validation_acc.append(cur_validation_acc)
        else:
          train_acc[i] += cur_train_acc
          test_acc[i] += cur_test_acc
          validation_acc[i] += cur_validation_acc
        if i % 100 == 0:
          print('fold %g batch %d: iter %d,  train_accuracy %g, validation_accuracy %g, test_accuracy %g'%(fold+1, batch_size, i, cur_train_acc, cur_validation_acc, cur_test_acc))
      
      time_taken += time_to_update/epochs

  # divide the accuracy element wise with the number of folds
  train_acc = np.divide(train_acc, num_folds)
  test_acc = np.divide(test_acc, num_folds)
  validation_acc = np.divide(validation_acc, num_folds)
  time_taken /= num_folds

  return train_acc, validation_acc, test_acc, time_taken

def q1():
  #--------------------------------------------------------------------------------------------------------------------------------
  # Q1
  # initialize variables
  batch_size = 32
  num_neurons = 10
  beta = 1e-6

  # perform experiments
  train_acc, validation_acc, test_acc, train_err, validation_err, test_err, time_taken = train_exp(batch_size, num_neurons, beta)

  # plot learning curves
  plt.figure(1)
  plt.plot(range(epochs), train_acc, label='train accuracy')
  plt.plot(range(epochs), validation_acc, label='validation accuracy')
  plt.plot(range(epochs), test_acc, label='test accuracy')
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.title('Accuracy against epochs')
  plt.legend()
  plt.show()

  # plot learning curves
  plt.figure(2)
  plt.plot(range(epochs), test_err, label='test error')
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('error')
  plt.title('Test Error against epochs')
  plt.legend()
  plt.show()

def q2():
  # --------------------------------------------------------------------------------------------------------------------
  # Q2
  # initialize variables 
  batch_size = [4,8,16,32,64]
  num_neurons = 10
  beta = 1e-6
  train_err = []
  validation_err = []
  test_err = []
  train_acc = []
  validation_acc = []
  test_acc = []
  time_taken = []

  # perform experiments
  for batch in batch_size:
    acc_train, acc_validation, acc_test, err_train, err_validation, err_test, time_update = train_exp(batch, num_neurons, beta)
    train_err.append(err_train)
    validation_err.append(err_validation)
    test_err.append(err_test)
    train_acc.append(acc_train)
    validation_acc.append(acc_validation)
    test_acc.append(acc_test)
    time_taken.append(time_update)

  # Q2a
  # plot learning curves
  plt.figure(3)
  for i in range(len(batch_size)):
    plt.plot(range(epochs), validation_acc[i], label='batch %d'%batch_size[i])
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('cross-validation accuracy')
  plt.legend()
  plt.title('Cross-Validation Accuracy for Different Batch Sizes')
  plt.show()

  # plot learning curves
  plt.figure(4)
  plt.plot(time_taken)
  plt.xticks(range(5), batch_size)  
  plt.xlabel('batch size')
  plt.ylabel('time')
  plt.legend()
  plt.title('Time Taken per Epochs for Different Batch Sizes')
  plt.show()  

  # Q2c
  # plot learning curves
  for i in range (len(batch_size)):
    plt.figure(i+5)
    plt.plot(range(epochs), train_acc[i], label='train accuracy')
    plt.plot(range(epochs), test_acc[i], label='test accuracy')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracies against epochs for batch size %d'%batch_size[i])
    plt.show()

def q3():
  #-----------------------------------------------------------------------------------------------------------------------
  # Q3
  # initialize variables
  num_neurons = [5,10,15,20,25]
  batch_size = 64
  beta = 1e-6
  train_err = []
  validation_err = []
  test_err = []
  train_acc = []
  validation_acc = []
  test_acc = []
  time_taken = []

  for hidden in num_neurons:
    acc_train, acc_validation, acc_test, err_train, err_validation, err_test, time_update = train_exp(batch_size, hidden, beta)
    train_err.append(err_train)
    validation_err.append(err_validation)
    test_err.append(err_test)
    train_acc.append(acc_train)
    validation_acc.append(acc_validation)
    test_acc.append(acc_test)
    time_taken.append(time_update)

  # plot learning curves
  plt.figure(10)
  for i in range(len(num_neurons)):
    plt.plot(range(epochs), validation_acc[i], label='neurons %d'%num_neurons[i])
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('cross-validation accuracy')
  plt.legend()
  plt.title('Cross-Validation Accuracies for Different Number of Hidden Neurons')
  plt.show()

  # Q3c
  # plot learning curves
  for i in range (len(num_neurons)):
    plt.figure(i+11)
    plt.plot(range(epochs), train_acc[i], label='train accuracy')
    plt.plot(range(epochs), test_acc[i], label='test accuracy')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracies against epochs for %d hidden neurons'%num_neurons[i])
    plt.show()

def q4():
    #---------------------------------------------------------------------------------------------------------------------
  # Q4
  # Initialize variables
  num_neurons = 20
  batch_size = 32
  beta_values = [0, 1e-3, 1e-6, 1e-9, 1e-12]
  train_err = []
  validation_err = []
  test_err = []
  train_acc = []
  validation_acc = []
  test_acc = []
  time_taken = []

  for beta in beta_values:
    acc_train, acc_validation, acc_test, err_train, err_validation, err_test, time_update = train_exp(batch_size, num_neurons, beta)
    train_err.append(err_train)
    validation_err.append(err_validation)
    test_err.append(err_test)
    train_acc.append(acc_train)
    validation_acc.append(acc_validation)
    test_acc.append(acc_test)
    time_taken.append(time_update)

 # plot learning curves
  plt.figure(16)
  for i in range(len(beta_values)):
    plt.plot(range(epochs), validation_acc[i], label='beta %g'%beta_values[i])
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('cross-validation accuracy')
  plt.legend()
  plt.title('Cross-Validation Accuracies for Different Beta Values')
  plt.show()

  # Q4c
  # plot learning curves
  for i in range (len(beta_values)):
    plt.figure(i+17)
    plt.plot(range(epochs), train_acc[i], label='train accuracy')
    plt.plot(range(epochs), test_acc[i], label='test accuracy')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracies against epochs for beta %g'%beta_values[i])
    plt.show()

def q5():
  #-------------------------------------------------------------------------------------------------------------------
  # Q5
  # Initialize variables for 4-layer network
  num_neurons_4 = 10
  batch_size_4 = 32
  beta_4 = 1e-6
  train_acc_4 = []
  validation_acc_4 = []
  test_acc_4 = []
  time_taken_4 = []

  # perform experiments
  train_acc_4, validation_acc_4, test_acc_4, time_taken_4 = train_exp_ffn_4(batch_size_4, num_neurons_4, num_neurons_4, beta_4)

  # Initialize variables for 3-layer network
  num_neurons_3 = 20
  batch_size_3 = 64
  beta_3 = 1e-3
  train_err_3 = []
  validation_err_3 = []
  test_err_3 = []
  train_acc_3 = []
  validation_acc_3 = []
  test_acc_3 = []
  time_taken_3 = []

  # perform experiment
  train_acc_3, validation_acc_3, test_acc_3, train_err_3, validation_err_3, test_err_3, time_taken_3 = train_exp(batch_size_3, num_neurons_3, beta_3)

  # plot learning curves
  plt.figure(22)
  plt.plot(range(epochs), train_acc_4, label='train accuracy')
  plt.plot(range(epochs), validation_acc_4, label='validation accuracy')
  plt.plot(range(epochs), test_acc_4, label='test accuracy')
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.legend()
  plt.show()

  # plot comparison curves
  plt.figure(23)
  plt.plot(range(epochs), train_acc_4, label='4-layer')
  plt.plot(range(epochs), train_acc_3, label='3-layer')
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('training accuracy')
  plt.legend()
  plt.title('Training accuracy for 3-layer and 4-layer network')
  plt.show()

  # plot comparison curves
  plt.figure(24)
  plt.plot(range(epochs), test_acc_4, label='4-layer')
  plt.plot(range(epochs), test_acc_3, label='3-layer')
  plt.xlabel(str(epochs) + ' iterations')
  plt.ylabel('test accuracy')
  plt.legend()
  plt.title('Test accuracy for 3-layer and 4-layer network')
  plt.show()

def main():

  q1()
  q2()
  q3()
  q4()
  q5()

if __name__ == '__main__':
    main()

