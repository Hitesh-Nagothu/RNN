#!/usr/bin/env python
# coding: utf-8

# Import all required libraries

# In[3]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Create the Sine Wave

# In[4]:


minimum=0
maximum=60
data_points= np.linspace(minimum, maximum, (maximum-minimum)*10)
dataset=np.sin(data_points)


# Normalizing the SInewave, range from 0 to 1

# In[8]:


dataset=dataset.reshape(-1,1)
scaler= MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)


# Initialize hyper-parameters and generate data

# In[12]:


def create_training_dataset(dataset, n_steps, n_outputs):
    dataX, dataY = [], []
    for i in range(500):
        x = dataset[i]
        y = dataset[i+1]
        dataX.append(x)
        dataY.append(y)
    dataX, dataY =  np.array(dataX), np.array(dataY)
    dataX = np.reshape(dataX, (-1, n_steps, n_outputs))
    dataY = np.reshape(dataY, (-1, n_steps, n_outputs))    
    return dataX, dataY


n_steps=100
n_iterations=10000
n_inputs = 1
n_neurons = 120
n_outputs =1
learning_rate = 0.0001
dataset= dataset.reshape(-1,)
dataX, dataY = create_training_dataset(dataset, n_steps, n_outputs)


# Setting up the Computational Graph

# In[ ]:


X= tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y= tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)


# In[ ]:


# initialize all variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = dataX, dataY
        # prediction dimension [batch_size x t_steps x n_inputs]
        _, prediction =sess.run((training_op, outputs), feed_dict={X: X_batch, y: y_batch})
        if iteration % 20 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE", mse)
            # roll out prediction dimension into a single dimension
            prediction = np.reshape(prediction, (-1,))
            plt.plot(prediction)
            plt.title('prediction over training data')
            plt.show()
            
            # simulate the prediction for some time steps
            #sequence = [0.]*n_steps
            num_batches = X_batch.shape[0]
            sequence = X_batch[num_batches-1,:,:].reshape(-1).tolist()
            prediction_iter = 100
            for iteration in range(prediction_iter):
                X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
                y_pred = sess.run(outputs, feed_dict={X: X_batch})
                sequence.append(y_pred[0, -1, 0])
            plt.plot(sequence[-prediction_iter:])
            plt.title('prediction')
            plt.show()


# In[ ]:




