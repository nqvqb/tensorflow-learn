#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

# linear regression
input_dim = 2
output_dim = 1
learning_rate = 0.01

# weight matrix
w = tf.Variable(tf.random.uniform(shape=(input_dim, output_dim)))
print('w.shape', w.shape)
print('w', w.numpy)

# bias vector
b = tf.Variable(tf.zeros(shape=(output_dim,)))
print('b.shape', b.shape)
print('b', b.numpy)

def compute_predictions(features):
    # matrix multiplication
    return tf.matmul(features, w) + b

def compute_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))

# complie the training function into a static graph
# by adding tf.function decorator on it
@tf.function
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = compute_predictions(x)
        loss = compute_loss(y, predictions)
        # tape.gradient works with list as well [w, b]
        dloss_dw, dloss_db = tape.gradient(loss, [w,b])

    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)

    return loss


# generate some artificial data
import numpy as np
import random
import matplotlib.pyplot as plt
# prepare a dataset
# the actual sample size is 20 as half pos half neg
num_samples = 10
negative_samples = np.random.multivariate_normal(mean=[0,3], cov=[[1,0.5],[0.5,1]], size=num_samples)
positive_samples = np.random.multivariate_normal(mean=[3,0], cov=[[1,0.5],[0.5,1]], size=num_samples)
print('negative_samples', negative_samples.shape) # n * 2
print('positive_samples', positive_samples.shape) # n * 2

features = np.vstack((negative_samples, positive_samples)).astype(np.float32)
labels = np.vstack((np.zeros((num_samples,1), dtype='float32'),
                      np.ones((num_samples,1), dtype='float32')))
print('labels',labels.shape)
# show plots
# plt.scatter(features[:,0], features[:,1], c=labels[:,0])
# plt.show()

# shuffle the data
indices = np.random.permutation(len(features))
print('indices', indices.shape, indices[0], indices[1])
features = features[indices]
labels = labels[indices]

# create a tf.data.Dataset object for eacy batched iteration
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#
# TensorSliceDataset object
print('dataset',dataset)

# shuffle and batch the dataset
# buffer_size greater or equals to the size of original dataset
# batch:
dataset = dataset.shuffle(buffer_size=1024).batch(5)
print('dataset',dataset.element_spec)

import time
t0 = time.time()
for epoch in range(2):
    #print('current epoch', epoch)
    # total sample size / batch = step
    for step, (x, y) in enumerate(dataset):
        #print('step', step)
        #print(x,y)
        # w and b are updated globally
        loss = train_on_batch(x,y)
    #print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
t_end = time.time() - t0
print('Time per epoch: %.3f s' % (t_end / 20,))


















