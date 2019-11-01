# !/usr/bin/python3

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras import layers

# You don't normally have to define by hand how to update your variables during gradient descent,
# like we did in our initial linear regression example.
# You would usually use one of the built-in Keras optimizer, like SGD, RMSprop, or Adam.
#
# Here's a simple MNSIT example that brings together loss classes, metric classes, and optimizers.


# Prepare a dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

x_train = x_train[:].reshape(60000, 784).astype('float32') / 255
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# instantiate a simple classification model
model = tf.keras.Sequential([
    # layers.Dense: output = activation(dot(input, kernel) + bias)
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(10)
])

# Instantiate a logistic loss function that expects integer targets.
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Instantiate an accuracy metric.
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam()

# Iterate over the batches of the dataset.
for step, (x, y) in enumerate(dataset):
    # Open a GradientTape.
    with tf.GradientTape() as tape:

        # Forward pass.
        logits = model(x)

        # Loss value for this batch.
        loss_value = loss(y, logits)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    print('gradients', len(gradients))
    # Update the weights of our linear layer.
    # update the weights with the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Update the running accuracy.
    accuracy.update_state(y, logits)

    # Logging.
    if step % 100 == 0:
        print('Step:', step)
        print('Loss from last step: %.3f' % loss_value)
        print('Total running accuracy so far: %.3f' % accuracy.result())



x_test = x_test[:].reshape(10000, 784).astype('float32') / 255
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(128)

accuracy.reset_states()  # This clears the internal state of the metric

for step, (x, y) in enumerate(test_dataset):
  logits = model(x)
  accuracy.update_state(y, logits)

print('Final test accuracy: %.3f' % accuracy.result())