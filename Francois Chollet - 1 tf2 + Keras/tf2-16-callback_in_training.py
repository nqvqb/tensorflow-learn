# !/usr/bin/python3

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras import layers

#Callbacks
#One of the neat features of fit (besides built-in support for sample weighting and class weighting) is that
# you can easily customize what happens during training and evaluation by using callbacks.

# A callback is an object that is called at different points during training (e.g. at the end of every batch or at the
# end of every epoch) and takes actions, such as saving a model, mutating variables on the model, loading a checkpoint, stopping training, etc.

# There's a bunch of built-in callback available, like ModelCheckpoint to save your models after each epoch during training,
# or EarlyStopping, which interrupts training when your validation metrics start stalling.

# And you can easily write your own callbacks.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255

num_val_samples = 10000
x_val = x_train[-num_val_samples:]
y_val = y_train[-num_val_samples:]
x_train = x_train[:-num_val_samples]
y_train = y_train[:-num_val_samples]



# Instantiate a simple classification model
model = tf.keras.Sequential([
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

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[accuracy])

# Instantiate some callbacks
callbacks = [tf.keras.callbacks.EarlyStopping(),
             tf.keras.callbacks.ModelCheckpoint(filepath='my_model.keras',
                                                save_best_only=True)]

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=30,
          batch_size=64,
          callbacks=callbacks)