#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Layer

# dont have to use object-oriented programming all the time
# layers can be composed functionally
# called functional api

class Linear(Layer):
    """y = w.x + b"""

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)

        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        print('self.w.shape', self.w.shape, 'self.b.shape', self.b.shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Dropout(Layer):

    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

# We use an `Input` object to describe the shape and dtype of the inputs.
# This is the deep learning equivalent of *declaring a type*.
# The shape argument is per-sample; it does not include the batch size.
# The functional API focused on defining per-sample transformations.
# The model we create will automatically batch the per-sample transformations,
# so that it can be called on batches of data.

# inputs
inputs = tf.keras.Input(shape=(16,))
print('inputs',inputs.shape)

# We call layers on these "type" objects
# and they return updated types (new shapes/dtypes).

x = Linear(32)(inputs) # We are reusing the Linear layer we defined earlier.
x = Dropout(0.5)(x) # We are reusing the Dropout layer we defined earlier.
outputs = Linear(10)(x)
print('outputs',outputs.shape)

# A functional `Model` can be defined by specifying inputs and outputs.
# A model is itself a layer like any other.
model = tf.keras.Model(inputs, outputs)

# A functional model already has weights, before being called on any data.
# That's because we defined its input shape in advance (in `Input`).
assert len(model.weights) == 4



# Let's call our model on some data.
y = model(tf.ones((2, 16)))
assert y.shape == (2, 10)















































