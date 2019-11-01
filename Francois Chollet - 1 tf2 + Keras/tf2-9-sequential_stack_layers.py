#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Layer

from tensorflow.keras import Sequential

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




model = Sequential([
    Linear(32),
    Dropout(0.5),
    Linear(10)
])



























