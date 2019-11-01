#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Layer

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


# batchNormalization and Dropout layer have different behaviours during training and inference

class Dropout(Layer):

    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


class MLPWithDropout(Layer):

    def __init__(self):
        super(MLPWithDropout, self).__init__()
        self.linear_1 = Linear(32)
        self.dropout = Dropout(0.5)
        self.linear_3 = Linear(10)

    def call(self, inputs, training=None):
        print('x_before_linear_1', inputs.shape)
        x = self.linear_1(inputs)
        print('x_after_linear_1', x.shape)
        x = tf.nn.relu(x)
        print('x_after_relu_1', x.shape)
        x = self.dropout(x, training=training)
        print('x_after_dropout', x.shape)
        return self.linear_3(x)


mlp = MLPWithDropout()
y_train = mlp(tf.ones((2, 2)), training=True)
print('y_train.shape', y_train.shape)

y_test = mlp(tf.ones((2, 2)), training=False)
print('y_test.shape', y_test.shape)

print('y_train',y_train)
print('y_test',y_test)



