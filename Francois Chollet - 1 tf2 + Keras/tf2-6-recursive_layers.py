#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Layer


# layers can be recursively nested to create bigger computation blocks
# each layer tracks the wieghts of its sublayers (both trainable and non-trainable)

# using build

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


class MLP(Layer):
    """stacks of linear layers"""

    def __init__(self):
        super(MLP, self).__init__()

        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)


    def call(self, inputs):
        print('x_before_linear_1', inputs.shape)
        x = self.linear_1(inputs)
        print('x_after_linear_1', x.shape)
        x = tf.nn.relu(x)
        print('x_after_relu_1', x.shape)
        x = self.linear_2(x)
        print('x_after_linear_2', x.shape)
        x = tf.nn.relu(x)
        print('x_after_relu_2', x.shape)
        return self.linear_3(x)

mlp = MLP()
inputs = tf.ones(shape=(3, 64))
print('inputs.shape',inputs.shape)
y = mlp(inputs)
print('y.shape',y.shape)

assert len(mlp.weights) == 6