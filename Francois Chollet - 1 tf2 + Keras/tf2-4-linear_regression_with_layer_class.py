#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Layer

# Layer
# encapsulates a state / weight
# and some computation, defined in the call method

class Linear(Layer):
    """Y = WX + B"""

    # init with number of sample and input dimension
    def __init__(self, units, input_dim, output_dim):
        super(Linear, self).__init__()

        w_init = tf.random_normal_initializer()

        self.w = tf.Variable(
            initial_value = w_init(shape=(input_dim, output_dim), dtype='float32'),
            trainable=True)
        print('self.w.shape', self.w.shape)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value = b_init(shape=(output_dim,), dtype='float32'),
            trainable=True)
        print('self.b.shape', self.b.shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# instantiate layer
# init with number of sample and input dimension
# TODO: remove the need to specify input dimension
input_dim = 3
output_dim = 2
number_of_sample = 1000
linear_layer = Linear(number_of_sample, input_dim, output_dim)

# inputs
inputs = tf.ones((number_of_sample, input_dim)) # dimension of input, number of inputs
print('inputs.shape', inputs.shape)

# call the inear layer object will call the call function automatically
y = linear_layer(inputs)
print(y.shape)
assert y.shape == (number_of_sample, output_dim)

# weights are automatically tracked under the "weights" property
# variables like weights and bias, are tracked automatically
# as a list
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
print('linear_layer.w.shape', linear_layer.w.shape)
print('linear_layer.b.shape', linear_layer.b.shape)


# use build function to build the weights upon call
# build is called lazily with the shape of the first inputs seen by layer
# prevents us from having to specify input_dim in the constructors
# this is a good practice to create weights in a seperate build method
class Linear(Layer):
    """y = w.x b"""

    def __init__(self, units):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        # the shape of last dimension(?)
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        print('self.w.shape', self.w.shape)
        print('input_shape',input_shape)

        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        print('self.b.shape', self.b.shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# input_dim = 3
# output_dim = 2
number_of_sample = 4
# instantiate our lazy layer
linear_layer = Linear(4)

y = linear_layer(tf.ones((2,3)))

print(y.shape)


# TODO: modify into linear regression, change the shape of weights and inputs and bias





