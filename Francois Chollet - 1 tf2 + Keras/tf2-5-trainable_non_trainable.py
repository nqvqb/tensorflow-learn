#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

# example of non_trainable weights
from tensorflow.keras.layers import Layer


class ComputeSum(Layer):
    """returns the sum of the inputs"""

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # create a non-trainable weight
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                 trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

my_sum = ComputeSum(2)
x = tf.ones((2,2))

y = my_sum(x)
print(y.numpy()) # [2, 2]

y = my_sum(x)
print(y.numpy()) # [4, 4]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []



