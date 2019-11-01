

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Layer


class Linear(Layer):

    def __init__(self, units, input_dim, output_dim):
        super(Linear, self).__init__()

        w_init = tf.random_normal_initializer()

        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype='float32'),
            trainable=True)
        print('self.w.shape', self.w.shape)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype='float32'),
            trainable=True)
        print('self.b.shape', self.b.shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b