#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

# construct an constant tensor
x = tf.constant([[5, 2],
                [1, 2]])
print('constant tensor', x)

# get tensor value by callinn numpy()
x_numpy = x.numpy()
print(x_numpy)
print(type(x_numpy))

# wiht numpy attributes
print('dtype', x.dtype)
print('shape', x.shape)

# create zeros and ones
print(tf.ones(shape=(2,1)))
print(tf.zeros(shape=(2,1)))

# random normal distributed number
print(tf.random.normal(shape=(2,2), mean=0., stddev=1.))
# integer version
print(tf.random.uniform(shape=(2,2), minval=0, maxval=10, dtype='int32'))

#