#!/usr/bin/python3

import tensorflow as tf
print(tf.__version__)

# tensor variable used to store mutable state
initial_value = tf.random.normal(shape=(2,2))
a = tf.Variable(initial_value)
print(a)
print(type(a))

# update the value of a variable by using methods like assign(value), assign_add(increment), assign_sub(decrement)
new_value = tf.random.normal(shape=(2,2))
a.assign(new_value)
print(a)
# assign value one by one
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i ,j]
print(a)

# added value
added_value = tf.random.normal(shape=(2,2))
a.assign_add(added_value)
print(a)
# assign value one by one
for i in range (2):
    for j in range(2):
        assert a[i,j] == new_value[i,j] + added_value[i, j]
print(a)


# computing gradients with gradient tape
a = tf.random.normal(shape=(2,2))
b = tf.random.normal(shape=(2,2))
#print(a)
#print(b)

with tf.GradientTape() as tape:
    # start recording the history of operations applied to a
    # by default all variables are watched automatically
    tape.watch(a)

    # do some math using a
    c = tf.sqrt(tf.square(a) + tf.square(b))
    # what's the gradient of c with respect to a
    dc_da = tape.gradient(c, a)
    print('gradient tape dc_da', dc_da)

# calculate the gradient of the above manually
# without using gradient tape
print('gradient calculate manually', tf.sqrt(1/(tf.square(a) + tf.square(b))) * a)


# without explicit watch, gradient is automatically watched
a = tf.Variable(a)
with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
    print('gradient tape dc_da', dc_da)


# nest higher order derivatives
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
    d2c_da2 = outer_tape.gradient(dc_da, a)
    print('nested d2c_da2', d2c_da2)


# linear regression
input_dim = 2
output_dim = 1
learning_rate = 0.01
# weight matrix
w = tf.Variable(tf.random.uniform(shape=(input_dim, output_dim)))
# bias vector














