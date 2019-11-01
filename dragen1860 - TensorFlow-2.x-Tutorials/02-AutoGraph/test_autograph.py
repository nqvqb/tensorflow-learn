
import  tensorflow as tf
import timeit

# AutoGraph helps you write complicated graph code using normal Python.
# Behind the scenes, AutoGraph automatically transforms your code into the equivalent TensorFlow graph code.
ReLU_Layer = tf.keras.layers.Dense(100, input_shape=(784,), activation=tf.nn.relu)
Logit_Layer = tf.keras.layers.Dense(10, input_shape=(100,))


# TensorFlow 1.0: Operations are added as nodes to the computational graph and are not actually executed until we call session.run(),
# much like defining a function that doesn't run until it is called.

#
# SGD_Trainer = tf.train.GradientDescentOptimizer(1e-2)

# inputs = tf.placeholder(tf.float32, shape=[None, 784])
# labels = tf.placeholder(tf.int16, shape=[None, 10])
# hidden = ReLU_Layer(inputs)
# logits = Logit_Layer(hidden)
# entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
# loss = tf.reduce_mean(entropy)
# train_step = SGD_Trainer.minimize(loss, var_list=ReLU_Layer.weights+Logit_Layer.weights)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# for step in range(1000):
#     sess.run(train_step, feed_dict={inputs:X, labels:y})


SGD_Trainer = tf.optimizers.SGD(1e-2)
@tf.function
def loss_fn(inputs=X, labels=y):
    hidden = ReLU_Layer(inputs)
    logits = Logit_Layer(hidden)
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    return tf.reduce_mean(entropy)

for step in range(1000):
    SGD_Trainer.minimize(loss_fn,
        var_list=ReLU_Layer.weights + Logit_Layer.weights)