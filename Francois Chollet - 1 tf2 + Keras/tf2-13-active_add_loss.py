# !/usr/bin/python3

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras import layers

# Sometimes you need to compute loss values on the fly during a foward pass (especially regularization losses).
# Keras allows you to compute loss values at any time, and to recursively keep track of them via the add_loss method.
# Here's an example of a layer that adds a sparsity regularization loss based on the L2 norm of the inputs:


from tensorflow.keras.layers import Layer


class ActivityRegularization(Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, rate=1e-2):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        # We use `add_loss` to create a regularization loss
        # that depends on the inputs.
        self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))
        return inputs




class SparseMLP(Layer):
    """Stack of Linear layers with a sparsity regularization loss."""

    def __init__(self, output_dim):
        super(SparseMLP, self).__init__()
        self.dense_1 = layers.Dense(32, activation=tf.nn.relu)
        self.regularization = ActivityRegularization(1e-2)
        self.dense_2 = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.regularization(x)
        return self.dense_2(x)


mlp = SparseMLP(1)
y = mlp(tf.ones((10, 10)))

print(mlp.losses)  # List containing one float32 scalar




# These losses are cleared by the top-level layer at the start of each forward pass -- they don't accumulate.
# So layer.losses always contain only the losses created during the last forward pass.
# You would typically use these losses by summing them before computing your gradients when writing a training loop.

# Losses correspond to the *last* forward pass.
mlp = SparseMLP(1)
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1  # No accumulation.

# Let's demonstrate how to use these losses in a training loop.

# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype('float32') / 255, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# A new MLP.
mlp = SparseMLP(10)

# Loss and optimizer.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:
        # Forward pass.
        logits = mlp(x)

        # External loss value for this batch.
        loss = loss_fn(y, logits)

        # Add the losses created during the forward pass.
        loss += sum(mlp.losses)

        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss, mlp.trainable_weights)

    # Update the weights of our linear layer.
    optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))

    # Logging.
    if step % 100 == 0:
        print('Loss at step %d: %.3f' % (step, loss))

