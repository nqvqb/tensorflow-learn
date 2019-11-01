
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

# only take the training set
# (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
def mnist_dataset():
    (x, y), _ = datasets.mnist.load_data()
    # x.shape (60000, 28, 28) y.shape (60000,)
    print('x.shape', x.shape, 'y.shape', y.shape)
    print('x.dtype', x.dtype, 'y.dtype', y.dtype)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    # x: TensorSpec(shape=(28, 28), dtype=tf.uint8, name=None)
    # y: TensorSpec(shape=(), dtype=tf.uint8, name=None)
    # print('element_spec of x', ds.element_spec[0])
    # print('element_spec of y', ds.element_spec[1])
    return ds

model = keras.Sequential([
    layers.Reshape(target_shape=(28 * 28,),
                   input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10)
])

optimizer = optimizers.Adam()

# use @tf.function AutoGraph decorator to pre-compile our methods
# as tensorflow computational graphs

# tf 2.0 is fully imperative
# the AutoGraph decorator isn't necessary for our code to work
# but it speeds up executation
# take advantage of graph execution
@tf.function
def compute_loss(logits, labels):
    print('logits sparse_softmax_cross_entropy_with_logits',logits.shape)
    print('labels sparse_softmax_cross_entropy_with_logits', labels.shape)
    # logits and labels are different
    # logits is the output of a logistic function
    # label is not one-hop encoded, but like 1,2,3,4,5
    tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    print('loss_tmp', tmp.shape)
    # calculate reduce mean
    loss = tf.reduce_mean(tmp)
    return loss

@tf.function
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    tmp = tf.equal(predictions, labels)
    tmp = tf.cast(tmp, tf.float32)
    accuracy = tf.reduce_mean(tmp)
    return accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        print('logits', logits.shape)
        #
        # print(logits)
        loss = compute_loss(logits, y)
    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    print('grads', len(grads))
    # update weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy

def train(epoch, model, optimizer):

    train_ds = mnist_dataset()
    loss = 0.0
    accuracy = 0.0

    for step, (x,y) in enumerate(train_ds):
        loss, accuracy = train_one_step(model, optimizer, x, y)
        print('loss',loss.numpy())


        if step % 500 == 0:
            print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

        return loss, accuracy

for epoch in range(1):
  loss, accuracy = train(epoch, model, optimizer)

print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

