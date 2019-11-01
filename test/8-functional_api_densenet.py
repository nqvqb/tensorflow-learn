

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# more fiexiable than sequentral
# can handle with non-linear topology
# models with shared layers
# models with multiple inputs or outputs

# deep learning model is usually a directed acyclic graph (DAG)
# the functional API a set of tools for building graphs of layers

# https://www.tensorflow.org/guide/keras/functional

# consider the following model:
# (input: 784-dimensional vectors)
#        ↧
# [Dense (64 units, relu activation)]
#        ↧
# [Dense (64 units, relu activation)]
#        ↧
# [Dense (10 units, softmax activation)]
#        ↧
# (output: probability distribution over 10 classes)

# it's a simple graph of 3 layers
# functional api
# start of input node
# batc size is always ommited
# only specify the shape of each sample
inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation='relu')
x = dense(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

mnist_model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
mnist_model.summary()
# get model plot
# keras.utils.plot_model(mnist_model, 'my_first_model.png')

# prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

mnist_model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
history = mnist_model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)
print('history', history)
test_scores = mnist_model.evaluate(x_test, y_test, verbose=2)

print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
mnist_model.save('mnist_model.h5')
del mnist_model
# Recreate the exact same model purely from the file:
mnist_model = keras.models.load_model('mnist_model.h5')
mnist_model.summary()


#  all models are callable just like layers
# treat any model as they are layers
# calling a model not just resusing the architecture of the model
# but also reusing its weights

encoder_input = keras.Input(shape=(28, 28, 1), name='original_img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

decoder_input = keras.Input(shape=(16,), name='encoded_img')
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

decoder = keras.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name='img')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
autoencoder.summary()

