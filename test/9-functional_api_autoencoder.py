

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://www.tensorflow.org/guide/keras/functional

# use the same stack of layers to instantiate two models:
# an encoder model that turns image inputs into 16-dimensional vectors,
# and an end-to-end autoencoder model for training
print('encoder')
encoder_input = keras.Input(shape=(28,28,1), name='img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
# global pull from 16 feature maps
# turn image into 16-dimensional vectors
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

print('decoder')
x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)


print('autoencoder')
autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()

