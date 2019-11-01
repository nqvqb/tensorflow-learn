import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)
tf.debugging.set_log_device_placement(True)


# function API makes it easy to manipulate non-linear connectivity topologies
# non-linear topologies: layers are not connected sequentially
# cannot be handled with sequential api

# residual connections
# TODO: change input
# input block
inputs = keras.Input(shape=(32, 32, 3), name='face')
x = layers.Conv2D(64, 7, activation='relu', strides=2, padding='same')(inputs)
x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
block_0_output = layers.BatchNormalization()(x)
# block_1_1
x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_0_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
block_1_1_output = layers.add([x, block_0_output])

# block_1_2
x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
block_1_2_output = layers.add([x, block_1_1_output])

# block_2_1
x = layers.Conv2D(128, 3, activation='relu', padding='same')(block_1_2_output)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
# shortcut using 1by1 conv2d
block_1_2_output = layers.Conv2D(128, 1, activation='relu', padding='same')(block_1_2_output)
block_1_2_output = layers.BatchNormalization()(block_1_2_output)

block_2_1_output = layers.add([x, block_1_2_output])
# block_2_2
x = layers.Conv2D(128, 3, activation='relu', padding='same')(block_2_1_output)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
block_2_2_output = layers.add([x, block_2_1_output])

# block_3_1
x = layers.Conv2D(256, 3, activation='relu', padding='same')(block_2_2_output)
x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
# shortcut using 1by1 conv2d
block_2_2_output = layers.Conv2D(256, 1, activation='relu', padding='same')(block_2_2_output)
block_2_2_output = layers.BatchNormalization()(block_2_2_output)
block_3_1_output = layers.add([x, block_2_2_output])
# block_3_2
x = layers.Conv2D(256, 3, activation='relu', padding='same')(block_3_1_output)
x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
block_3_2_output = layers.add([x, block_3_1_output])

# block_4_1
x = layers.Conv2D(512, 3, activation='relu', padding='same')(block_3_2_output)
x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
# shortcut using 1by1 conv2d
block_3_2_output = layers.Conv2D(512, 1, activation='relu', padding='same')(block_3_2_output)
block_3_2_output = layers.BatchNormalization()(block_3_2_output)

block_4_1_output = layers.add([x, block_3_2_output])
# block_4_2
x = layers.Conv2D(512, 3, activation='relu', padding='same')(block_4_1_output)
x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
block_4_2_output = layers.add([x, block_4_1_output])

# average pool softmax outputs
x = layers.GlobalAveragePooling2D()(block_4_2_output)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

resnet18 = keras.Model(inputs, outputs, name='resnet18')
resnet18.summary()

keras.utils.plot_model(resnet18, 'resnet18.png', show_shapes=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

resnet18.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss='categorical_crossentropy', metrics=['acc'])
resnet18.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2)
