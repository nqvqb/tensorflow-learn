
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import math
home_dir = os.getenv("HOME")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x/255.) - 0.5) * 2.
        return x



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)
tf.debugging.set_log_device_placement(True)

# TODO:
# the model has too many feature map, makes it hard to train
#


# function API makes it easy to manipulate non-linear connectivity topologies
# non-linear topologies: layers are not connected sequentially
# cannot be handled with sequential api

# residual connections
# TODO: change input
# input block
inputs = keras.Input(shape=(32, 32, 3), name='face')
x = layers.Conv2D(64, 7, activation='relu', strides=2, padding='same',use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
block_0_output = layers.BatchNormalization()(x)
# block_1_1
x = layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_0_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
block_1_1_output = layers.add([x, block_0_output])

# block_1_2
x = layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_1_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
block_1_2_output = layers.add([x, block_1_1_output])

# block_2_1
x = layers.Conv2D(128, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_1_2_output)
x = layers.Conv2D(128, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
# shortcut using 1by1 conv2d
block_1_2_output = layers.Conv2D(128, 1, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_1_2_output)
block_1_2_output = layers.BatchNormalization()(block_1_2_output)

block_2_1_output = layers.add([x, block_1_2_output])
# block_2_2
x = layers.Conv2D(128, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_2_1_output)
x = layers.Conv2D(128, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
block_2_2_output = layers.add([x, block_2_1_output])

# block_3_1
x = layers.Conv2D(256, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_2_2_output)
x = layers.Conv2D(256, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
# shortcut using 1by1 conv2d
block_2_2_output = layers.Conv2D(256, 1, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_2_2_output)
block_2_2_output = layers.BatchNormalization()(block_2_2_output)
block_3_1_output = layers.add([x, block_2_2_output])
# block_3_2
x = layers.Conv2D(256, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_3_1_output)
x = layers.Conv2D(256, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
block_3_2_output = layers.add([x, block_3_1_output])

# block_4_1
x = layers.Conv2D(512, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_3_2_output)
x = layers.Conv2D(512, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
# shortcut using 1by1 conv2d
block_3_2_output = layers.Conv2D(512, 1, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_3_2_output)
block_3_2_output = layers.BatchNormalization()(block_3_2_output)

block_4_1_output = layers.add([x, block_3_2_output])
# block_4_2
x = layers.Conv2D(512, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(block_4_1_output)
x = layers.Conv2D(512, 3, activation='relu', padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
block_4_2_output = layers.add([x, block_4_1_output])

# average pool softmax outputs
x = layers.GlobalAveragePooling2D()(block_4_2_output)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

resnet18 = keras.Model(inputs, outputs, name='resnet18')
resnet18.summary()

keras.utils.plot_model(resnet18, 'resnet18.png', show_shapes=True)

NUM_TRAINING_SAMPLE = 50000
NUM_EPOCHES = 50
BATCH_SIZE = 64
STEPS_PER_EPOCH = int(NUM_TRAINING_SAMPLE/BATCH_SIZE)
print('STEPS_PER_EPOCH', STEPS_PER_EPOCH)

# TODO: visualize more validation

data_gen_args = dict(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
                     channel_shift_range=0.1, fill_mode='nearest',horizontal_flip=True)
train_datagen = ImageDataGenerator(**data_gen_args)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(home_dir + '/datasets/cifar-10/train', target_size=(32, 32), batch_size=BATCH_SIZE, shuffle=True, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(home_dir + '/datasets/cifar-10/test', target_size=(32, 32), batch_size=BATCH_SIZE, class_mode='binary')


data_dir = home_dir + '/datasets/cifar-10/train'
data_dir = pathlib.Path(data_dir)
print('data_dir', data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('image_count', image_count)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print('CLASS_NAMES', CLASS_NAMES)


def show_batch(image_batch, label_batch):
    square = int(math.ceil(math.sqrt(BATCH_SIZE)))
    plt.axis('off')
    fig, axs = plt.subplots(square, square, figsize=(15,15))
    n = 0
    for i in range(square):
        for j in range(square):
            axs[i, j].title.set_text(str(int(label_batch[n])))
            axs[i, j].imshow(image_batch[n])
            n +=1
    fig.savefig('dataset_preview.png')

image_batch, label_batch = next(train_generator)
print('image_batch', image_batch.shape)
print('label_batch', label_batch.shape)

show_batch(image_batch, label_batch)

opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
resnet18.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
# deprecated
# resnet18.fit_generator(train_generator, epochs=NUM_EPOCHES, validation_data=validation_generator, validation_freq=1)
resnet18.fit(train_generator, epochs=NUM_EPOCHES, validation_data=validation_generator, validation_freq=1)
