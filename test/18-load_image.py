


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

num_classes = 10



# load from directory
# real world senario
# .flow_from_directory
data_gen_args = dict(featurewise_center=True, featurewise_std_normalization=True, rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

train_datagen = ImageDataGenerator(**data_gen_args)
test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

train_data = train_datagen.flow_from_directory('/home/' + os.environ.get('USERNAME') + '/datasets/cifar-10/train', target_size=(150, 150), batch_size=32, shuffle=True, class_mode='binary')
test_data = test_datagen.flow_from_directory('/home/' + os.environ.get('USERNAME') + '/datasets/cifar-10/test', target_size=(150, 150), batch_size=32, class_mode='binary')

for image_batch, label_batch in train_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

data_dir = '/home/' + os.environ.get('USERNAME') + '/datasets/cifar-10/train'
data_dir = pathlib.Path(data_dir)
print('data_dir', data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('image_count', image_count)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print('CLASS_NAMES', CLASS_NAMES)


def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')



image_batch, label_batch = next(train_generator)
show_batch(image_batch, label_batch)






# https://www.tensorflow.org/tutorials/load_data/images
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# https://www.tensorflow.org/guide/data

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# print('y_train_before', y_train.shape, 'y_test_after', y_test.shape)
# print('y_train_before_sample', y_train[0], 'y_test_after_sample', y_test[0])

# y_train = utils.np_utils.to_categorical(y_train, num_classes)
# y_test =  utils.np_utils.to_categorical(y_test, num_classes)
# print('y_train_before', y_train.shape, 'y_test_after', y_test.shape)
# print('y_train_after_sample', y_train[0], 'y_test_afer_sample', y_test[0])

# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# datagen.fit(x_train)
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
# for e in range(epochs):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
#         model.fit(x_batch, y_batch)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break













# data_gen_args = dict(featurewise_center=True, featurewise_std_normalization=True, rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)
# test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# Provide the same seed and keyword arguments to the fit and flow methods
# seed = 1
# image_datagen.fit(images, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)
# image_generator = image_datagen.flow_from_directory('data/images', class_mode=None, seed=seed)
# mask_generator = mask_datagen.flow_from_directory( 'data/masks', class_mode=None, seed=seed)
# combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)

























