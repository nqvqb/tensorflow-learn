import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import resnet

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.callbacks import ModelCheckpoint

home_dir = os.getenv("HOME")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

dataset_name = 'cifar-10'
NUM_CLASSES = 10
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)
# training meta
NUM_EPOCHS = 200
FREQ_SAVE_CHECKPOINT = 20
BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 2), (0.01, 50), (0.001, 300)]
# machine meta
NUM_GPUS = 1
BS_PER_GPU = 1024
# batch size, TODO: gpu memory
BATCH_SIZE = BS_PER_GPU * NUM_GPUS

# learning rate scheduler
def schedule(epoch):
    initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
        if epoch >= start_epoch:
            learning_rate = initial_learning_rate * mult
        else:
            break
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate




if NUM_GPUS == 1:
    model = resnet.resnet20(img_input=img_input, classes=NUM_CLASSES)
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model = resnet.resnet20(img_input=img_input, classes=NUM_CLASSES)

# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
#               loss='sparse_categorical_crossentropy',
#               metrics=['sparse_categorical_accuracy'])
# use SGD
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

# TODO
# tensorboard_callback = TensorBoard(log_dir=log_dir, update_freq='batch', histogram_freq=1)

lr_schedule_callback = LearningRateScheduler(schedule)


data_gen_args = dict(rescale=1./255,
                     rotation_range=15,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     channel_shift_range=0.1,
                     brightness_range=[0.5, 1.5],
                     shear_range=10,
                     fill_mode='nearest',
                     horizontal_flip=True)
train_datagen = ImageDataGenerator(**data_gen_args)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(home_dir + '/datasets/'+dataset_name+'/train', target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, shuffle=True, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(home_dir + '/datasets/'+dataset_name+'/test', target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, class_mode='binary')

data_dir = home_dir + '/datasets/'+dataset_name+'/train'
data_dir = pathlib.Path(data_dir)
print('data_dir', data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('image_count', image_count)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print('CLASS_NAMES', CLASS_NAMES)

def show_batch(image_batch, label_batch):
    square = int(math.ceil(math.sqrt(BATCH_SIZE)))
    plt.axis('off')
    fig, axs = plt.subplots(square, square, figsize=(20,20))
    n = 0
    for i in range(square):
        for j in range(square):
            try:
                axs[i, j].title.set_text(str(int(label_batch[n])))
                axs[i, j].imshow(image_batch[n])
            except IndexError:
                continue
            n +=1
    fig.savefig('dataset_preview.png')

image_batch, label_batch = next(train_generator)
print('image_batch', image_batch.shape)
print('label_batch', label_batch.shape)

show_batch(image_batch, label_batch)

model.summary()
# keras.utils.plot_model(model, 'resnet20.png', show_shapes=True)

# get current utc timestamp
st = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]
path_checkpoint_best = home_dir + '/models/'+dataset_name+'/train/'+st+'/best'
path_checkpoint_period = home_dir + '/models/'+dataset_name+'/train/'+st+'/{epoch:04d}'

# save best checkpoint
cp_callback_best = ModelCheckpoint(path_checkpoint_best, monitor='val_loss', verbose=1, save_best_only=True)
# save checkpoints periodically
cp_callback_period = ModelCheckpoint(path_checkpoint_period, verbose=1, period=FREQ_SAVE_CHECKPOINT)

model.fit(train_generator,
          epochs=NUM_EPOCHS,
          validation_data=validation_generator,
          validation_freq=1,
          callbacks=[lr_schedule_callback, cp_callback_best, cp_callback_period]
          )


#model.save(dataset_name + '.h5')
#new_model = keras.models.load_model(dataset_name + '.h5')
#snew_model.evaluate(validation_generator)













