
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, metrics
import pydot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create a dense model
input_size = 10
output_dense = 32
output_dense_1 = 10
dense_model = tf.keras.Sequential([
    layers.Dense(output_dense, input_shape=(input_size,), activation=tf.nn.relu, use_bias=False),
    layers.Dense(output_dense_1)
])

print('dense_model.summary()', dense_model.summary())
# assertion check on number of parameters
# dense layer, with bias, number of parameters in a layer will increase
assert output_dense * (input_size) == 320
# dense layer 1 without bias
assert output_dense_1 * (output_dense + 1) == 330

# dense_model parameters
#for layer in dense_model.layers:
#    print('layer.name', layer.name, 'layer.output_shape', layer.output_shape)
# plot the layers
# keras.utils.plot_model(dense_model, to_file='dense_model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)



# create a conv2D model
img_rows, img_cols = 224, 224
colors = 3
conv2d_filter_shape = 5
conv2d_feature_map = 4

conv2d_1_filter_shape = 50
conv2d_1_feature_map = 20

conv2d_2_filter_shape = 30
conv2d_2_feature_map = 1
conv2d_2_stride = 2

output_dense = 10

conv2d_model = tf.keras.Sequential([
    # receive image input, with 3 feature maps (channel)
    # img_rows,img_cols,colors, reshape the input if necessary
    layers.Reshape(target_shape=[img_rows,img_cols,colors], input_shape=(img_rows,img_cols,colors)),
    # shorter version to create a conv2D layer
    # where 4 is the number of feature maps
    # 5, shorted from (5,5) 5 by 5 filter
    # padding='same', the input and output feature map shape is the same
    # noted: bias only added to the final featuer map to allow it shift away from origin
    # no extra trainable params after add padding
    # filter parameters are always the same for each feature map
    # activation if not specified will become a(x) = x
    layers.Conv2D(conv2d_feature_map, conv2d_filter_shape, padding='same', activation=tf.nn.relu, use_bias=False),
    # with bias by default
    layers.Conv2D(conv2d_1_feature_map, (conv2d_1_filter_shape, conv2d_1_filter_shape), activation=tf.nn.relu),
    # strides dont change the parameters
    layers.Conv2D(conv2d_2_feature_map, (conv2d_2_filter_shape,conv2d_2_filter_shape), activation=tf.nn.relu, strides = conv2d_2_stride),
    layers.MaxPooling2D((2,2), (2,2)),
    # all the parameters
    layers.Flatten(),
    layers.Dense(output_dense)
])
print('conv2d_model.summary()', conv2d_model.summary())
# if with bias, bias only added to output feature map
# param # of conv2d layer
assert conv2d_feature_map * (colors * (conv2d_filter_shape * conv2d_filter_shape)) == 300
# output of conv2d layer
# param # of conv2d_1 layer
assert conv2d_1_feature_map * (conv2d_feature_map * (conv2d_1_filter_shape * conv2d_1_filter_shape) + 1) == 200020
# output of conv2d_1 layer
assert (img_rows - conv2d_1_filter_shape + 1) * (img_cols - conv2d_1_filter_shape + 1) == 175 * 175
# param # of conv2d_2 layer
assert conv2d_2_feature_map * (conv2d_1_feature_map * (conv2d_2_filter_shape * conv2d_2_filter_shape) + 1) == 18001
# output of conv2d_2 layer
# equation to calculate output size
padding =  0 -1
assert ((175 - conv2d_2_filter_shape + padding + conv2d_2_stride)/conv2d_2_stride) * \
       ((175 - conv2d_2_filter_shape + padding + conv2d_2_stride)/conv2d_2_stride) == 73 * 73
# output of maxpool layer
# 0
# equation to calculate output size
maxpoll_filter_shape = 2
maxpoll_stride = 2
assert ((73 - maxpoll_filter_shape + padding + maxpoll_stride)/conv2d_2_stride) * \
       ((73 - maxpoll_filter_shape + padding + maxpoll_stride)/conv2d_2_stride) == 36 * 36
# output of flatten layer
# 0
# equation to calculate output size
assert 36 * 36 == 1296
# output of dense_2 layer
# 0
# equation to calculate output size
assert output_dense * (1296 + 1) == 12970

