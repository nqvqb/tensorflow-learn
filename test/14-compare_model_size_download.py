import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow import keras

# no available resent20
# from tensorflow.keras.applications import ResNet20

resnet50 = ResNet50()
# total parameters 25,583,592
print(resnet50.summary())

# TODO: why vgg16 even more parameters?
# vgg16 = VGG16()
# Total params: 138,357,544
# print(vgg16.summary())

#keras.utils.plot_model(resnet50, 'resnet50.png', show_shapes=True)



# TODO: where the models are downloaded?
