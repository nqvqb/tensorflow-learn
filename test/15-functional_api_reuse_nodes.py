
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.applications import VGG19

#  graph of layers you are manipulating in the Functional API is a static datastructure

vgg19 = VGG19()

features_list = [layer.output for layer in vgg19.layers]
