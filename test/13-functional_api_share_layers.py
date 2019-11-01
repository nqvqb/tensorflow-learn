
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# unctional API are models that use shared layers
# Shared layers are layer instances that get reused multiple times in a same model:
# they learn features that correspond to multiple paths in the graph-of-layers.

# parameters are also shared ?


# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype='int32')

# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype='int32')

# We reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)