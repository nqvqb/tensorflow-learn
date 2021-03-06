
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# building a system for ranking custom issue tickets by priority and routing them to the right department

# You model will have 3 inputs:
# Title of the ticket (text input)
# Text body of the ticket (text input)
# Any tags added by the user (categorical input)

# It will have two outputs:
# Priority score between 0 and 1 (scalar sigmoid output)
# The department that should handle the ticket (softmax output over the set of departments)
# Let's built this model in a few lines with the Functional API.


# number of unique issue tags
num_tags = 12
# size of vocabulary obtained when preprocessing text data
num_words = 10000
# number of departments for predictions
num_departments = 4


# variable-len sequence of ints
title_input = keras.Input(shape=(None,), name='title')
# Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name='body')
# Binary vectors of size `num_tags`
tags_input = keras.Input(shape=(num_tags,), name='tags')


# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)


# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)


# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])


# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, activation='softmax', name='department')(x)


# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(inputs=[title_input, body_input, tags_input], outputs=[priority_pred, department_pred])

# keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
model.summary()

#
model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss=['binary_crossentropy', 'categorical_crossentropy'], loss_weights=[1., 0.2])
# or
model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss={'priority': 'binary_crossentropy', 'department': 'categorical_crossentropy'}, loss_weights=[1., 0.2])

# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets}, epochs=2, batch_size=32)




