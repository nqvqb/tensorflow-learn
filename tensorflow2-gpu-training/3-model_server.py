

from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

model = keras.models.load_model('model.h5')

# '/1' specifies the version of a model, or "servable" we want to export
path_to_saved_model = 'servable'

# Saving the keras model in SavedModel format
keras.experimental.export_saved_model(model, path_to_saved_model)

# Load the saved keras model back
restored_saved_model = keras.experimental.load_from_saved_model(path_to_saved_model)