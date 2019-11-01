
import os
import time
import numpy as np
import tensorflow as tf

mean_metric = tf.keras.metrics.Mean()
mean_metric.update_state(2.0)
mean_metric.update_state(3.0)
mean_metric.update_state(4.0)

# print the average result = 3.0
print('mean_metric.result() after update', mean_metric.result().numpy())

# reset the metric
mean_metric.reset_states()
print('mean_metric.result() after reset', mean_metric.result().numpy())