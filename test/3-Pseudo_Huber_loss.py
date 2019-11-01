
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# L1 norm loss/ Absolute loss function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Y_pred = tf.linspace(-1.0, 1.0, 500)
print('Y_pred', Y_pred.shape)
Y_truth = tf.constant(0.)
print('Y_truth', Y_truth)

# Pseudo-huber loss is a variant of the Huber loss function
# It takes the best properties of the L1 and L2 loss by being convex close to the target and less steep for extreme values
# This loss depends on an extra parameter delta δ which dictates how steep the function will be
delta = tf.constant(0.24)
loss_pseudo_huber = tf.multiply(tf.square(delta),tf.sqrt(1. + tf.square((Y_truth - Y_pred) / delta)) - 1. )
#ploting the predicted values against the L2 loss
plt.plot(Y_pred, loss_pseudo_huber, 'g-' )
plt.title('Pseudo Huber loss')
plt.xlabel('$Y_{pred}$', fontsize=15)
plt.ylabel('$Y_{true}$', fontsize=15)
plt.show()
