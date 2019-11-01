


import tensorflow as tf
import matplotlib.pyplot as plt
import os

# L1 norm loss/ Absolute loss function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Y_pred = tf.linspace(-4., 6., 500)
Y_label = tf.constant(1.)
Y_labels = tf.fill([500,], 1.)

# Cross entropy loss is sometimes referred to as the logistic loss function.
# Cross entropy loss for binary classification is used when we are predicting two classes 0 and 1.
# Here we wish to measure the distance from the actual class (0 or 1) to the predicted value,
# which is usually a real number between 0 and 1.

#applying sigmoid
x_entropy_loss = - tf.multiply(Y_label, tf.math.log(Y_pred)) - tf.multiply((1. - Y_label), tf.math.log(1. - Y_pred))
#ploting the predicted values against the cross entropy loss
plt.plot(Y_pred, x_entropy_loss, 'r-' )
plt.title('Cross entropy loss')
plt.xlabel('$Y_{pred}$', fontsize=15)
plt.ylabel('$Y_{label}$', fontsize=15)
plt.ylim(-2, 5)
plt.show()





import tensorflow as tf
import matplotlib.pyplot as plt
import os

# L1 norm loss/ Absolute loss function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Y_pred = tf.linspace(-4., 6., 500)
Y_label = tf.constant(1.)
Y_labels = tf.fill([500,], 1.)

# very similar to the cross entropy loss function
# except that we transform the x-values by the sigmoid function before applying the cross entropy loss

loss_x_entropy_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(labels= Y_labels, logits=Y_pred)
#ploting the predicted values against the Sigmoid cross entropy loss
plt.plot(Y_pred, loss_x_entropy_sigmoid, 'y-' )
plt.title('Sigmoid cross entropy loss')
plt.xlabel('$Y_{pred}$', fontsize=15)
plt.ylabel('$Y_{label}$', fontsize=15)
plt.ylim(-2, 5)
plt.show()


# weighted version of the sigmoid cross entropy loss function
weight = tf.constant(1.)
loss_x_entropy_weighted_sigmoid = tf.nn.weighted_cross_entropy_with_logits(labels=Y_labels, logits=Y_pred, pos_weight=weight)

weight2 = tf.constant(0.5)
loss_x_entropy_weighted_sigmoid_1 = tf.nn.weighted_cross_entropy_with_logits(labels=Y_labels, logits=Y_pred, pos_weight=weight2)

#ploting the predicted values against the Sigmoid cross entropy loss
plt.plot(Y_pred, loss_x_entropy_weighted_sigmoid, 'b-', label=' weight = 1.0' )
plt.plot(Y_pred, loss_x_entropy_weighted_sigmoid_1, 'r--', label='weight = 0.5' )
plt.title('Weighted cross entropy loss')
plt.legend(loc=4)
plt.xlabel('$Y_{pred}$', fontsize=15)
plt.ylabel('$Y_{label}$', fontsize=15)
plt.ylim(-2, 5)
plt.show()