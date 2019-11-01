
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# L1 norm loss/ Absolute loss function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Y_pred = tf.linspace(-1.0, 1.0, 500)
print('Y_pred', Y_pred.shape)
Y_truth = tf.constant(0.)
print('Y_truth', Y_truth)


# calculate a L1 loss
# L1-norm loss function, least absolute deviations (LAD)
# least absolute erros (LAE)
# Lasso Regression
# https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
# minimizing the sun of the abosoulte differencs between the target value and the estimated values

# absolute value
# L1 loss is better in detecting outliers than the L2 norm because it is not steep for very large values
# L1 loss is not smooth when close to the target/minimum and this can cause non-convergence for algorithms
loss_L1 = tf.abs(Y_truth - Y_pred)
#ploting the predicted values against the L2 loss
plt.plot(Y_pred, loss_L1, 'r-' )
plt.title('L1 loss')
plt.xlabel('$Y_{pred}$', fontsize=15)
plt.ylabel('$Y_{true}$', fontsize=15)
plt.show()
