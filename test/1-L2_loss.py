
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u/47227886
# If you don't have a GPU and want to utilize CPU as much as possible,
# you should build tensorflow from the source optimized for your CPU with AVX, AVX2,
# and FMA enabled if your CPU supports them.
# It's been discussed in this question and also this GitHub issue.
# Tensorflow uses an ad-hoc build system called bazel and building it is not that trivial,
# but is certainly doable. After this, not only will the warning disappear,
# tensorflow performance should also improve
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create sample data with linspace from -1.0 to 1.0 (float)
Y_pred = tf.linspace(-1.0, 1.0, 500)
print('Y_pred', Y_pred.shape)
# create traget as zero constant tensro
Y_truth = tf.constant(0.)
print('Y_truth', Y_truth)

# calculate l2 loss
# L2-norm loss function
# least squares error (LSE)
# Ridge Regression
# https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
# minimizing the sum of the square of the differences between the target value and the estimated values

# pros: converge near the target
# cons: converge slowly near the target, avoid over-shooting the minimum

# TODO: try tf.nn.l2_loss()
loss_L2 = tf.square(Y_truth - Y_pred)
print('loss_L2', loss_L2.shape)
# visualise
#plt.plot(Y_pred, loss_L2, 'b-', label='loss L2')
plt.title('L2 loss')
plt.xlabel('$Y_{pred}$', fontsize=15)
plt.ylabel('$Y_{true}$', fontsize=15)
plt.show()
