import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing


batch_size = 2
num_total_points = 3
x_size = 1
y_size = 1
l1_scale = 0.4
sigma_scale = 1.0
sigma_noise=2e-2

sigma_f = tf.ones(shape=[batch_size, y_size]) * sigma_scale

x_values = tf.random_uniform([batch_size, num_total_points, x_size], -2, 2)
l1 = tf.ones(shape=[batch_size, y_size, x_size]) * l1_scale

xdata1 = tf.expand_dims(x_values, axis=1)  # [B, 1, num_total_points, x_size]
xdata2 = tf.expand_dims(x_values, axis=2)  # [B, num_total_points, 1, x_size]
diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
norm = tf.reduce_sum(norm, -1)

kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

# Add some noise to the diagonal to make the cholesky work.
noise = (sigma_noise ** 2) * tf.eye(num_total_points)
kernel += noise

sess = tf.Session()
print(sess.run(sigma_f))
print(sess.run(diff))
print(sess.run(norm))
print(sess.run(noise))
print(sess.run(kernel))


