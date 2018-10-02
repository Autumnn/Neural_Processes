import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

'''
batch_size = 2
num_total_points = 7
num_target = 4
num_context = 3
x_size = 1
y_size = 1
l1_scale = 0.4
sigma_scale = 1.0
sigma_noise = 2e-2


#x_values = tf.random_uniform([5, ], -2, 2)
x = np.random.uniform(-2, 2, size=[num_total_points, 1])
#x = np.random.randint(2, 8, size=[num_total_points, 1])

xdata1 = tf.constant(x, tf.float32)
xdata2 = tf.constant(x.T, tf.float32)
diff = xdata1 - xdata2
noise = (sigma_noise ** 2) * tf.eye(num_total_points)
norm = tf.square(diff + noise)
kernel = tf.exp(-0.5 * norm)
#kernel = tf.square(diff)
Cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)
y_values = tf.matmul(Cholesky, tf.random_normal([num_total_points, 1]))

sess = tf.Session()
print(x)
print(sess.run(xdata1))
print(sess.run(xdata2))
print(sess.run(diff))
print(sess.run(noise))
print(sess.run(norm))
print(sess.run(kernel))
print(sess.run(Cholesky))
print(sess.run(y_values))

idx = tf.random_shuffle(tf.range(num_target))
context_x = tf.gather(x, idx[:num_context], axis=0)
print(sess.run(idx))
print(sess.run(context_x))

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
'''


x = np.random.normal(0,1,20)
print(x)

ind = np.argpartition(x, -10)
print(ind)

s = ind[-10:]
print(s)

x_s = x[s]
print(x_s)


