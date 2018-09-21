import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

x_input = tf.placeholder(tf.float64, shape=(None, 1))
y_input = tf.placeholder(tf.float64, shape=(None, 1))


x1 = tf.contrib.layers.fully_connected(x_input, 100)
x2 = tf.contrib.layers.fully_connected(x1, 100)
y_output = tf.contrib.layers.fully_connected(x2, 1, activation_fn=None)

loss = tf.nn.l2_loss(y_output - y_input)


#optimizer = tf.train.AdadeltaOptimizer()
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()
sess.run(init)

n_iter = 10000
for i in range(n_iter):
    x_t = np.random.rand(100)*10
    y_t = np.sin(x_t)
    a = sess.run([train_op, loss], feed_dict={x_input: x_t[:, None], y_input: y_t[:, None]})
    if i%100 == 0:
        print(a[1])


t = np.linspace(0, 10, 100)
t_x = np.expand_dims(t, axis=1)
t_y = sess.run(y_output, feed_dict={x_input: t_x})

plt.scatter(t, np.sin(t), marker='o', color='#539caf', label='1', s=3, alpha=1)
plt.scatter(t, t_y, marker='+', color='r', label='2', s=3)
plt.savefig('compare.png')