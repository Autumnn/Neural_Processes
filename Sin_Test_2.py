import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def custom_objective(y_pred, y_real):
    #loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, y_real)))
    loss = tf.nn.l2_loss(y_pred-y_real)
    return loss

x = np.linspace(-4, 4, 100)
#print(x.shape)
x_a = np.expand_dims(x, axis=1)
#print(x_a.shape)
#print(x)

y = np.sin(x_a)
#y = np.square(x_a)
#print(y)
#print(y.shape)
#learning_rate = 0.01
sess = tf.Session()

'''
N = len(y)
ori = np.linspace(1, N, N, dtype=int)
print(ori)
N_context = np.random.randint(1, 4, 1)
print(N_context[0])
sel = np.random.choice(N, N_context, replace=False) + 1
print(sel, x[sel-1])
lef = np.setdiff1d(ori, sel)
print(lef, x[lef-1])
'''

x_input = tf.placeholder(tf.float64, shape=(None, 1))
y_input = tf.placeholder(tf.float64, shape=(None, 1))

h_1 = tf.layers.dense(x_input, 100, tf.nn.sigmoid, name='encoder_1', reuse=tf.AUTO_REUSE)
#h_2 = tf.layers.dense(h_1, 100, tf.nn.sigmoid, name='encoder_2', reuse=tf.AUTO_REUSE)
#y_output = tf.layers.dense(h_1, 1, tf.nn.relu, name='encoder_3', reuse=tf.AUTO_REUSE)
y_output = tf.layers.dense(h_1, 1, name='encoder_3', reuse=tf.AUTO_REUSE)


loss = custom_objective(y_output, y_input)
#loss = tf.constant(msq, dtype=tf.float32)

#optimizer = tf.train.AdadeltaOptimizer()
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

n_iter = 10000
for i in range(n_iter):
    #x_t = np.random.rand(100)*10
    #y_t = np.sin(x_t)
    #a = sess.run([train_op, loss], feed_dict={x_input: x_t[:, None], y_input: y_t[:, None]})

    a = sess.run([train_op, loss], feed_dict={x_input: x_a, y_input: y})
    if i%100 == 0:
        print(a[1])

t = np.linspace(-4, 4, 100)
#print(x.shape)
t_x = np.expand_dims(t, axis=1)
t_y = sess.run(y_output, feed_dict={x_input: t_x})

plt.scatter(x, y, marker='o', color='#539caf', label='1', s=3, alpha=1)
plt.scatter(t, t_y, marker='+', color='r', label='2', s=3)
plt.savefig('compare.png')