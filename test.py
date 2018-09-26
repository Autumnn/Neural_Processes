import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

x_1 = np.linspace(0, 3, 20)
x_1 = np.expand_dims(x_1, axis=1)
print(x_1)

sess = tf.Session()
x_target = tf.placeholder(tf.float32, shape=(None, 1))
p_normal = tf.distributions.Normal(loc=0.0, scale=1.0)
p_star = p_normal.log_prob(x_target)
p_star_1 = p_normal.prob(x_target)
p_c = tf.concat([p_star_1, p_star], axis=1)

print(sess.run(p_c, {x_target: x_1}))

y = tf.nn.moments(p_c, axes=[1])
print(sess.run(y, {x_target: x_1}))

