import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

x_1 = np.linspace(0, 1, 10)
x_1 = np.expand_dims(x_1, axis=1)
print(x_1)

sess = tf.Session()
x_1_i = tf.placeholder(tf.float32, shape=(None, 1))
p_normal = tf.distributions.Normal(loc=0.0, scale=1.0)
p_star = p_normal.log_prob(x_1_i)

print(sess.run(p_star, {x_1_i: x_1}))


