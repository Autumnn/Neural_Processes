import numpy as np
import pandas as pd
from sklearn import preprocessing


df = pd.read_csv('data.csv', header=None)
y_ori = df.values
min_max_scalar = preprocessing.MinMaxScaler()
y = min_max_scalar.fit_transform(y_ori)*2-1
print(y)
#z_sample = tf.add(mu, tf.multiply(epsilon, sigma))