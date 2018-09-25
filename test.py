import numpy as np
import pandas as pd
from sklearn import preprocessing

x = [{'x_i': 1}, {'x_i': 2}, {'x_i': 3}, {'x_i': 4}]
print(x[0:2])
for item in x[0:2]:
    print(item['x_i'])