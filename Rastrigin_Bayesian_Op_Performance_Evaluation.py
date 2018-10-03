import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from bayes_opt import BayesianOptimization


def rastrigin(x_1, x_2):
    A = 3
    return -(A + ((x_1**2 - A * np.cos(2 * math.pi * x_1)))+
             (x_2 ** 2 - A * np.cos(2 * math.pi * x_2)))


warnings.filterwarnings('ignore')

x_1 = np.linspace(-2, 6, 200)
x_2 = np.linspace(-2, 6, 200)

x_1, x_2 = np.meshgrid(x_1, x_2)
y = rastrigin(x_1, x_2)

bound_lo = np.min(x_1)
bound_up = np.max(x_1)

Bo = BayesianOptimization(rastrigin, {'x_1': (bound_lo, bound_up), 'x_2': (bound_lo, bound_up)})
Bo.maximize(init_points=40, n_iter=30)

target_list = Bo.res['all']['values']       # list --> [value_1, value_2, ..., value_n]
para_list = Bo.res['all']['params']         # list --> [{'x_1': , 'x_2':}, {'x_1': , 'x_2': }, ... , {'x_1': , 'x_2': }]
time_stamp = Bo.res['all']['timestamp']     # list --> [t_1, t_2, ..., t_n]
init_para = Bo.res['init']['params'][0]

n = len(target_list)
value_list = []
spot_location = np.zeros((n, 2))
max_spot = []
for i in range(n):
    if i == 0:
        max_value = target_list[i]
        value_list.append(max_value)
        max_spot.append(para_list[i]['x_1'])
        max_spot.append(para_list[i]['x_2'])
    else:
        if target_list[i] > max_value:
            max_value = target_list[i]
            max_spot[0] = para_list[i]['x_1']
            max_spot[1] = para_list[i]['x_2']
        value_list.append(max_value)
    spot_location[i, 0] = para_list[i]['x_1']
    spot_location[i, 1] = para_list[i]['x_2']


fig = plt.figure()
plt.plot(time_stamp, value_list, color='#539caf', linestyle='-', linewidth=0.6)
fig_name = 'Figures/Rastrigin_Bayesian_Op_Performance.png'
plt.savefig(fig_name)

fig = plt.figure()
con = plt.contourf(x_1, x_2, y, 10, cmap="PuBuGn")
fig.colorbar(con)
plt.scatter(init_para[:, 0], init_para[:, 1], marker='o', color='orchid', label='1', s=30, alpha=0.8)
plt.scatter(spot_location[:, 0], spot_location[:, 1], marker='+', color='r', label='1', s=30, alpha=0.8)
plt.scatter(max_spot[0], max_spot[1], marker='+', color='gold', label='1', s=30, alpha=1)
plt.xlim((bound_lo - 0.1, bound_up + 0.1))
plt.ylim((bound_lo - 0.1, bound_up + 0.1))
fig_name = 'Figures/Rastrigin_Bayesian_Op_Searched_Spots.png'
plt.savefig(fig_name)



