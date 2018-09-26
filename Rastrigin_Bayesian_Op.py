import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from bayes_opt import BayesianOptimization


def rastrigin(x_1, x_2):
    A = 6
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
Bo.maximize(init_points=10, n_iter=30)

target_list = Bo.res['all']['values']
para_list = Bo.res['all']['params']

n = len(target_list)
for i in range(n):
    fig = plt.figure()
    con = plt.contourf(x_1, x_2, y, 10, cmap="RdBu_r")
    fig.colorbar(con)
    if i != 0:
        x_1_p = []
        x_2_p = []
        for item in para_list[0:i]:
            x_1_p.append(item['x_1'])
            x_2_p.append(item['x_2'])
        plt.scatter(x_1_p, x_2_p, marker='o', color='lightblue', label='1', s=16, alpha=1)
    plt.scatter(para_list[i]['x_1'], para_list[i]['x_2'], marker='+', color='yellow', label='1', s=40, alpha=1)
    plt.xlim((bound_lo, bound_up))
    plt.ylim((bound_lo, bound_up))
    fig_name = 'Figures/Rastrigin_Bayesian_Op_iter_' + str(i) + '.png'
    plt.savefig(fig_name)


'''
warnings.filterwarnings('ignore')
path = "WTI_Baseline.npz"
r = np.load(path)
x = r['x']
y_mat = r['y_mat']
n_draws = r['n_draws']
y = np.mean(y_mat, axis=1)

bound_lo = np.min(x)
bound_up = np.max(x)

Bo = BayesianOptimization(eva, {'x_i': (bound_lo, bound_up)})
Bo.maximize(init_points=10, n_iter=30)

target_list = Bo.res['all']['values']
para_list = Bo.res['all']['params']

n = len(target_list)
for i in range(n):
    if i != 0:
        x_t = []
        for item in para_list[0:i]:
            x_t.append(item['x_i'])
        plt.scatter(x_t, target_list[0:i], marker='o', color='#539caf', label='1', s=16, alpha=1)
    plt.scatter(para_list[i]['x_i'], target_list[i], marker='+', color='r', label='1', s=40, alpha=1)
    plt.plot(x, y, color='g', linestyle='--', linewidth=0.6)
    fig_name = 'Figures/WTI_experiment_Bayesian_Op_iter_' + str(i) + '.png'
    plt.savefig(fig_name)

'''



