import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from bayes_opt import BayesianOptimization


def eva(x_i):
    idx = np.where(x >= x_i)[0][0]
    y_i = y[idx]
    print(y_i)
    return y_i


warnings.filterwarnings('ignore')
path = "WTI_Baseline.npz"
r = np.load(path)
x = r['x']
y_mat = r['y_mat']
n_draws = r['n_draws']
y = np.mean(y_mat, axis=1)

bound_lo = np.min(x)
bound_up = np.max(x)

'''
n = len(x)
index_ori = np.linspace(1, n, n, dtype=int)
n_init = 10
index_init = np.random.choice(index_ori, n_init, replace=False)
x_init = x[index_init]
y_init = y[index_init]
index_lef = np.setdiff1d(index_ori, index_init)
'''

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




