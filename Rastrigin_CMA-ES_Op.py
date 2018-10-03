import os
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings

from cma_es import CmaEs


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

pop_num = 5
iter_num = 30
cma = CmaEs(rastrigin, {'x_1': (bound_lo, bound_up), 'x_2': (bound_lo, bound_up)})
time_stamp, target_list, para_list, pop_list = cma.run(max_iter=iter_num, pop_size=pop_num, sigma=0.8)

print(target_list[-1], para_list[-1])

folder_name = 'Figures/Rastrigin_CMA-ES_Op_' + str(pop_num) + '_pop_' + str(iter_num) + '_iter'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

n = len(target_list)
x_1_p = []
x_2_p = []
for i in range(n):
    fig = plt.figure()
    con = plt.contourf(x_1, x_2, y, 10, cmap="PuBuGn")
    fig.colorbar(con)
    if i != 0:
        plt.scatter(x_1_p, x_2_p, marker='o', color='orchid', label='1', s=30, alpha=0.6)
    for item in pop_list[i]:
        x_1_p.append(item[0])
        x_2_p.append(item[1])
    plt.scatter(x_1_p[-pop_num:], x_2_p[-pop_num:], marker='+', color='r', label='1', s=30, alpha=1)
    plt.scatter(para_list[i][0], para_list[i][1], marker='+', color='gold', label='1', s=30, alpha=1)
    plt.xlim((bound_lo-0.1, bound_up+0.1))
    plt.ylim((bound_lo-0.1, bound_up+0.1))
    fig_name = folder_name + '/Rastrigin_CMA-ES_Op_iter_' + str(i) + '.png'
    plt.savefig(fig_name)


fig = plt.figure()
plt.plot(time_stamp, target_list, color='#539caf', linestyle='-', linewidth=0.6)
fig_name = 'Figures/Rastrigin_CMA-ES_Op_Performance_' + str(pop_num) + '_pop_' + str(iter_num) + '_iter.png'
plt.savefig(fig_name)

fig = plt.figure()
con = plt.contourf(x_1, x_2, y, 10, cmap="PuBuGn")
fig.colorbar(con)
plt.scatter(x_1_p[0:pop_num], x_2_p[0:pop_num], marker='o', color='r', label='1', s=30, alpha=0.8)
plt.scatter(x_1_p[pop_num:], x_2_p[pop_num:], marker='+', color='orchid', label='1', s=30, alpha=0.6)
plt.scatter(para_list[-1][0], para_list[-1][1], marker='+', color='gold', label='1', s=30, alpha=1)
plt.xlim((bound_lo - 0.1, bound_up + 0.1))
plt.ylim((bound_lo - 0.1, bound_up + 0.1))
fig_name = 'Figures/Rastrigin_CMA-ES_Op_Searched_Spots_' + str(pop_num) + '_pop_' + str(iter_num) + '_iter.png'
plt.savefig(fig_name)