import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
#from neural_processes import NeuralProcess
from conditional_neural_processes_my import NeuralProcess


def rastrigin(x_1, x_2):
    A = 3
    return -(A + ((x_1**2 - A * np.cos(2 * math.pi * x_1)))+
             (x_2 ** 2 - A * np.cos(2 * math.pi * x_2)))


num_star = 200
x_1 = np.linspace(-2, 6, num_star)
x_2 = np.linspace(-2, 6, num_star)
x_1, x_2 = np.meshgrid(x_1, x_2)
y = rastrigin(x_1, x_2)

target_list = []
time_stamp = []
para_list = []

start_time = datetime.now()
num_init = 10
bound_lo = np.min(x_1)
bound_up = np.max(x_1)
x_init = np.random.uniform(bound_lo, bound_up, size=(num_init, 2))
#x_init_plot = x_init
y_init = rastrigin(x_init[:,0], x_init[:,1])
y_init = np.expand_dims(y_init, axis=1)


elapse_time = (datetime.now() - start_time).total_seconds()
time_stamp.append(elapse_time)
y_init_max = np.max(y_init)
target_list.append(y_init_max)
id_x = np.where(y_init == y_init_max)[0]
x_init_max = np.squeeze(x_init[id_x])
para_list.append(x_init_max.tolist())


dim_r = 4
dim_h_hidden = 128
dim_g_hidden = 128

sess = tf.Session()
x_context = tf.placeholder(tf.float32, shape=(None, 2))
y_context = tf.placeholder(tf.float32, shape=(None, 1))
x_target = tf.placeholder(tf.float32, shape=(None, 2))
y_target = tf.placeholder(tf.float32, shape=(None, 1))
neural_process = NeuralProcess(x_context, y_context, x_target, y_target, dim_r, dim_h_hidden, dim_g_hidden)
train_op_and_loss = neural_process.init_NP(learning_rate = 0.001)

init = tf.global_variables_initializer()
sess.run(init)

n_iter = 10001
plot_freq = 1000

for iter in range(n_iter):
    N_context = np.random.randint(1, num_init, 1)
    # create feed_dict containing context and target sets
    feed_dict = neural_process.helper_context_and_target(x_init, y_init, N_context, x_context, y_context, x_target, y_target)
    # optimisation step
    a = sess.run(train_op_and_loss, feed_dict= feed_dict)
    if iter % plot_freq == 0:
        print(a[1])
print('Initialization Completed !')

n_op_iter = 6
for iter_op in range(n_op_iter):
    num_candidate = 100
    x_candidate = np.random.uniform(bound_lo, bound_up, size=(num_candidate, 2))
    predict_candidate = neural_process.posterior_predict(x_init, y_init, x_candidate)
    _, y_candidate_mu, y_candidate_sigma = sess.run(predict_candidate)

    num_select = 10
    y_candidate_mu = np.squeeze(y_candidate_mu)
    y_candidate_sigma = np.squeeze(y_candidate_sigma)
    ind_mu = np.argpartition(y_candidate_mu, -num_select)[-num_select:]
    x_mu_select = x_candidate[ind_mu]
    ind_sigma = np.argpartition(y_candidate_sigma, -num_select)[-num_select:]
    x_sigma_select = x_candidate[ind_sigma]

    x_select = np.unique(np.concatenate((x_mu_select, x_sigma_select), axis=0), axis=0)
    _, idx_d = np.unique(np.concatenate((x_init, x_select), axis=0), axis=0, return_index=True) #remove the same item
    idx_d = idx_d - x_init.shape[0]
    idx_d = np.delete(idx_d, np.where(idx_d < 0))
    x_select = x_select[idx_d]
    y_select = rastrigin(x_select[:, 0], x_select[:, 1])
    y_select = np.expand_dims(y_select, axis=1)
    x_init = np.concatenate((x_init, x_select), axis=0)
    y_init = np.concatenate((y_init, y_select), axis=0)
    num_iter = x_init.shape[0]

    elapse_time = (datetime.now() - start_time).total_seconds()
    time_stamp.append(elapse_time)
    y_select_max = np.max(y_select)
    print("The Max value is: ", y_select_max)
    target_list.append(y_select_max)
    id_s = np.where(y_select == y_select_max)[0]
    x_select_max = np.squeeze(x_select[id_s])
    para_list.append(x_select_max.tolist())

    for iter in range(n_iter):
        N_context = np.random.randint(1, num_iter, 1)
        # create feed_dict containing context and target sets
        feed_dict = neural_process.helper_context_and_target(x_init, y_init, N_context, x_context, y_context, x_target,
                                                             y_target)
        # optimisation step
        a = sess.run(train_op_and_loss, feed_dict=feed_dict)
        if iter % plot_freq == 0:
            print(a[1])
    print('The %d-th iteration is Completed!' %iter_op)

n = len(target_list)
value_list = []
spot_location = np.zeros((n, 2))
max_spot = []
for i in range(n):
    if i == 0:
        max_value = target_list[i]
        value_list.append(max_value)
        max_spot.append(para_list[i][0])
        max_spot.append(para_list[i][1])
    else:
        if target_list[i] > max_value:
            max_value = target_list[i]
            max_spot[0] = para_list[i][0]
            max_spot[1] = para_list[i][1]
        value_list.append(max_value)
    spot_location[i, 0] = para_list[i][0]
    spot_location[i, 1] = para_list[i][1]


fig = plt.figure()
plt.plot(time_stamp, value_list, color='#539caf', linestyle='-', linewidth=0.6)
fig_name = 'Figures/Rastrigin_CNPs_Op_Performance.png'
plt.savefig(fig_name)

fig = plt.figure()
con = plt.contourf(x_1, x_2, y, 10, cmap="PuBuGn")
fig.colorbar(con)
plt.scatter(x_init[:, 0], x_init[:, 1], marker='o', color='orchid', label='1', s=30, alpha=0.8)
plt.scatter(spot_location[:, 0], spot_location[:, 1], marker='+', color='r', label='1', s=30, alpha=0.8)
plt.scatter(max_spot[0], max_spot[1], marker='+', color='gold', label='1', s=30, alpha=1)
plt.xlim((bound_lo - 0.1, bound_up + 0.1))
plt.ylim((bound_lo - 0.1, bound_up + 0.1))
fig_name = 'Figures/Rastrigin_CNPs_Op_Searched_Spots.png'
plt.savefig(fig_name)

