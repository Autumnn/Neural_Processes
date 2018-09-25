import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from bayes_opt import BayesianOptimization
from neural_processes import NeuralProcess


path = "WTI_Baseline.npz"
r = np.load(path)
x = r['x']
#x = np.expand_dims(x, axis=1)
y_mat = r['y_mat']
n_draws = r['n_draws']
y = np.mean(y_mat, axis=1)
y = np.expand_dims(y, axis=1)
bound_lo = np.min(x)
bound_up = np.max(x)

n = len(x)
index_ori = np.linspace(1, n, n, dtype=int)-1
n_init = 10
index_init = np.random.choice(index_ori, n_init, replace=False)
x_init = x[index_init]
y_init = y[index_init]
index_lef = np.setdiff1d(index_ori, index_init)

dim_r = 2
dim_z = 2
dim_h_hidden = 8
dim_g_hidden = 8

sess = tf.Session()
x_context = tf.placeholder(tf.float32, shape=(None, 1))
y_context = tf.placeholder(tf.float32, shape=(None, 1))
x_target = tf.placeholder(tf.float32, shape=(None, 1))
y_target = tf.placeholder(tf.float32, shape=(None, 1))
neural_process = NeuralProcess(x_context, y_context, x_target, y_target,
                                  dim_r, dim_z, dim_h_hidden, dim_g_hidden)

train_op_and_loss = neural_process.init_NP(learning_rate = 0.001)

init = tf.global_variables_initializer()
sess.run(init)

n_iter = 1000
x_star_temp = np.linspace(bound_lo, bound_up, n)
x_star = np.expand_dims(x_star_temp, axis=1)
eps_value = np.random.normal(size=(n_draws, dim_r))
epsilon = tf.constant(eps_value, dtype=tf.float32)
predict_op = neural_process.posterior_predict(x_init, y_init, x_star, epsilon=epsilon, n_draws=n_draws)

for iter in range(n_iter):
    N_context = np.random.randint(2, n_init, 1)
    # create feed_dict containing context and target sets
    feed_dict = neural_process.helper_context_and_target(x_init, y_init, N_context, x_context, y_context, x_target, y_target)
    # optimisation step
    a = sess.run(train_op_and_loss, feed_dict= feed_dict)
    if iter % 100 == 0:
        print(a[1])

y_star_mat = sess.run(predict_op['mu'])
plt.plot(x, y, color='g', linestyle='--', linewidth=0.6)
for ii in range(n_draws):
    plt.plot(x_star, y_star_mat[:, ii], color='#539caf')
plt.scatter(x_init, y_init, marker='+', color='r', label='1', s=30, alpha=1)
fig_name = 'Figures/WTI_experiment_Bayesian_Op_iter_init.png'
plt.savefig(fig_name)

print('Initialization Completed !')

n_op_iter = 101
for iter_op in range(n_op_iter):
    y_star_mean = np.max(y_star_mat, axis=1)
    y_max = np.max(y_star_mean)
    idx_sel = np.where(y_star_mean == y_max)

    x_init = np.concatenate((x_init, x[idx_sel]))
    y_init = np.concatenate((y_init, y[idx_sel]))
    n_init += 1

    for iter in range(n_iter):
        N_context = np.random.randint(2, n_init, 1)
        # create feed_dict containing context and target sets
        feed_dict = neural_process.helper_context_and_target(x_init, y_init, N_context, x_context, y_context, x_target,
                                                             y_target)
        # optimisation step
        a = sess.run(train_op_and_loss, feed_dict=feed_dict)
        if iter % 100 == 0:
            print(a[1])

    y_star_mat = sess.run(predict_op['mu'])
    plt.figure()
    plt.plot(x, y, color='g', linestyle='--', linewidth=0.6)
    for ii in range(n_draws):
        plt.plot(x_star, y_star_mat[:, ii], color='#539caf')
    plt.scatter(x_init[0:n_init-1], y_init[0:n_init-1], marker='o', color='b', label='1', s=10, alpha=1)
    plt.scatter(x_init[-1], y_init[-1], marker='+', color='r', label='1', s=40, alpha=1)
    plt.ylim((-1, 1))
    fig_name = 'Figures/WTI_experiment_Bayesian_Op_iter_' + str(iter_op) + '.png'
    plt.savefig(fig_name)

    print('The %d-th iteration is completed!' % iter_op)





'''
n = len(target_list)
for i in range(n):
    if i != 0:
        x_t = []
        for item in para_list[0:i]:
            x_t.append(item['x_i'])
        plt.scatter(x_t, target_list[0:i], marker='o', color='b', label='1', s=16, alpha=1)
    plt.scatter(para_list[i]['x_i'], target_list[i], marker='+', color='r', label='1', s=16, alpha=1)
    plt.plot(x, y, color='#539caf', label='1')
    fig_name = 'Figures/WTI_experiment_Bayesian_Op_iter_' + str(i) + '.png'
    plt.savefig(fig_name)
'''



