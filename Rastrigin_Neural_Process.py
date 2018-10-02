import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from bayes_opt import BayesianOptimization

#from neural_processes import NeuralProcess
from conditional_neural_processes_my import NeuralProcess


def rastrigin(x_1, x_2):
    A = 3
    return -(A + ((x_1**2 - A * np.cos(2 * math.pi * x_1)))+
             (x_2 ** 2 - A * np.cos(2 * math.pi * x_2)))

num_star = 200


x_1 = np.linspace(-2, 6, num_star)
x_2 = np.linspace(-2, 6, num_star)
bound_lo = np.min(x_1)
bound_up = np.max(x_1)

x_1, x_2 = np.meshgrid(x_1, x_2)
y = rastrigin(x_1, x_2)

'''
y = np.random.uniform(bound_lo, bound_up, size=(num_star, num_star))
print(y)
y_t = np.reshape(y, (-1,1))
print(y_t)
y_c = np.reshape(y_t, (num_star,-1))
print(y_c)
'''

num_init = 10
x_init = np.random.uniform(bound_lo, bound_up, size=(num_init, 2))
y_init = rastrigin(x_init[:,0], x_init[:,1])
y_init = np.expand_dims(y_init, axis=1)

'''
x = np.concatenate((x_1, x_2), axis=1)
#x = np.expand_dims(x, axis=1)
y_mat = r['y_mat']
n_draws = r['n_draws']
y = np.mean(y_mat, axis=1)
y = np.expand_dims(y, axis=1)

n = len(x)
index_ori = np.linspace(1, n, n, dtype=int)-1
n_init = 10
index_init = np.random.choice(index_ori, n_init, replace=False)
x_init = x[index_init]
y_init = y[index_init]
index_lef = np.setdiff1d(index_ori, index_init)
'''

dim_r = 4
#dim_z = 2
dim_h_hidden = 128
dim_g_hidden = 128

sess = tf.Session()
x_context = tf.placeholder(tf.float32, shape=(None, 2))
y_context = tf.placeholder(tf.float32, shape=(None, 1))
x_target = tf.placeholder(tf.float32, shape=(None, 2))
y_target = tf.placeholder(tf.float32, shape=(None, 1))
neural_process = NeuralProcess(x_context, y_context, x_target, y_target,
                                  dim_r, dim_h_hidden, dim_g_hidden)

train_op_and_loss = neural_process.init_NP(learning_rate = 0.001)

init = tf.global_variables_initializer()
sess.run(init)

n_iter = 1001
x_1_t = np.reshape(x_1, (-1,1))
x_2_t = np.reshape(x_2, (-1,1))
x_star = np.concatenate((x_1_t, x_2_t), axis=1)
# eps_value = np.random.normal(size=(n_draws, dim_r))
# epsilon = tf.constant(eps_value, dtype=tf.float32)
predict_op = neural_process.posterior_predict(x_init, y_init, x_star)

ini_folder = 'Figures/Initiation'
if not os.path.exists(ini_folder):
    os.makedirs(ini_folder)

for iter in range(n_iter):
    N_context = np.random.randint(1, num_init, 1)
    # create feed_dict containing context and target sets
    feed_dict = neural_process.helper_context_and_target(x_init, y_init, N_context, x_context, y_context, x_target, y_target)
    # optimisation step
    a = sess.run(train_op_and_loss, feed_dict= feed_dict)
    if iter % 100 == 0:
        print(a[1])
        _, y_star_mu, y_star_sigma = sess.run(predict_op)
        y_mu_t = np.reshape(y_star_mu, (num_star, -1))
        y_sigma_t = np.reshape(y_star_sigma, (num_star, -1))

        fig = plt.figure(figsize=(12, 4))

        ax_1 = plt.subplot(121)
        con_1 = plt.contourf(x_1, x_2, y_mu_t, 10, cmap="PuBuGn")
        fig.colorbar(con_1, ax=ax_1)
        plt.scatter(x_init[:, 0], x_init[:, 1], marker='+', color='r', label='1', s=30, alpha=1)
        ax_1.set_title('mean')

        ax_2 = plt.subplot(122)
        con_2 = plt.contourf(x_1, x_2, y_sigma_t, 10, cmap="YlOrRd")
        fig.colorbar(con_2, ax=ax_2)
        plt.scatter(x_init[:, 0], x_init[:, 1], marker='o', color='#539caf', label='1', s=30, alpha=1)
        ax_2.set_title('variance')

        fig_name = ini_folder + '/Rastrigin_CNP_Op_iter_' + str(iter) + '.png'
        plt.savefig(fig_name)

print('Initialization Completed !')


n_op_iter = 9
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

    fig = plt.figure(figsize=(12, 4))

    ax_1 = plt.subplot(121)
    con_1 = plt.contourf(x_1, x_2, y_mu_t, 10, cmap="PuBuGn")
    fig.colorbar(con_1, ax=ax_1)
    plt.scatter(x_init[:, 0], x_init[:, 1], marker='+', color='pink', label='1', s=10, alpha=0.3)
    plt.scatter(x_candidate[:, 0], x_candidate[:, 1], marker='+', color='r', label='1', s=20, alpha=0.3)
    plt.scatter(x_mu_select[:, 0], x_mu_select[:, 1], marker='o', color='r', label='1', s=30, alpha=1)
    ax_1.set_title('mean')

    ax_2 = plt.subplot(122)
    con_2 = plt.contourf(x_1, x_2, y_sigma_t, 10, cmap="YlOrRd")
    fig.colorbar(con_2, ax=ax_2)
    plt.scatter(x_init[:, 0], x_init[:, 1], marker='o', color='powderblue', label='1', s=10, alpha=0.3)
    plt.scatter(x_candidate[:, 0], x_candidate[:, 1], marker='o', color='#539caf', label='1', s=20, alpha=0.3)
    plt.scatter(x_sigma_select[:, 0], x_sigma_select[:, 1], marker='+', color='#539caf', label='1', s=30, alpha=1)
    ax_2.set_title('variance')

    fig_name = 'Figures/Rastrigin_CNP_Op_' + str(iter_op) + '_candidate.png'
    plt.savefig(fig_name)

    iter_folder = 'Figures/The_' + str(iter_op) + '_iteration'
    if not os.path.exists(iter_folder):
        os.makedirs(iter_folder)

    x_select = np.unique(np.concatenate((x_mu_select, x_sigma_select), axis=0), axis=0)
    x_init = np.unique(np.concatenate((x_init, x_select), axis=0), axis=0)
    num_iter = x_init.shape[0]
    y_init = rastrigin(x_init[:, 0], x_init[:, 1])
    y_init = np.expand_dims(y_init, axis=1)


    for iter in range(n_iter):
        N_context = np.random.randint(1, num_iter, 1)
        # create feed_dict containing context and target sets
        feed_dict = neural_process.helper_context_and_target(x_init, y_init, N_context, x_context, y_context, x_target,
                                                             y_target)
        # optimisation step
        a = sess.run(train_op_and_loss, feed_dict=feed_dict)
        if iter % 100 == 0:
            print(a[1])
            _, y_star_mu, y_star_sigma = sess.run(predict_op)
            y_mu_t = np.reshape(y_star_mu, (num_star, -1))
            y_sigma_t = np.reshape(y_star_sigma, (num_star, -1))

            fig = plt.figure(figsize=(12, 4))

            ax_1 = plt.subplot(121)
            con_1 = plt.contourf(x_1, x_2, y_mu_t, 10, cmap="PuBuGn")
            fig.colorbar(con_1, ax=ax_1)
            plt.scatter(x_init[:, 0], x_init[:, 1], marker='+', color='r', label='1', s=30, alpha=1)
            ax_1.set_title('mean')

            ax_2 = plt.subplot(122)
            con_2 = plt.contourf(x_1, x_2, y_sigma_t, 10, cmap="YlOrRd")
            fig.colorbar(con_2, ax=ax_2)
            plt.scatter(x_init[:, 0], x_init[:, 1], marker='o', color='#539caf', label='1', s=30, alpha=1)
            ax_2.set_title('variance')

            fig_name = iter_folder + '/Rastrigin_CNP_Op_iter_' + str(iter) + '.png'
            plt.savefig(fig_name)

    print('The %d-th iteration is Completed!' %iter_op)

'''

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



