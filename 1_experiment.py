import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from neural_processes import NeuralProcess

N = 5
#x = np.linspace(-4, 4, 17)
x = np.linspace(-2, 2, 5)
x = np.expand_dims(x, axis=1)
y = np.sin(x)

'''
plt.scatter(x, y, marker='o', color='#539caf', label='1', s=3, alpha=1)
plt.savefig('observed_data.png')
'''

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

n_iter = 5000
plot_freq = 100

n_draws = 20
x_star_temp = np.linspace(-4, 4, 100)
x_star = np.expand_dims(x_star_temp, axis=1)
eps_value = np.random.normal(size=(n_draws, dim_r))
epsilon = tf.constant(eps_value, dtype=tf.float32)
predict_op = neural_process.posterior_predict(x, y, x_star, epsilon=epsilon, n_draws=n_draws)

df_pred_list = []
for iter in range(n_iter):
    N_context = np.random.randint(2, 4, 1)
    # create feed_dict containing context and target sets
    feed_dict = neural_process.helper_context_and_target(x, y, N_context, x_context, y_context, x_target, y_target)
    # optimisation step
    a = sess.run(train_op_and_loss, feed_dict= feed_dict)

    # plotting
    if iter%plot_freq == 0:
        y_star_mat = sess.run(predict_op['mu'])         # 'Mu' --> dim = [N_star, n_draws] --> [100, 50]
        df_pred_list.append(y_star_mat)
        print(a[1])
        #see_shape = sess.run(predict_op['size'])
        #print(see_shape)
        plt.figure()
        plt.scatter(x, y, marker='o', color='r', label='1', s=10, alpha=1)
        n_draws = y_star_mat.shape[1]
        for ii in range(n_draws):
            plt.plot(x_star, y_star_mat[:, ii], color='#539caf')
            plt.ylim((-2, 2))
            #print(y_star_mat[:,ii])
        fig_name = 'Figures/experiment_1_iter_' + str(iter) + '.png'
        plt.savefig(fig_name)

