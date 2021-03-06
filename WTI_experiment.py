import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from neural_processes import NeuralProcess
from conditional_neural_processes_my import NeuralProcess

df = pd.read_csv('data.csv', header=None)
y_ori = df.values
nn = y_ori.shape[0]
y_ori = y_ori[nn-20:,]
min_max_scalar = preprocessing.MinMaxScaler()
y_f = min_max_scalar.fit_transform(y_ori)*2-1
n_f = y_f.shape[0]
x_f = (np.linspace(1, n_f, n_f) - n_f/2)/1
x_min = np.min(x_f)
x_max = np.max(x_f)
x_f = np.expand_dims(x_f, axis=1)

y = y_f[::2]
N = y.shape[0]
x = x_f[::2]

'''
plt.scatter(x, y, marker='o', color='#539caf', label='1', s=3, alpha=1)
plt.savefig('observed_data.png')
'''

dim_r = 2
#dim_z = 2
dim_h_hidden = 8
dim_g_hidden = 8

sess = tf.Session()

x_context = tf.placeholder(tf.float32, shape=(None, 1))
y_context = tf.placeholder(tf.float32, shape=(None, 1))
x_target = tf.placeholder(tf.float32, shape=(None, 1))
y_target = tf.placeholder(tf.float32, shape=(None, 1))

#neural_process = NeuralProcess(x_context, y_context, x_target, y_target,
#                                  dim_r, dim_z, dim_h_hidden, dim_g_hidden)
neural_process = NeuralProcess(x_context, y_context, x_target, y_target,
                                  dim_r, dim_h_hidden, dim_g_hidden)

train_op_and_loss = neural_process.init_NP(learning_rate = 0.001)

init = tf.global_variables_initializer()
sess.run(init)

n_iter = 5000
plot_freq = 200

n_draws = 50
x_star_temp = np.linspace(x_min, x_max, n_f*10)
x_star = np.expand_dims(x_star_temp, axis=1)
#eps_value = np.random.normal(size=(n_draws, dim_r))
eps_value = np.random.normal(size=(n_draws, 1))
epsilon = tf.constant(eps_value, dtype=tf.float32)
predict_op = neural_process.posterior_predict(x, y, x_star, epsilon=epsilon, n_draws=n_draws)

df_pred_list = []
for iter in range(n_iter):
    N_context = np.random.randint(1, N, 1)
    # create feed_dict containing context and target sets
    feed_dict = neural_process.helper_context_and_target(x, y, N_context, x_context, y_context, x_target, y_target)
    # optimisation step
    a = sess.run(train_op_and_loss, feed_dict= feed_dict)

    # plotting
    if iter%plot_freq == 0:
        #y_star_mat = sess.run(predict_op['mu'])         # 'Mu' --> dim = [N_star, n_draws] --> [100, 50]
        #y_star_mat = sess.run(predict_op)
        _, y_mu, y_sigma = sess.run(predict_op)
        #df_pred_list.append(y_star_mat)
        print(a[1])
        #see_shape = sess.run(predict_op['size'])
        #print(see_shape)
        plt.figure()
        plt.scatter(x, y, marker='o', color='b', label='1', s=10, alpha=1)
        plt.scatter(x_f[1::2], y_f[1::2], marker='+', color='r', label='1', s=20, alpha=1)
        plt.plot(x_f, y_f, color='g', linestyle='--')
        plt.plot(x_star, y_mu, 'b', linewidth=2)
        mu_temp = np.squeeze(y_mu)
        sigma_temp = np.squeeze(y_sigma)
        #sigma_temp = np.reshape(y_sigma.T, [1, -1])
        plt.fill_between(x_star_temp, mu_temp - sigma_temp, mu_temp + sigma_temp,
            alpha=0.2, facecolor='#65c9f7', interpolate=True)
        '''
        n_draws = y_star_mat.shape[1]
        for ii in range(n_draws):
            plt.plot(x_star, y_star_mat[:, ii], color='#539caf', linewidth=0.3)
            #plt.ylim((-2, 2))
            #print(y_star_mat[:,ii])
        '''

        fig_name = 'Figures/experiment_1_iter_' + str(iter) + '.png'
        plt.savefig(fig_name)

