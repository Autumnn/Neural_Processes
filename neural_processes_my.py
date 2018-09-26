import tensorflow as tf
import numpy as np


class NeuralProcess:

    def __init__(self, x_context, y_context, x_target, y_target,
                 dim_r, dim_z, dim_h_hidden, dim_g_hidden):

        self.dim_r = dim_r
        self.dim_z = dim_z
        self.dim_h_hidden = dim_h_hidden
        self.dim_g_hidden = dim_g_hidden
        self.x_context = x_context
        self.y_context = y_context
        self.x_target = x_target
        self.y_target = y_target
        self.x_all = tf.concat([self.x_context, self.x_target], axis=0)
        self.y_all = tf.concat([self.y_context, self.y_target], axis=0)

    def map_xy_to_z_params(self, x, y):
        inp = tf.concat([x, y], axis=1)
        t_x_1 = tf.layers.dense(inp, self.dim_h_hidden, tf.nn.sigmoid, name='encoder_layer_1', reuse=tf.AUTO_REUSE)
        t_x = tf.layers.dense(t_x_1, self.dim_r, name="encoder_layer_2", reuse=tf.AUTO_REUSE)
        r_1 = tf.reduce_mean(t_x, axis=0)
        r = tf.reshape(r_1, shape=(1, -1))   ######################
        mu = tf.layers.dense(r, self.dim_z, name='z_params_mu', reuse=tf.AUTO_REUSE)
        sigma_t = tf.layers.dense(r, self.dim_z, name='z_params_sigma', reuse=tf.AUTO_REUSE)
        sigma = tf.nn.softplus(sigma_t)

        #size = tf.shape(t_x)

        #return {'mu': mu, 'sigma': sigma, 'size': size}
        return {'mu': mu, 'sigma': sigma}


    def g(self, z_sample, x_star, noise_sd = 0.05):
        # inputs dimensions
        # z_sample has dim [n_draws, dim_z]
        # x_star has dim [N_star, dim_x]

        n_draws = z_sample.get_shape().as_list()[0]
        N_star = tf.shape(x_star)[0]

        # z_sample_rep will have dim [n_draws, N_star, dim_z]
        z_sample_rep = tf.tile(tf.expand_dims(z_sample, axis=1), [1, N_star, 1])
        # x_star_rep will have dim [n_draws, N_star, dim_x]
        x_star_rep = tf.tile(tf.expand_dims(x_star, axis=0), [n_draws, 1, 1])

        inp = tf.concat([x_star_rep, z_sample_rep], axis=2)

        hidden = tf.layers.dense(inp, self.dim_g_hidden, tf.nn.sigmoid, name="decoder_layer_1", reuse=tf.AUTO_REUSE)

        y_star = tf.layers.dense(hidden, 1, name="decoder_layer_2", reuse=tf.AUTO_REUSE)
        #size = tf.shape(mu_star)
        y_star = tf.squeeze(y_star, axis=2)
        y_star = tf.transpose(y_star)         # dim = [N_star, n_draws]

        #return {'mu': mu_star, 'sigma': sigma_star, 'size': size}
        return {'y_star': y_star}

    def klqp_gaussian(self, mu_q, sigma_q, mu_p, sigma_p):
        sigma2_q = tf.square(sigma_q) + 1e-16
        sigma2_p = tf.square(sigma_p) + 1e-16
        temp = sigma2_q / sigma2_p + tf.square(mu_q - mu_p) / sigma2_p - 1.0 + tf.log(sigma2_p / sigma2_q + 1e-16)
        temp = 0.5 * tf.reduce_sum(temp)
        return temp

    def custom_objective(self, y_pred_params, z_all, z_context):
        y_star_para = y_pred_params['y_star']     # dim = [N_star, n_draws]
        mu, sigma = tf.nn.moments(y_star_para, axes=[1])    # dim = N_star
        sdv = tf.sqrt(sigma)
        p_normal = tf.distributions.Normal(loc=mu, scale=sdv)
        p_star = p_normal.log_prob(tf.transpose(self.y_target))
        loglik = tf.reduce_sum(p_star)
        KL_loss = self.klqp_gaussian(z_all['mu'], z_all['sigma'], z_context['mu'], z_context['sigma'])
        loss = tf.negative(loglik) + KL_loss
        return loss


    def init_NP(self, learning_rate=0.001):
        z_context = self.map_xy_to_z_params(self.x_context, self.y_context)
        z_all = self.map_xy_to_z_params(self.x_all, self.y_all)

        epsilon = tf.random_normal(shape=(7, self.dim_z))
        #epsilon = tf.random_uniform(shape=(7, self.dim_z), minval=-10, maxval=10)
        z_sample = tf.add(z_all['mu'], tf.multiply(epsilon, z_all['sigma']))

        y_pred_params = self.g(z_sample, self.x_target)     # dim = [N_star, n_draws]

        loss = self.custom_objective(y_pred_params=y_pred_params, z_all=z_all, z_context=z_context)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(loss=loss)

        return [train_op, loss]

    def prior_predict(self, x_star_value, epsilon=None, n_draws=1):
        x_star = tf.constant(x_star_value, dtype=tf.float32)

        # the source of randomness can be optionally passed as an argument
        if not epsilon:
            epsilon = tf.random_normal(shape=(n_draws, self.dim_z))

        # draw z ~ N(0, 1)
        z_sample = epsilon

        # y ~ g(z, x*)
        y_star = self.g(z_sample, x_star)

        return y_star

    def posterior_predict(self, x, y, x_star_value, epsilon=None, n_draws=1):
        # inputs for prediction time
        x_obs = tf.constant(x, dtype=tf.float32)
        y_obs = tf.constant(y, dtype=tf.float32)
        x_star = tf.constant(x_star_value, dtype=tf.float32)

        # for out-of-sample new points
        z_params = self.map_xy_to_z_params(x_obs, y_obs)
        #z_params_shape = tf.shape(z_params['mu'])

        # the source of randomness can be optionally passed as an argument
        if epsilon is None:
            epsilon = tf.random_normal(shape=(n_draws, self.dim_z))

        # sample z using reparametrisation
        z_sample = tf.add(z_params['mu'], tf.multiply(epsilon, z_params['sigma']))

        # predictions
        y_star = self.g(z_sample, x_star)

        #return y_star, z_params_shape, z_params['size']
        return y_star


    def helper_context_and_target(self, x, y, N_context, x_context, y_context, x_target, y_target):
        N = len(y)
        ori = np.linspace(1, N, N, dtype=int)
        context_set = np.random.choice(N, N_context, replace=False) + 1
        context_lef = np.setdiff1d(ori, context_set)

        dic = {x_context: x[context_set-1], y_context: y[context_set-1],
               x_target: x[context_lef-1], y_target: y[context_lef-1]}

        return dic






