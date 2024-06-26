from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import random

from opt_guarantees.algo_steps import (
    k_steps_eval_fista,
    k_steps_eval_lista,
    k_steps_train_lista,
)
from opt_guarantees.l2o_model import L2Omodel
from opt_guarantees.utils.nn_utils import (
    compute_single_param_KL,
)


class LISTAmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(LISTAmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'lista'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        D, W = input_dict['D'], input_dict['W']
        lambd = input_dict['lambd']
        self.lambd = lambd
        # ista_step = input_dict['ista_step']
        self.D, self.W = D, W
        self.m, self.n = D.shape
        self.output_size = self.n

        evals, evecs = jnp.linalg.eigh(D.T @ D)
        L = evals.max()
        self.L = L
        # step = 1 / evals.max()
        # lambd = 0.1
        self.ista_step = lambd / evals.max()

        self.k_steps_train_fn = partial(k_steps_train_lista,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lista,
                                       jit=self.jit)
        self.out_axes_length = 5

        self.prior = self.W

        self.W_1 = D.T / L
        self.W_2 = jnp.eye(self.n) - D.T @ D / L

        self.W_1_tensor = jnp.tile(
            self.W_1[:, :, jnp.newaxis], (1, 1, self.train_unrolls))
        self.W_2_tensor = jnp.tile(
            self.W_2[:, :, jnp.newaxis], (1, 1, self.train_unrolls))

    def init_params(self):
        # self.mean_params = (jnp.ones((self.train_unrolls, 2)),
        #                         jnp.ones((self.m, self.n)))
        self.mean_params = (self.lambd / self.L * jnp.ones((self.train_unrolls)),
                            self.W_1_tensor + .001,
                            self.W_2_tensor + .001
                            )
        self.sigma_params = (-jnp.ones((self.train_unrolls)) * 10,
                             -jnp.ones((self.n, self.m,
                                       self.train_unrolls)) * 10,
                             -jnp.ones((self.n, self.n, self.train_unrolls)) * 10)

        self.prior_param = jnp.log(self.init_var) * jnp.ones(3)

        self.params = [self.mean_params, self.sigma_params, self.prior_param]

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            z0 = jnp.zeros(z_star.size)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # w_key = random.split(key)
            w_key = random.PRNGKey(key)
            perturb1 = random.normal(w_key, (self.train_unrolls,))
            perturb2 = random.normal(
                w_key, (self.n, self.m, self.train_unrolls))
            perturb3 = random.normal(
                w_key, (self.n, self.n, self.train_unrolls))
            # return scale * random.normal(w_key, (n, m))
            if self.deterministic:
                stochastic_params = params[0]
            else:
                stochastic_params = (params[0][0] + jnp.sqrt(jnp.exp(params[1][0])) * perturb1,
                                     params[0][1] +
                                     jnp.sqrt(
                                         jnp.exp(params[1][1])) * perturb2,
                                     params[0][2] + jnp.sqrt(jnp.exp(params[1][2])) * perturb3)

            if bypass_nn:
                eval_out = k_steps_eval_fista(k=iters,
                                              z0=z0,
                                              q=q,
                                              lambd=0.1,
                                              A=self.D,
                                              ista_step=self.ista_step,
                                              supervised=True,
                                              z_star=z_star,
                                              jit=True)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            else:
                if diff_required:
                    z_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    params=stochastic_params,
                                                    supervised=supervised,
                                                    z_star=z_star)
                else:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       params=stochastic_params,
                                       supervised=supervised,
                                       z_star=z_star)
                    z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                    angles = None

            loss = self.final_loss(loss_method, z_final,
                                   iter_losses, supervised, z0, z_star)

            if diff_required:
                # calculate_total_penalty(self.N_train, params, self.b, self.c,
                #                                         self.delta,
                #                                         prior=self.W)
                # q = jnp.array([loss / self.penalty_coeff, 1 - loss / self.penalty_coeff])
                # c = jnp.reshape(penalty, (1,))
                # loss = self.kl_inv_layer(q, c)
                q = loss / self.penalty_coeff
            else:
                # calculate_pinsker_penalty(self.N_train, params, self.b, self.c,
                penalty_loss = 0
                #  self.delta,
                #  prior=self.W)
                loss = loss + self.penalty_coeff * penalty_loss

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1,
                              angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn

    def calculate_total_penalty(self, N_train, params, c, b, delta):
        # prior = self.W
        # pi_pen = jnp.log(jnp.pi ** 2 * N_train / (6 * delta))
        # log_pen = 2 * jnp.log(b * jnp.log(c / jnp.exp(params[2][0])))
        # penalty_loss = self.compute_all_params_KL(params[0], params[1],
        #                                     params[2], prior=prior) + pi_pen + log_pen
        # return penalty_loss /  N_train
        # priors are already rounded
        rounded_priors = params[2]

        # second: calculate the penalties
        num_groups = len(rounded_priors)
        pi_pen = jnp.log(jnp.pi ** 2 * num_groups * N_train / (6 * delta))
        log_pen = 0
        for i in range(num_groups):
            curr_lambd = jnp.clip(jnp.exp(rounded_priors[i]), a_max=c)
            log_pen += 2 * jnp.log(b * jnp.log((c+1e-6) / curr_lambd))

        # calculate the KL penalty
        prior = self.W
        penalty_loss = self.compute_all_params_KL(params[0], params[1],
                                                  params[2], prior=prior) + pi_pen + log_pen
        return penalty_loss / N_train

    def compute_all_params_KL(self, mean_params, sigma_params, lambd, prior=None):
        total_pen = 0

        # thresholds
        total_pen += compute_single_param_KL(mean_params[0],
                                             jnp.exp(sigma_params[0]), jnp.exp(lambd[0]))

        # matrix W_1
        total_pen += compute_single_param_KL(mean_params[1],
                                             jnp.exp(sigma_params[1]),
                                             jnp.exp(lambd[1]),
                                             prior=self.W_1_tensor
                                             )

        # matrix W_2
        total_pen += compute_single_param_KL(mean_params[2],
                                             jnp.exp(sigma_params[2]),
                                             jnp.exp(lambd[2]),
                                             prior=self.W_2_tensor
                                             )

        return total_pen

    def compute_weight_norm_squared(self, nn_params):
        weight_norms = np.zeros(len(nn_params))
        weight_norms[0] = jnp.linalg.norm(nn_params[0]) ** 2
        weight_norms[1] = jnp.linalg.norm(nn_params[1] - self.W_1_tensor) ** 2
        weight_norms[2] = jnp.linalg.norm(nn_params[2] - self.W_2_tensor) ** 2
        num_weights = weight_norms[0].size + \
            weight_norms[1].size + weight_norms[2].size
        return weight_norms.sum(), num_weights

    def calculate_avg_posterior_var(self, params):
        sigma_params = params[1]
        flattened_params = jnp.concatenate(
            [jnp.ravel(sigma_params[0]), jnp.ravel(sigma_params[1])])
        variances = jnp.exp(flattened_params)
        avg_posterior_var = variances.mean()
        stddev_posterior_var = variances.std()
        return avg_posterior_var, stddev_posterior_var
