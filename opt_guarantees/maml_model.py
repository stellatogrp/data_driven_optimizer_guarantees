from functools import partial

import jax.numpy as jnp
from jax import grad, random, vmap

from opt_guarantees.algo_steps import (
    k_steps_eval_maml,
    k_steps_train_maml,
)
from opt_guarantees.l2o_model import L2Omodel
from opt_guarantees.utils.nn_utils import (
    get_perturbed_weights,
    predict_y,
)


class MAMLmodel(L2Omodel):
    def __init__(self, **kwargs):
        super(MAMLmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'maml'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']

        self.output_size = 1
        gamma = input_dict['gamma']

        neural_net_grad = grad(neural_net_fwd, argnums=0, has_aux=True)

        custom_loss = input_dict.get('custom_loss', False)
        if custom_loss:
            neural_net_fwd2 = partial(neural_net_fwd, norm='inf')
        else:
            neural_net_fwd2 = partial(neural_net_fwd, norm='MSE')

        self.k_steps_train_fn = partial(k_steps_train_maml,
                                        neural_net_fwd=neural_net_fwd,
                                        neural_net_grad=neural_net_grad,
                                        gamma=gamma,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_maml,
                                       neural_net_fwd=neural_net_fwd2,
                                       neural_net_grad=neural_net_grad,
                                       gamma=gamma,
                                       jit=self.jit)
        self.out_axes_length = 5

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
            perturb = random.normal(w_key, (self.train_unrolls, 2))
            # return scale * random.normal(w_key, (n, m))
            if self.deterministic:
                stochastic_params = params[0]
            else:
                mean_params, sigma_params = params[0], params[1]
                perturb = get_perturbed_weights(
                    random.PRNGKey(key), self.layer_sizes, 1)
                stochastic_params = [
                    (
                        perturb[i][0] *
                        jnp.sqrt(
                            jnp.exp(sigma_params[i][0])) + mean_params[i][0],
                        perturb[i][1] *
                        jnp.sqrt(
                            jnp.exp(sigma_params[i][1])) + mean_params[i][1]
                    )
                    for i in range(len(mean_params))
                ]
            z0 = stochastic_params  # params[0]

            if bypass_nn:
                eval_out = k_steps_eval_maml(k=iters,
                                             z0=z0,
                                             q=q,
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
                                                    supervised=supervised,
                                                    z_star=z_star)
                else:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       supervised=supervised,
                                       z_star=z_star)
                    z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                    angles = None

            loss = self.final_loss(loss_method, z_final,
                                   iter_losses, supervised, z0, z_star)

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1,
                              angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn


def neural_net_fwd(z, theta, norm='MSE'):
    num = int(theta.size / 2)
    inputs = theta[:num]
    outputs = jnp.reshape(theta[num:], (num, 1))
    neural_net_single_input_batch = vmap(
        neural_net_single_input, in_axes=(None, 0), out_axes=(0))
    inputs_reshaped = jnp.reshape(inputs, (inputs.size, 1))
    predicted_outputs = neural_net_single_input_batch(z, inputs_reshaped)
    # loss = jnp.linalg.norm(outputs - predicted_outputs) ** 2 / num
    if norm == 'MSE':
        loss = jnp.mean((outputs - predicted_outputs)**2)
    elif norm == 'inf':
        loss = jnp.max(outputs - predicted_outputs)
    return loss, (predicted_outputs, outputs)


def neural_net_single_input(z, x):
    # here z is the weights
    y = predict_y(z, x)
    return y
