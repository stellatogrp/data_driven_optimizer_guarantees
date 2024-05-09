import csv
import gc
import os
import time

import hydra
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import lax, vmap
from jax.config import config
from scipy.sparse import load_npz
from scipy.spatial import distance_matrix

from opt_guarantees.algo_steps import (
    create_projection_fn,
    form_osqp_matrix,
    get_psd_sizes,
    unvec_symm,
)
from opt_guarantees.alista_model import ALISTAmodel
from opt_guarantees.gd_model import GDmodel
from opt_guarantees.glista_model import GLISTAmodel
from opt_guarantees.ista_model import ISTAmodel
from opt_guarantees.lista_model import LISTAmodel
from opt_guarantees.maml_model import MAMLmodel
from opt_guarantees.osqp_model import OSQPmodel
from opt_guarantees.scs_model import SCSmodel
from opt_guarantees.tilista_model import TILISTAmodel
from opt_guarantees.utils.generic_utils import (
    count_files_in_directory,
    sample_plot,
    setup_permutation,
)
from opt_guarantees.utils.nn_utils import (
    invert_kl,
)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 16,
})
config.update("jax_enable_x64", True)


class Workspace:
    def __init__(self, algo, cfg, static_flag, static_dict, example,
                 traj_length=None,
                 custom_visualize_fn=None,
                 custom_loss=None,
                 shifted_sol_fn=None,
                 closed_loop_rollout_dict=None):
        '''
        cfg is the run_cfg from hydra
        static_flag is True if the matrices P and A don't change from problem to problem
        static_dict holds the data that doesn't change from problem to problem
        example is the string (e.g. 'robust_kalman')
        '''
        self.algo = algo
        if cfg.get('custom_loss', False):
            if self.algo == 'maml':
                self.custom_loss = True
            else:
                self.custom_loss = custom_loss
            # self.custom_loss = custom_loss
        else:
            self.custom_loss = None
        pac_bayes_cfg = cfg.get('pac_bayes_cfg', {})
        self.skip_pac_bayes_full = pac_bayes_cfg.get('skip_full', True)

        pac_bayes_accs = pac_bayes_cfg.get(
            'frac_solved_accs', [0.1, 0.01, 0.001, 0.0001])
        self.pac_bayes_num_samples = pac_bayes_cfg.get(
            'pac_bayes_num_samples', 20)

        self.nmse = False
        if pac_bayes_accs == 'fp_full':
            start = -6  # Start of the log range (log10(10^-5))
            end = 2  # End of the log range (log10(1))
            pac_bayes_accs = list(np.round(np.logspace(start, end, num=81), 6))
        elif pac_bayes_accs == 'nmse_full':
            start = 0
            end = -80
            self.nmse = True
            pac_bayes_accs = list(np.round(np.linspace(start, end, num=81), 6))
        elif pac_bayes_accs == 'maml_full':
            start = -3  # Start of the log range (log10(10^-5))
            end = 1  # End of the log range (log10(1))
            pac_bayes_accs = list(np.round(np.logspace(start, end, num=81), 6))
        self.frac_solved_accs = pac_bayes_accs
        self.rep = pac_bayes_cfg.get('rep', True)
        self.sigma_nn_grid = np.array(cfg.get('sigma_nn', []))
        self.sigma_beta_grid = np.array(cfg.get('sigma_beta', []))

        self.pac_bayes_hyperparameter_opt_flag = cfg.get(
            'pac_bayes_flag', False)

        self.key_count = 0

        self.static_flag = static_flag
        self.example = example
        self.eval_unrolls = cfg.eval_unrolls + 1
        self.eval_every_x_epochs = cfg.eval_every_x_epochs
        self.save_every_x_epochs = cfg.save_every_x_epochs
        self.num_samples = cfg.get('num_samples', 10)

        self.num_samples_test = cfg.get('num_samples_test', self.num_samples)
        self.num_samples_train = cfg.get(
            'num_samples_train', self.num_samples_test)

        self.eval_batch_size_test = cfg.get(
            'eval_batch_size_test', self.num_samples_test)
        self.eval_batch_size_train = cfg.get(
            'eval_batch_size_train', self.num_samples_train)

        self.pretrain_cfg = cfg.pretrain
        self.key_count = 0
        self.save_weights_flag = cfg.get('save_weights_flag', False)
        self.load_weights_datetime = cfg.get('load_weights_datetime', None)
        self.nn_load_type = cfg.get('nn_load_type', 'deterministic')
        self.shifted_sol_fn = shifted_sol_fn
        self.plot_iterates = cfg.plot_iterates
        self.normalize_inputs = cfg.get('normalize_inputs', True)
        self.epochs_jit = cfg.epochs_jit
        self.accs = cfg.get('accuracies')

        # custom visualization
        self.init_custom_visualization(cfg, custom_visualize_fn)
        self.vis_num = cfg.get('vis_num', 20)

        # from the run cfg retrieve the following via the data cfg
        N_train, N_test = cfg.N_train, cfg.N_test
        N = N_train + N_test

        # for control problems only
        self.closed_loop_rollout_dict = closed_loop_rollout_dict
        self.traj_length = traj_length

        self.train_unrolls = cfg.train_unrolls

        # load the data from problem to problem
        jnp_load_obj = self.load_setup_data(
            example, cfg.data.datetime, N_train, N)
        thetas = jnp.array(jnp_load_obj['thetas'])
        self.thetas_train = thetas[:N_train, :]
        self.thetas_test = thetas[N_train:N, :]

        train_inputs, test_inputs = self.normalize_inputs_fn(
            thetas, N_train, N_test)
        self.train_inputs, self.test_inputs = train_inputs, test_inputs
        self.skip_startup = cfg.get('skip_startup', False)
        self.setup_opt_sols(algo, jnp_load_obj, N_train, N)

        # everything below is specific to the algo
        if algo == 'ista':
            # get b_mat
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]

            self.create_ista_model(cfg, static_dict)
        elif algo == 'osqp':
            self.create_osqp_model(cfg, static_dict)
        elif algo == 'scs':
            self.create_scs_model(cfg, static_dict)
        elif algo == 'gd':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_gd_model(cfg, static_dict)
        elif algo == 'alista':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_alista_model(cfg, static_dict)
        elif algo == 'tilista':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_tilista_model(cfg, static_dict)
        elif algo == 'lista':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_lista_model(cfg, static_dict)
        elif algo == 'glista':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_glista_model(cfg, static_dict)
        elif algo == 'maml':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_maml_model(cfg, static_dict)

        # write th z_stars_max
        # Scalar value to be saved
        # include the worst-case
        z_star_max = 1.0 * jnp.max(jnp.linalg.norm(self.z_stars_train, axis=1))
        theta_max = jnp.max(jnp.linalg.norm(self.thetas_train, axis=1))
        # worst_case = jnp.zeros(frac_solved.size)
        # steps = jnp.arange(frac_solved.size)
        # indices = 1 / jnp.sqrt(steps + 2) * z_star_max / 100 < self.frac_solved_accs[i]
        # worst_case = worst_case.at[indices].set(1.0)

        # Specify the CSV file name
        filename = 'z_star_max.csv'

        # Open the file in write mode
        if self.l2ws_model.algo in ['gd', 'ista', 'scs', 'osqp']:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)

                # Write the scalar value to the file
                writer.writerow([z_star_max])
                writer.writerow([theta_max])
                for i in range(len(self.l2ws_model.params[0])):
                    U, S, VT = jnp.linalg.svd(self.l2ws_model.params[0][i][0])
                    ti = S.max()
                    writer.writerow([ti])

    def create_ista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        A, lambd = static_dict['A'], static_dict['lambd']
        ista_step = static_dict['ista_step']

        input_dict = dict(algorithm='ista',
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          lambd=lambd,
                          ista_step=ista_step,
                          A=A
                          )
        # self.l2ws_model = ISTAmodel(input_dict)
        self.l2ws_model = ISTAmodel(train_unrolls=self.train_unrolls,
                                    eval_unrolls=self.eval_unrolls,
                                    train_inputs=self.train_inputs,
                                    test_inputs=self.test_inputs,
                                    regression=cfg.supervised,
                                    nn_cfg=cfg.nn_cfg,
                                    pac_bayes_cfg=cfg.pac_bayes_cfg,
                                    z_stars_train=self.z_stars_train,
                                    z_stars_test=self.z_stars_test,
                                    algo_dict=input_dict)

    def create_alista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        W, D = static_dict['W'], static_dict['D']
        alista_cfg = {'step': static_dict['step'], 'eta': static_dict['eta']}
        # ista_step = static_dict['ista_step']

        input_dict = dict(algorithm='alista',
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          D=D,
                          W=W
                          )
        self.l2ws_model = ALISTAmodel(train_unrolls=self.train_unrolls,
                                      eval_unrolls=self.eval_unrolls,
                                      train_inputs=self.train_inputs,
                                      test_inputs=self.test_inputs,
                                      regression=cfg.supervised,
                                      nn_cfg=cfg.nn_cfg,
                                      pac_bayes_cfg=cfg.pac_bayes_cfg,
                                      z_stars_train=self.z_stars_train,
                                      z_stars_test=self.z_stars_test,
                                      alista_cfg=alista_cfg,
                                      algo_dict=input_dict)

    def create_glista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        W, D = static_dict['W'], static_dict['D']
        alista_cfg = {'step': static_dict['step'], 'eta': static_dict['eta']}
        # ista_step = static_dict['ista_step']

        input_dict = dict(algorithm='glista',
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          D=D,
                          W=W
                          )
        self.l2ws_model = GLISTAmodel(train_unrolls=self.train_unrolls,
                                      eval_unrolls=self.eval_unrolls,
                                      train_inputs=self.train_inputs,
                                      test_inputs=self.test_inputs,
                                      regression=cfg.supervised,
                                      nn_cfg=cfg.nn_cfg,
                                      pac_bayes_cfg=cfg.pac_bayes_cfg,
                                      z_stars_train=self.z_stars_train,
                                      z_stars_test=self.z_stars_test,
                                      alista_cfg=alista_cfg,
                                      algo_dict=input_dict)

    def create_tilista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        W, D = static_dict['W'], static_dict['D']
        alista_cfg = {'step': static_dict['step'], 'eta': static_dict['eta']}
        # ista_step = static_dict['ista_step']

        input_dict = dict(algorithm='tilista',
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          D=D,
                          W=W
                          )
        self.l2ws_model = TILISTAmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       pac_bayes_cfg=cfg.pac_bayes_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       alista_cfg=alista_cfg,
                                       algo_dict=input_dict)

    def create_lista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        W, D = static_dict['W'], static_dict['D']
        alista_cfg = {'step': static_dict['step'], 'eta': static_dict['eta']}
        # ista_step = static_dict['ista_step']

        input_dict = dict(algorithm='lista',
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          D=D,
                          W=W,
                          lambd=cfg.lambd
                          )
        self.l2ws_model = LISTAmodel(train_unrolls=self.train_unrolls,
                                     eval_unrolls=self.eval_unrolls,
                                     train_inputs=self.train_inputs,
                                     test_inputs=self.test_inputs,
                                     regression=cfg.supervised,
                                     nn_cfg=cfg.nn_cfg,
                                     pac_bayes_cfg=cfg.pac_bayes_cfg,
                                     z_stars_train=self.z_stars_train,
                                     z_stars_test=self.z_stars_test,
                                     alista_cfg=alista_cfg,
                                     algo_dict=input_dict)

    def create_maml_model(self, cfg, static_dict):
        input_dict = dict(algorithm='maml',
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          gamma=cfg.gamma,
                          custom_loss=self.custom_loss
                          )
        self.l2ws_model = MAMLmodel(train_unrolls=self.train_unrolls,
                                    eval_unrolls=self.eval_unrolls,
                                    train_inputs=self.train_inputs,
                                    test_inputs=self.test_inputs,
                                    regression=cfg.supervised,
                                    nn_cfg=cfg.nn_cfg,
                                    pac_bayes_cfg=cfg.pac_bayes_cfg,
                                    z_stars_train=self.z_stars_train,
                                    z_stars_test=self.z_stars_test,
                                    algo_dict=input_dict)

    def create_gd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        P = static_dict['P']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='gd',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          P=P
                          )
        self.l2ws_model = GDmodel(train_unrolls=self.train_unrolls,
                                  eval_unrolls=self.eval_unrolls,
                                  train_inputs=self.train_inputs,
                                  test_inputs=self.test_inputs,
                                  regression=cfg.supervised,
                                  nn_cfg=cfg.nn_cfg,
                                  pac_bayes_cfg=cfg.pac_bayes_cfg,
                                  z_stars_train=self.z_stars_train,
                                  z_stars_test=self.z_stars_test,
                                  algo_dict=input_dict)


    def create_osqp_model(self, cfg, static_dict):
        if self.static_flag:
            factor = static_dict['factor']
            A = static_dict['A']
            P = static_dict['P']
            m, n = A.shape
            self.m, self.n = m, n
            rho = static_dict['rho']
            input_dict = dict(factor_static_bool=True,
                              supervised=cfg.supervised,
                              rho=rho,
                              q_mat_train=self.q_mat_train,
                              q_mat_test=self.q_mat_test,
                              A=A,
                              P=P,
                              m=m,
                              n=n,
                              factor=factor,
                              custom_loss=self.custom_loss,
                              #   train_inputs=self.train_inputs,
                              #   test_inputs=self.test_inputs,
                              #   train_unrolls=self.train_unrolls,
                              #   eval_unrolls=self.eval_unrolls,
                              #   nn_cfg=cfg.nn_cfg,
                              #   z_stars_train=self.z_stars_train,
                              #   z_stars_test=self.z_stars_test,
                              #   jit=True,
                              plateau_decay=cfg.plateau_decay)
        else:
            self.m, self.n = static_dict['m'], static_dict['n']
            m, n = self.m, self.n
            rho_vec = jnp.ones(m)
            l0 = self.q_mat_train[0, n: n + m]
            u0 = self.q_mat_train[0, n + m: n + 2 * m]
            # rho_vec = rho_vec.at[l0 == u0].set(1000)
            rho_vec = rho_vec.at[l0 == u0].set(1)

            t0 = time.time()

            # form matrices (N, m + n, m + n) to be factored
            nc2 = int(n * (n + 1) / 2)
            q_mat = jnp.vstack([self.q_mat_train, self.q_mat_test])
            N_train, _ = self.q_mat_train.shape[0], self.q_mat_test[0]
            N = q_mat.shape[0]
            unvec_symm_batch = vmap(
                unvec_symm, in_axes=(0, None), out_axes=(0))
            P_tensor = unvec_symm_batch(
                q_mat[:, 2 * m + n: 2 * m + n + nc2], n)
            A_tensor = jnp.reshape(q_mat[:, 2 * m + n + nc2:], (N, m, n))
            sigma = 1
            batch_form_osqp_matrix = vmap(
                form_osqp_matrix, in_axes=(0, 0, None, None), out_axes=(0))

            # try batching
            cutoff = 4000
            matrices1 = batch_form_osqp_matrix(
                P_tensor[:cutoff, :, :], A_tensor[:cutoff, :, :], rho_vec, sigma)
            matrices2 = batch_form_osqp_matrix(
                P_tensor[cutoff:, :, :], A_tensor[cutoff:, :, :], rho_vec, sigma)
            # matrices =

            # do factors
            # factors0, factors1 = self.batch_factors(self.q_mat_train)
            batch_lu_factor = vmap(jsp.linalg.lu_factor,
                                   in_axes=(0,), out_axes=(0, 0))
            factors10, factors11 = batch_lu_factor(matrices1)
            factors20, factors21 = batch_lu_factor(matrices2)
            factors0 = jnp.vstack([factors10, factors20])
            factors1 = jnp.vstack([factors11, factors21])

            t1 = time.time()
            print('batch factor time', t1 - t0)

            self.factors_train = (
                factors0[:N_train, :, :], factors1[:N_train, :])
            self.factors_test = (
                factors0[N_train:N, :, :], factors1[N_train:N, :])

            input_dict = dict(factor_static_bool=False,
                              supervised=cfg.supervised,
                              rho=rho_vec,
                              q_mat_train=self.q_mat_train,
                              q_mat_test=self.q_mat_test,
                              m=self.m,
                              n=self.n,
                              train_inputs=self.train_inputs,
                              test_inputs=self.test_inputs,
                              factors_train=self.factors_train,
                              factors_test=self.factors_test,
                              custom_loss=self.custom_loss,
                              #   train_unrolls=self.train_unrolls,
                              #   eval_unrolls=self.eval_unrolls,
                              #   nn_cfg=cfg.nn_cfg,
                              #   z_stars_train=self.z_stars_train,
                              #   z_stars_test=self.z_stars_test,
                              jit=True)
        self.x_stars_train = self.z_stars_train[:, :self.n]
        self.x_stars_test = self.z_stars_test[:, :self.n]
        self.l2ws_model = OSQPmodel(train_unrolls=self.train_unrolls,
                                    eval_unrolls=self.eval_unrolls,
                                    train_inputs=self.train_inputs,
                                    test_inputs=self.test_inputs,
                                    regression=cfg.supervised,
                                    nn_cfg=cfg.nn_cfg,
                                    pac_bayes_cfg=cfg.pac_bayes_cfg,
                                    z_stars_train=self.z_stars_train,
                                    z_stars_test=self.z_stars_test,
                                    algo_dict=input_dict)

    def create_scs_model(self, cfg, static_dict):
        # get_M_q = None
        # if get_M_q is None:
        #     q_mat = jnp_load_obj['q_mat']
        if self.static_flag:
            static_M = static_dict['M']

            static_algo_factor = static_dict['algo_factor']
            cones = static_dict['cones_dict']

        else:
            pass

        rho_x = cfg.get('rho_x', 1)
        scale = cfg.get('scale', 1)
        alpha_relax = cfg.get('alpha_relax', 1)

        # save cones
        self.cones = static_dict['cones_dict']

        self.M = static_M
        proj = create_projection_fn(cones, self.n)
        psd_sizes = get_psd_sizes(cones)

        self.psd_size = psd_sizes[0]

        algo_dict = {'proj': proj,
                     'q_mat_train': self.q_mat_train,
                     'q_mat_test': self.q_mat_test,
                     'm': self.m,
                     'n': self.n,
                     'static_M': static_M,
                     'static_flag': self.static_flag,
                     'static_algo_factor': static_algo_factor,
                     'rho_x': rho_x,
                     'scale': scale,
                     'alpha_relax': alpha_relax,
                     'cones': cones,
                     'lightweight': cfg.get('lightweight', False),
                     'custom_loss': self.custom_loss
                     }
        self.l2ws_model = SCSmodel(train_unrolls=self.train_unrolls,
                                   eval_unrolls=self.eval_unrolls,
                                   train_inputs=self.train_inputs,
                                   test_inputs=self.test_inputs,
                                   z_stars_train=self.z_stars_train,
                                   z_stars_test=self.z_stars_test,
                                   x_stars_train=self.x_stars_train,
                                   x_stars_test=self.x_stars_test,
                                   y_stars_train=self.y_stars_train,
                                   y_stars_test=self.y_stars_test,
                                   regression=cfg.get('supervised', False),
                                   nn_cfg=cfg.nn_cfg,
                                   pac_bayes_cfg=cfg.pac_bayes_cfg,
                                   algo_dict=algo_dict)
        # self.l2ws_model = SCSmodel(input_dict)

    def setup_opt_sols(self, algo, jnp_load_obj, N_train, N, num_plot=5):
        if algo != 'scs':
            z_stars = jnp_load_obj['z_stars']
            z_stars_train = z_stars[:N_train, :]
            z_stars_test = z_stars[N_train:N, :]
            self.plot_samples(num_plot, self.thetas_train,
                              self.train_inputs, z_stars_train)
            self.z_stars_test = z_stars_test
            self.z_stars_train = z_stars_train

            # if algo == 'osqp':
            #     self.x_stars_train = z_stars_train[:, :self.n]
            #     self.x_stars_test = z_stars_test[:, :self.n]
        else:
            if 'x_stars' in jnp_load_obj.keys():
                x_stars = jnp_load_obj['x_stars']
                y_stars = jnp_load_obj['y_stars']
                s_stars = jnp_load_obj['s_stars']
                z_stars = jnp.hstack([x_stars, y_stars + s_stars])
                x_stars_train = x_stars[:N_train, :]
                y_stars_train = y_stars[:N_train, :]

                self.x_stars_train = x_stars[:N_train, :]
                self.y_stars_train = y_stars[:N_train, :]

                z_stars_train = z_stars[:N_train, :]
                self.x_stars_test = x_stars[N_train:N, :]
                self.y_stars_test = y_stars[N_train:N, :]
                z_stars_test = z_stars[N_train:N, :]
                self.m, self.n = y_stars_train.shape[1], x_stars_train[0, :].size
            else:
                x_stars_train, self.x_stars_test = None, None
                y_stars_train, self.y_stars_test = None, None
                z_stars_train, z_stars_test = None, None
                self.m, self.n = int(jnp_load_obj['m']), int(jnp_load_obj['n'])
            self.plot_samples_scs(num_plot, self.thetas_train, self.train_inputs,
                                  x_stars_train, y_stars_train, z_stars_train)
            self.z_stars_train = z_stars_train
            self.z_stars_test = z_stars_test

    def save_weights(self):
        nn_weights = self.l2ws_model.params
        if len(nn_weights) == 3 and not isinstance(nn_weights[2], tuple):
            if self.l2ws_model.algo in ['alista', 'glista']:
                self.save_weights_stochastic_alista()
            elif self.l2ws_model.algo in ['lista', 'tilista']:
                self.save_weights_stochastic_lista()
            else:
                self.save_weights_stochastic()
        else:
            self.save_weights_deterministic()

    def save_weights_stochastic_tilista(self):
        nn_weights = self.l2ws_model.params
        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')
            os.mkdir('nn_weights/mean')
            os.mkdir('nn_weights/variance')
            os.mkdir('nn_weights/prior')

        # Save mean weights
        mean_params = nn_weights[0]
        scalar_params, matrix_params = mean_params[0], mean_params[1]
        jnp.savez("nn_weights/mean/mean_params.npz",
                  scalar_params=scalar_params,
                  matrix_params=matrix_params)

        # Save variance weights
        variance_params = nn_weights[1]
        jnp.savez("nn_weights/variance/variance_params.npz",
                  scalar_params=variance_params[0],
                  matrix_params=variance_params[1])

        # save prior
        jnp.savez("nn_weights/prior/prior_val.npz", prior=nn_weights[2])

    def save_weights_stochastic_alista(self):
        nn_weights = self.l2ws_model.params
        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')
            os.mkdir('nn_weights/mean')
            os.mkdir('nn_weights/variance')
            os.mkdir('nn_weights/prior')

        # Save mean weights
        mean_params = nn_weights[0]
        jnp.savez("nn_weights/mean/mean_params.npz", mean_params=mean_params)

        # Save variance weights
        variance_params = nn_weights[1]
        jnp.savez("nn_weights/variance/variance_params.npz",
                  variance_params=variance_params)

        # save prior
        jnp.savez("nn_weights/prior/prior_val.npz", prior=nn_weights[2])

    def save_weights_stochastic_lista(self):
        nn_weights = self.l2ws_model.params
        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')
            os.mkdir('nn_weights/mean')
            os.mkdir('nn_weights/variance')
            os.mkdir('nn_weights/prior')

        # Save mean weights
        mean_params = nn_weights[0]

        # Save variance weights
        variance_params = nn_weights[1]

        if len(mean_params) == 2:
            jnp.savez("nn_weights/mean/mean_params.npz",
                      thresh_step_params=mean_params[0],
                      W1_params=mean_params[1])

            jnp.savez("nn_weights/variance/variance_params.npz",
                      thresh_step_params=variance_params[0],
                      W1_params=variance_params[1])
        elif len(mean_params) == 3:
            jnp.savez("nn_weights/mean/mean_params.npz", thresh_params=mean_params[0],
                      W1_params=mean_params[1], W2_params=mean_params[2])

            jnp.savez("nn_weights/variance/variance_params.npz", thresh_params=variance_params[0],
                      W1_params=variance_params[1], W2_params=variance_params[2])

        # save prior
        jnp.savez("nn_weights/prior/prior_val.npz", prior=nn_weights[2])

    def save_weights_stochastic(self):
        nn_weights = self.l2ws_model.params
        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')
            os.mkdir('nn_weights/mean')
            os.mkdir('nn_weights/variance')
            os.mkdir('nn_weights/prior')

        # Save mean weights
        mean_params = nn_weights[0]
        for i, params in enumerate(mean_params):
            weight_matrix, bias_vector = params
            jnp.savez(f"nn_weights/mean/layer_{i}_params.npz", weight=weight_matrix,
                      bias=bias_vector)

        # Save variance weights
        variance_params = nn_weights[1]
        for i, params in enumerate(variance_params):
            weight_matrix, bias_vector = params
            jnp.savez(f"nn_weights/variance/layer_{i}_params.npz", weight=weight_matrix,
                      bias=bias_vector)

        # save prior
        jnp.savez("nn_weights/prior/prior_val.npz", prior=nn_weights[2])

    def save_weights_deterministic(self):
        nn_weights = self.l2ws_model.params
        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')

        if self.l2ws_model.algo == 'tilista':
            jnp.savez("nn_weights/params.npz",
                      scalar_params=nn_weights[0],
                      matrix=nn_weights[1])
        else:
            # Save each weight matrix and bias vector separately using jnp.savez
            for i, params in enumerate(nn_weights):
                weight_matrix, bias_vector = params
                jnp.savez(f"nn_weights/layer_{i}_params.npz", weight=weight_matrix,
                          bias=bias_vector)

    def weight_stats(self):
        # record statistics about the weights
        weights_df = pd.DataFrame()

        num_layers = len(self.l2ws_model.params)
        mean_weights = np.zeros((num_layers, 2))
        min_weights = np.zeros((num_layers, 2))
        max_weights = np.zeros((num_layers, 2))
        std_dev_weights = np.zeros((num_layers, 2))
        norm_sq_weights = np.zeros((num_layers, 2))
        for i, params in enumerate(self.l2ws_model.params):
            weight_matrix, bias_vector = params
            mean_weights[i, 0] = weight_matrix.mean()
            mean_weights[i, 1] = bias_vector.mean()
            max_weights[i, 0] = weight_matrix.max()
            max_weights[i, 1] = bias_vector.max()
            min_weights[i, 0] = weight_matrix.min()
            min_weights[i, 1] = bias_vector.min()
            std_dev_weights[i, 0] = weight_matrix.std()
            std_dev_weights[i, 1] = bias_vector.std()

            norm_sq_weights[i, 0] = jnp.linalg.norm(weight_matrix) ** 2
            norm_sq_weights[i, 1] = jnp.linalg.norm(bias_vector) ** 2
        weights_df['means_weights'] = mean_weights[:, 0]
        weights_df['std_dev_weights'] = std_dev_weights[:, 0]
        weights_df['norm_sq_weights'] = norm_sq_weights[:, 0]
        weights_df['min_weghts'] = min_weights[:, 0]
        weights_df['max_weights'] = max_weights[:, 0]

        weights_df['means_bias'] = mean_weights[:, 1]
        weights_df['std_dev_bias'] = std_dev_weights[:, 1]
        weights_df['norm_sq_bias'] = norm_sq_weights[:, 1]
        weights_df['min_bias'] = min_weights[:, 1]
        weights_df['max_bias'] = max_weights[:, 1]
        weights_df.to_csv('weights_stats.csv')

    def finalize_genL2O(self, train, num_samples):
        '''
        part 1: first obtain H (num_samples) samples
        '''
        K = self.l2ws_model.eval_unrolls
        N = self.l2ws_model.N_train if train else self.l2ws_model.N_test
        # hist = np.zeros((num_samples, N, K-1))
        # hist = np.zeros((num_samples, K-1))

        # round the priors
        priors = self.l2ws_model.params[2]
        rounded_priors = self.l2ws_model.round_priors(
            priors, self.l2ws_model.c, self.l2ws_model.b)
        self.l2ws_model.params[2] = rounded_priors

        sample_conv_penalty = jnp.log(2 / self.l2ws_model.delta2) / num_samples
        mcallester_penalty = self.l2ws_model.calculate_total_penalty(self.l2ws_model.N_train,
                                                                     self.l2ws_model.params,
                                                                     self.l2ws_model.c,
                                                                     self.l2ws_model.b,
                                                                     self.l2ws_model.delta
                                                                     )

        sum_frac_solved = np.zeros((K - 1, len(self.frac_solved_accs)))
        all_losses = []
        for i in range(num_samples):
            eval_out = self.evaluate_only(fixed_ws=False, num=N, train=train,
                                          col='pac_bayes', batch_size=N)
            loss_train, out_train, train_time = eval_out
            losses_over_examples = out_train[1].T
            if i < 20:
                all_losses.append(losses_over_examples.T)
            self.l2ws_model.key += 1

            for j in range(len(self.frac_solved_accs)):
                fs = (out_train[1] < self.frac_solved_accs[j])
                frac_solved = fs.sum(axis=0)  # fs.mean(axis=0)
                sum_frac_solved[:, j] += frac_solved
        mean_frac_solved = sum_frac_solved / (N * num_samples)
        # hist[i, :, :] = out_train[1]
        # hist[i, :] = out_train[1].mean(axis=0)

        col = "train_epoch_0"

        # enter the percentiles
        # losses_over_examples = out_train[1].T
        # self.update_percentiles(losses_over_examples.T, train, col)
        self.update_percentiles(np.vstack(all_losses), train, col)

        '''
        part 2: finalize by looping over steps and tolerances  
        2.1: threshold 
        2.2: sample convergence bound
        2.3: McAllester bound
        '''

        # frac_solved_list = []
        if train:
            frac_solved_df_list = self.frac_solved_df_list_train
        else:
            frac_solved_df_list = self.frac_solved_df_list_test

        for i in range(len(self.frac_solved_accs)):
            # 2.1: compute frac solved
            # fs = (out_train[1] < self.frac_solved_accs[i])
            # frac_solved = fs.mean(axis=0)
            # frac_solved_list.append(frac_solved)
            frac_solved = mean_frac_solved[:, i]
            final_pac_bayes_loss = jnp.zeros(frac_solved.size)
            for j in range(frac_solved.size):
                # 2.2: sample convergence bound
                # R_bar = invert_kl(1 - frac_solved[j], sample_conv_penalty)
                R_bar = invert_kl(
                    1 - mean_frac_solved[j, i], sample_conv_penalty)

                # 2.3: McAllester
                if train:
                    R_star = invert_kl(R_bar, mcallester_penalty)
                else:
                    # R_star = invert_kl(1 - frac_solved[j], mcallester_penalty)
                    R_star = invert_kl(
                        1 - mean_frac_solved[j, i], mcallester_penalty)

                final_pac_bayes_loss = final_pac_bayes_loss.at[j].set(
                    1 - R_star)
                # print('risk', 1 - frac_solved[j], 'R_bar', R_bar, 'R_star', R_star)

            final_pac_bayes_frac_solved = jnp.clip(
                final_pac_bayes_loss, a_min=0)

            # update the df
            frac_solved_df_list[i][col] = frac_solved
            frac_solved_df_list[i][col + '_pinsker'] = jnp.clip(
                frac_solved - jnp.sqrt(mcallester_penalty / 2), a_min=0)
            # jnp.clip(frac_solved - penalty, a_min=0)
            frac_solved_df_list[i][col +
                                   '_pac_bayes'] = final_pac_bayes_frac_solved
            ylabel = f"frac solved tol={self.frac_solved_accs[i]}"
            filename = f"frac_solved/tol={self.frac_solved_accs[i]}"
            curr_df = frac_solved_df_list[i]

            # plot and update csv
            self.plot_eval_iters_df(curr_df, train, col, ylabel, filename, yscale='standard',
                                    pac_bayes=True)
            csv_filename = filename + '_train.csv' if train else filename + '_test.csv'
            curr_df.to_csv(csv_filename)

    def load_weights(self, example, datetime, nn_type):
        if nn_type == 'deterministic':
            # if self.l2ws_model.algo == 'alista':
            if self.l2ws_model.algo in ['alista', 'glista', 'lista', 'tilista']:
                self.load_weights_deterministic_alista(example, datetime)
            else:
                self.load_weights_deterministic(example, datetime)
        elif nn_type == 'stochastic':
            if self.l2ws_model.algo in ['alista', 'glista']:
                self.load_weights_stochastic_alista(example, datetime)
            elif self.l2ws_model.algo in ['lista', 'tilista']:
                self.load_weights_stochastic_lista(example, datetime)
            else:
                self.load_weights_stochastic(example, datetime)

    def load_weights_stochastic_alista(self, example, datetime):
        # get the appropriate folder
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights"

        # load the mean
        loaded_mean = jnp.load(f"{folder}/mean/mean_params.npz")
        mean_params = loaded_mean['mean_params']

        # load the variance
        loaded_variance = jnp.load(f"{folder}/variance/variance_params.npz")
        variance_params = loaded_variance['variance_params']

        # load the prior
        loaded_prior = jnp.load(f"{folder}/prior/prior_val.npz")
        prior = loaded_prior['prior']

        self.l2ws_model.params = [mean_params, variance_params, prior]

    def load_weights_stochastic_lista(self, example, datetime):
        # get the appropriate folder
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights"

        # load the mean
        loaded_mean = jnp.load(f"{folder}/mean/mean_params.npz")

        # load the variance
        loaded_variance = jnp.load(f"{folder}/variance/variance_params.npz")

        if 'thresh_params' in loaded_mean.keys():
            # case of lista
            mean_params = (loaded_mean['thresh_params'],
                           loaded_mean['W1_params'],
                           loaded_mean['W2_params'])

            variance_params = (loaded_variance['thresh_params'],
                               loaded_variance['W1_params'],
                               loaded_variance['W2_params'])
        else:
            # case of tilista
            mean_params = (loaded_mean['thresh_step_params'],
                           loaded_mean['W1_params'])

            variance_params = (loaded_variance['thresh_step_params'],
                               loaded_variance['W1_params'])

        # load the prior
        loaded_prior = jnp.load(f"{folder}/prior/prior_val.npz")
        prior = loaded_prior['prior']

        self.l2ws_model.params = [mean_params, variance_params, prior]

    def load_weights_deterministic_alista(self, example, datetime):
        # get the appropriate folder
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights"

        # load the mean
        loaded_mean = jnp.load(f"{folder}/mean/mean_params.npz")
        mean_params = loaded_mean['mean_params']
        self.l2ws_model.params[0] = mean_params

        # load the variance
        # loaded_variance = jnp.load(f"{folder}/variance/variance_params.npz")
        # variance_params = loaded_variance['variance_params']
        variance_params = jnp.log(jnp.abs(mean_params / 100))
        self.l2ws_model.params[1] = variance_params

        # load the prior
        # loaded_prior = jnp.load(f"{folder}/prior/prior_val.npz")
        # prior = loaded_prior['prior']

        # self.l2ws_model.params = [mean_params, variance_params, prior]

    def load_weights_stochastic(self, example, datetime):
        # get the appropriate folder
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights"

        # find the number of layers based on the number of files
        num_layers = count_files_in_directory(folder + "/mean")

        # load the mean
        mean_params = []
        for i in range(num_layers):
            layer_file = f"{folder}/mean/layer_{i}_params.npz"
            loaded_layer = jnp.load(layer_file)
            weight_matrix, bias_vector = loaded_layer['weight'], loaded_layer['bias']
            weight_bias_tuple = (weight_matrix, bias_vector)
            mean_params.append(weight_bias_tuple)

        # load the variance
        variance_params = []
        for i in range(num_layers):
            layer_file = f"{folder}/variance/layer_{i}_params.npz"
            loaded_layer = jnp.load(layer_file)
            weight_matrix, bias_vector = loaded_layer['weight'], loaded_layer['bias']
            weight_bias_tuple = (weight_matrix, bias_vector)
            variance_params.append(weight_bias_tuple)

        # load the prior
        loaded_prior = jnp.load(f"{folder}/prior/prior_val.npz")
        prior = loaded_prior['prior']

        self.l2ws_model.params = [mean_params, variance_params, prior]

    def load_weights_deterministic(self, example, datetime):
        # get the appropriate folder
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights"

        # len(nn_weights) == 3 and not isinstance(nn_weights[2], tuple)
        if os.path.isdir(folder + "/mean"):
            folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights/mean"

        # find the number of layers based on the number of files
        num_layers = count_files_in_directory(folder)

        # iterate over the files/layers
        params = []
        for i in range(num_layers):
            layer_file = f"{folder}/layer_{i}_params.npz"
            loaded_layer = jnp.load(layer_file)
            weight_matrix, bias_vector = loaded_layer['weight'], loaded_layer['bias']
            weight_bias_tuple = (weight_matrix, bias_vector)
            params.append(weight_bias_tuple)

        # store the weights as the l2ws_model params
        self.l2ws_model.params[0] = params

        # load the variances proportional to the means
        for i, params in enumerate(params):
            weight_matrix, bias_vector = params
            self.l2ws_model.params[1][i] = (jnp.log(jnp.abs(weight_matrix / 100)),
                                            jnp.log(jnp.abs(bias_vector / 100)))

    def normalize_inputs_fn(self, thetas, N_train, N_test):
        # normalize the inputs if the option is on
        N = N_train + N_test
        if self.normalize_inputs:
            col_sums = thetas.mean(axis=0)
            std_devs = thetas.std(axis=0)
            inputs_normalized = (thetas - col_sums) / \
                std_devs  # thetas.std(axis=0)
            inputs = jnp.array(inputs_normalized)

            # save the col_sums and std deviations
            self.normalize_col_sums = col_sums
            self.normalize_std_dev = std_devs
        else:
            inputs = jnp.array(thetas)
        train_inputs = inputs[:N_train, :]
        test_inputs = inputs[N_train:N, :]

        return train_inputs, test_inputs

    def normalize_theta(self, theta):
        normalized_input = (theta - self.normalize_col_sums) / \
            self.normalize_std_dev
        return normalized_input

    def load_setup_data(self, example, datetime, N_train, N):
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime}"
        filename = f"{folder}/data_setup.npz"

        if self.static_flag:
            jnp_load_obj = jnp.load(filename)
        else:
            jnp_load_obj = jnp.load(filename)
            q_mat = jnp.array(load_npz(f"{filename[:-4]}_q.npz").todense())

            # randomize for quadcopter
            # self.q_mat_train = q_mat[:N_train, :]
            # self.q_mat_test = q_mat[N_train:N, :]
            train_indices = np.random.choice(
                q_mat.shape[0], N_train, replace=False)
            self.q_mat_train = q_mat[train_indices, :]

            test_indices = np.random.choice(
                q_mat.shape[0], N - N_train, replace=False)
            self.q_mat_test = q_mat[test_indices, :]

            # load factors
            # factors0, factors1 = jnp_load_obj['factors0'], jnp_load_obj['factors1']
            # factors = (factors0, factors1)
            # jnp_load_obj['factors'] = factors

            # compute factors
            # all_factors_train is a tuple with shapes ((N, n + m, n + m), (N, n + m))
            # factors0, factors1 = self.batch_factors(q_mat)

            # if we are in the dynamic case, then we need to get q from the sparse format
            # jnp_load_obj['q_mat'] = jnp.array(q_mat_sparse)

            # self.factors_train = (jnp.array(factors0[:N_train, :, :]),
            #                       jnp.array(factors1[:N_train, :]))
            # self.factors_test = (jnp.array(factors0[N_train:N, :, :]),
            #                      jnp.array(factors1[N_train:N, :]))

        if 'q_mat' in jnp_load_obj.keys():
            q_mat = jnp.array(jnp_load_obj['q_mat'])
            q_mat_train = q_mat[:N_train, :]
            q_mat_test = q_mat[N_train:N, :]
            self.q_mat_train, self.q_mat_test = q_mat_train, q_mat_test
        # elif self.algo == 'extragradient':
        #     q_mat = jnp.array(jnp_load_obj['thetas'])
        #     q_mat_train = q_mat[:N_train, :]
        #     q_mat_test = q_mat[N_train:N, :]
        #     self.q_mat_train, self.q_mat_test = q_mat_train, q_mat_test

        # load the closed_loop_rollout trajectories
        if 'ref_traj_tensor' in jnp_load_obj.keys():
            # load all of the goals
            self.closed_loop_rollout_dict['ref_traj_tensor'] = jnp_load_obj['ref_traj_tensor']

        return jnp_load_obj

    def plot_samples(self, num_plot, thetas, train_inputs, z_stars):
        sample_plot(thetas, 'theta', num_plot)
        sample_plot(train_inputs, 'input', num_plot)
        if z_stars is not None:
            sample_plot(z_stars, 'z_stars', num_plot)

    def plot_samples_scs(self, num_plot, thetas, train_inputs, x_stars, y_stars, z_stars):
        sample_plot(thetas, 'theta', num_plot)
        sample_plot(train_inputs, 'input', num_plot)
        if x_stars is not None:
            sample_plot(x_stars, 'x_stars', num_plot)
            sample_plot(y_stars, 'y_stars', num_plot)
            sample_plot(z_stars, 'z_stars', num_plot)

    def init_custom_visualization(self, cfg, custom_visualize_fn):
        iterates_visualize = cfg.get('iterates_visualize', 0)
        if custom_visualize_fn is None or iterates_visualize == 0:
            self.has_custom_visualization = False
        else:
            self.has_custom_visualization = True
            self.custom_visualize_fn = custom_visualize_fn
            self.iterates_visualize = iterates_visualize

    def _init_logging(self):
        self.logf = open('log.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'val_loss',
                      'test_loss', 'time_train_per_epoch']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.writer.writeheader()

        self.test_logf = open('log_test.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'test_loss', 'time_per_iter']
        self.test_writer = csv.DictWriter(
            self.test_logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.test_writer.writeheader()

        self.logf = open('train_results.csv', 'a')

        fieldnames = ['train_loss', 'moving_avg_train', 'time_train_per_epoch']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('train_results.csv').st_size == 0:
            self.writer.writeheader()

        self.test_logf = open('train_test_results.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'test_loss', 'penalty', 'avg_posterior_var',
                      'stddev_posterior_var', 'prior', 'mean_squared_w', 'time_per_iter']
        self.test_writer = csv.DictWriter(
            self.test_logf, fieldnames=fieldnames)
        if os.stat('train_results.csv').st_size == 0:
            self.test_writer.writeheader()

    def evaluate_iters(self, num, col, train=False, plot=True, plot_pretrain=False):
        fixed_ws = col == 'nearest_neighbor'

        # do the actual evaluation (most important step in thie method)
        eval_batch_size = self.eval_batch_size_train if train else self.eval_batch_size_test
        eval_out = self.evaluate_only(
            fixed_ws, num, train, col, eval_batch_size)

        # extract information from the evaluation
        loss_train, out_train, train_time = eval_out
        iter_losses_mean = out_train[1].mean(axis=0)
        # angles = out_train[3]
        # iter_losses_mean = out_train[2].mean(axis=0)
        # angles = out_train[3]
        # primal_residuals = out_train[4].mean(axis=0)
        # dual_residuals = out_train[5].mean(axis=0)

        # plot losses over examples
        # losses_over_examples = out_train[2].T
        losses_over_examples = out_train[1].T

        yscalelog = False if self.l2ws_model.algo in [
            'glista', 'lista_cpss', 'lista', 'alista', 'tilista'] else True
        if min(self.frac_solved_accs) < 0:
            yscalelog = False
        self.plot_losses_over_examples(
            losses_over_examples, train, col, yscalelog=yscalelog)

        # update the eval csv files
        primal_residuals, dual_residuals, obj_vals_diff = None, None, None
        if len(out_train) == 6 or len(out_train) == 8:
            primal_residuals = out_train[4].mean(axis=0)
            dual_residuals = out_train[5].mean(axis=0)
        elif len(out_train) == 5:
            obj_vals_diff = out_train[4].mean(axis=0)

        self.update_percentiles(losses_over_examples.T, train, col)

        df_out = self.update_eval_csv(
            iter_losses_mean, train, col,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
            obj_vals_diff=obj_vals_diff
        )
        iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df = df_out

        if not self.skip_startup:
            # write accuracies dataframe to csv
            self.write_accuracies_csv(iter_losses_mean, train, col)

        # plot the evaluation iterations
        self.plot_eval_iters(iters_df, primal_residuals_df,
                             dual_residuals_df, plot_pretrain, obj_vals_diff_df, train, col)

        # take care of frac_solved
        frac_solved_list = []
        frac_solved_df_list = self.frac_solved_df_list_train if train else self.frac_solved_df_list_test  # noqa: E501

        cache = {}
        for i in range(len(self.frac_solved_accs)):
            # compute frac solved
            fs = (out_train[1] < self.frac_solved_accs[i])
            frac_solved = fs.mean(axis=0)
            frac_solved_list.append(frac_solved)

            if col in ['no_train', 'nearest_neighbor']:
                penalty = jnp.log(2 / self.l2ws_model.delta) / \
                    self.l2ws_model.N_train
            else:
                penalty = self.l2ws_model.calculate_total_penalty(self.l2ws_model.N_train,
                                                                  self.l2ws_model.params,
                                                                  self.l2ws_model.c,
                                                                  self.l2ws_model.b,
                                                                  self.l2ws_model.delta
                                                                  )
            final_pac_bayes_loss = jnp.zeros(frac_solved.size)
            for j in range(frac_solved.size):
                if self.rep:
                    if float(frac_solved[j]) in cache.keys():
                        kl_inv = cache[float(frac_solved[j])]
                    else:
                        kl_inv = invert_kl(1 - frac_solved[j], penalty)
                        cache[float(frac_solved[j])] = kl_inv
                    final_pac_bayes_loss = final_pac_bayes_loss.at[j].set(
                        1 - kl_inv)
                else:
                    kl_inv = jnp.clip(
                        frac_solved[j] - jnp.sqrt(penalty / 2), a_min=0)
                    final_pac_bayes_loss = final_pac_bayes_loss.at[j].set(
                        kl_inv)
            final_pac_bayes_frac_solved = jnp.clip(
                final_pac_bayes_loss, a_min=0)

            # update the df
            frac_solved_df_list[i][col] = frac_solved
            frac_solved_df_list[i][col + '_pinsker'] = jnp.clip(
                frac_solved - jnp.sqrt(penalty / 2), a_min=0)
            # jnp.clip(frac_solved - penalty, a_min=0)
            frac_solved_df_list[i][col +
                                   '_pac_bayes'] = final_pac_bayes_frac_solved
            ylabel = f"frac solved tol={self.frac_solved_accs[i]}"
            filename = f"frac_solved/tol={self.frac_solved_accs[i]}"
            curr_df = frac_solved_df_list[i]

            # plot and update csv
            self.plot_eval_iters_df(curr_df, train, col, ylabel, filename, yscale='standard',
                                    pac_bayes=True)
            csv_filename = filename + '_train.csv' if train else filename + '_test.csv'
            curr_df.to_csv(csv_filename)

        # plot the warm-start predictions
        z_all = out_train[2]

        if isinstance(self.l2ws_model, SCSmodel):
            out_train[6]

            z_plot = z_all[:, :, :-1] / z_all[:, :, -1:]
        else:
            z_plot = z_all

        if self.l2ws_model.algo != 'maml':
            self.plot_warm_starts(None, z_plot, train, col)
        else:
            self.plot_maml(z_plot, train, col)

        # custom visualize
        if self.has_custom_visualization:
            if self.vis_num > 0:
                self.custom_visualize(z_plot, train, col)

        # closed loop control rollouts
        if not train:
            if self.closed_loop_rollout_dict is not None:
                self.run_closed_loop_rollouts(col)

        # Specify the CSV file name
        filename = 'z_star_max2.csv'

        if self.l2ws_model.algo == 'gd':
            # Open the file in write mode
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)

                # Write the scalar value to the file
                # writer.writerow([z_star_max])
                # writer.writerow([theta_max])
                for i in range(len(self.l2ws_model.params[0])):
                    U, S, VT = jnp.linalg.svd(self.l2ws_model.params[0][i][0])

                    max_size = jnp.max(
                        jnp.array(self.l2ws_model.params[0][i][0].shape))

                    sigma = jnp.exp(jnp.max(self.l2ws_model.params[1][i][0]))
                    ti = S.max() + max_size * sigma * jnp.log(100 * 2 * max_size)
                    writer.writerow([ti])

        if self.save_weights_flag:
            self.save_weights()
        gc.collect()

        return out_train

    def get_xys_from_z(self, z_init):
        """
        z = (x, y + s, 1)
        we always set the last entry of z to be 1
        we allow s to be zero (we just need z[n:m + n] = y + s)
        """
        m, n = self.l2ws_model.m, self.l2ws_model.n
        x = z_init[:n]
        y = z_init[n:n + m]
        s = jnp.zeros(m)
        return x, y, s

    def custom_visualize(self, z_all, train, col):
        """
        x_primals has shape [N, eval_iters]
        """
        visualize_path = 'visualize_train' if train else 'visualize_test'

        if not os.path.exists(visualize_path):
            os.mkdir(visualize_path)
        if not os.path.exists(f"{visualize_path}/{col}"):
            os.mkdir(f"{visualize_path}/{col}")

        visual_path = f"{visualize_path}/{col}"

        # call custom visualize fn
        if train:
            z_stars = self.z_stars_train
            thetas = self.thetas_train
            if 'z_nn_train' in dir(self):
                z_nn = self.z_nn_train
        else:
            z_stars = self.z_stars_test
            thetas = self.thetas_test
            if 'z_nn_test' in dir(self):
                z_nn = self.z_nn_test
            if 'z_prev_sol_test' in dir(self):
                z_prev_sol = self.z_prev_sol_test
            else:
                z_prev_sol = None

        if col == 'no_train':
            if train:
                self.z_no_learn_train = z_all  # [:self.vis_num, :, :]
            else:
                self.z_no_learn_test = z_all  # x_primals[:self.vis_num, :, :]
        elif col == 'nearest_neighbor':
            if train:
                self.z_nn_train = z_all  # x_primals[:self.vis_num, :, :]
            else:
                self.z_nn_test = z_all  # x_primals[:self.vis_num, :, :]
        if train:
            z_no_learn = self.z_no_learn_train
        else:
            z_no_learn = self.z_no_learn_test

        if train:
            if col != 'nearest_neighbor' and col != 'no_train' and col != 'prev_sol':
                self.custom_visualize_fn(z_all, z_stars, z_no_learn, z_nn,
                                         thetas, self.iterates_visualize, visual_path)
            else:
                # try plotting
                self.custom_visualize_fn(z_all, z_stars, z_no_learn, None,
                                         thetas, self.iterates_visualize, visual_path)
        else:
            if col != 'nearest_neighbor' and col != 'no_train' and col != 'prev_sol':
                if z_prev_sol is None:
                    self.custom_visualize_fn(z_all, z_stars, z_no_learn, z_nn,
                                             thetas, self.iterates_visualize, visual_path,
                                             num=self.vis_num)
                else:
                    self.custom_visualize_fn(z_all, z_stars, z_prev_sol, z_nn,
                                             thetas, self.iterates_visualize, visual_path,
                                             num=self.vis_num)

    def run(self):
        # setup logging and dataframes
        self._init_logging()
        self.setup_dataframes()

        # set pretrain_on boolean
        self.pretrain_on = self.pretrain_cfg.pretrain_iters > 0

        if not self.skip_startup:
            # fixed ws evaluation
            if self.l2ws_model.z_stars_train is not None and self.l2ws_model.algo != 'maml':
                self.eval_iters_train_and_test('nearest_neighbor', False)

            # no learning evaluation
            self.eval_iters_train_and_test('no_train', False)

            # pretrain evaluation
            if self.pretrain_on:
                self.pretrain()

        # load the weights AFTER the cold-start
        if self.load_weights_datetime is not None:
            self.load_weights(
                self.example, self.load_weights_datetime, self.nn_load_type)

        # eval test data to start
        self.test_eval_write()

        # do all of the training
        test_zero = True if self.skip_startup else False
        self.train(test_zero=test_zero)

    def train(self, test_zero=False):
        """
        does all of the training
        jits together self.epochs_jit number of epochs together
        writes results to filesystem
        """
        num_epochs_jit = int(self.l2ws_model.epochs / self.epochs_jit)
        loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)

        # key_count updated to get random permutation for each epoch
        # key_count = 0
        if not self.skip_pac_bayes_full:
            self.finalize_genL2O(
                train=True, num_samples=self.pac_bayes_num_samples)
            self.finalize_genL2O(train=False, num_samples=500)
            return

        if self.pac_bayes_hyperparameter_opt_flag:
            self.pac_bayes_hyperparameter_opt(self.pac_bayes_num_samples,
                                              self.sigma_nn_grid, self.sigma_beta_grid)

        for epoch_batch in range(num_epochs_jit):
            epoch = int(epoch_batch * self.epochs_jit)
            if (test_zero and epoch == 0) or (epoch % self.eval_every_x_epochs == 0 and epoch > 0):
                self.eval_iters_train_and_test(
                    f"train_epoch_{epoch}", self.pretrain_on)

            # setup the permutations
            permutation = setup_permutation(
                self.key_count, self.l2ws_model.N_train, self.epochs_jit)

            # train the jitted epochs
            params, state, epoch_train_losses, time_train_per_epoch = self.train_jitted_epochs(
                permutation, epoch)

            # reset the global (params, state)
            self.key_count += 1
            self.l2ws_model.epoch += self.epochs_jit
            self.l2ws_model.params, self.l2ws_model.state = params, state

            gc.collect()

            prev_batches = len(self.l2ws_model.tr_losses_batch)
            self.l2ws_model.tr_losses_batch = self.l2ws_model.tr_losses_batch + \
                list(epoch_train_losses)

            # write train results
            self.write_train_results(loop_size, prev_batches,
                                     epoch_train_losses, time_train_per_epoch)

            # evaluate the test set and write results
            self.test_eval_write()

            # plot the train / test loss so far
            if epoch % self.save_every_x_epochs == 0:
                self.plot_train_test_losses()

    def train_jitted_epochs(self, permutation, epoch):
        """
        train self.epochs_jit at a time
        special case: the first time we call train_batch (i.e. epoch = 0)
        """
        epoch_batch_start_time = time.time()
        loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)
        epoch_train_losses = jnp.zeros(loop_size)
        if epoch == 0:
            # unroll the first iterate so that This allows `init_val` and `body_fun`
            #   below to have the same output type, which is a requirement of
            #   lax.while_loop and lax.scan.
            batch_indices = lax.dynamic_slice(
                permutation, (0,), (self.l2ws_model.batch_size,))

            train_loss_first, params, state = self.l2ws_model.train_batch(
                batch_indices, self.l2ws_model.params, self.l2ws_model.state)

            epoch_train_losses = epoch_train_losses.at[0].set(train_loss_first)
            start_index = 1
            # self.train_over_epochs_body_simple_fn_jitted = jit(self.train_over_epochs_body_simple_fn)  # noqa
            self.train_over_epochs_body_simple_fn_jitted = self.train_over_epochs_body_simple_fn
        else:
            start_index = 0
            params, state = self.l2ws_model.params, self.l2ws_model.state

        init_val = epoch_train_losses, params, state, permutation
        val = lax.fori_loop(start_index, loop_size,
                            self.train_over_epochs_body_simple_fn_jitted, init_val)
        epoch_batch_end_time = time.time()
        time_diff = epoch_batch_end_time - epoch_batch_start_time
        time_train_per_epoch = time_diff / self.epochs_jit
        epoch_train_losses, params, state, permutation = val

        self.l2ws_model.key = state.iter_num

        return params, state, epoch_train_losses, time_train_per_epoch

    def train_over_epochs_body_simple_fn(self, batch, val):
        """
        to be used as the body_fn in lax.fori_loop
        need to call partial for the specific permutation
        """
        train_losses, params, state, permutation = val
        start_index = batch * self.l2ws_model.batch_size
        batch_indices = lax.dynamic_slice(
            permutation, (start_index,), (self.l2ws_model.batch_size,))
        train_loss, params, state = self.l2ws_model.train_batch(
            batch_indices, params, state)
        train_losses = train_losses.at[batch].set(train_loss)
        val = train_losses, params, state, permutation
        return val

    def train_over_epochs_body_fn(self, batch, val, permutation):
        """
        to be used as the body_fn in lax.fori_loop
        need to call partial for the specific permutation
        """
        train_losses, params, state = val
        start_index = batch * self.l2ws_model.batch_size
        batch_indices = lax.dynamic_slice(
            permutation, (start_index,), (self.l2ws_model.batch_size,))
        train_loss, params, state = self.l2ws_model.train_batch(
            batch_indices, params, state)
        train_losses = train_losses.at[batch].set(train_loss)
        val = train_losses, params, state
        return val

    def write_accuracies_csv(self, losses, train, col):
        df_acc = pd.DataFrame()
        df_acc['accuracies'] = np.array(self.accs)

        if train:
            accs_path = 'accuracies_train'
        else:
            accs_path = 'accuracies_test'
        if not os.path.exists(accs_path):
            os.mkdir(accs_path)
        if not os.path.exists(f"{accs_path}/{col}"):
            os.mkdir(f"{accs_path}/{col}")

        # accuracies
        iter_vals = np.zeros(len(self.accs))
        for i in range(len(self.accs)):
            if losses.min() < self.accs[i]:
                iter_vals[i] = int(np.argmax(losses < self.accs[i]))
            else:
                iter_vals[i] = losses.size
        int_iter_vals = iter_vals.astype(int)
        df_acc[col] = int_iter_vals
        df_acc.to_csv(f"{accs_path}/{col}/accuracies.csv")

        # save no learning accuracies
        if not hasattr(self, 'no_learning_accs'):  # col == 'no_train':
            self.no_learning_accs = int_iter_vals

        # percent reduction
        df_percent = pd.DataFrame()
        df_percent['accuracies'] = np.array(self.accs)

        for col in df_acc.columns:
            if col != 'accuracies':
                val = 1 - df_acc[col] / self.no_learning_accs
                df_percent[col] = np.round(val, decimals=2)
        df_percent.to_csv(f"{accs_path}/{col}/reduction.csv")

    def eval_iters_train_and_test(self, col, pretrain_on):
        self.evaluate_iters(
            self.num_samples_test, col, train=False, plot_pretrain=pretrain_on)
        self.evaluate_iters(
            self.num_samples_train, col, train=True, plot_pretrain=pretrain_on)

    def write_train_results(self, loop_size, prev_batches, epoch_train_losses,
                            time_train_per_epoch):
        for batch in range(loop_size):
            start_window = prev_batches - 10 + batch
            end_window = prev_batches + batch
            last10 = np.array(
                self.l2ws_model.tr_losses_batch[start_window:end_window])
            moving_avg = last10.mean()
            self.writer.writerow({
                'train_loss': epoch_train_losses[batch],
                'moving_avg_train': moving_avg,
                'time_train_per_epoch': time_train_per_epoch
            })
            self.logf.flush()

    def evaluate_only(self, fixed_ws, num, train, col, batch_size):
        tag = 'train' if train else 'test'
        if self.static_flag:
            factors = None
        else:
            if train:
                factors = (
                    self.factors_train[0][:num, :, :], self.factors_train[1][:num, :])
            else:
                factors = (
                    self.factors_test[0][:num, :, :], self.factors_test[1][:num, :])
        if self.l2ws_model.z_stars_train is None:
            z_stars = None
        else:
            if train:
                z_stars = self.l2ws_model.z_stars_train[:num, :]
            else:
                z_stars = self.l2ws_model.z_stars_test[:num, :]
        if col == 'prev_sol':
            if train:
                q_mat_full = self.l2ws_model.q_mat_train[:num, :]
            else:
                q_mat_full = self.l2ws_model.q_mat_test[:num, :]
            non_first_indices = jnp.mod(jnp.arange(num), self.traj_length) != 0
            q_mat = q_mat_full[non_first_indices, :]
            z_stars = z_stars[non_first_indices, :]
            if factors is not None:
                factors = (factors[0][non_first_indices, :, :],
                           factors[1][non_first_indices, :])
        else:
            q_mat = self.l2ws_model.q_mat_train[:num,
                                                :] if train else self.l2ws_model.q_mat_test[:num, :]

        inputs = self.get_inputs_for_eval(fixed_ws, num, train, col)

        # do the batching
        num_batches = int(num / batch_size)
        full_eval_out = []
        if num_batches <= 1:
            eval_out = self.l2ws_model.evaluate(
                self.eval_unrolls, inputs, q_mat, z_stars, fixed_ws, factors=factors, tag=tag)
            return eval_out

        for i in range(num_batches):
            print('evaluating batch num', i)
            start = i * batch_size
            end = (i + 1) * batch_size
            curr_inputs = inputs[start: end]
            curr_q_mat = q_mat[start: end]

            if factors is not None:
                curr_factors = (
                    factors[0][start:end, :, :], factors[1][start:end, :])
            else:
                curr_factors = None
            if z_stars is not None:
                curr_z_stars = z_stars[start: end]
            else:
                curr_z_stars = None
            eval_out = self.l2ws_model.evaluate(
                self.eval_unrolls, curr_inputs, curr_q_mat, curr_z_stars, fixed_ws,
                factors=curr_factors, tag=tag)
            # full_eval_out.append(eval_out)
            # eval_out_cpu = tuple(item.copy_to_host() for item in eval_out)
            # full_eval_out.append(eval_out_cpu)
            eval_out1_list = [eval_out[1][i] for i in range(len(eval_out[1]))]
            eval_out1_list[2] = eval_out1_list[2][:, :25, :]
            if isinstance(self.l2ws_model, SCSmodel):
                eval_out1_list[6] = eval_out1_list[6][:, :25, :]
            eval_out_cpu = (eval_out[0], tuple(eval_out1_list), eval_out[2])
            full_eval_out.append(eval_out_cpu)
            del eval_out
            del eval_out_cpu
            del eval_out1_list
            gc.collect()
        loss = np.array([curr_out[0] for curr_out in full_eval_out]).mean()
        time_per_prob = np.array([curr_out[2]
                                 for curr_out in full_eval_out]).mean()
        out = self.stack_tuples([curr_out[1] for curr_out in full_eval_out])

        flattened_eval_out = (loss, out, time_per_prob)
        return flattened_eval_out

    def stack_tuples(self, tuples_list):
        result = []
        num_tuples = len(tuples_list)
        tuple_length = len(tuples_list[0])

        for i in range(tuple_length):
            stacked_entry = []
            for j in range(num_tuples):
                stacked_entry.append(tuples_list[j][i])
            # result.append(tuple(stacked_entry))
            if tuples_list[j][i] is None:
                result.append(None)
            elif tuples_list[j][i].ndim == 2:
                result.append(jnp.vstack(stacked_entry))
            elif tuples_list[j][i].ndim == 1:
                result.append(jnp.hstack(stacked_entry))
            # elif tuples_list[j][i].ndim == 3 and i == 0:
            #     result.append(jnp.vstack(stacked_entry))
            elif tuples_list[j][i].ndim == 3:
                result.append(jnp.vstack(stacked_entry))
        return result

    def get_inputs_for_eval(self, fixed_ws, num, train, col):
        if fixed_ws:
            if col == 'nearest_neighbor':
                inputs = self.get_nearest_neighbors(train, num)
        else:
            if train:
                inputs = self.l2ws_model.train_inputs[:num, :]
            else:
                inputs = self.l2ws_model.test_inputs[:num, :]
        return inputs

    def theta_2_nearest_neighbor(self, theta):
        """
        given a new theta returns the closest training problem solution
        """
        # first normalize theta
        test_input = self.normalize_theta(theta)

        # make it a matrix
        test_inputs = jnp.expand_dims(test_input, 0)

        distances = distance_matrix(
            np.array(test_inputs),
            np.array(self.l2ws_model.train_inputs))
        indices = np.argmin(distances, axis=1)
        if isinstance(self.l2ws_model, OSQPmodel):
            return self.l2ws_model.z_stars_train[indices, :self.m + self.n]
        else:
            return self.l2ws_model.z_stars_train[indices, :]

    def get_nearest_neighbors(self, train, num):
        if train:
            distances = distance_matrix(
                np.array(self.l2ws_model.train_inputs[:num, :]),
                np.array(self.l2ws_model.train_inputs))
        else:
            distances = distance_matrix(
                np.array(self.l2ws_model.test_inputs[:num, :]),
                np.array(self.l2ws_model.train_inputs))
        indices = np.argmin(distances, axis=1)
        plt.plot(indices)
        if train:
            plt.savefig("indices_train_plot.pdf", bbox_inches='tight')
        else:
            plt.savefig("indices_train_plot.pdf", bbox_inches='tight')
        plt.clf()
        if isinstance(self.l2ws_model, OSQPmodel):
            return self.l2ws_model.z_stars_train[indices, :self.m + self.n]
        return self.l2ws_model.z_stars_train[indices, :]

    def setup_dataframes(self):
        self.iters_df_train = pd.DataFrame(
            columns=['iterations', 'no_train'])
        self.iters_df_train['iterations'] = np.arange(1, self.eval_unrolls+1)

        self.iters_df_test = pd.DataFrame(
            columns=['iterations', 'no_train'])
        self.iters_df_test['iterations'] = np.arange(1, self.eval_unrolls+1)

        # primal and dual residuals
        self.primal_residuals_df_train = pd.DataFrame(
            columns=['iterations'])
        self.primal_residuals_df_train['iterations'] = np.arange(
            1, self.eval_unrolls+1)
        self.dual_residuals_df_train = pd.DataFrame(
            columns=['iterations'])
        self.dual_residuals_df_train['iterations'] = np.arange(
            1, self.eval_unrolls+1)

        self.primal_residuals_df_test = pd.DataFrame(
            columns=['iterations'])
        self.primal_residuals_df_test['iterations'] = np.arange(
            1, self.eval_unrolls+1)
        self.dual_residuals_df_test = pd.DataFrame(
            columns=['iterations'])
        self.dual_residuals_df_test['iterations'] = np.arange(
            1, self.eval_unrolls+1)

        # obj_vals_diff
        self.obj_vals_diff_df_train = pd.DataFrame(
            columns=['iterations'])
        self.obj_vals_diff_df_train['iterations'] = np.arange(
            1, self.eval_unrolls+1)
        self.obj_vals_diff_df_test = pd.DataFrame(
            columns=['iterations'])
        self.obj_vals_diff_df_test['iterations'] = np.arange(
            1, self.eval_unrolls+1)

        # setup solve times
        self.agg_solve_times_df_train = pd.DataFrame()
        self.agg_solve_times_df_train['rel_tol'] = self.rel_tols
        self.agg_solve_times_df_train['abs_tol'] = self.abs_tols
        self.agg_solve_iters_df_train = pd.DataFrame()
        self.agg_solve_iters_df_train['rel_tol'] = self.rel_tols
        self.agg_solve_iters_df_train['abs_tol'] = self.abs_tols

        self.agg_solve_times_df_test = pd.DataFrame()
        self.agg_solve_times_df_test['rel_tol'] = self.rel_tols
        self.agg_solve_times_df_test['abs_tol'] = self.abs_tols
        self.agg_solve_iters_df_test = pd.DataFrame()
        self.agg_solve_iters_df_test['rel_tol'] = self.rel_tols
        self.agg_solve_iters_df_test['abs_tol'] = self.abs_tols

        self.frac_solved_df_list_train = []
        for i in range(len(self.frac_solved_accs)):
            self.frac_solved_df_list_train.append(
                pd.DataFrame(columns=['iterations']))
        self.frac_solved_df_list_test = []
        for i in range(len(self.frac_solved_accs)):
            self.frac_solved_df_list_test.append(
                pd.DataFrame(columns=['iterations']))
        if not os.path.exists('frac_solved'):
            os.mkdir('frac_solved')

        self.percentiles = [10, 20, 30, 40, 50, 60, 70, 80,
                            90, 95, 96, 97, 98, 99]  # [30, 50, 80, 90, 95, 99]
        self.percentiles_df_list_train = []
        self.percentiles_df_list_test = []
        for i in range(len(self.percentiles)):
            self.percentiles_df_list_train.append(
                pd.DataFrame(columns=['iterations']))
        for i in range(len(self.percentiles)):
            self.percentiles_df_list_test.append(
                pd.DataFrame(columns=['iterations']))

    def train_full(self):
        print("Training full...")
        pretrain_on = True
        self.full_train_df = pd.DataFrame(
            columns=['pretrain_loss', 'pretrain_test_loss'])
        pretrain_out = self.l2ws_model.train_full(self.l2ws_model.static_algo_factor,
                                                  self.proj,
                                                  self.train_unrolls,
                                                  1000,
                                                  stepsize=1e-4,
                                                  df_fulltrain=self.full_train_df,
                                                  batches=10)
        train_pretrain_losses, test_pretrain_losses = pretrain_out
        self.evaluate_iters(
            self.num_samples_train, 'pretrain', train=True, plot_pretrain=pretrain_on)
        self.evaluate_iters(
            self.num_samples_test, 'pretrain', train=False, plot_pretrain=pretrain_on)
        plt.plot(train_pretrain_losses, label='train')
        plt.plot(test_pretrain_losses, label='test')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('full train loss')
        plt.legend()
        plt.savefig('losses.pdf')
        plt.clf()

    def test_eval_write(self):
        test_loss, time_per_iter = self.l2ws_model.short_test_eval()
        last_epoch = np.array(
            self.l2ws_model.tr_losses_batch[-self.l2ws_model.num_batches:])
        moving_avg = last_epoch.mean()

        # do penalty calculation
        total_pen = self.l2ws_model.calculate_total_penalty(self.l2ws_model.N_train,
                                                            self.l2ws_model.params,
                                                            self.l2ws_model.c,
                                                            self.l2ws_model.b,
                                                            self.l2ws_model.delta
                                                            )
        pen = jnp.sqrt(total_pen / 2)

        # calculate avg posterior var
        avg_posterior_var, stddev_posterior_var = self.l2ws_model.calculate_avg_posterior_var(
            self.l2ws_model.params)

        mean_squared_w, dim = self.l2ws_model.compute_weight_norm_squared(
            self.l2ws_model.params[0])

        print('mean', self.l2ws_model.params[0])
        print('var', self.l2ws_model.params[1])

        if self.test_writer is not None:
            self.test_writer.writerow({
                'iter': self.l2ws_model.state.iter_num,
                'train_loss': moving_avg,
                'test_loss': test_loss,
                'penalty': pen,
                'avg_posterior_var': avg_posterior_var,
                'stddev_posterior_var': stddev_posterior_var,
                'prior': jnp.exp(self.l2ws_model.params[2]),
                'mean_squared_w': mean_squared_w,
                'time_per_iter': time_per_iter
            })
            self.test_logf.flush()

    def plot_train_test_losses(self):
        batch_losses = np.array(self.l2ws_model.tr_losses_batch)
        te_losses = np.array(self.l2ws_model.te_losses)
        num_data_points = batch_losses.size
        epoch_axis = np.arange(num_data_points) / \
            self.l2ws_model.num_batches

        epoch_test_axis = self.epochs_jit * np.arange(te_losses.size)
        plt.plot(epoch_axis, batch_losses, label='train')
        plt.plot(epoch_test_axis, te_losses, label='test')
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('fixed point residual average')
        plt.legend()
        plt.savefig('losses_over_training.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(epoch_axis, batch_losses, label='train')

        # include when learning rate decays
        if len(self.l2ws_model.epoch_decay_points) > 0:
            epoch_decay_points = self.l2ws_model.epoch_decay_points
            epoch_decay_points_np = np.array(epoch_decay_points)
            batch_decay_points = epoch_decay_points_np * self.l2ws_model.num_batches

            batch_decay_points_int = batch_decay_points.astype('int')
            decay_vals = batch_losses[batch_decay_points_int]
            plt.scatter(epoch_decay_points_np, decay_vals,
                        c='r', label='lr decay')
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('fixed point residual average')
        plt.legend()
        plt.savefig('train_losses_over_training.pdf', bbox_inches='tight')
        plt.clf()

    def update_percentiles(self, losses, train, col):
        # update the percentiles
        path = 'percentiles'
        if not os.path.exists(path):
            os.mkdir(path)
        for i in range(len(self.percentiles)):
            if train:
                filename = f"{path}/train_{self.percentiles[i]}.csv"
                curr_percentile = np.percentile(
                    losses, self.percentiles[i], axis=0)
                self.percentiles_df_list_train[i][col] = curr_percentile
                self.percentiles_df_list_train[i].to_csv(filename)
            else:
                filename = f"{path}/test_{self.percentiles[i]}.csv"
                curr_percentile = np.percentile(
                    losses, self.percentiles[i], axis=0)
                self.percentiles_df_list_test[i][col] = curr_percentile
                self.percentiles_df_list_test[i].to_csv(filename)

    def update_eval_csv(self, iter_losses_mean, train, col, primal_residuals=None,
                        dual_residuals=None, obj_vals_diff=None):
        # def update_eval_csv(self, iter_losses_mean, train, col):
        """
        update the eval csv files
            fixed point residuals
            primal residuals
            dual residuals
        returns the new dataframes
        """
        primal_residuals_df, dual_residuals_df = None, None
        obj_vals_diff_df = None
        if train:
            self.iters_df_train[col] = iter_losses_mean
            self.iters_df_train.to_csv('iters_compared_train.csv')
            if primal_residuals is not None:
                self.primal_residuals_df_train[col] = primal_residuals
                self.primal_residuals_df_train.to_csv(
                    'primal_residuals_train.csv')
                self.dual_residuals_df_train[col] = dual_residuals
                self.dual_residuals_df_train.to_csv('dual_residuals_train.csv')
                primal_residuals_df = self.primal_residuals_df_train
                dual_residuals_df = self.dual_residuals_df_train
            if obj_vals_diff is not None:
                self.obj_vals_diff_df_train[col] = obj_vals_diff
                self.obj_vals_diff_df_train.to_csv('obj_vals_diff_train.csv')
                obj_vals_diff_df = self.obj_vals_diff_df_train
            iters_df = self.iters_df_train

        else:
            self.iters_df_test[col] = iter_losses_mean
            self.iters_df_test.to_csv('iters_compared_test.csv')
            if primal_residuals is not None:
                self.primal_residuals_df_test[col] = primal_residuals
                self.primal_residuals_df_test.to_csv(
                    'primal_residuals_test.csv')
                self.dual_residuals_df_test[col] = dual_residuals
                self.dual_residuals_df_test.to_csv('dual_residuals_test.csv')
                primal_residuals_df = self.primal_residuals_df_test
                dual_residuals_df = self.dual_residuals_df_test
            if obj_vals_diff is not None:
                self.obj_vals_diff_df_test[col] = obj_vals_diff
                self.obj_vals_diff_df_test.to_csv('obj_vals_diff_test.csv')
                obj_vals_diff_df = self.obj_vals_diff_df_test

            iters_df = self.iters_df_test

        return iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df

    def plot_eval_iters_df(self, df, train, col, ylabel, filename,
                           xlabel='evaluation iterations',
                           xvals=None,
                           yscale='log', pac_bayes=False):
        if xvals is None:
            xvals = np.arange(self.eval_unrolls)
        # plot the cold-start if applicable
        if 'no_train' in df.keys():

            plt.plot(xvals, df['no_train'], 'k-', label='no learning')

        # plot the nearest_neighbor if applicable
        if col != 'no_train' and 'nearest_neighbor' in df.keys():
            plt.plot(xvals, df['nearest_neighbor'],
                     'm-', label='nearest neighbor')

        # plot the learned warm-start if applicable
        if col != 'no_train' and col != 'pretrain' and col != 'nearest_neighbor' and col != 'prev_sol':  # noqa
            plt.plot(xvals, df[col], label=f"train k={self.train_unrolls}")
            if pac_bayes:
                plt.plot(xvals, df[col + '_pac_bayes'], label="pac_bayes")
        if yscale == 'log':
            plt.yscale('log')
        # plt.xlabel('evaluation iterations')
        plt.xlabel(xlabel)
        plt.ylabel(f"test {ylabel}")
        plt.legend()
        if train:
            plt.title('train problems')
            plt.savefig(f"{filename}_train.pdf", bbox_inches='tight')
        else:
            plt.title('test problems')
            plt.savefig(f"{filename}_test.pdf", bbox_inches='tight')
        plt.clf()

    def plot_eval_iters(self, iters_df, primal_residuals_df, dual_residuals_df, plot_pretrain,
                        obj_vals_diff_df,
                        train, col):
        # self.plot_eval_iters_df(curr_df, train, col, ylabel, filename, yscale=yscale,
        #                         pac_bayes=True)
        yscale = 'standard' if self.l2ws_model.algo in [
            'glista', 'lista_cpss', 'lista', 'alista', 'tilista'] else 'log'
        if self.nmse:
            yscale = 'standard'
        self.plot_eval_iters_df(
            iters_df, train, col, 'fixed point residual', 'eval_iters', yscale=yscale)
        if primal_residuals_df is not None:
            self.plot_eval_iters_df(primal_residuals_df, train, col,
                                    'primal residual', 'primal_residuals')
            self.plot_eval_iters_df(dual_residuals_df, train, col,
                                    'dual residual', 'dual_residuals')
        if obj_vals_diff_df is not None:
            self.plot_eval_iters_df(
                obj_vals_diff_df, train, col, 'obj diff', 'obj_diffs')

    def plot_losses_over_examples(self, losses_over_examples, train, col, yscalelog=True):
        """
        plots the fixed point residuals over eval steps for each individual problem
        """
        if train:
            loe_folder = 'losses_over_examples_train'
        else:
            loe_folder = 'losses_over_examples_test'
        if not os.path.exists(loe_folder):
            os.mkdir(loe_folder)

        plt.plot(losses_over_examples)

        if yscalelog:
            plt.yscale('log')
        plt.savefig(f"{loe_folder}/losses_{col}_plot.pdf", bbox_inches='tight')
        plt.clf()

    def plot_maml(self, z_all, train, col):
        if train:
            ws_path = 'warm-starts_train'
        else:
            ws_path = 'warm-starts_test'
        if not os.path.exists(ws_path):
            os.mkdir(ws_path)
        if not os.path.exists(f"{ws_path}/{col}"):
            os.mkdir(f"{ws_path}/{col}")

        num = int(self.l2ws_model.z_stars_train[0, :].size / 2)
        num_theta = int(self.thetas_train[0, :].size / 2)

        for i in range(20):
            for j in self.plot_iterates:

                # plt.plot(z_all[i, j, :], label=f"prediction_{j}")
                prediction = z_all[i, j, :]
                if train:
                    x_vals = self.l2ws_model.z_stars_train[i, :num]
                    y_vals = self.l2ws_model.z_stars_train[i, num:]
                    x_grad_points = self.thetas_train[i, :num_theta]
                    y_grad_points = self.thetas_train[i, num_theta:]
                else:
                    x_vals = self.l2ws_model.z_stars_test[i, :num]
                    y_vals = self.l2ws_model.z_stars_test[i, num:]
                    x_grad_points = self.thetas_test[i, :num_theta]
                    y_grad_points = self.thetas_test[i, num_theta:]
                sorted_indices = np.argsort(x_vals)
                plt.plot(x_vals[sorted_indices],
                         y_vals[sorted_indices], label='optimal')
                plt.plot(x_vals[sorted_indices],
                         prediction[sorted_indices], label=f"prediction_{j}")

                plt.fill_between(x_vals[sorted_indices], y_vals[sorted_indices] - 1,
                                 y_vals[sorted_indices] + 1, color='red', alpha=0.2)

                # mark the points used for gradients
                plt.scatter(x_grad_points, y_grad_points,
                            color='green', marker='^')

                plt.legend()
                plt.savefig(f"{ws_path}/{col}/prob_{i}_z_ws.pdf")
                plt.clf()
                df_acc = pd.DataFrame()
                df_acc['x_vals'] = x_vals[sorted_indices]
                df_acc['y_vals'] = y_vals[sorted_indices]
                df_acc['predicted_y_vals'] = prediction[sorted_indices]
                df_acc.to_csv(f"{ws_path}/{col}/prob_{i}_z_ws.csv")

                df_grad_points = pd.DataFrame()
                df_grad_points['x_grad_points'] = x_grad_points
                df_grad_points['y_grad_points'] = y_grad_points
                df_grad_points.to_csv(f"{ws_path}/{col}/grad_points_{i}.csv")

    def plot_warm_starts(self, u_all, z_all, train, col):
        """
        plots the warm starts for the given method

        we give plots for
            x: primal variable
            y: dual variable
            z: base Douglas-Rachford iterate (dual of primal-dual variable)

        train is a boolean

        plots the first 5 problems and

        self.plot_iterates is a list
            e.g. [0, 10, 20]
            tells us to plot
                (z^0, z^10, z^20, z_opt) for each of the first 5 problems
                AND do a separate plot for
                (z^0 - z_opt, z^10 - z_opt, z^20 - z_opt) for each of the first 5 problems
        """
        if train:
            ws_path = 'warm-starts_train'
        else:
            ws_path = 'warm-starts_test'
        if not os.path.exists(ws_path):
            os.mkdir(ws_path)
        if not os.path.exists(f"{ws_path}/{col}"):
            os.mkdir(f"{ws_path}/{col}")
        # m, n = self.l2ws_model.m, self.l2ws_model.n
        for i in range(5):
            # plot for z
            for j in self.plot_iterates:
                plt.plot(z_all[i, j, :], label=f"prediction_{j}")
            if train:
                plt.plot(self.l2ws_model.z_stars_train[i, :], label='optimal')
            else:
                plt.plot(self.l2ws_model.z_stars_test[i, :], label='optimal')
            plt.legend()
            plt.savefig(f"{ws_path}/{col}/prob_{i}_z_ws.pdf")
            plt.clf()

            for j in self.plot_iterates:
                if isinstance(self.l2ws_model, OSQPmodel):
                    try:
                        plt.plot(z_all[i, j, :self.l2ws_model.m + self.l2ws_model.n] -
                                 self.l2ws_model.z_stars_train[i, :],
                                 label=f"prediction_{j}")
                    except:
                        plt.plot(z_all[i, j, :self.l2ws_model.m + self.l2ws_model.n] -
                                 self.l2ws_model.z_stars_train[i, :self.l2ws_model.m + self.l2ws_model.n],  # noqa
                                 label=f"prediction_{j}")
                else:
                    if train:
                        plt.plot(z_all[i, j, :] - self.l2ws_model.z_stars_train[i, :],
                                 label=f"prediction_{j}")
                    else:
                        plt.plot(z_all[i, j, :] - self.l2ws_model.z_stars_test[i, :],
                                 label=f"prediction_{j}")
            plt.legend()
            plt.title('diffs to optimal')
            plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_z.pdf")
            plt.clf()
