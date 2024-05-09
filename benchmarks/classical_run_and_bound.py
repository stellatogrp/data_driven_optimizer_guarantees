import sys

import hydra

import opt_guarantees.examples.lasso as lasso
import opt_guarantees.examples.mnist as mnist
import opt_guarantees.examples.quadcopter as quadcopter
import opt_guarantees.examples.robust_kalman as robust_kalman
from opt_guarantees.utils.data_utils import copy_data_file, recover_last_datetime


@hydra.main(config_path='configs/lasso', config_name='lasso_run.yaml')
def main_run_lasso(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'lasso'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    lasso.run(cfg)


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_run.yaml')
def main_run_quadcopter(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'quadcopter'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    quadcopter.run(cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_run.yaml')
def main_run_mnist(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'mnist'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    mnist.run(cfg)




@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_run.yaml')
def main_run_robust_kalman(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'robust_kalman'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    robust_kalman.run(cfg)



if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'robust_kalman':
        sys.argv[1] = base + 'robust_kalman/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_robust_kalman()
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_quadcopter()
    elif sys.argv[1] == 'mnist':
        sys.argv[1] = base + 'mnist/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_mnist()

