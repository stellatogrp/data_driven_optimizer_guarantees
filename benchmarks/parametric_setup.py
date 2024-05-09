import sys

import hydra

import opt_guarantees.examples.lasso as lasso
import opt_guarantees.examples.mnist as mnist
import opt_guarantees.examples.quadcopter as quadcopter
import opt_guarantees.examples.robust_kalman as robust_kalman
import opt_guarantees.examples.sine as sine
import opt_guarantees.examples.sparse_coding as sparse_coding
import opt_guarantees.examples.unconstrained_qp as unconstrained_qp


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_setup.yaml')
def main_setup_quadcopter(cfg):
    quadcopter.setup_probs(cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_setup.yaml')
def main_setup_mnist(cfg):
    mnist.setup_probs(cfg)


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_setup.yaml')
def main_setup_robust_kalman(cfg):
    robust_kalman.setup_probs(cfg)


@hydra.main(config_path='configs/lasso', config_name='lasso_setup.yaml')
def main_setup_lasso(cfg):
    lasso.setup_probs(cfg)


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_setup.yaml')
def main_setup_unconstrained_qp(cfg):
    unconstrained_qp.setup_probs(cfg)


@hydra.main(config_path='configs/sparse_coding', config_name='sparse_coding_setup.yaml')
def main_setup_sparse_coding(cfg):
    sparse_coding.setup_probs(cfg)


@hydra.main(config_path='configs/sine', config_name='sine_setup.yaml')
def main_setup_sine(cfg):
    sine.setup_probs(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'robust_kalman':
        sys.argv[1] = base + 'robust_kalman/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_robust_kalman()
    elif sys.argv[1] == 'lasso':
        sys.argv[1] = base + 'lasso/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_lasso()
    elif sys.argv[1] == 'unconstrained_qp':
        sys.argv[1] = base + 'unconstrained_qp/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_unconstrained_qp()
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_quadcopter()
    elif sys.argv[1] == 'mnist':
        sys.argv[1] = base + 'mnist/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_mnist()
    elif sys.argv[1] == 'sparse_coding':
        sys.argv[1] = base + 'sparse_coding/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_sparse_coding()
    elif sys.argv[1] == 'sine':
        sys.argv[1] = base + 'sine/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_sine()
