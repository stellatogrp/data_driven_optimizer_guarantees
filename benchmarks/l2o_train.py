import sys

import hydra

import opt_guarantees.examples.sine as sine
import opt_guarantees.examples.sparse_coding as sparse_coding
import opt_guarantees.examples.unconstrained_qp as unconstrained_qp
from opt_guarantees.utils.data_utils import copy_data_file, recover_last_datetime


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_run.yaml')
def main_run_unconstrained_qp(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'unconstrained_qp'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    unconstrained_qp.run(cfg)


@hydra.main(config_path='configs/sparse_coding', config_name='sparse_coding_run.yaml')
def main_run_sparse_coding(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'sparse_coding'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    sparse_coding.run(cfg)


@hydra.main(config_path='configs/sine', config_name='sine_run.yaml')
def main_run_sine(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'sine'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    sine.run(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'unconstrained_qp':
        sys.argv[1] = base + 'unconstrained_qp/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_unconstrained_qp()
    elif sys.argv[1] == 'sparse_coding':
        sys.argv[1] = base + 'sparse_coding/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_sparse_coding()
    elif sys.argv[1] == 'sine':
        sys.argv[1] = base + 'sine/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_sine()
