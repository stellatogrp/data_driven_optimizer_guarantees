# dataDrivenOptimizerGuarantees

This repository is by
[Rajiv Sambharya](https://rajivsambharya.github.io/) and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"[Data-Driven Performance Guarantees for Classical and Learned Optimizers](https://arxiv.org/pdf/2404.13831)."


If you find this repository helpful in your publications, please consider citing our papers.

# Abstract
We introduce a data-driven approach to analyze the performance of continuous optimization algorithms using generalization guarantees from statistical learning theory. We study classical and learned optimizers to solve families of parametric optimization problems. We build generalization guarantees for classical optimizers, using a sample convergence bound, and for learned optimizers, using the Probably Approximately Correct (PAC)-Bayes framework. To train learned optimizers, we use a gradient-based algorithm to directly minimize the PAC-Bayes upper bound. Numerical experiments in signal processing, control, and meta-learning showcase the ability of our framework to provide strong generalization guarantees for both classical and learned optimizers given a fixed budget of iterations. For classical optimizers, our bounds are much tighter than those that worst-case guarantees provide. For learned optimizers, our bounds outperform the empirical outcomes observed in their non-learned counterparts.

## Installation
To install the opt_guarantees package, run
```
git clone https://github.com/stellatogrp/data_driven_optimizer_guarantees.git
$ pip install -e ".[dev]"
```

## Guarantees for classical optimizers

### Running experiments for classical optimizers for robust Kalman filtering

Run the following commands to obtain the guarantees for the fixed-point residual:

- ```python benchmarks/parametric_setup.py robust_kalman local```
- ```python benchmarks/classical_run_and_bound.py robust_kalman_fp local``` (with `N_train` set to `10` in the file `benchmarks/configs/robust_kalman/robust_kalman_run_fp.yaml`)
- ```python benchmarks/classical_run_and_bound.py robust_kalman_fp local``` (with `N_train` set to `100` in the file `benchmarks/configs/robust_kalman/robust_kalman_run_fp.yaml`)
- ```python benchmarks/classical_run_and_bound.py robust_kalman_fp local``` (with `N_train` set to `1000` in the file `benchmarks/configs/robust_kalman/robust_kalman_run_fp.yaml`)
- ```python benchmarks/plot_genL2O.py robust_kalman_fp local``` (with `cold_start_datetimes` set to a list of the datetimes that correspond to the folders of the previous three commands in the file `benchmarks/configs/robust_kalman/robust_kalman_plot_fp.yaml`)

After this, run the following commands to obtain the guarantees for the maximum Euclidean metric:
- ```python benchmarks/classical_run_and_bound.py robust_kalman_custom local``` (with `N_train` set to `10` in the file `benchmarks/configs/robust_kalman/robust_kalman_run_custom.yaml`)
- ```python benchmarks/classical_run_and_bound.py robust_kalman_custom local``` (with `N_train` set to `100` in the file `benchmarks/configs/robust_kalman/robust_kalman_run_custom.yaml`)
- ```python benchmarks/classical_run_and_bound.py robust_kalman_custom local``` (with `N_train` set to `1000` in the file `benchmarks/configs/robust_kalman/robust_kalman_run_custom.yaml`)
- ```python benchmarks/plot_genL2O.py robust_kalman_custom local``` (with `cold_start_datetimes` set to a list of the datetimes that correspond to the folders of the previous three commands in the file `benchmarks/configs/robust_kalman/robust_kalman_plot_custom.yaml`)


### Running experiments for classical optimizers for image deblurring

Run the following commands to obtain the guarantees for the fixed-point residual:

- ```python benchmarks/parametric_setup.py mnist local```
- ```python benchmarks/classical_run_and_bound.py mnist_fp local``` (with `N_train` set to `10` in the file `benchmarks/configs/mnist/mnist_run_fp.yaml`)
- ```python benchmarks/classical_run_and_bound.py mnist_fp local``` (with `N_train` set to `100` in the file `benchmarks/configs/mnist/mnist_run_fp.yaml`)
- ```python benchmarks/classical_run_and_bound.py mnist_fp local``` (with `N_train` set to `1000` in the file `benchmarks/configs/mnist/mnist_run_fp.yaml`)
- ```python benchmarks/plot_genL2O.py mnist_fp local``` (with `cold_start_datetimes` set to a list of the datetimes that correspond to the folders of the previous three commands in the file `benchmarks/configs/mnist/mnist_plot_fp.yaml`)

After this, run the following commands to obtain the guarantees for the maximum Euclidean metric:
- ```python benchmarks/classical_run_and_bound.py mnist_custom local``` (with `N_train` set to `10` in the file `benchmarks/configs/mnist/mnist_run_custom.yaml`)
- ```python benchmarks/classical_run_and_bound.py mnist_custom local``` (with `N_train` set to `100` in the file `benchmarks/configs/mnist/mnist_run_custom.yaml`)
- ```python benchmarks/classical_run_and_bound.py mnist_custom local``` (with `N_train` set to `1000` in the file `benchmarks/configs/mnist/mnist_run_custom.yaml`)
- ```python benchmarks/plot_genL2O.py mnist_custom local``` (with `cold_start_datetimes` set to a list of the datetimes that correspond to the folders of the previous three commands in the file `benchmarks/configs/mnist/mnist_plot_custom.yaml`)

### Running experiments for classical optimizers for quadcopter control

Run the following commands to obtain the guarantees for the fixed-point residual:

- ```python benchmarks/parametric_setup.py quadcopter local```
- ```python benchmarks/classical_run_and_bound.py quadcopter local``` (with `N_train` set to `10` in the file `benchmarks/configs/quadcopter/quadcopter_run.yaml`)
- ```python benchmarks/classical_run_and_bound.py quadcopter local``` (with `N_train` set to `100` in the file `benchmarks/configs/quadcopter/quadcopter_run.yaml`)
- ```python benchmarks/classical_run_and_bound.py quadcopter local``` (with `N_train` set to `1000` in the file `benchmarks/configs/quadcopter/quadcopter_run.yaml`)
- ```python benchmarks/plot_genL2O.py quadcopter local``` (with `cold_start_datetimes` set to a list of the datetimes that correspond to the folders of the previous three commands in the file `benchmarks/configs/mnist/quadcopter_plot.yaml`)

## Guarantees for learned optimizers

***
#### ```parametric_setup.py```

The first script ```parametric_setup.py``` creates all of the problem instances and solves them.
The number of problems that are being solved is set in the setup config file.
That config file also includes other parameters that define the problem instances. 
This only needs to be run once for each example.
After running this script, the results are saved a file in
```
outputs/robust_kalman/data_setup_outputs/2024-05-03/14-54-32/
```

***
#### ```l2o_train.py```

The second script ```l2o_train.py``` trains the learned optimizer using the output from the prevous setup command.
- Runs the fixed-point algorithm for ```eval_unrolls``` across the ```N_train``` number of problems.
- Evaluates the empirical risk up to ```eval_unrolls``` number of iterations across $81$ tolerances evenly spaced out on a log-scale between $10^{-6}$ and $10^2$ for the fixed-point residual. For other metrics we use a different set of tolerances (see our paper for details).
- Computes the KL inverse for each algorithm step and each tolerance.

In particular, in the config file, it takes a datetime that points to the setup output.
By default, it takes the most recent setup if this pointer is empty.
The config file holds information about the actual training process.
Run this file for each $k$ value to train for that number of fixed-point steps.
Each run for a given $k$ and the loss function creates an output folder like
To replicate our results in the paper, the only input that needs to be changed is the one that determines the number of samples.
- ```N_train``` (an integer that is the number of samples)


#### ```l2o_calibrate.py```
The third script ```l2o_calibrate.py``` does the calibration step to obtain the final generalization guarantees for learned optimizers.
That is, given the trained optimizer (whose weights were saved in the last step)

```
outputs/robust_kalman/train_outputs/2024-05-04/15-14-05/
```
In this folder there are many metrics that are stored.
We highlight the mains ones here (both the raw data in csv files and the corresponding plots in pdf files).


- Fixed-point residuals over the test problems 

    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/plots/iters_compared_test.csv```
    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/plots/eval_iters_test.pdf```

- Fixed-point residuals over the training problems 

    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/plots/iters_compared_train.csv```
    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/eval_iters_train.pdf```

- Losses over epochs: for training this holds the average loss (for either loss function), for testing we plot the fixed-point residual at $k$ steps

    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/train_test_results.csv```
    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/losses_over_training.pdf```

- The ```accuracies``` folder holds the results that are used for the tables. First, it holds the average number of iterations to reach the desired accuracies ($0.1$, $0.01$, $0.001$, and $0.0001$ by default).
Second, it holds the reduction in iterations in comparison to the cold start.

    ```outputs/quadcopter/2022-12-03/14-54-32/plots/accuracies```





***
#### ```plot_genL2O.py```

The third script ```plot.py``` plots the results across many different training runs.
Each train run creates a new folder 
```
outputs/quadcopter/plots/2022-06-04/15-14-05/
```



For the image deblurring task, we use the EMNIST dataset found at https://www.nist.gov/itl/products-and-services/emnist-dataset and use pip to install emnist (https://pypi.org/project/emnist/). 


Adjust the config files to try different settings; for example, the number of train/test data, number of evaluation iterations, and the number of training steps.
Additionally, the neural network and problem setup configurations can be updated.
We automatically use the most recent output after each stage, but the specific datetime can be inputted. Additionally, the final evaluation plot can take in multiple training datetimes in a list. See the commented out lines in the config files.

***


# Important files in the backend
To reproduce our results, this part is not needed.

- The ```opt_guarantees/examples``` folder holds the code for each of the numerical experiments we run. The main purpose is to be used in conjunction with the ```parametric_setup.py```.

- An important note is that the code is set to periodically evaluate the train and test sets; this is set in the ```eval_every_x_epochs``` entry in the run config file.
When we evaluate, the fixed-point curves are updated (see the above files for the run config).


***

The ```opt_guarantees``` folder holds the code that implements our architecture and allows for the training. In particular,

- ```opt_guarantees/launcher.py``` is the workspace which holds the L2Omodel below.
All of the evaluation and training is run through

- ```opt_guarantees/algo_steps.py``` holds all of the code that runs the algorithms

    - the fixed-point algorithms follow the same form in case you want to try your own algorithm

- ```opt_guarantees/l2o_model.py``` holds the L2Omodel object, i.e., the architecture. This code allows us to 
    - evaluate the problems (both test and train) for any initialization technique
    - train the learned optimizers with given parameters
