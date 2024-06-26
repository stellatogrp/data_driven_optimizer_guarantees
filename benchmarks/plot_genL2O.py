import sys

import hydra
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import yaml
from pandas import read_csv

from opt_guarantees.utils.data_utils import recover_last_datetime

from PEPit import PEP
from PEPit.operators import LipschitzOperator
from PEPit.examples.fixed_point_problems import wc_krasnoselskii_mann_constant_step_sizes

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 25,
    # "font.size": 16,
})
cmap = plt.cm.Set1
colors = cmap.colors
titles_2_colors = dict(cold_start='black', 
                       nearest_neighbor=colors[3],
                       worst=colors[6], 
                       prev_sol=colors[4],
                       reg_k0=colors[3],
                       reg_k2=colors[0],
                       reg_k5=colors[0],
                       reg_k5_deterministic=colors[2],
                       reg_k10=colors[0],
                       reg_k10_deterministic=colors[2],
                       reg_k15=colors[0],
                       reg_k15_deterministic=colors[2],
                       reg_k30=colors[5],
                       reg_k60=colors[2],
                    #    reg_k120=colors[0],
                       obj_k0=colors[3],
                       obj_k5=colors[0],
                       obj_k15=colors[0],
                       obj_k15_deterministic=colors[2],
                       obj_k30=colors[5],
                       obj_k60=colors[2])
                    #    obj_k120='gray')
titles_2_styles = dict(cold_start='-.', 
                       nearest_neighbor='-.', 
                       worst='-.',
                       prev_sol='-.',
                       reg_k0='-',
                       reg_k2='-',
                       reg_k5='-',
                       reg_k5_deterministic='-',
                       reg_k10='-',
                       reg_k10_deterministic='-',
                       reg_k15='-',
                       reg_k15_deterministic='-',
                       reg_k30='-',
                       reg_k60='-',
                       reg_k120='-',
                       obj_k0='-',
                       obj_k5='-',
                       obj_k15='-',
                       obj_k15_deterministic='-',
                       obj_k30='-',
                       obj_k60='-')
                    #    obj_k120='-')
titles_2_markers = dict(cold_start='v', 
                        nearest_neighbor='^',
                       worst='<', 
                       prev_sol='^',
                       reg_k0='>',
                       reg_k2='o',
                       reg_k5='o',
                       reg_k5_deterministic='D',
                       reg_k10='o',
                       reg_k10_deterministic='D',
                       reg_k15='o',
                       reg_k15_deterministic='D',
                       reg_k30='x',
                       reg_k60='D',
                    #    reg_k120='-',
                       obj_k0='>',
                       obj_k5='o',
                       obj_k15='o',
                       obj_k15_deterministic='D',
                       obj_k30='x',
                       obj_k60='D')
titles_2_marker_starts = dict(cold_start=0, 
                       nearest_neighbor=16, 
                       prev_sol=23,
                       reg_k0=8,
                       reg_k2=4,
                       reg_k5=4,
                       reg_k10=4,
                       reg_k15=12,
                       reg_k30=0,
                       reg_k60=20,
                    #    reg_k120='-',
                       obj_k0=8,
                       obj_k5=4,
                       obj_k15=12,
                       obj_k30=0,
                       obj_k60=20)


@hydra.main(config_path='configs/sparse_coding', config_name='sparse_coding_plot.yaml')
def sparse_coding_plot_eval_iters(cfg):
    example = 'sparse_coding'
    percentile_plots_maml(example, cfg)
    create_gen_l2o_results_maml(example, cfg)
    return



def create_classical_results(example, cfg):
    # metrics, timing_data, titles = get_all_data(example, cfg, train=False)
    # eval_iters = int(cfg.eval_iters)

    # if len(titles) == 4:
    #     titles[-2] = titles[-2] + '_deterministic'
    # nmse = metrics[0]
    # for i in range(len(nmse)):
    #     plt.plot(nmse[i][:eval_iters],
    #              linestyle=titles_2_styles[titles[i]], 
    #              color=titles_2_colors[titles[i]],
    #              marker=titles_2_markers[titles[i]],
    #              markevery=(0, 100)
    #              )
    # plt.tight_layout()
    # plt.xlabel('iterations')
    # plt.ylabel("fixed-point residual")
    # plt.yscale('log')
    # plt.savefig('fp_res.pdf', bbox_inches='tight')
    # plt.clf()

    percentile_plots(example, cfg)
    risk_plots(example, cfg)
    if cfg.worst_case:
        worst_case_gap_plot(example, cfg)
    # pep()


def worst_case_gap_plot(example, cfg):
    plt.figure(figsize=(10,6))

    # get the cold start results
    # metrics, timing_data, titles = get_all_data(example, cfg, train=False)
    # cold_start = metrics[0][0]
    
    if example == 'mnist':
        num_steps = 8000
    elif example == 'robust_kalman':
        num_steps = 500
    

    # get the worst case results
    z_star_max, theta_max = get_worst_case_datetime(example, cfg)
    steps = np.arange(num_steps)
    worst_case = z_star_max / np.sqrt((steps + 2) / np.exp(1))
    plt.plot(worst_case, label='Theoretical worst-case bound',
             linestyle=titles_2_styles['worst'], 
                 color=titles_2_colors['worst'],
                 marker=titles_2_markers['worst'],
                 markevery=0.1)
    
    
    
    cold_start_results, guarantee_results = get_frac_solved_data_classical(example, cfg)
    # percentile_results = get_percentiles(example, cfg)
    train = True
    col = 'no_train'
    percentiles = cfg.get('percentiles', [30, 50, 90, 99])
    percentile_results = get_percentiles(example, cfg, cfg.percentile_datetime, train, col, percentiles)

    cold_start_quantile = percentile_results[3][:8000]
    print('cold_start_quantile', cold_start_quantile)

    # 99th quantile
    plt.plot(cold_start_quantile, label='Sample 99th quantile',
             linestyle=titles_2_styles['nearest_neighbor'], 
                 color=titles_2_colors['nearest_neighbor'],
                 marker=titles_2_markers['nearest_neighbor'],
                 markevery=0.1)

    # 50th quantile
    median = percentile_results[1][:num_steps]
    plt.plot(median, label='Sample average',
             linestyle=titles_2_styles['cold_start'], 
                 color=titles_2_colors['cold_start'],
                 marker=titles_2_markers['cold_start'],
                 markevery=0.1)

    # plt.tight_layout()
    plt.legend(loc='lower left')
    if example == 'mnist':
        plt.xscale('log')
    plt.xlabel('iterations')
    plt.ylabel("fixed-point residual")
    plt.yscale('log')
    plt.savefig('worst_case_analysis_gap.pdf', bbox_inches='tight')
    plt.clf()
    return


def get_steps(cold_start_size, eval_iters, worst_case):
    # steps1 = np.arange(cold_start_size)
    steps1 = np.arange(min(cold_start_size, eval_iters))
    steps2 = np.logspace(np.log10(cold_start_size), np.log10(eval_iters), 100000) # np.linspace(cold_start_size, eval_iters, 100000)
    if worst_case:
        steps = np.concatenate([steps1, steps2])
    else:
        steps = steps1
    return steps


def construct_worst(z_star_max, steps, lin_conv_rate):
    avg = 1 / np.sqrt((steps + 2) / np.exp(1)) * z_star_max * 1.1
    worst = avg
    for i in range(1, steps.size):
        worst[i] = min(worst[i], worst[i-1] * lin_conv_rate ** (steps[i] - steps[i - 1]))
    return worst


def get_worse_indices(z_star_max, steps, acc, lin_conv_rate, custom_loss=False):
    worst = construct_worst(z_star_max, steps, lin_conv_rate)
    worst_case_indices = worst < acc
    if custom_loss:
        worst_case_indices = worst < -100
    return worst_case_indices


def construct_full_curve(partial_curve, steps, worst_case_indices):
    full_curve = np.zeros(steps.size)
    curr_size = partial_curve.size

    # prob bounds
    full_curve[:curr_size] = partial_curve

    # will only improve as more iterations are evaluated
    full_curve[curr_size:] = partial_curve.max()

    # covered by worst case analysis
    full_curve[worst_case_indices] = 1.0
    return full_curve


def risk_plots(example, cfg):
    accuracies = get_accs(cfg)
    if cfg.worst_case:
        eval_iters = int(cfg.get('worst_case_iters', cfg.eval_iters))
    else:
        eval_iters = cfg.eval_iters
    if example == 'quadcopter':
        cold_start_results, guarantee_results, aux_results = get_frac_solved_data_classical_aux(example, cfg)
    else:
        cold_start_results, guarantee_results = get_frac_solved_data_classical(example, cfg)
        aux_results = None

    # if example == 'quadcopter':
    #     train = False
    #     col = 'no_train'
    #     aux_results = get_percentiles(example, cfg, cfg.cold_start_percentile_datetime, train, col, percentiles)

    cold_start_size = cold_start_results[-1].size
    steps = get_steps(cold_start_size, eval_iters, cfg.worst_case)

    if cfg.worst_case:
        z_star_max, theta_max = get_worst_case_datetime(example, cfg)

    acc_list, bounds_list_list = [], []
    cold_start_list, worst_list = [], []
    aux_quantile_list = []

    for i in range(len(accuracies)):
        acc = accuracies[i]
        if example == 'mnist':
            lin_conv_rate = 0.99995
        elif example == 'robust_kalman':
            lin_conv_rate = 1.0
        else:
            lin_conv_rate = 1.0
        
        if cfg.worst_case:
            worst_case_indices = get_worse_indices(z_star_max, steps, acc, lin_conv_rate, custom_loss=cfg.custom_loss)
            cold_start_curve = construct_full_curve(cold_start_results[i], steps, worst_case_indices)
        else:
            cold_start_curve = cold_start_results[i][:eval_iters]

        if aux_results is not None:
            aux_quantile = aux_results[i][:eval_iters]
        else:
            aux_quantile = None
        aux_quantile_list.append(aux_quantile)

        if cfg.worst_case:
            worst_case_curve = np.zeros(steps.size)
            worst_case_curve[worst_case_indices] = 1.0
        else:
            worst_case_curve = None

        # plot the bounds
        curr_pac_bayes_results = guarantee_results[i]
        bounds_list = []
        for j in range(len(curr_pac_bayes_results)):
            if cfg.worst_case:
                curr_curve = construct_full_curve(curr_pac_bayes_results[j], 
                                                 steps, worst_case_indices)
            else:
                curr_curve = curr_pac_bayes_results[j][:eval_iters]
            bounds_list.append(curr_curve)
        if acc in cfg.get('risk_accs', [0.1, 0.01, 0.001]):
            plot_final_classical_risk_bounds(acc, steps, bounds_list, cold_start_curve, 
                                            worst_case_curve, cfg.worst_case, cfg.custom_loss)
        
            acc_list.append(acc)
            bounds_list_list.append(bounds_list)
            cold_start_list.append(cold_start_curve)
            worst_list.append(worst_case_curve)
    plot_final_classical_risk_bounds_together(example, acc_list, steps, bounds_list_list, cold_start_list,
                                     worst_list, cfg.worst_case, cfg.custom_loss, aux_quantile_list=aux_quantile_list)



def plot_final_classical_risk_bounds_together(example, acc_list, steps, bounds_list_list, cold_start_list,
                                     worst_list, worst_case, custom_loss, aux_quantile_list=None):
    markers = ['o', 's', '>', 'D'] #['o', 's', '<', 'D']
    cmap = plt.cm.Set1
    colors = cmap.colors
    fig, axes = plt.subplots(nrows=1, ncols=len(acc_list), figsize=(20, 5), sharey='row')

    fontsize = 30
    title_fontsize = 30

    # y-label
    ylabel = r'$1 - r_{\mathcal{X}}$'
    # ylabel = r'prob. of reaching $\epsilon$'
    axes[0].set_ylabel(ylabel, fontsize=fontsize)

    for k in range(len(acc_list)):
        loc = len(acc_list) - 1 - k
        acc = acc_list[k]
        bounds_list = bounds_list_list[k]
        cold_start = cold_start_list[k]
        worst = worst_list[k]

        num_bounds = len(bounds_list)
        for j in range(num_bounds):
            axes[loc].plot(steps, bounds_list[j], 
                            color=colors[j], 
                            alpha=0.6,
                            markevery=0.1,
                            marker=markers[j])
        axes[loc].plot(steps,
                    cold_start, 
                    linestyle=titles_2_styles['cold_start'], 
                    color=titles_2_colors['cold_start'],
                    marker=titles_2_markers['cold_start'],
                    linewidth=2.0,
                    markevery=(0.05, 0.1)
                    )
        if aux_quantile_list is not None and aux_quantile_list[0] is not None:
            aux_quantile = aux_quantile_list[k]
            axes[loc].plot(aux_quantile, 
                        color=titles_2_colors['cold_start'], 
                        linestyle=titles_2_styles['cold_start'],
                        marker=titles_2_markers['cold_start'],
                        markevery=(0.05, 0.1))
            axes[loc].plot(cold_start, 
                        color=titles_2_colors['nearest_neighbor'], 
                        linestyle=titles_2_styles['nearest_neighbor'],
                        marker=titles_2_markers['nearest_neighbor'],
                        markevery=(0.05, 0.1))
        if worst_case and not custom_loss:
            axes[loc].plot(steps,
                        worst, 
                        linestyle=titles_2_styles['worst'], 
                        color=titles_2_colors['worst'],
                        marker=titles_2_markers['worst'],
                        linewidth=2.0,
                        markevery=(0.05, 0.1)
                        )
        axes[loc].set_xlabel('iterations', fontsize=fontsize)
        
        if custom_loss:
            if example == 'robust_kalman':
                title = r'max Euclidean distance: $\epsilon={}$'.format(acc)
            elif example == 'mnist':
                title = r'NMSE (dB): $\epsilon={}$'.format(np.round(acc, 1))
        else:
            title = r'fixed-point residual: $\epsilon={}$'.format(acc)
        axes[loc].set_title(title, fontsize=title_fontsize)
        if worst_case:
            axes[k].set_xscale('log')
        
        # axes[loc].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("risk_together.pdf", bbox_inches='tight')
    plt.clf()


def plot_final_classical_risk_bounds(acc, steps, bounds_list, cold_start, worst, 
                                     worst_case, custom_loss):
    markers = ['o', 's', '<', 'D']
    cmap = plt.cm.Set1
    colors = cmap.colors
    num_bounds = len(bounds_list)
    for j in range(num_bounds):
        plt.plot(steps, bounds_list[j], 
                        color=colors[j], 
                        alpha=0.6,
                        markevery=0.1,
                        marker=markers[j])
    plt.plot(steps,
                cold_start, 
                linestyle=titles_2_styles['cold_start'], 
                color=titles_2_colors['cold_start'],
                marker=titles_2_markers['cold_start'],
                linewidth=2.0,
                markevery=(0.05, 0.1)
                )
    if worst_case:
        plt.plot(steps,
                    worst, 
                    linestyle=titles_2_styles['nearest_neighbor'], 
                    color=titles_2_colors['nearest_neighbor'],
                    marker=titles_2_markers['nearest_neighbor'],
                    linewidth=2.0,
                    markevery=(0.05, 0.1)
                    )
    plt.tight_layout()
    plt.xlabel('iterations')
    ylabel = r'$1 - r_{\mathcal{X}}$'
    plt.ylabel(ylabel)
    if custom_loss:
        title = r'max Euclidean distance: $\epsilon={}$'.format(acc)
    else:
        title = r'fixed-point residual: $\epsilon={}$'.format(acc)
    plt.title(title)
    if worst_case:
        plt.xscale('log')
    plt.savefig(f"acc_{acc}.pdf", bbox_inches='tight')
    plt.clf()


def get_accs(cfg):
    accuracies = cfg.accuracies
    if accuracies == 'fp_full':
        start = -6  # Start of the log range (log10(10^-5))
        end = 2  # End of the log range (log10(1))
        accuracies = list(np.round(np.logspace(start, end, num=81), 6))
    if accuracies == 'nmse_full':
        start = -80  # Start of the log range (log10(10^-5))
        end = 0  # End of the log range (log10(1))
        accuracies = list(np.round(np.linspace(start, end, num=81), 6))
    if accuracies == 'maml_full':
        start = -3  # Start of the log range (log10(10^-5))
        end = 1  # End of the log range (log10(1))
        accuracies = list(np.round(np.logspace(start, end, num=81), 6))
    return accuracies


def get_e_stars(guarantee_results, accuracies, eval_iters):
    num_N = len(guarantee_results[0])
    e_stars = np.zeros((num_N, len(accuracies), eval_iters))
    for i in range(len(accuracies)):
        curr_pac_bayes_results = guarantee_results[i]
        for j in range(len(curr_pac_bayes_results)):
            curr = curr_pac_bayes_results[j][:eval_iters]
            e_stars[j, i, :curr.size] = curr #curr_pac_bayes_results[j][:eval_iters]
    return e_stars


def get_quantile(e_stars, percentile, eval_iters, worst, accuracies):
    quantile_curve = np.zeros(eval_iters)
    for k in range(eval_iters):
        where = np.where(e_stars[:,k] > percentile / 100)[0]
        if where.size == 0:
            if worst is None:
                quantile_curve[k] = max(accuracies)
            else:
                quantile_curve[k] = min(max(accuracies), worst[k])
        else:
            quantile_curve[k] = accuracies[np.min(where)]
    return quantile_curve


def percentile_plots(example, cfg):
    accuracies, eval_iters = get_accs(cfg), int(cfg.eval_iters)

    cold_start_results, guarantee_results = get_frac_solved_data_classical(example, cfg)
    # percentile_results = get_percentiles(example, cfg)
    if example == 'quadcopter':
        train = False
        col = 'nearest_neighbor'
    else:
        train = True
        col = 'no_train'
    percentiles = cfg.get('percentiles', [30, 90, 99])
    percentile_results = get_percentiles(example, cfg, cfg.percentile_datetime, train, col, percentiles)

    if example == 'quadcopter':
        train = False
        col = 'no_train'
        aux_results = get_percentiles(example, cfg, cfg.cold_start_percentile_datetime, train, col, percentiles)
    else:
        aux_results = None

    steps1 = np.arange(cold_start_results[-1].size)[:eval_iters]
    

    # fill in e_star tensor
    num_N = len(guarantee_results[0])
    e_stars = get_e_stars(guarantee_results, accuracies, eval_iters)

    percentiles = cfg.get('percentiles', [30, 90, 99])
    corrected_indices = [0, 1, 2] #[0, 2, 4]

    if example == 'mnist':
        lin_conv_rate = 0.99995
    elif example == 'robust_kalman':
        lin_conv_rate = 1.0
    else:
        lin_conv_rate = 1.0
    
    # worst case
    if cfg.worst_case:
        z_star_max, theta_max = get_worst_case_datetime(example, cfg)
        worst = construct_worst(z_star_max, steps1, lin_conv_rate)
        steps = get_steps(steps1.size, cfg.worst_case_iters, worst_case=True)
        worst_full = construct_worst(z_star_max, steps, lin_conv_rate)
    else:
        worst = None
        worst_full = None
        steps = get_steps(steps1.size, steps1.size, worst_case=False)

    cold_start_quantile_list, worst_list = [], []
    worst_full_list = []
    aux_quantile_list = []
    bounds_list_list, plot_bool_list_list = [], []
    
    for i in range(len(percentiles)):
        percentile = percentiles[i]
        bounds_list = []
        plot_bool_list = []
        for j in range(num_N):
            curr_bound = get_quantile(e_stars[j, :, :], percentile, eval_iters, worst, accuracies)
            bounds_list.append(curr_bound)
            if cfg.custom_loss and e_stars[j, :, :].max() < percentile / 100:
                plot_bool_list.append(False)
            else:
                plot_bool_list.append(True)
        correct_index = corrected_indices[i]
        cold_start_quantile = percentile_results[correct_index][:eval_iters]
        if aux_results is not None:
            aux_quantile = aux_results[correct_index][:eval_iters]
        else:
            aux_quantile = None
        
        percentile_final_plots(percentile, cold_start_quantile, worst, 
                            bounds_list, cfg.custom_loss, plot_bool_list, 
                            cfg.get('percentile_ylabel', 'fixed-point residual'))
        cold_start_quantile_list.append(cold_start_quantile)
        worst_list.append(worst)
        worst_full_list.append(worst_full)
        bounds_list_list.append(bounds_list)
        plot_bool_list_list.append(plot_bool_list)
        aux_quantile_list.append(aux_quantile)
    percentile_final_plots_together(example, percentiles, cold_start_quantile_list, worst_list, 
                            bounds_list_list, cfg.custom_loss, plot_bool_list_list, 
                            aux_quantile_list=aux_quantile_list)
    create_tables_classical(example, steps, percentiles, cold_start_quantile_list, 
                            worst_full_list, bounds_list_list, cfg.custom_loss, 
                            aux_quantile_list=aux_quantile_list)





def percentile_final_plots_together(example, percentiles, cold_start_quantile_list, worst_list, 
                            bounds_list_list, custom_loss, plot_bool_list_list, aux_quantile_list=None):
    markers = ['o', 's', '>', 'D'] #['o', 's', '<', 'D']
    cmap = plt.cm.Set1
    colors = cmap.colors
    offsets = [0, .03, .06]
    fig, axes = plt.subplots(nrows=1, ncols=len(percentiles), figsize=(20, 5), sharey='row')
    # axes[0].set_ylabel(ylabel)

    fontsize = 30
    title_fontsize = 30

    # y-label
    if custom_loss:
        if example == 'robust_kalman':
            ylabel = 'max Euclidean dist.'
        elif example == 'mnist':
            ylabel = 'NMSE (dB)'
    else:
        ylabel = 'fixed-point residual'
    axes[0].set_ylabel(ylabel, fontsize=fontsize)

    for k in range(len(percentiles)):
        percentile = percentiles[k]
        bounds_list = bounds_list_list[k]
        cold_start_quantile = cold_start_quantile_list[k]
        worst = worst_list[k]
        plot_bool_list = plot_bool_list_list[k]

        loc = k
        num_N = len(bounds_list)
        for j in range(num_N):
            curr = bounds_list[j]
            if plot_bool_list[j]:
                axes[loc].plot(curr, color=colors[j], marker=markers[j], 
                            alpha=0.6, markevery=(0.00 + offsets[j], 0.1))
                
        if aux_quantile_list is not None and aux_quantile_list[0] is not None:
            aux_quantile = aux_quantile_list[k]
            # print('aux_quantile', aux_quantile)
            # print('cold start_quantile', cold_start_quantile)
            # flip the colors
            axes[loc].plot(aux_quantile, 
                        color=titles_2_colors['cold_start'], 
                        linestyle=titles_2_styles['cold_start'],
                        marker=titles_2_markers['cold_start'],
                        markevery=(0.05, 0.1))
            axes[loc].plot(cold_start_quantile, 
                        color=titles_2_colors['nearest_neighbor'], 
                        linestyle=titles_2_styles['nearest_neighbor'],
                        marker=titles_2_markers['nearest_neighbor'],
                        markevery=(0.05, 0.1))
        else:
            if example == 'mnist' and custom_loss:
                cold_start_quantile = np.clip(cold_start_quantile, a_max=0, a_min=-1000)
            axes[loc].plot(cold_start_quantile, 
                        color=titles_2_colors['cold_start'], 
                        linestyle=titles_2_styles['cold_start'],
                        marker=titles_2_markers['cold_start'],
                        markevery=(0.05, 0.1))
            

        # plot the worst case
        if not custom_loss:
            axes[loc].plot(worst, 
                        color=titles_2_colors['worst'], 
                        linestyle=titles_2_styles['worst'],
                        marker=titles_2_markers['worst'],
                        markevery=(0.05, 0.1)
                        )
            

        if cold_start_quantile.min() > 0:
            axes[k].set_yscale('log')
        if example == 'mnist':
            axes[k].set_xscale('log')
        axes[k].set_xlabel('iterations', fontsize=fontsize)
        # axes[k].set_ylabel(ylabel)

        axes[loc].set_title(r'${}$th quantile bound'.format(percentile), fontsize=title_fontsize)
        
    plt.tight_layout()
    plt.savefig("percentile_together.pdf", bbox_inches='tight')
    plt.clf()


def percentile_final_plots(percentile, cold_start_quantile, worst, bounds_list, 
                           custom_loss, plot_bool_list, ylabel):
    markers = ['o', 's', '<', 'D']
    colors = plt.cm.Set1.colors
    offsets = [0, .03, .06]
    num_N = len(bounds_list)
    for j in range(num_N):
        curr = bounds_list[j]
        if plot_bool_list[j]:
            plt.plot(curr, color=colors[j], marker=markers[j], 
                        alpha=0.6, markevery=(0.00 + offsets[j], 0.1))
            
    plt.plot(cold_start_quantile, 
                 titles_2_colors['cold_start'], 
                 linestyle=titles_2_styles['cold_start'],
                 marker=titles_2_markers['cold_start'],
                 markevery=(0.05, 0.1))

    # plot the worst case
    if not custom_loss:
        plt.plot(worst, 
                    color=titles_2_colors['nearest_neighbor'], 
                    linestyle=titles_2_styles['nearest_neighbor'],
                    marker=titles_2_markers['nearest_neighbor'],
                    markevery=(0.05, 0.1))
    if cold_start_quantile.min() > 0:
        plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel(ylabel)

    plt.title(r'${}$th quantile bound'.format(percentile))
    plt.savefig(f"percentile_{percentile}.pdf", bbox_inches='tight')
    plt.clf()


def pep():
    gamma = 0.5
    n = 70
    import time
    t0 = time.time()
    pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(n=n, gamma=gamma, verbose=1)
    t1 = time.time()
    print('time to solve', t1 - t0)
    # # Instantiate PEP
    # problem = PEP()

    # # Declare a non expansive operator
    # A = problem.declare_function(LipschitzOperator, L=1.)

    # # Start by defining its unique optimal point xs = x_*
    # xs, _, _ = A.fixed_point()

    # # Then define the starting point x0 of the algorithm
    # x0 = problem.set_initial_point()

    # # Set the initial constraint that is the difference between x0 and xs
    # problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # x = x0
    # for i in range(n):
    #     x = (1 - gamma) * x + gamma * A.gradient(x)

    # # Set the performance metric to distance between xN and AxN
    # problem.set_performance_metric((1 / 2 * (x - A.gradient(x))) ** 2)

    # # Solve the PEP
    # verbose = 1
    # pepit_verbose = max(verbose, 0)
    # pepit_tau = problem.solve(verbose=pepit_verbose)

    # # Compute theoretical guarantee (for comparison)
    # if 1 / 2 <= gamma <= 1 / 2 * (1 + np.sqrt(n / (n + 1))):
    #     theoretical_tau = 1 / (n + 1) * (n / (n + 1)) ** n / (4 * gamma * (1 - gamma))
    # elif 1 / 2 * (1 + np.sqrt(n / (n + 1))) < gamma <= 1:
    #     theoretical_tau = (2 * gamma - 1) ** (2 * n)
    # else:
    #     raise ValueError("{} is not a valid value for the step-size \'gamma\'."
    #                      " \'gamma\' must be a number between 1/2 and 1".format(gamma))

    # # Print conclusion if required
    # print('*** Example file: worst-case performance of Kranoselskii-Mann iterations ***')
    # print('\tPEPit guarantee:\t 1/4||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
    # print('\tTheoretical guarantee:\t 1/4||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))



def get_worst_case_datetime(example, cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    datetime = cfg.worst_case_datetime
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/z_star_max.csv"

    # fp_file = f"tol={acc}_train.csv" if train else f"tol={acc}_test.csv"
    # df = read_csv(f"{path}/{fp_file}")
    # filename = cfg.worst_case_datetime

    # Open the file in read mode
    with open(path, 'r') as file:
        # reader = csv.reader(file)
        reader = read_csv(file)
        z_star_max = float(reader.columns[0])
        theta_max = reader[str(z_star_max)][0]

        
        # Read the first row and extract the scalar value
        # for row in reader:
        #     z_star_max = row[0]
    return z_star_max, theta_max

def create_gen_l2o_results(example, cfg):
    # example = 'sparse_coding'
    # overlay_training_losses(example, cfg)
    # create_journal_results(example, cfg, train=False)
    # create_genL2O_results()
    metrics, timing_data, titles = get_all_data(example, cfg, train=False)

    if len(titles) == 4:
        titles[-2] = titles[-2] + '_deterministic'
    nmse = metrics[0]
    for i in range(len(nmse)):
        # if titles[i] != 'cold_start' and titles[i] != 'nearest_neighbor':
        #     plt.plot(nmse[i])
        plt.plot(nmse[i][:cfg.eval_iters],
                 linestyle=titles_2_styles[titles[i]], 
                 color=titles_2_colors[titles[i]],
                 marker=titles_2_markers[titles[i]],
                 markevery=(0, 100)
                 )
    plt.tight_layout()
    plt.xlabel('iterations')
    plt.ylabel("fixed-point residual")
    plt.yscale('log')
    plt.savefig('fp_res.pdf', bbox_inches='tight')
    plt.clf()

    z_star_max, theta_max = get_worst_case_datetime(example, cfg)

    out = get_frac_solved_data(example, cfg)
    all_test_results, all_pac_bayes_results, cold_start_results, nearest_neighbor_results = out
    markers = ['o', 's']
    cmap = plt.cm.Set1
    colors = cmap.colors
    styles = ['-', '-']
    for i in range(len(cfg.accuracies)):
        # plot ista and fista
        mark_start = titles_2_marker_starts['cold_start']
        plt.plot(cold_start_results[i][:cfg.eval_iters], 
                 linestyle=titles_2_styles['cold_start'], 
                 color=titles_2_colors['cold_start'],
                 marker=titles_2_markers['cold_start'],
                #  markevery=(30, 100)
                markevery=0.1
                 )
        mark_start = titles_2_marker_starts['nearest_neighbor']
        plt.plot(nearest_neighbor_results[i][:cfg.eval_iters], 
                linestyle=titles_2_styles['nearest_neighbor'], 
                color=titles_2_colors['nearest_neighbor'],
                marker=titles_2_markers['nearest_neighbor'],
                # markevery=(60, 100)
                markevery=0.1
                )

        # plot the learned variants
        acc = cfg.accuracies[i]
        curr_test_results = all_test_results[i]
        curr_pac_bayes_results = all_pac_bayes_results[i]
        for j in range(len(curr_test_results)):
            curr_size = curr_pac_bayes_results[j].size
            curr_test = np.ones(cfg.eval_iters)
            curr_test[:curr_size] = curr_test_results[j]
            plt.plot(curr_test, 
                     linestyle='-', 
                     color=colors[0], 
                     marker=markers[0],
                     markevery=0.1
                     )
            # curr_pac = curr_pac_bayes_results[j]

            # worst-case
            # prob bounds
            # curr_curve[:curr_size] = curr_pac_bayes_results[j] #[:cfg.eval_iters]

            # worst-case bounds
            
            steps = np.arange(cfg.eval_iters)

            init_diff = z_star_max * 1.1 + 1.1 * theta_max * 30
            indices = .995 ** steps * init_diff  < acc

            # cold_start_results[i][:cfg.eval_iters]

            curr_pac = np.zeros(cfg.eval_iters)
            curr_pac[:curr_size] = curr_pac_bayes_results[j]
            curr_pac[curr_size:] = curr_pac_bayes_results[j].max()
            curr_pac[indices] = 1.0


            plt.plot(curr_pac, 
                     linestyle='-', 
                     color=colors[1], 
                     markevery=0.1,
                     marker=markers[1])
        plt.tight_layout()
        plt.xlabel('iterations')
        plt.xscale('log')
        plt.ylabel(f"frac. at {acc} fp res")
        plt.savefig(f"acc_{acc}.pdf", bbox_inches='tight')
        plt.clf()


def percentile_plots_maml(example, cfg):
    accuracies, eval_iters = get_accs(cfg), int(cfg.eval_iters)

    # cold_start_results, guarantee_results = get_frac_solved_data_classical(example, cfg)
    out = get_frac_solved_data(example, cfg)
    all_test_results, all_pac_bayes_results, cold_start_results, pretrain_results = out

    # fill in e_star tensor
    num_N = len(all_pac_bayes_results[0])
    e_stars = get_e_stars(all_pac_bayes_results, accuracies, eval_iters)

    percentiles = cfg.get('percentiles', [10, 30, 50, 60, 80, 90, 95]) #[30, 80, 90])
    plot_percentiles = cfg.get('plot_percentiles', [30, 80, 90])
    plot_indices = cfg.get('plot_indices', [0, 1, 2]) #[1, 4, 5]
    # corrected_indices = [0, 2, 3]
    worst = None

    cold_start_quantile_list, second_baseline_quantile_list = [], []
    worst_list = []
    bounds_list_list, plot_bool_list_list = [], []
    emp_list_list = []

    # get the empirical quantiles
    percentiles_list_list = get_percentiles_learned(example, cfg, percentiles)

    if cfg.nearest_neighbor_datetime != '':
        # percentiles_list_nn = get_percentiles(example, cfg, first=False)
        percentiles_list_nn = get_percentiles(example, cfg, cfg.percentile_nn_dt, 
                                              train=False, col='nearest_neighbor', percentiles=percentiles)

    baseline_col = 'last' if example == 'sine' else 'no_train' 
    percentiles_list_cold = get_percentiles(example, cfg, cfg.percentile_dt, 
                                            train=False, col=baseline_col, percentiles=percentiles)
    
    for i in range(len(percentiles)):
        percentile = percentiles[i]
        bounds_list = []
        plot_bool_list = []

        # get the current bound
        for j in range(num_N):
            curr_bound = get_quantile(e_stars[j, :, :], percentile, eval_iters, worst, accuracies)
            bounds_list.append(curr_bound)
            if cfg.custom_loss and e_stars[j, :, :].max() < percentile / 100:
                plot_bool_list.append(False)
            else:
                plot_bool_list.append(True)
        correct_index = i #corrected_indices[i]

        if example == 'sine':
            # pretrain quantile
            percentile_results = percentiles_list_list[0]
            cold_start_quantile = percentiles_list_cold[i] #percentile_results[correct_index][:eval_iters]
            second_baseline_quantile = None
            worst = None

            # learned quantile
            emp_list = [percentiles_list_list[0][correct_index][:eval_iters]]
        elif example == 'sparse_coding':
            cold_start_quantile = percentiles_list_cold[i]
            emp_list = [percentiles_list_list[k][correct_index][:eval_iters] for k in range(len(percentiles_list_list))]
            second_baseline_quantile = percentiles_list_nn[i]
        elif example == 'unconstrained_qp':
            cold_start_quantile = percentiles_list_cold[i]
            emp_list = [percentiles_list_list[k][correct_index][:eval_iters] for k in range(len(percentiles_list_list))]
            second_baseline_quantile = percentiles_list_nn[i]

        learned_percentile_final_plots(example, percentile, cold_start_quantile, second_baseline_quantile, 
                                       worst, emp_list, bounds_list, cfg.custom_loss, 
                                       plot_bool_list)
        
        # append to lists
        cold_start_quantile_list.append(cold_start_quantile)
        second_baseline_quantile_list.append(second_baseline_quantile)
        worst_list.append(worst)
        bounds_list_list.append(bounds_list)
        emp_list_list.append(emp_list)
        plot_bool_list_list.append(plot_bool_list)
    # second_baseline_quantile_list = None
    # learned_percentile_final_plots_together(example, percentiles, cold_start_quantile_list, 
    #                                         second_baseline_quantile_list,
    #                                         worst_list, emp_list_list,
    #                         bounds_list_list, cfg.custom_loss, plot_bool_list_list)
    plot_cold_start_quantile_list = [cold_start_quantile_list[i] for i in plot_indices]
    plot_second_baseline_quantile_list = [second_baseline_quantile_list[i] for i in plot_indices]
    plot_worst_list = [worst_list[i] for i in plot_indices]
    plot_emp_list_list = [emp_list_list[i] for i in plot_indices]
    plot_bounds_list_list = [bounds_list_list[i] for i in plot_indices]
    plot_plot_bool_list_list = [plot_bool_list_list[i] for i in plot_indices]
    learned_percentile_final_plots_together(example, plot_percentiles, plot_cold_start_quantile_list, 
                                            plot_second_baseline_quantile_list,
                                            plot_worst_list, plot_emp_list_list,
                            plot_bounds_list_list, cfg.custom_loss, plot_plot_bool_list_list)
    create_percentile_table(example, percentiles, cold_start_quantile_list, 
                                            second_baseline_quantile_list,
                                            worst_list, emp_list_list,
                            bounds_list_list, cfg.custom_loss, plot_bool_list_list)
    

def get_ylabel_percentile(example, custom_loss):
    if example == 'sparse_coding':
        return 'NMSE (dB)'
    if example == 'sine':
        if custom_loss:
            return 'Infinity norm'
        else:
            return 'MSE'
    if custom_loss:
        if example == 'robust_kalman':
            ylabel = 'max Euclidean dist.'
        elif example == 'mnist' or example == 'sparse_coding':
            ylabel = 'NMSE (dB)'
    else:
        ylabel = 'fixed-point residual'

    return ylabel


def create_tables_classical(example, steps, percentiles, cold_start_quantile_list, 
                            worst_list, bounds_list_list, custom_loss, aux_quantile_list=None):
    num_N = len(bounds_list_list)
    num_samples = [10, 100, 1000]
    if custom_loss and example == 'mnist':
        tols = np.array([-20, -30, -40])
    if not custom_loss and example == 'mnist':
        tols = np.array([0.1, 0.01, 0.001, 0.0001])
    elif custom_loss and example == 'robust_kalman':
        tols = np.array([0.01, 0.001, 0.0001])
    else:
        tols = np.array([0.01, 0.001, 0.0001])
    for i in range(len(percentiles)):
        curr_df = pd.DataFrame()
        quantile = percentiles[i]

        cold_start_vals = np.zeros(tols.size)
        nearest_neighbor_vals = np.zeros(tols.size)
        worst_vals = np.zeros(tols.size)
        # curr_emp_vals = np.zeros(tols.size)
        # curr_bound_vals = np.zeros(tols.size)

        cold_start_curve = cold_start_quantile_list[i]
        worst_curve = worst_list[i]

        samples_curve_list = [np.zeros(tols.size) for i in range(num_N)]
        for j in range(tols.size):
            cold_start_vals[j] = get_cutoff_tol(cold_start_curve, tols[j])

            if example == 'quadcopter':
                nearest_neighbor_vals[j] = get_cutoff_tol(aux_quantile_list[i], tols[j])

            if worst_curve is not None:
                worst_cutoff = get_cutoff_tol(worst_curve, tols[j])
                if worst_cutoff is not None:
                    worst_vals[j] = steps[worst_cutoff]

            # iterate over the bounds
            
            for k in range(num_N):
                samples_curve_list[k][j] = get_cutoff_tol(bounds_list_list[i][k], tols[j])
        curr_df['tol'] = tols

        if example == 'quadcopter':
            curr_df['cold_start'] = nearest_neighbor_vals.astype(int)
            curr_df['nearest_neighbor'] = cold_start_vals.astype(int)
        else:
            curr_df['cold_start'] = cold_start_vals.astype(int)
        if not custom_loss:
            curr_df['worst'] = worst_vals.astype(int)
        for k in range(num_N):
            curr_df[f"samples_{num_samples[k]}"] = samples_curve_list[k].astype(int)

        curr_df.to_csv(f"quantiles_{quantile}.csv")



def create_percentile_table(example, percentiles, cold_start_quantile_list,
                                            second_baseline_quantile_list, 
                                            worst_list, 
                            emp_list_list, bounds_list_list, custom_loss, plot_bool_list_list):
    df = pd.DataFrame()
    df['quantiles'] = np.array(percentiles)
    

    if example == 'sparse_coding':
        algorithms = ['alista', 'tilista', 'glista', 'lista']
        index = cold_start_quantile_list[0].size - 1
        ista_vals = np.zeros(len(percentiles))
        fista_vals = np.zeros(len(percentiles))
        for i in range(len(percentiles)):
            ista_vals[i] = cold_start_quantile_list[i][index]
            fista_vals[i] = second_baseline_quantile_list[i][index]
        df['ista'] = np.round(ista_vals, 2)
        df['fista'] = np.round(fista_vals, 2)
        
        for j in range(len(bounds_list_list[0])):
            algo = algorithms[j]
            curr_emp_vals = np.zeros(len(percentiles))
            curr_bound_vals = np.zeros(len(percentiles))
            for i in range(len(percentiles)):
                curr_emp_vals[i] = emp_list_list[i][j][index]
                curr_bound_vals[i] = bounds_list_list[i][j][index]
        
            df[f"{algo}_emp"] = np.round(curr_emp_vals, 2)
            df[f"{algo}_bound"] = np.round(curr_bound_vals, 2)
        df.to_csv('quantiles.csv')
    elif example == 'sine':
        algorithms = ['maml']
        index = cold_start_quantile_list[0].size - 1
        pretrain_vals = np.zeros(len(percentiles))
        for i in range(len(percentiles)):
            pretrain_vals[i] = cold_start_quantile_list[i][index]
        df['pretrain'] = np.round(pretrain_vals, 3)
        
        for j in range(len(bounds_list_list[0])):
            algo = algorithms[j]
            curr_emp_vals = np.zeros(len(percentiles))
            curr_bound_vals = np.zeros(len(percentiles))
            for i in range(len(percentiles)):
                curr_emp_vals[i] = emp_list_list[i][j][index]
                curr_bound_vals[i] = bounds_list_list[i][j][index]
        
            df[f"{algo}_emp"] = np.round(curr_emp_vals, 3)
            df[f"{algo}_bound"] = np.round(curr_bound_vals, 3)
        df.to_csv('quantiles.csv')
    elif example == 'unconstrained_qp':
        # quantiles = [30, 80, 90]
        tols = np.array([0.01, 0.001, 0.0001])

        algo = 'l2ws'

        # for j in range(len(quantiles)):
        for i in range(len(percentiles)):
            curr_df = pd.DataFrame()
            # df['quantiles'] = np.array(percentiles)
            quantile = percentiles[i]

            cold_start_vals = np.zeros(tols.size)
            nn_vals = np.zeros(tols.size)
            curr_emp_vals = np.zeros(tols.size)
            curr_bound_vals = np.zeros(tols.size)

            cold_start_curve = cold_start_quantile_list[i]
            nn_curve = second_baseline_quantile_list[i]
            for j in range(tols.size):
                # where =  np.where(cold_start_curve < tols[j])[0]
                # if where.size == 0:
                #     pass
                # else:
                #     cold_start_vals[j] = where[0]
                cold_start_vals[j] = get_cutoff_tol(cold_start_curve, tols[j])
                nn_vals[j] = get_cutoff_tol(nn_curve, tols[j])
                curr_emp_vals[j] = get_cutoff_tol(emp_list_list[i][0], tols[j])
                curr_bound_vals[j] = get_cutoff_tol(bounds_list_list[i][0], tols[j])
            curr_df['tol'] = tols
            curr_df['cold_start'] = cold_start_vals.astype(int)
            curr_df['nearest_neighbor'] = nn_vals.astype(int)
            curr_df[f"{algo}_emp"] = curr_emp_vals.astype(int)
            curr_df[f"{algo}_bound"] = curr_bound_vals.astype(int)

            # for j in range(len(bounds_list_list[0])):
            # algo = algorithms[j]
            
            # for i in range(len(percentiles)):
            # curr_emp_vals[i] = emp_list_list[i][j][index]
            # curr_bound_vals[i] = bounds_list_list[i][j][index]
            
            
            

            # index = cold_start_quantile_list[0].size - 1
            # cold_start_vals = np.zeros(len(percentiles))
            # for i in range(len(percentiles)):
            #     cold_start_vals[i] = cold_start_quantile_list[i][index]
            # df['cold_start'] = np.round(cold_start_vals, 3)
            
            # for j in range(len(bounds_list_list[0])):
            #     algo = algorithms[j]
            #     curr_emp_vals = np.zeros(len(percentiles))
            #     curr_bound_vals = np.zeros(len(percentiles))
            #     for i in range(len(percentiles)):
            #         curr_emp_vals[i] = emp_list_list[i][j][index]
            #         curr_bound_vals[i] = bounds_list_list[i][j][index]
            
            # df[f"{algo}_emp"] = np.round(curr_emp_vals, 3)
            # df[f"{algo}_bound"] = np.round(curr_bound_vals, 3)
            curr_df.to_csv(f"quantiles_{quantile}.csv")
    
def get_cutoff_tol(curve, tol):
    where =  np.where(curve < tol)[0]
    if where.size == 0:
        0
    else:
        return where[0]


def learned_percentile_final_plots_together(example, percentiles, cold_start_quantile_list,
                                            second_baseline_quantile_list, 
                                            worst_list, 
                            emp_list_list, bounds_list_list, custom_loss, plot_bool_list_list):
    markers = ['o', 's', '>', '^', 'D', 'X', 'P', '*']
    cmap = plt.cm.Set1
    colors = cmap.colors
    fig, axes = plt.subplots(nrows=1, ncols=len(percentiles), figsize=(20, 5), sharey='row')
    fontsize = 30
    title_fontsize = 30

    ylabel = get_ylabel_percentile(example, custom_loss)
    axes[0].set_ylabel(ylabel, fontsize=fontsize)

    for k in range(len(percentiles)):
        percentile = percentiles[k]
        bounds_list = bounds_list_list[k]
        emp_list = emp_list_list[k]
        cold_start_quantile = cold_start_quantile_list[k]
        worst = worst_list[k]
        plot_bool_list = plot_bool_list_list[k]

        loc = k
        num_N = len(bounds_list)
        if cold_start_quantile.size < 20:
            markevery = None
        else:
            markevery = 0.1
        for j in range(num_N):
            # plot the empirical
            emp = emp_list[j]
            if plot_bool_list[j]:
                marker = markers[2 * j] #if example == 'sparse_coding' else markers[j]
                axes[loc].plot(emp, linestyle='dotted', markerfacecolor='none',
                               color=colors[j], marker=marker, markevery=markevery)

            # plot the bound
            # curr = bounds_list[j]
            if plot_bool_list[j]:
                marker = markers[2 * j + 1] #if example == 'sparse_coding' else markers[j + 1]
                # bound = np.maximum(bounds_list[j], emp_list[j]) #bounds_list[j] #np.maximum(bounds_list[j], emp_list[j])
                bound = bounds_list[j]
                axes[loc].plot(bound, color=colors[j], marker=marker, 
                            markevery=markevery)
        # plot the cold start
        axes[loc].plot(cold_start_quantile[:emp_list[0].size], 
                    titles_2_colors['cold_start'], 
                    linestyle=titles_2_styles['cold_start'],
                    marker=titles_2_markers['cold_start'],
                    markevery=markevery
                    )
        
        # plot the secondary baseline
        if second_baseline_quantile_list is not None:
            print('len(second_baseline_quantile_list)', len(second_baseline_quantile_list))
            print('second_baseline_quantile_list[k].size', second_baseline_quantile_list[k].size)
            axes[loc].plot(second_baseline_quantile_list[k][:emp_list[0].size],
                        color=titles_2_colors['nearest_neighbor'], 
                        linestyle=titles_2_styles['nearest_neighbor'],
                        marker=titles_2_markers['nearest_neighbor'],
                        markevery=markevery
                        )
        
        # plot the worst case
        # if not custom_loss and worst is not None:
        #     axes[loc].plot(worst) #, 
                        # color=titles_2_colors['nearest_neighbor'], 
                        # linestyle=titles_2_styles['nearest_neighbor'],
                        # marker=titles_2_markers['nearest_neighbor'],
                        # markevery=markevery)
        if cold_start_quantile.min() > 0:
            axes[k].set_yscale('log')
        axes[k].set_xlabel('iterations', fontsize=fontsize)
        axes[loc].set_title(r'${}$th quantile bound'.format(percentile), fontsize=title_fontsize)
        if example != 'unconstrained_qp':
            axes[k].xaxis.set_major_locator(MaxNLocator(integer=True))
        
    plt.tight_layout()
    plt.savefig("percentile_together.pdf", bbox_inches='tight')
    plt.clf()


def learned_percentile_final_plots(example, percentile, cold_start_quantile, second_baseline_quantile, 
                                   worst, emp_list, bounds_list, custom_loss, plot_bool_list):
    markers = ['o', 's', '>', '^', 'D', 'X', 'P', '*']
    colors = plt.cm.Set1.colors
    offsets = [0, .03, .06]
    num_N = len(bounds_list)

    percentile_ylabel = get_ylabel_percentile(example, custom_loss)
    for j in range(num_N):
        # plot the empirical quantity
        emp = emp_list[j]
        if plot_bool_list[j]:
            plt.plot(emp, linestyle='dotted', markerfacecolor='none',
                     color=colors[j], marker=markers[2*j], markevery=None)

        # plot the bound
        curr = bounds_list[j]
        if plot_bool_list[j]:
            plt.plot(curr, color=colors[j], marker=markers[2*j+1], markevery=None)
            
    # plot the cold start
    plt.plot(cold_start_quantile, 
                 titles_2_colors['cold_start'], 
                 linestyle=titles_2_styles['cold_start'],
                 marker=titles_2_markers['cold_start'],
                 markevery=None)
    
    # plot the secondary baseline
    # if second_baseline_quantile is not None:
    #     plt.plot(second_baseline_quantile, 
    #                 titles_2_colors['nearest_neighbor'], 
    #                 linestyle=titles_2_styles['nearest_neighbor'],
    #                 marker=titles_2_markers['nearest_neighbor'],
    #                 markevery=None)

    # plot the worst case
    # if not custom_loss:
    #     plt.plot(worst, 
    #                 color=titles_2_colors['nearest_neighbor'], 
    #                 linestyle=titles_2_styles['nearest_neighbor'],
    #                 marker=titles_2_markers['nearest_neighbor'],
    #                 markevery=(0.05, 0.1))
    if cold_start_quantile.min() > 0:
        plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel(percentile_ylabel)

    plt.title(r'${}$th quantile bound'.format(percentile))
    plt.savefig(f"percentile_{percentile}.pdf", bbox_inches='tight')
    plt.clf()
    
    

def create_gen_l2o_results_maml(example, cfg):
    out = get_frac_solved_data(example, cfg)
    all_test_results, all_pac_bayes_results, cold_start_results, pretrain_results = out

    plot_acc_list = cfg.plot_acc_list
    steps = np.arange(cfg.eval_iters)
    worst_case, worst_list = False, []

    accs = get_accs(cfg)
    acc_list, emp_list_list, bounds_list_list, cold_start_list = [], [], [], []
    secondary_baseline_list = []
    for i in range(len(accs)):
        acc = accs[i]

        if acc in plot_acc_list:
            if example == 'sine':
                # get the pretrained model
                cold_start_curve = pretrain_results[i]
                secondary_baseline_curve = None

                # get the pac_bayes plot
                curr_pac_bayes_results = all_pac_bayes_results[i][0]

                # get the empirical plot
                curr_test_results = all_test_results[i][0]

                emp_list = [curr_test_results]
                bounds_list = [curr_pac_bayes_results]
            elif example == 'sparse_coding':
                # get the pretrained model
                cold_start_curve = cold_start_results[i] #pretrain_results[i]
                secondary_baseline_curve = None

                # get the pac_bayes plot
                curr_pac_bayes_results = all_pac_bayes_results[i]

                # get the empirical plot
                curr_test_results = all_test_results[i]

                emp_list = curr_test_results
                bounds_list = curr_pac_bayes_results
            elif example == 'unconstrained_qp':
                # get the pretrained model
                cold_start_curve = cold_start_results[i] #pretrain_results[i]
                secondary_baseline_curve = pretrain_results[i] #None

                # get the pac_bayes plot
                curr_pac_bayes_results = all_pac_bayes_results[i]

                # get the empirical plot
                curr_test_results = all_test_results[i]

                emp_list = curr_test_results
                bounds_list = curr_pac_bayes_results
            
            worst_case_curve = None
            # plot_final_learned_risk_bounds(acc, steps, bounds_list, emp_list, cold_start_curve, 
            #                                secondary_baseline_curve,
            #                                 worst_case_curve, False, cfg.custom_loss)
            acc_list.append(acc)
            emp_list_list.append(emp_list)
            bounds_list_list.append(bounds_list)
            cold_start_list.append(cold_start_curve)
            worst_list.append(worst_case_curve)
            secondary_baseline_list.append(secondary_baseline_curve)

    plot_final_learned_risk_bounds_together(example, acc_list, steps, bounds_list_list, 
                                            emp_list_list, cold_start_list, secondary_baseline_list,
                                     worst_list, worst_case, cfg.custom_loss)


        
def plot_final_learned_risk_bounds_together(example, plot_acc_list, steps, bounds_list_list, 
                                            emp_list_list, cold_start_list, secondary_baseline_list,
                                     worst_list, worst_case, custom_loss):

    markers = ['o', 's', '>', '^', 'D', 'X', 'P', '*']
    cmap = plt.cm.Set1
    colors = cmap.colors

    fig, axes = plt.subplots(nrows=1, ncols=len(plot_acc_list), figsize=(20, 5), sharey='row')

    fontsize = 30
    title_fontsize = 30

    # y-label
    # ylabel = r'$1 - r_{\mathcal{X}}$'
    ylabel = r'$1 - R_{\mathcal{X}}(P)$'
    # ylabel = r'prob. of reaching $\epsilon$'
    axes[0].set_ylabel(ylabel, fontsize=fontsize)

    if steps.size < 20:
        markevery = None
    else:
        markevery = 0.1

    for k in range(len(plot_acc_list)):
        loc = len(plot_acc_list) - 1 - k
        # loc = k
        acc = plot_acc_list[k]
        bounds_list = bounds_list_list[k]
        emp_list = emp_list_list[k]
        cold_start = cold_start_list[k]
        worst = worst_list[k]

        num_bounds = len(bounds_list)

        for j in range(num_bounds):
            # plot empirical results
            emp_vals = emp_list[j][emp_list[j].size-1] * np.ones(steps.size)
            emp_vals[:emp_list[j].size] = emp_list[j][:steps.size]
            axes[loc].plot(steps, emp_vals, 
                            color=colors[j], 
                            markerfacecolor='none',
                            linestyle='dotted',
                            # alpha=0.6,
                            markevery=markevery,
                            marker=markers[2*j])
            
            # plot pac bayes bounds
            # bound = np.minimum(bounds_list[j], emp_list[j])
            # if False: #k == 0:
            #     val = bounds_list_list[k+1][j][bounds_list[j].size-1]
            # else:
            #     val = bounds_list[j][bounds_list[j].size-1]
            val = bounds_list[j][bounds_list[j].size-1]
            bound = val * np.ones(steps.size)
            bound[:bounds_list[j].size] = bounds_list[j][:steps.size]
            axes[loc].plot(steps, bound, 
                            color=colors[j], 
                            # alpha=0.6,
                            markevery=markevery,
                            marker=markers[2*j+1])

        cold_start_full = cold_start[cold_start.size-1] * np.ones(steps.size)
        cold_start_full[:cold_start.size] = cold_start[:steps.size]
        axes[loc].plot(steps,
                    cold_start_full, 
                    linestyle=titles_2_styles['cold_start'], 
                    color=titles_2_colors['cold_start'],
                    marker=titles_2_markers['cold_start'],
                    linewidth=2.0,
                    markevery=markevery
                    )
        if secondary_baseline_list[k] is not None:
            axes[loc].plot(steps,
                    secondary_baseline_list[k][:steps.size], 
                    linestyle=titles_2_styles['nearest_neighbor'], 
                    color=titles_2_colors['nearest_neighbor'],
                    marker=titles_2_markers['nearest_neighbor'],
                    linewidth=2.0,
                    markevery=markevery
                    )
        if worst_case:
            axes[loc].plot(steps,
                        worst, 
                        linestyle=titles_2_styles['nearest_neighbor'], 
                        color=titles_2_colors['nearest_neighbor'],
                        marker=titles_2_markers['nearest_neighbor'],
                        linewidth=2.0,
                        markevery=(0.05, 0.1)
                        )
        axes[loc].set_xlabel('iterations', fontsize=fontsize)

        acc = round_acc(acc)
        
        
        if example == 'sparse_coding':
            title = r'NMSE (dB): $\epsilon={}$'.format(np.round(acc, 1))
        elif example == 'sine':
            if custom_loss:
                title = r'Infinity norm: $\epsilon={}$'.format(acc)
            else:
                title = r'MSE: $\epsilon={}$'.format(acc)
        elif custom_loss:
            if example == 'robust_kalman':
                title = r'max Euclidean distance: $\epsilon={}$'.format(acc)
            elif example == 'mnist' or example == 'sparse_coding':
                title = r'NMSE (dB): $\epsilon={}$'.format(np.round(acc, 1))
        else:
            title = r'fixed-point residual: $\epsilon={}$'.format(acc)
        axes[loc].set_title(title, fontsize=title_fontsize)
        if worst_case:
            axes[k].set_xscale('log')
        if steps.size < 20:
            axes[loc].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        
    plt.tight_layout()
    plt.savefig("risk_together.pdf", bbox_inches='tight')
    plt.clf()


def round_acc(acc):
    if acc == 0.501187:
        return 0.50
    if acc == 1.995262:
        return 2.0
    return acc


def plot_final_learned_risk_bounds(acc, steps, bounds_list, emp_list, cold_start, secondary_baseline_curve, worst, 
                                     worst_case, custom_loss):
    # markers = ['o', 's', '<', 'D']
    markers = ['o', 's', '>', '^', 'D', 'X', 'P', '*']
    
    cmap = plt.cm.Set1
    colors = cmap.colors
    num_bounds = len(bounds_list)
    for j in range(num_bounds):
        # plot the stochastic test curve
        plt.plot(steps[:emp_list[j].size], emp_list[j], 
                 linestyle='dotted',
                        color=colors[j], 
                        markerfacecolor='none',
                        alpha=0.6,
                        markevery=0.1,
                        marker=markers[2*j])

        # plot the bounds
        plt.plot(steps[:bounds_list[j].size], bounds_list[j], 
                        color=colors[j], 
                        alpha=0.6,
                        markevery=0.1,
                        marker=markers[2*j+1])

    plt.plot(steps,
                cold_start, 
                linestyle=titles_2_styles['cold_start'], 
                color=titles_2_colors['cold_start'],
                marker=titles_2_markers['cold_start'],
                linewidth=2.0,
                markevery=(0.05, 0.1)
                )
    if secondary_baseline_curve is not None:
        plt.plot(steps,
                cold_start, 
                linestyle=titles_2_styles['nearest_neighbor'], 
                color=titles_2_colors['nearest_neighbor'],
                marker=titles_2_markers['nearest_neighbor'],
                linewidth=2.0,
                markevery=(0.05, 0.1)
                )
    if worst_case:
        plt.plot(steps,
                    worst, 
                    linestyle=titles_2_styles['nearest_neighbor'], 
                    color=titles_2_colors['nearest_neighbor'],
                    marker=titles_2_markers['nearest_neighbor'],
                    linewidth=2.0,
                    markevery=(0.05, 0.1)
                    )
    plt.tight_layout()
    plt.xlabel('iterations')
    # ylabel = r'$1 - r_{\mathcal{X}}$'
    ylabel = r'$1 - R_{\mathcal{X}}(P)$'
    plt.ylabel(ylabel)
    
    rounded_acc = round(acc, 2)
    if custom_loss:
        title = r'max Euclidean distance: $\epsilon={}$'.format(rounded_acc)
    else:
        title = r'fixed-point residual: $\epsilon={}$'.format(rounded_acc)
    plt.title(title)
    if worst_case:
        plt.xscale('log')
    plt.savefig(f"acc_{acc}.pdf", bbox_inches='tight')
    plt.clf()


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_plot_fp.yaml')
def robust_kalman_plot_eval_iters_fp(cfg):
    example = 'robust_kalman'
    create_classical_results(example, cfg)


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_plot_custom.yaml')
def robust_kalman_plot_eval_iters_custom(cfg):
    example = 'robust_kalman'
    create_classical_results(example, cfg)

@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_plot.yaml')
def unconstrained_qp_plot_eval_iters(cfg):
    example = 'unconstrained_qp'

    percentile_plots_maml(example, cfg)
    create_gen_l2o_results_maml(example, cfg)
    


@hydra.main(config_path='configs/sine', config_name='sine_plot.yaml')
def sine_plot_eval_iters(cfg):
    example = 'sine'
    create_gen_l2o_results_maml(example, cfg)
    percentile_plots_maml(example, cfg)
    get_maml_visualization_data(example, cfg)



@hydra.main(config_path='configs/mnist', config_name='mnist_plot_fp.yaml')
def mnist_plot_eval_iters_fp(cfg):
    example = 'mnist'
    create_classical_results(example, cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_plot_custom.yaml')
def mnist_plot_eval_iters_custom(cfg):
    example = 'mnist'
    create_classical_results(example, cfg)



@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_plot.yaml')
def quadcopter_plot_eval_iters(cfg):
    example = 'quadcopter'
    # plot_eval_iters(example, cfg, train=False)
    # overlay_training_losses(example, cfg)
    # plot_eval_iters(example, cfg, train=False)
    # create_journal_results(example, cfg, train=False)
    create_classical_results(example, cfg)



def plot_sparse_coding(metrics, titles, eval_iters, vert_lines=False):
    """
    metrics is a list of lists

    e.g.
    metrics = [metric_fp, metric_pr, metric_dr]
    metric_fp = [cs, nn-ws, ps-ws, k=5, k=10, ..., k=120]
        where cs is a numpy array
    same for metric_pr and metric_dr

    each metric has a title

    each line within each metric has a style

    note that we do not explicitly care about the k values
        we will manually create the legend in latex later
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 13), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharey='row')

    # for i in range(2):

    # yscale
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')

    # x-label
    # axes[0, 0].set_xlabel('evaluation iterations')
    # axes[0, 1].set_xlabel('evaluation iterations')
    fontsize = 40
    title_fontsize = 40
    axes[1, 0].set_xlabel('evaluation iterations', fontsize=fontsize)
    axes[1, 1].set_xlabel('evaluation iterations', fontsize=fontsize)

    # y-label
    axes[0, 0].set_ylabel('test fixed-point residual', fontsize=fontsize)
    axes[1, 0].set_ylabel('test gain to cold start', fontsize=fontsize)

    axes[0, 0].set_title('training with fixed-point residual losses', fontsize=title_fontsize)
    axes[0, 1].set_title('training with regression losses', fontsize=title_fontsize)


    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])


    if len(metrics) == 3:
        start = 1
    else:
        start = 0

    # plot the fixed-point residual
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if title[:3] != 'reg':
                axes[0, 0].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
                # if vert_lines:
                #     if title[0] == 'k':
                #         k = int(title[1:])
                #         axes[i].axvline(k, color=color)
            if title[:3] != 'obj':
                axes[0, 1].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
                # if vert_lines:
                #     if title[0] == 'k':
                #         k = int(title[1:])
                #         axes[i].axvline(k, color=color)

    # plot the gain
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            # if j > 0:
            #     gain = cs / np.array(curr_metric[j])[start:eval_iters + start]
            # else:
            #     cs = np.array(curr_metric[j])[start:eval_iters + start]
            if j == 0:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            else:
                gain = np.clip(cs / np.array(curr_metric[j])[start:eval_iters + start], 
                               a_min=0, a_max=1500)
                if title[:3] != 'reg':
                    axes[1, 0].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))
                if title[:3] != 'obj':
                    axes[1, 1].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))

            # if vert_lines:
            #     if title[0] == 'k':
            #         k = int(title[1:])
            #         plt.axvline(k, color=color)
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    
    fig.tight_layout()
    if vert_lines:
        plt.savefig('all_metric_plots_vert.pdf', bbox_inches='tight')
    else:
        plt.savefig('all_metric_plots.pdf', bbox_inches='tight')
    
    plt.clf()


def determine_scs_or_osqp(example):
    if example == 'unconstrained_qp' or example == 'lasso' or example == 'jamming' or example == 'sparse_coding' or example == 'sine':
        return False
    return True


# def get_percentiles(example, cfg, first=True):
#     orig_cwd = hydra.utils.get_original_cwd()

#     if first:
#         if example == 'sine':
#             percentile_dt = cfg.percentile_datetime
#         else:
#             percentile_dt = cfg.percentile_datetime
#     else:
#         percentile_dt = cfg.nearest_neighbor_datetime
#     path = f"{orig_cwd}/outputs/{example}/train_outputs/{percentile_dt}"
#     # no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"

#     # return the first column
#     percentiles_list = []
#     percentiles = cfg.get('percentiles', [30, 50, 90, 95, 99])
#     for i in range(len(percentiles)):
#         # filename = f"percentiles/train_{percentiles[i]}.csv"
#         filename = f"percentiles/test_{percentiles[i]}.csv"
#         df = read_csv(f"{path}/{filename}")
#         if first:
#             curr_percentile_curve = df['no_train']
#         else:
#             # curr_percentile_curve = df.iloc[-1]
#             curr_percentile_curve = df['nearest_neighbor']
#         percentiles_list.append(curr_percentile_curve)
#     return percentiles_list


def get_percentiles(example, cfg, percentile_dt, train, col, percentiles):
    orig_cwd = hydra.utils.get_original_cwd()

    # if first:
    #     if example == 'sine':
    #         percentile_dt = cfg.percentile_datetime
    #     else:
    #         percentile_dt = cfg.percentile_datetime
    # else:
    #     percentile_dt = cfg.nearest_neighbor_datetime
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{percentile_dt}"
    # no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"

    # return the first column
    percentiles_list = []
    # percentiles = cfg.get('percentiles', [30, 50, 90, 95, 99])
    for i in range(len(percentiles)):
        # filename = f"percentiles/train_{percentiles[i]}.csv"
        if train:
            filename = f"percentiles/train_{percentiles[i]}.csv"
        else:
            filename = f"percentiles/test_{percentiles[i]}.csv"
        df = read_csv(f"{path}/{filename}")
        if col == 'last':
            curr_percentile_curve = df.iloc[:, -1]
        else:
            curr_percentile_curve = df[col]
        #     curr_percentile_curve = df['no_train']
        # else:
        #     # 
        #     curr_percentile_curve = df['nearest_neighbor']
        percentiles_list.append(curr_percentile_curve)
    return percentiles_list


def get_percentiles_learned(example, cfg, percentiles, first=False):
    orig_cwd = hydra.utils.get_original_cwd()
    percentile_dt_list = cfg.percentile_datetime_list

    percentiles_list_list = []
    for j in range(len(percentile_dt_list)):
        percentile_dt = percentile_dt_list[j]
        path = f"{orig_cwd}/outputs/{example}/train_outputs/{percentile_dt}"

        # return the last column
        percentiles_list = []
        # percentiles = cfg.get('percentiles', [30, 50, 90, 95, 99])
        for i in range(len(percentiles)):
            filename = f"percentiles/test_{percentiles[i]}.csv"
            df = read_csv(f"{path}/{filename}")
            if first:
                curr_percentile_curve = df['no_train']
            else:
                curr_percentile_curve = df.iloc[:, -1]
            percentiles_list.append(curr_percentile_curve)
        percentiles_list_list.append(percentiles_list)
    return percentiles_list_list


def get_frac_solved_data_classical(example, cfg):
    # setup
    cold_start_datetimes = cfg.cold_start_datetimes
    
    cold_start_results = []
    guarantee_results = []

    accuracies = get_accs(cfg)
    for acc in accuracies:
        cold_title = 'nearest_neighbor' if example == 'quadcopter' else 'no_train'
        train = example != 'quadcopter'
        curr_cold_start_results = load_frac_solved(example, cold_start_datetimes[-1], acc, train=train, 
                                                    title=cold_title)
        cold_start_results.append(curr_cold_start_results)
        curr_guarantee_results = []
        for datetime in cold_start_datetimes:
            # guarantee_title = 'nearest_neighbor_pac_bayes' if example == 'quadcopter' else 'no_train'
            guarantee_title = cold_title + '_pac_bayes'
            single_guarantee_results = load_frac_solved(example, datetime, acc, train=train, 
                                                        title=guarantee_title)
            curr_guarantee_results.append(single_guarantee_results)
        guarantee_results.append(curr_guarantee_results)

    return cold_start_results, guarantee_results


def get_frac_solved_data_classical_aux(example, cfg):
    # setup
    cold_start_datetimes = cfg.cold_start_datetimes

    aux_datetime = cfg.cold_start_datetime
    
    cold_start_results = []
    guarantee_results = []
    aux_results = []

    accuracies = get_accs(cfg)
    for acc in accuracies:
        # base
        cold_title = 'nearest_neighbor' if example == 'quadcopter' else 'no_train'
        train = example != 'quadcopter'
        curr_cold_start_results = load_frac_solved(example, cold_start_datetimes[-1], acc, train=train, 
                                                    title=cold_title)
        cold_start_results.append(curr_cold_start_results)
        
        # aux
        if example == 'quadcopter':
            curr_aux_results = load_frac_solved(example, aux_datetime, acc, train=False, 
                                                        title='no_train')
            aux_results.append(curr_aux_results)

        curr_guarantee_results = []
        for datetime in cold_start_datetimes:
            # guarantee_title = 'nearest_neighbor_pac_bayes' if example == 'quadcopter' else 'no_train'
            guarantee_title = cold_title + '_pac_bayes'
            single_guarantee_results = load_frac_solved(example, datetime, acc, train=train, 
                                                        title=guarantee_title)
            curr_guarantee_results.append(single_guarantee_results)
        guarantee_results.append(curr_guarantee_results)

    return cold_start_results, guarantee_results, aux_results



def get_frac_solved_data(example, cfg):
    # setup
    orig_cwd = hydra.utils.get_original_cwd()

    cold_start_datetime = cfg.cold_start_datetime
    

    nn_datetime = cfg.nearest_neighbor_datetime

    pretrain_datetime = cfg.pretrain_datetime

    

    # get the datetimes
    learn_datetimes = cfg.output_datetimes
    if learn_datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        learn_datetimes = [dt]

    all_test_results = []
    all_pac_bayes_results = []
    cold_start_results = []
    nearest_neighbor_results = []

    accs = get_accs(cfg)
    for acc in accs: #cfg.accuracies:
        if cold_start_datetime != '':
            # cold_start_datetime = recover_last_datetime(orig_cwd, example, 'train')
            curr_cold_start_results = load_frac_solved(example, cold_start_datetime, acc, train=False, title='no_train')
            cold_start_results.append(curr_cold_start_results)
        if nn_datetime != '':
            # nn_datetime = recover_last_datetime(orig_cwd, example, 'train')
            curr_nearest_neighbor_results = load_frac_solved(example, nn_datetime, acc, train=False, title='nearest_neighbor')
            nearest_neighbor_results.append(curr_nearest_neighbor_results)
        if pretrain_datetime != '':
            # nn_datetime = recover_last_datetime(orig_cwd, example, 'train')
            # curr_nearest_neighbor_results = load_frac_solved(example, nn_datetime, acc, train=False, title='nearest_neighbor')
            curr_nearest_neighbor_results = load_frac_solved(example, pretrain_datetime, acc, train=False)
            nearest_neighbor_results.append(curr_nearest_neighbor_results)
        curr_pac_bayes_results = []
        curr_test_results = []
        for datetime in learn_datetimes:
            pac_bayes_curve = load_frac_solved(example, datetime, acc, train=True)
            test_curve = load_frac_solved(example, datetime, acc, train=False)
            curr_pac_bayes_results.append(pac_bayes_curve)
            curr_test_results.append(test_curve)
        all_pac_bayes_results.append(curr_pac_bayes_results)
        all_test_results.append(curr_test_results)
    return all_test_results, all_pac_bayes_results, cold_start_results, nearest_neighbor_results


def get_all_data(example, cfg, train=False):
    # setup
    orig_cwd = hydra.utils.get_original_cwd()

    # get the datetimes
    learn_datetimes = cfg.output_datetimes
    if learn_datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        learn_datetimes = [dt]

    cold_start_datetime = cfg.cold_start_datetime
    if cold_start_datetime == '':
        cold_start_datetime = recover_last_datetime(orig_cwd, example, 'train')

    nn_datetime = cfg.nearest_neighbor_datetime
    if nn_datetime == '':
        nn_datetime = recover_last_datetime(orig_cwd, example, 'train')

    metrics_list = []
    timing_data = []

    # check if prev_sol exists
    # if 'prev_sol' in cfg.keys():
    # prev_sol_bool = cfg.prev_sol_datetime
    prev_sol_bool = 'prev_sol_datetime' in cfg.keys()

    # benchmarks = ['cold_start', 'nearest_neighbor']
    # benchmark_dts = [cold_start_datetime, nn_datetime]
    benchmarks, benchmark_dts = [], []
    if 'cold_start_datetime' in cfg.keys():
        benchmarks.append('cold_start')
        benchmark_dts.append(cold_start_datetime)
    if 'nearest_neighbor_datetime' in cfg.keys() and example != 'sine':
        benchmarks.append('nearest_neighbor')
        benchmark_dts.append(nn_datetime)
    if prev_sol_bool:
        benchmarks.append('prev_sol')
        benchmark_dts.append(cfg.prev_sol_datetime)

    # for init in ['cold_start', 'nearest_neighbor', 'prev_sol']:

    for i in range(len(benchmarks)):
        init = benchmarks[i]
        datetime = benchmark_dts[i]
        metric, timings = load_data_per_title(example, init, datetime)
        metrics_list.append(metric)
        timing_data.append(timings)

    # learned warm-starts
    k_vals = np.zeros(len(learn_datetimes))
    loss_types = []
    for i in range(len(k_vals)):
        datetime = learn_datetimes[i]
        loss_type = get_loss_type(orig_cwd, example, datetime)
        loss_types.append(loss_type)
        k = get_k(orig_cwd, example, datetime)
        k_vals[i] = k
        metric, timings = load_data_per_title(example, k, datetime)
        metrics_list.append(metric)
        timing_data.append(timings)

    k_vals_new = []
    for i in range(k_vals.size):
        k = k_vals[i]
        new_k = k if k >= 2 else 0
        k_vals_new.append(new_k)
    # titles = benchmarks + [f"k{int(k)}" for k in k_vals_new]
    titles = benchmarks
    for i in range(len(loss_types)):
        loss_type = loss_types[i]
        k = k_vals_new[i]
        titles.append(f"{loss_type}_k{int(k)}")

    metrics = [[row[i] for row in metrics_list] for i in range(len(metrics_list[0]))]

    return metrics, timing_data, titles



def get_maml_vis_df(example, maml_pretrain_visualization_dt, maml_visualization_dt, iter):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{maml_pretrain_visualization_dt}/warm-starts_test"

    import os
    folders = os.listdir(path)
    folders.sort()
    last_time = folders[len(folders)-1]

    # f"{last_date}/{last_time}"

    pretrain_test_df = read_csv(f"{path}/{last_time}/prob_{iter}_z_ws.csv")
    pretrain_grad_points_df = read_csv(f"{path}/{last_time}/grad_points_{iter}.csv")

    path2 = f"{orig_cwd}/outputs/{example}/train_outputs/{maml_visualization_dt}/warm-starts_test"
    folders2 = os.listdir(path2)
    folders2.sort()
    last_time2 = folders2[len(folders2)-1]

    maml_test_df = read_csv(f"{path2}/{last_time2}/prob_{iter}_z_ws.csv")
    maml_grad_points_df = read_csv(f"{path2}/{last_time2}/grad_points_{iter}.csv")
    return pretrain_test_df, pretrain_grad_points_df, maml_test_df, maml_grad_points_df


def get_maml_visualization_data(example, cfg, train=False):
    # setup
    # orig_cwd = hydra.utils.get_original_cwd()

    # maml_pretrain_visualization_dt:
    # maml_visualization_dt:
    maml_pretrain_visualization_dt = cfg.maml_pretrain_visualization_dt
    maml_visualization_dt = cfg.maml_visualization_dt

    cmap = plt.cm.Set1
    colors = cmap.colors

    import os
    os.mkdir('vis')

    for i in range(20):
        out = get_maml_vis_df(example, maml_pretrain_visualization_dt, maml_visualization_dt, i)
        pretrain_test_df, pretrain_grad_points_df, maml_test_df, maml_grad_points_df = out

        x_vals = pretrain_test_df['x_vals']
        y_vals = pretrain_test_df['y_vals']
        x_grad_points = maml_grad_points_df['x_grad_points']
        y_grad_points = maml_grad_points_df['y_grad_points']
        pretrain = pretrain_test_df['predicted_y_vals']
        maml = maml_test_df['predicted_y_vals']
        plt.plot(x_vals, y_vals, color=colors[0], zorder=2) #, label='optimal')
        plt.plot(x_vals, pretrain, color=colors[1], zorder=3) #, label=f"prediction_{j}")
        plt.plot(x_vals, maml, color=colors[2], zorder=4)

        plt.fill_between(x_vals, y_vals - 1, y_vals + 1, color=colors[4], alpha = 0.2, zorder=1)
        # mark the points used for gradients
        plt.scatter(x_grad_points, y_grad_points, color=colors[3], marker='^', s=100, zorder=5)
        plt.ylim((-6, 6))
        plt.savefig(f"vis/maml_vis_{i}.pdf")
        plt.clf()

        # generate grad only plot
        # plt.plot(x_vals, y_vals, color=colors[0], zorder=2)
        plt.scatter(x_grad_points, y_grad_points, color=colors[3], marker='^', s=100, zorder=5)
        plt.ylim((-6, 6))
        plt.savefig(f"vis/maml_vis_grad_only_{i}.pdf")
        plt.clf()

        # generate grads and ground truth
        plt.plot(x_vals, y_vals, color=colors[0], zorder=2)
        plt.scatter(x_grad_points, y_grad_points, color=colors[3], marker='^', s=100, zorder=5)
        plt.ylim((-6, 6))
        plt.savefig(f"vis/maml_vis_metatrain_{i}.pdf")
        plt.clf()

        # generate grads and approx sol
        plt.plot(x_vals, y_vals, color=colors[0], zorder=2)
        plt.plot(x_vals, maml, color=colors[2], zorder=4)
        plt.scatter(x_grad_points, y_grad_points, color=colors[3], marker='^', s=100, zorder=5)
        plt.ylim((-6, 6))
        plt.savefig(f"vis/maml_vis_grad_and_approx_{i}.pdf")
        plt.clf()





def load_frac_solved(example, datetime, acc, train, title=None):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/frac_solved"

    fp_file = f"tol={acc}_train.csv" if train else f"tol={acc}_test.csv"
    df = read_csv(f"{path}/{fp_file}")
    if title is None:
        if train:
            results = df.iloc[:, -1]
        else:
            results = df.iloc[:, -3]
    else:
        results = df[title]
    return results

def load_data_per_title(example, title, datetime, train=False):
    scs_or_osqp = determine_scs_or_osqp(example)

    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}"
    # no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"

    # read the eval iters csv
    # fp_file = 'eval_iters_train.csv' if train else 'eval_iters_test.csv'
    fp_file = 'iters_compared_train.csv' if train else 'iters_compared_test.csv'
    fp_df = read_csv(f"{path}/{fp_file}")
    fp = get_eval_array(fp_df, title)
    # fp = fp_df[title]

    # read the primal and dual residausl csv
    if scs_or_osqp:
        pr_file = 'primal_residuals_train.csv' if train else 'primal_residuals_test.csv'
        pr_df = read_csv(f"{path}/{pr_file}")
        pr = get_eval_array(pr_df, title)
        # pr = pr_df[title]

        dr_file = 'dual_residuals_train.csv' if train else 'dual_residuals_test.csv'
        dr_df = read_csv(f"{path}/{dr_file}")
        # dr = dr_df[title]
        dr = get_eval_array(dr_df, title)
        metric = [fp, pr, dr]

    # read the obj_diffs csv
    else:
        metric = [fp, fp]
    
    # do timings
    try:
        if scs_or_osqp:
            train_str = 'train' if train else 'test'
            timings_file = f"solve_C/{train_str}_aggregate_solve_times.csv"
            timings_df = read_csv(f"{path}/{timings_file}")
            # timings = timings_df[title]
            timings = get_eval_array(timings_df, title)
        else:
            timings = None
    except:
        timings = None

    return metric, timings


def get_eval_array(df, title):
    if title == 'cold_start' or title == 'no_learn':
        data = df['no_train']
    elif title == 'nearest_neighbor':
        data = df['nearest_neighbor']
    elif title == 'prev_sol':
        data = df['prev_sol']
    else:
        # case of the learned warm-start, take the latest column
        data = df.iloc[:, -1]
    return data



def get_loss_type(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml" # noqa
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # k = int(out_dict['train_unrolls'])
    loss_type = 'reg' if bool(out_dict['supervised']) else 'obj'
    return loss_type



def get_k(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml" # noqa
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    k = int(out_dict['train_unrolls'])
    return k


def get_data(example, datetime, csv_title, eval_iters):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/iters_compared.csv"
    df = read_csv(path)
    if csv_title == 'last':
        last_column = df.iloc[:, -1]
    else:
        last_column = df[csv_title]
    return last_column[:eval_iters]


def get_loss_data(example, datetime):
    orig_cwd = hydra.utils.get_original_cwd()

    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/train_test_results.csv"
    df = read_csv(path)
    # if csv_title == 'last':
    #     last_column = df.iloc[:, -1]
    # else:
    #     last_column = df[csv_title]
    # return last_column[:eval_iters]
    train_losses = df['train_loss']
    test_losses = df['test_loss']
    return train_losses, test_losses




def create_title(example):
    if example == 'robust_kalman':
        title = 'Robust Kalman filtering'
    elif example == 'robust_ls':
        title = 'Robust non-negative least squares'
    elif example == 'sparse_pca':
        title = 'Sparse PCA'
    elif example == 'phase_retrieval':
        title = 'Phase retrieval'
    elif example == 'mnist':
        title = 'Image deblurring'
    elif example == 'quadcopter':
        title = 'Quadcopter'
    elif example == 'lasso':
        title = 'Lasso'
    elif example == 'unconstrained_wp':
        title = 'Unconstrained QP'
    return title




def update_acc(df_acc, accs, col, losses):
    iter_vals = np.zeros(len(accs))
    for i in range(len(accs)):
        if losses.min() < accs[i]:
            iter_vals[i] = int(np.argmax(losses < accs[i]))
        else:
            iter_vals[i] = losses.size
    int_iter_vals = iter_vals.astype(int)
    df_acc[col] = int_iter_vals
    return df_acc




if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'robust_kalman_fp':
        sys.argv[1] = base + 'robust_kalman/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        robust_kalman_plot_eval_iters_fp()
    if sys.argv[1] == 'robust_kalman_custom':
        sys.argv[1] = base + 'robust_kalman/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        robust_kalman_plot_eval_iters_custom()
    elif sys.argv[1] == 'unconstrained_qp':
        sys.argv[1] = base + 'unconstrained_qp/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        unconstrained_qp_plot_eval_iters()
    elif sys.argv[1] == 'mnist_fp':
        sys.argv[1] = base + 'mnist/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        mnist_plot_eval_iters_fp()
    elif sys.argv[1] == 'mnist_custom':
        sys.argv[1] = base + 'mnist/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        mnist_plot_eval_iters_custom()
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        quadcopter_plot_eval_iters()
    elif sys.argv[1] == 'sparse_coding':
        sys.argv[1] = base + 'sparse_coding/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        sparse_coding_plot_eval_iters()
    elif sys.argv[1] == 'sine':
        sys.argv[1] = base + 'sine/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        sine_plot_eval_iters()

