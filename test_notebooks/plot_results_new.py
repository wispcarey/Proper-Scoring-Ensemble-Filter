import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


TICK_STEPSIZE = {
    'lorenz63':[0.25,0.04],
    'lorenz96':[0.5,0.3],
    'ks':[0.5,0.4]
}

ENSEMBLE_SIZE = [5,10,15,20,40,60,100]

DATASET_FOLDERS = {
    'lorenz63': [
        "2025-04-10_17-18lorenz63_1.0_10_60_8192_norm_EnST_joint",
        "2025-04-10_19-40lorenz63_0.7_10_60_8192_norm_EnST_joint",
        "2025-04-11_12-18lorenz63_1.0_20_60_8192_norm_EnST_tuned_joint",
        "2025-04-11_12-39lorenz63_0.7_20_60_8192_norm_EnST_tuned_joint"],
    'lorenz96': [
        "2025-04-10_09-31lorenz96_1.0_10_60_8192_norm_EnST_joint",
        "2025-04-10_13-18lorenz96_0.7_10_60_8192_norm_EnST_joint",
        "2025-04-11_12-59lorenz96_1.0_20_60_8192_norm_EnST_tuned_joint",
        "2025-04-11_13-31lorenz96_0.7_20_60_8192_norm_EnST_tuned_joint"],
    'ks': [
        "2025-04-09_18-18ks_1.0_10_60_8192_norm_EnST_joint",
        "2025-04-10_01-52ks_0.7_10_60_8192_norm_EnST_joint",
        "2025-04-11_14-02ks_1.0_20_60_8192_norm_EnST_tuned_joint",
        "2025-04-11_15-09ks_0.7_20_60_8192_norm_EnST_tuned_joint"],
}

DATASET_METHODS = {
    'lorenz63': ["EnKF_PertObs", 
                "EnKF_Sqrt", 
                "iEnKF_PertObs", 
                # "MLF",
                ],
    'lorenz96': ["EnKF_loc", 
                "EnKF_Sqrt", 
                "LETKF", 
                "iEnKF_PertObs", 
                # "MLF",
                ],
    'ks': ["EnKF_loc", 
        "EnKF_Sqrt", 
        "LETKF", 
        "iEnKF_PertObs", 
        # "MLF",
        ]
}

TEST_DATA = {
    'lorenz63': '../data/lorenz63/test_0_96000_v_0.150step.npy',
    'lorenz96': '../data/lorenz96/test_0_96000_v_0.150step.npy',
    'ks': '../data/ks/test_0_128000_v_1.000step.npy'
}

COLOR_DICT = {
    'Pretrain': "green",
    'Tuned': "blue",
    'LETKF': "red",
    'ESRF': "cyan",
    'EnKF': "orange",
    'IEnKF': "brown",
    'MLEF': "purple"
}

# Default font size will be overridden by tick_fontsize parameter

def replace_nan_with_value(nparray, value=6):
    """Replace NaN values in numpy array with specified value"""
    nparray[np.isnan(nparray)] = value
    return nparray

def collect_results(args):
    """Collect neural network results from saved files"""
    folder_list = DATASET_FOLDERS[args.dataset]
    
    results = np.zeros((4, len(ENSEMBLE_SIZE), 5), dtype=float)
    
    for i in range(4):
        for j, N in enumerate(ENSEMBLE_SIZE):
            data_dic = torch.load(f'../save/{folder_list[i]}/output_records_{N}.pt', weights_only=True)
            results[i, j, 0] = data_dic['nn']['mean_rmse']
            results[i, j, 1] = data_dic['nn']['std_rmse']
            results[i, j, 2] = data_dic['nn']['mean_rrmse']
            results[i, j, 3] = data_dic['nn']['std_rrmse']
            # results[i, j, 4] = data_dic['nn']['mean_rmv']
            # results[i, j, 5] = data_dic['nn']['std_rmv']
            results[i, j, 4] = data_dic['nn']['valid_percent'] < 1
    
    rmse_max, rrmse_max = np.nanmax(results[:,:,0]), np.nanmax(results[:,:,2])
    nn_results = {'Pretrain': results[:2], 'Tuned': results[2:]}
    return nn_results, rmse_max, rrmse_max
        
def plot_with_type(x_inds, mean_values, std_values, key, args, ax=None, highlight_key=False):
    """
    Plot different types of visualization based on args.type parameter
    
    Parameters:
    - x_inds: x-axis indices
    - mean_values: mean data values
    - std_values: standard deviation data values
    - key: data series key, used for color and label
    - args: command line arguments, containing type attribute
    - ax: optional Axes object, defaults to current axes
    
    Returns:
    - The plotted line object, can be used for legend configuration
    """
    if ax is None:
        ax = plt.gca()
    
    # Get the color from the dictionary
    color = COLOR_DICT[key]
    
    if highlight_key:
        label = f"$\\mathbf{{{key}}}^*$"  # Bold with asterisk using LaTeX
    else:
        label = f"{key}"
    
    alpha = 0.7
    # Determine plot type based on args.type
    if args.plot_type == 'mean_std':
        # Plot with error bars showing standard deviation
        line = ax.errorbar(
            x_inds, mean_values, yerr=std_values,
            linestyle='-', capsize=3, capthick=2, 
            color=color, marker='o', markersize=args.marker_size,
            linewidth=args.line_width, 
            alpha=alpha, label=label, 
        )
    elif args.plot_type == 'mean':
        # Plot only the mean values
        line, = ax.plot(
            x_inds, mean_values,
            linestyle='-', color=color, marker='o', markersize=args.marker_size,
            linewidth=args.line_width, alpha=alpha, label=label
        )
    elif args.plot_type == 'std':
        # Plot only the standard deviation values
        line, = ax.plot(
            x_inds, std_values,
            linestyle='--', color=color, marker='o', markersize=args.marker_size,
            linewidth=args.line_width, alpha=alpha, label=label
        )
    else:
        # Default to mean_std if args.type is not recognized
        line = ax.errorbar(
            x_inds, mean_values, yerr=std_values,
            linestyle='-', capsize=3, capthick=2, 
            color=color, marker='o', markersize=args.marker_size,
            linewidth=args.line_width, 
            alpha=alpha, label=label
        )
    
    return line

def get_benchmarks(args):
    """
    Process benchmark data from CSV file, splitting by sigma_y values 1 and 0.7,
    and extracting specific columns for each method, then combining them into a 2*N*5 numpy array.
    
    Args:
        args: An object or namespace with a dataset attribute specifying the dataset name.
        
    Returns:
        result_dict: A dictionary where each key is a method, and the value is a 2*N*5 numpy array.
    """
    file_path = f'../save/benchmark/benchmarks_{args.dataset}.csv'
    df = pd.read_csv(file_path, usecols=['method', 'sigma_y', 'rmse', 'rmse_dstd', 'rrmse_mean', 'rrmse_mean_dstd', 'nan_exist'])

    result_dict = {}

    # methods = df['method'].unique()
    methods = DATASET_METHODS[args.dataset]

    rmse_max, rrmse_max = 0, 0
    for method in methods:
        method_data = df[df['method'] == method]
        
        # Filter rows where sigma_y == 1 and 0.7
        sigma_y_1 = method_data[method_data['sigma_y'] == 1][['rmse', 'rmse_dstd', 'rrmse_mean', 'rrmse_mean_dstd', 'nan_exist']]
        sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['rmse', 'rmse_dstd', 'rrmse_mean', 'rrmse_mean_dstd', 'nan_exist']]
        
        # Convert to numpy arrays
        sigma_y_1_array = sigma_y_1.to_numpy()
        sigma_y_0_7_array = sigma_y_0_7.to_numpy()
        
        # Combine into a 2*N*5 array
        combined_array = np.stack([sigma_y_1_array, sigma_y_0_7_array], axis=0).astype(float)
        
        # Store in result dictionary
        if method == 'iEnKF_PertObs':
            method_name = 'IEnKF'
        elif method == 'EnKF_Sqrt':
            method_name = 'ESRF'
        elif method == 'EnKF_loc':
            method_name = 'EnKF'
        elif method == 'MLF':
            method_name = 'MLEF'
        else:
            method_name = method
        result_dict[method_name] = combined_array
        
        rmse_m, rrmse_m = np.nanmax(combined_array[:,:,0]), np.nanmax(combined_array[:,:,2])
        rmse_max, rrmse_max = np.maximum(rmse_m, rmse_max), np.maximum(rrmse_m, rrmse_max)

    return result_dict, rmse_max, rrmse_max

def save_legend(args):
    """Save a separate horizontal legend containing all methods"""
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    
    # Create dummy plots for legend
    legend_items = ['Pretrain', 'Tuned', 'EnKF', 'ESRF', 'IEnKF']
    if args.dataset != 'lorenz63':
        legend_items.insert(-1, 'LETKF')  # Insert LETKF before IEnKF
    
    lines = []
    for item in legend_items:
        if item in ['Pretrain', 'Tuned']:
            item_name = rf"$\mathbf{{{item}}}^*$"
        else:
            item_name = item
        line, = ax.plot([], [], linestyle='-', color=COLOR_DICT[item], markersize=args.marker_size,
                       marker='o', linewidth=args.line_width, alpha=0.75, label=item_name)
        lines.append(line)
    
    # Create horizontal legend
    legend = ax.legend(handles=lines, loc='center', ncol=len(legend_items), 
                      frameon=False, fontsize=args.tick_fontsize)
    
    # Save the legend
    plt.savefig(os.path.join('../save/figures', f'{args.dataset}_legend.pdf'), 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.savefig(os.path.join('../save/figures', f'{args.dataset}_legend.png'), 
                bbox_inches='tight', dpi=300, pad_inches=0.1)
    print("Legend saved to", os.path.join('../save/figures', f'{args.dataset}_legend.png'))
    plt.close()

def plot_results_all(args):
    """Main plotting function for all results"""
    
    x_inds = ENSEMBLE_SIZE
    
    nn_results, nn_rmse_max, nn_rrmse_max = collect_results(args)
    benchmark_results, b_rmse_max, b_rrmse_max = get_benchmarks(args)
    
    tick_stepsize = TICK_STEPSIZE[args.dataset]
    
    nan_rmse_val = (int(np.maximum(nn_rmse_max, b_rmse_max) / tick_stepsize[0]) + 0.5) * tick_stepsize[0]
    nan_rrmse_val = (int(np.maximum(nn_rrmse_max, b_rrmse_max) / tick_stepsize[1]) + 0.5) * tick_stepsize[1]
    
    dataset_name = args.dataset.upper()
    if args.plot_type == 'mean_std':
        titles = [
            f"{dataset_name}:" + r"$\sigma_y=1$; RMSE Mean $\pm$ 1std", 
            f"{dataset_name}:" + r"$\sigma_y=1$; R-RMSE Mean $\pm$ 1std",
            f"{dataset_name}:" + r"$\sigma_y=0.7$; RMSE Mean $\pm$ 1std",
            f"{dataset_name}:" + r"$\sigma_y=0.7$; R-RMSE Mean $\pm$ 1std"
        ]
    elif args.plot_type == 'mean':
        titles = [
            f"{dataset_name}:" + r"$\sigma_y=1$; RMSE Mean", 
            f"{dataset_name}:" + r"$\sigma_y=1$; R-RMSE Mean",
            f"{dataset_name}:" + r"$\sigma_y=0.7$; RMSE Mean",
            f"{dataset_name}:" + r"$\sigma_y=0.7$; R-RMSE Mean"
        ]
    elif args.plot_type == 'std':
        titles = [
            f"{dataset_name}:" + r"$\sigma_y=1$; RMSE Std", 
            f"{dataset_name}:" + r"$\sigma_y=1$; R-RMSE Std",
            f"{dataset_name}:" + r"$\sigma_y=0.7$; RMSE Std",
            f"{dataset_name}:" + r"$\sigma_y=0.7$; R-RMSE Std"
        ]

    y_labels = ["RMSE", "R-RMSE", "RMSE", "R-RMSE"]
    save_figure_suffix = ["RMSE_10", "R_RMSE_10", "RMSE_07", "R_RMSE_07"]

    nan_replace_vals = [nan_rmse_val, 0, nan_rrmse_val, 0]
    
    # R-RMSE plots
    figsize = (10,5)
    fig1 = plt.figure(1, figsize=figsize) 
    fig2 = plt.figure(2, figsize=figsize) 
    fig3 = plt.figure(3, figsize=figsize)
    fig4 = plt.figure(4, figsize=figsize)
    
    value_max = np.zeros((2,2))
    value_min = 100 * np.ones((2,2))
    
    std_record = {'dataset':dataset_name}
    
            
    nan_in_others = np.zeros(2, dtype=bool)
    other_results = {}
    for key, value in benchmark_results.items():
        if key == "EnKF":
            other_results['EnKF'] = value
        elif key == "LETKF":
            other_results['LETKF'] = value
        elif key == "IEnKF":
            other_results['IEnKF'] = value
        
        nan_in_others = np.logical_or(nan_in_others, np.any(np.isnan(value), axis=(1,2)))
        for i in [0,1,2,3]:
            value[:, :, i] = replace_nan_with_value(value[:, :, i], nan_replace_vals[i])
        
        max_val = np.max(value[:,:,[0,2]], axis=1)
        value_max = np.maximum(max_val, value_max)
        min_val = np.min(value[:,:,[0,2]], axis=1)
        value_min = np.minimum(min_val, value_min)
        
        mean_rmse_1, std_rmse_1, mean_rrmse_1, std_rrmse_1, nan_exists_1 = value[0, :, 0], value[0, :, 1], value[0, :, 2], value[0, :, 3], value[0, :, 4]

        plt.figure(1)
        plot_with_type(x_inds, mean_rmse_1, std_rmse_1, key, args)
        
        plt.figure(2)
        plot_with_type(x_inds, mean_rrmse_1, std_rrmse_1, key, args)
        
        if args.require_rmse:
            std_record[f'{key}_10_rmse'] = np.nanmean(std_rmse_1)
        std_record[f'{key}_10_rrmse'] = np.nanmean(std_rrmse_1)
        
        mean_rmse_07, std_rmse_07, mean_rrmse_07, std_rrmse_07, nan_exists_07 = value[1, :, 0], value[1, :, 1], value[1, :, 2], value[1, :, 3], value[1, :, 4]
        
        plt.figure(3)
        plot_with_type(x_inds, mean_rmse_07, std_rmse_07, key, args)
        
        plt.figure(4)
        plot_with_type(x_inds, mean_rrmse_07, std_rrmse_07, key, args)
        
        if args.require_rmse:
            std_record[f'{key}_07_rmse'] = np.nanmean(std_rmse_07)
        std_record[f'{key}_07_rrmse'] = np.nanmean(std_rrmse_07)
    
    nan_in_ours = np.zeros(2, dtype=bool)
    for key, value in nn_results.items():
        if key.startswith('Tuned'):
            ours_best = value
        # change nan to values
        nan_in_ours = np.logical_or(nan_in_ours, np.any(np.isnan(value), axis=(1,2)))
        
        for i in [0,1,2,3]:
            value[:, :, i] = replace_nan_with_value(value[:, :, i], nan_replace_vals[i])

        max_val = np.max(value[:,:,[0,2]], axis=1)
        value_max = np.maximum(max_val, value_max)
                
        mean_rmse_1, std_rmse_1, mean_rrmse_1, std_rrmse_1, nan_exists_1 = value[0, :, 0], value[0, :, 1], value[0, :, 2], value[0, :, 3], value[0, :, 4]

        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=1$:", mean_rmse_1)

        plt.figure(1)
        plot_with_type(x_inds, mean_rmse_1, std_rmse_1, key, args, highlight_key=True)
        
        plt.figure(2)
        plot_with_type(x_inds, mean_rrmse_1, std_rrmse_1, key, args, highlight_key=True)
        
        if args.require_rmse:
            std_record[f'{key}_10_rmse'] = np.nanmean(std_rmse_1)
        std_record[f'{key}_10_rrmse'] = np.nanmean(std_rrmse_1)
        
        mean_rmse_07, std_rmse_07, mean_rrmse_07, std_rrmse_07, nan_exists_07 = value[1, :, 0], value[1, :, 1], value[1, :, 2], value[1, :, 3], value[1, :, 4]
        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=0.7$:", mean_rmse_07)

        plt.figure(3)
        plot_with_type(x_inds, mean_rmse_07, std_rmse_07, key, args, highlight_key=True)
        
        plt.figure(4)
        plot_with_type(x_inds, mean_rrmse_07, std_rrmse_07, key, args, highlight_key=True)
        
        if args.require_rmse:
            std_record[f'{key}_07_rmse'] = np.nanmean(std_rmse_07)
        std_record[f'{key}_07_rrmse'] = np.nanmean(std_rrmse_07)

    
    if args.dataset == 'lorenz63':
        std_record[f'LETKF_10_rrmse'], std_record[f'LETKF_07_rrmse'] = "", ""
        if args.require_rmse:
            std_record[f'LETKF_10_rmse'], std_record[f'LETKF_07_rmse'] = "", ""
    
    value_max, value_min = value_max.flatten(), value_min.flatten()
    for i in range(4):
        plt.figure(i+1)

        if i in [0,2]:
            nan_val = nan_rmse_val
        else:
            nan_val = nan_rrmse_val        
        
        ticksize = tick_stepsize[i%2]
            
        vmax = value_max[i]
        vmin = value_min[i] 
        if nan_in_ours[i//2] or nan_in_others[i//2]:
            y_ticks = list(np.arange(int(vmin / ticksize + 0.5) * ticksize, nan_val + ticksize/2, ticksize))
            if int(nan_val) != nan_val:
                y_ticks.append(nan_val)
            new_y_ticks = ["NaN" if y == nan_val else f"{y:.2f}".rstrip('0').rstrip('.') if '.' in f"{y:.2f}" else f"{y}" for y in y_ticks]
        else:
            y_ticks = list(np.arange(int(vmin / ticksize + 0.5) * ticksize, vmax + ticksize/2, ticksize))
            new_y_ticks = [f"{y:.2f}".rstrip('0').rstrip('.') if '.' in f"{y:.2f}" else f"{y}" for y in y_ticks]
            
        plt.yticks(ticks=y_ticks, labels=new_y_ticks, fontsize=args.tick_fontsize)
        # Remove legend from individual plots
        # plt.legend(loc='upper right')
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        # plt.title(titles[i])
        # Remove x and y labels
        # plt.xlabel("Ensemble Size")
        # plt.ylabel(y_labels[i])
        plt.xticks(ENSEMBLE_SIZE, fontsize=args.tick_fontsize)
        plt.grid(True)
        plt.tight_layout()
        
        suffix = ''
        if args.require_rmse or i % 2:
            plt.savefig(os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}{suffix}.png"), bbox_inches="tight", dpi=200)
            plt.savefig(os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}{suffix}.pdf"), bbox_inches="tight", dpi=200)
            print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}.png"))
        plt.close()  
    

    # =========================
    # Relative improvement plots
    # Now produce THREE figures:
    #   (1) sigma_y = 1.0 only
    #   (2) sigma_y = 0.7 only
    #   (3) Combined (overlay 1.0 and 0.7)
    # =========================
    figsize = (10, 5)

    # ---- Compute relative improvement arrays for sigma_y = 1.0 ----
    relative_imp_1_enkf = (other_results['EnKF'][0,:,2] - ours_best[0,:,2]) / other_results['EnKF'][0,:,2]
    relative_imp_1_ienkf = (other_results['IEnKF'][0,:,2] - ours_best[0,:,2]) / other_results['IEnKF'][0,:,2]
    min_val_1 = np.minimum(np.min(relative_imp_1_enkf), np.min(relative_imp_1_ienkf))
    if args.dataset != 'lorenz63':
        relative_imp_1_letkf = (other_results['LETKF'][0,:,2] - ours_best[0,:,2]) / other_results['LETKF'][0,:,2]
        min_val_1 = np.minimum(min_val_1, np.min(relative_imp_1_letkf))
    else:
        relative_imp_1_letkf = None  # Not available

    # ---- Plot sigma_y = 1.0 only ----
    plt.figure(figsize=figsize)
    plt.plot(x_inds, relative_imp_1_enkf, linestyle='-', marker='o', markersize=args.marker_size, 
             linewidth=args.line_width, color=COLOR_DICT['EnKF'], label="EnKF")
    plt.plot(x_inds, relative_imp_1_ienkf, linestyle='-', marker='o', markersize=args.marker_size, 
             linewidth=args.line_width, color=COLOR_DICT['IEnKF'], label="IEnKF")
    if relative_imp_1_letkf is not None:
        plt.plot(x_inds, relative_imp_1_letkf, linestyle='-', marker='o', markersize=args.marker_size, 
                 linewidth=args.line_width, color=COLOR_DICT['LETKF'], label="LETKF")

    # Keep style consistent with existing figures (no legend text on plot)
    plt.xticks([5, 10, 15, 20, 40, 60, 100], fontsize=args.tick_fontsize)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.yticks(fontsize=args.tick_fontsize)
    if min_val_1 > 0:
        plt.gca().set_ylim(0, None)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_1.0.pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_1.0.png"), bbox_inches="tight", dpi=300)
    print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_1.0.png"))
    plt.close()

    # ---- Compute relative improvement arrays for sigma_y = 0.7 ----
    relative_imp_07_enkf = (other_results['EnKF'][1,:,2] - ours_best[1,:,2]) / other_results['EnKF'][1,:,2]
    relative_imp_07_ienkf = (other_results['IEnKF'][1,:,2] - ours_best[1,:,2]) / other_results['IEnKF'][1,:,2]
    min_val_07 = np.minimum(np.min(relative_imp_07_enkf), np.min(relative_imp_07_ienkf))
    if args.dataset != 'lorenz63':
        relative_imp_07_letkf = (other_results['LETKF'][1,:,2] - ours_best[1,:,2]) / other_results['LETKF'][1,:,2]
        min_val_07 = np.minimum(min_val_07, np.min(relative_imp_07_letkf))
    else:
        relative_imp_07_letkf = None  # Not available

    # ---- Plot sigma_y = 0.7 only ----
    plt.figure(figsize=figsize)
    plt.plot(x_inds, relative_imp_07_enkf, linestyle='-', marker='o', markersize=args.marker_size, 
             linewidth=args.line_width, color=COLOR_DICT['EnKF'], label="EnKF")
    plt.plot(x_inds, relative_imp_07_ienkf, linestyle='-', marker='o', markersize=args.marker_size, 
             linewidth=args.line_width, color=COLOR_DICT['IEnKF'], label="IEnKF")
    if relative_imp_07_letkf is not None:
        plt.plot(x_inds, relative_imp_07_letkf, linestyle='-', marker='o', markersize=args.marker_size, 
                 linewidth=args.line_width, color=COLOR_DICT['LETKF'], label="LETKF")

    # Keep style consistent with existing figures (no legend text on plot)
    plt.xticks([5, 10, 15, 20, 40, 60, 100], fontsize=args.tick_fontsize)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.yticks(fontsize=args.tick_fontsize)
    if min_val_07 > 0:
        plt.gca().set_ylim(0, None)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_0.7.pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_0.7.png"), bbox_inches="tight", dpi=300)
    print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_0.7.png"))
    plt.close()

    # ---- Plot combined figure: overlay sigma_y = 1.0 (solid) and 0.7 (dashed) ----
    plt.figure(figsize=(12,5))

    # EnKF
    plt.plot(x_inds, relative_imp_1_enkf, linestyle='-', marker='o', markersize=args.marker_size,
             linewidth=args.line_width, color=COLOR_DICT['EnKF'], label="EnKF (1.0)")
    plt.plot(x_inds, relative_imp_07_enkf, linestyle='--', marker='o', markersize=args.marker_size,
             linewidth=args.line_width, color=COLOR_DICT['EnKF'], label="EnKF (0.7)")

    # IEnKF
    plt.plot(x_inds, relative_imp_1_ienkf, linestyle='-', marker='o', markersize=args.marker_size,
             linewidth=args.line_width, color=COLOR_DICT['IEnKF'], label="IEnKF (1.0)")
    plt.plot(x_inds, relative_imp_07_ienkf, linestyle='--', marker='o', markersize=args.marker_size,
             linewidth=args.line_width, color=COLOR_DICT['IEnKF'], label="IEnKF (0.7)")

    # LETKF if available
    if relative_imp_1_letkf is not None and relative_imp_07_letkf is not None:
        plt.plot(x_inds, relative_imp_1_letkf, linestyle='-', marker='o', markersize=args.marker_size,
                 linewidth=args.line_width, color=COLOR_DICT['LETKF'], label="LETKF (1.0)")
        plt.plot(x_inds, relative_imp_07_letkf, linestyle='--', marker='o', markersize=args.marker_size,
                 linewidth=args.line_width, color=COLOR_DICT['LETKF'], label="LETKF (0.7)")

    # >>> NEW: add legend on the right, outside the axes
    # Keep frame off to match existing minimalist style; adjust fontsize to tick size.
    legend = plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=args.tick_fontsize * 0.8
    )
    # <<< END NEW

    plt.xticks([5, 10, 15, 20, 40, 60, 100], fontsize=args.tick_fontsize)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.yticks(fontsize=args.tick_fontsize)

    # Determine lower bound across both sigmas to optionally start from 0
    all_mins = [np.min(relative_imp_1_enkf), np.min(relative_imp_07_enkf),
                np.min(relative_imp_1_ienkf), np.min(relative_imp_07_ienkf)]
    if relative_imp_1_letkf is not None and relative_imp_07_letkf is not None:
        all_mins.extend([np.min(relative_imp_1_letkf), np.min(relative_imp_07_letkf)])
    if np.min(all_mins) > 0:
        plt.gca().set_ylim(0, None)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_Both.pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_Both.png"), bbox_inches="tight", dpi=300)
    print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_Both.png"))
    plt.close()

    # Save unified legend
    save_legend(args)

    if args.save_csv:
        df = pd.DataFrame([std_record])
        csv_file = os.path.join("../save/figures", f"std_results.csv")
        directory = os.path.dirname(csv_file)
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist
        if os.path.isfile(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode='w', index=False)
        print("STD records saved to", csv_file)
        
    
    

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process benchmark dataset.")
    
    # Add dataset argument with choices and default value
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["ks", "lorenz96", "lorenz63"], 
        default="lorenz96", 
        help="Specify the dataset to process. Options: 'ks', 'lorenz96', 'lorenz63'. Default is 'lorenz96'."
    )
    parser.add_argument(
        '--plot_type', 
        type=str,
        choices=['mean_std', 'mean', 'std'],
        default='mean',
        help='Visualization type: mean_std (error bars), mean (only mean values), or std (only standard deviation)'
    )
    parser.add_argument(
        '--require_rmse', 
        action='store_true',
        help='Output RMSE in addition to R-RMSE'
    )
    parser.add_argument(
        '--save_csv', 
        action='store_true',
        help='Output to a csv'
    )
    parser.add_argument(
        '--tick_fontsize',
        type=int,
        default=25,
        help='Font size for x and y axis ticks (default: 25)'
    )
    parser.add_argument(
        '--line_width',
        type=float,
        default=3,
        help='Width of all lines in plots (default: 3)'
    )
    
    parser.add_argument(
        '--marker_size',
        type=float,
        default=8,
        help='Size of markers in plots (default: 8)'
    )
    
    args = parser.parse_args()
    
    plot_results_all(args)
