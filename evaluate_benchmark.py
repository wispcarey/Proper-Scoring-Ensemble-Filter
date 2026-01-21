import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

from config.cli import get_parameters
from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output
from train_test_utils import test_ClassicFilter, print_test_results

def get_benchmarks(args, source='dapper'):
    """
    Retrieve benchmark parameters and metrics.
    
    Args:
        args: Argument namespace.
        source (str): 'dapper' to read from CSV, 'torch' to read from grid search .pt file.
        
    Returns:
        If source is 'dapper': Returns (sigma_y_1_array, sigma_y_0_7_array)
        If source is 'torch': Returns (infl, loc_radius)
    """
    if source == 'dapper':
        # --- Original CSV Reading Logic ---
        file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
        df = pd.read_csv(file_path, usecols=['method', 'N', 'sigma_y', 'best_loc_rad','best_infl','rmse', 'rrmse_mean'])

        # Hardcoded method selection from original snippet (can be adjusted if needed)
        method = "LETKF" 
        method_data = df[(df['method'] == method) & (df['N'] == args.N)]
        
        # Filter rows for specific sigma_y
        sigma_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
        sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
        
        return sigma_y_1.to_numpy(), sigma_y_0_7.to_numpy()

    elif source == 'torch':
        # --- New Torch Grid Search Reading Logic ---
        # Construct path: save/{dataset}_benchmarks/benchmark_{dataset}_{sigma}_{method}/grid_search_results_{N}.pt
        folder_name = os.path.join(
            "save", 
            f"{args.dataset}_benchmarks", 
            f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}"
        )
        data_path = os.path.join(folder_name, f"grid_search_results_{args.N}.pt")
        
        print(f"Loading benchmarks from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Benchmark file not found: {data_path}")

        benchmark_data = torch.load(data_path, weights_only=False)
        
        # Extract optimal parameters
        best_params = benchmark_data['best_params']
        infl = best_params['infl']
        loc_radius = best_params['loc_radius']
        
        return infl, loc_radius

    else:
        raise ValueError(f"Unknown benchmark source: {source}")


def write_benchmark_row(args, loc_radius, infl, rmse, rrmse, rmse_std=None, rrmse_std=None):
    """
    Append or upsert a row into the benchmark CSV.
    """
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'

    if not os.path.isfile(file_path):
        base_cols = [
            'method', 'N', 'sigma_y',
            'best_loc_rad', 'best_infl',
            'nan_exist',
            'rmse', 'rmse_dstd', 'rmse_std', 'rmse_std_dstd',
            'rmv_mean', 'rmv_mean_dstd', 'rmv_std', 'rmv_std_dstd',
            'rrmse_mean', 'rrmse_mean_dstd', 'rrmse_std', 'rrmse_std_dstd'
        ]
        df_full = pd.DataFrame(columns=base_cols)
    else:
        df_full = pd.read_csv(file_path)

    candidate = {
        'method': f'{args.v}_loc',
        'N': int(args.N),
        'sigma_y': float(args.sigma_y),
        'best_loc_rad': float(loc_radius) if loc_radius is not None else np.nan,
        'best_infl': float(infl) if infl is not None else np.nan,
        'rmse': float(rmse) if rmse is not None else np.nan,
        'rrmse_mean': float(rrmse) if rrmse is not None else np.nan,
        'rmse_std': float(rmse_std) if rmse_std is not None else np.nan,
        'rrmse_std': float(rrmse_std) if rrmse_std is not None else np.nan,
    }

    for col in df_full.columns:
        if col not in candidate:
            candidate[col] = np.nan

    extra_cols = [k for k in candidate.keys() if k not in df_full.columns]
    if extra_cols:
        for c in extra_cols:
            df_full[c] = np.nan

    has_keys = all(k in df_full.columns for k in ['method', 'N', 'sigma_y'])
    if has_keys and len(df_full) > 0:
        mask = (
            (df_full['method'] == candidate['method']) &
            (df_full['N'] == candidate['N']) &
            (df_full['sigma_y'] == candidate['sigma_y'])
        )
        if mask.any():
            idxs = np.flatnonzero(mask)
            idx = idxs[0]
            print(f"[write_benchmark_row] Overwriting existing row for {candidate['method']}, N={candidate['N']}.")
            for k, v in candidate.items():
                df_full.at[idx, k] = v
            if len(idxs) > 1:
                df_full = df_full.drop(index=idxs[1:]).reset_index(drop=True)
        else:
            df_full = pd.concat([df_full, pd.DataFrame([candidate])], ignore_index=True)
    else:
        df_full = pd.concat([df_full, pd.DataFrame([candidate])], ignore_index=True)

    try:
        df_full = df_full.sort_values(by=['method', 'N', 'sigma_y']).reset_index(drop=True)
    except Exception:
        pass

    df_full.to_csv(file_path, index=False)


if __name__ == "__main__":
    args = get_parameters()
    
    # === Configuration ===
    # Set the benchmark source here: 'dapper' (CSV) or 'torch' (Grid Search .pt)
    BENCHMARK_SOURCE = 'dapper' 
    # =====================

    folder_name = os.path.join("save", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=folder_name, filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # modify test_batch_size
        if args.N == 100 and args.dataset == 'ks':
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.\n"
              f"Method: {args.v}")
    
        # --- Retrieve Optimal Parameters ---
        if BENCHMARK_SOURCE == 'dapper':
            sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(args, source='dapper')
            
            # Select appropriate array based on sigma_y
            if args.sigma_y == 1:
                dapper_array = sigma_y_1_array
            elif args.sigma_y == 0.7:
                dapper_array = sigma_y_0_7_array
            else:
                raise NotImplementedError(f"Dapper benchmark not configured for sigma_y={args.sigma_y}")
            
            # Unpack array [best_loc_rad, best_infl, rmse, rrmse_mean]
            if dapper_array.shape[0] > 0:
                loc_radius = dapper_array[0, 0]
                infl = dapper_array[0, 1]
                rmse_baseline = dapper_array[0, 2]
                rrmse_baseline = dapper_array[0, 3]
                print(f"RMSE from DAPPER: {rmse_baseline:.3f}.")
                print(f"RRMSE from DAPPER: {rrmse_baseline:.3f}.")
            else:
                print("Warning: No DAPPER benchmark found for this configuration.")
                loc_radius, infl = None, None

        elif BENCHMARK_SOURCE == 'torch':
            infl, loc_radius = get_benchmarks(args, source='torch')
            print(f"Loaded from Torch Grid Search -> Inflation: {infl}; Localization Radius: {loc_radius}")

        print(f"Using Inflation: {infl}; Localization Radius: {loc_radius}")
        
        # test
        t_start = time.time()
        loss_list_nn = []
        test_results = \
            test_ClassicFilter(test_loader, 
                            args, 
                            H_info=H_info, 
                            plot_figures=False, 
                            # fig_name=f'{folder_name}/test_{args.N}', 
                            infl=infl, 
                            loc_radius=loc_radius, 
                            # save_pdf=True
                            )
        print_test_results(test_results)
        t_inference = time.time() - t_start
        print(f"Inference finished with time {t_inference: .2f}s.")

        # save results
        tensor_dict = {
            'nn': {
                'mean_rmse': test_results.get('mean_rmse', float('nan')),
                'std_rmse': test_results.get('std_rmse', float('nan')),
                'mean_rrmse': test_results.get('mean_rrmse', float('nan')),
                'std_rrmse': test_results.get('std_rrmse', float('nan')),
                'mean_crps': test_results.get('mean_crps', float('nan')),
                'std_crps': test_results.get('std_crps', float('nan')),
                'mean_rcrps': test_results.get('mean_rcrps', float('nan')),
                'std_rcrps': test_results.get('std_rcrps', float('nan')),
                'valid_percent': test_results.get('no_nan_percent', 0.0),
                'cov_diff': test_results.get('mean_cov_diff', float('nan')),
                'rcov_diff': test_results.get('mean_rcov_diff', float('nan')),
                'pf_rmse': test_results.get('mean_pf_rmse', float('nan')),
                'pf_rrmse': test_results.get('mean_pf_rrmse', float('nan')),
                'loc_diff_dist': getattr(args, 'diff_dist', None),
                'mean_assim_time_w':test_results.get('assim_step_time_mean_weighted', float('nan')),
                'std_assim_time_w':test_results.get('assim_step_time_std_weighted', float('nan')),
                'mean_assim_time':test_results.get('assim_step_time_mean', float('nan')),
                'std_assim_time':test_results.get('assim_step_time_std', float('nan')),
            },
            'cp_load_path': getattr(args, 'cp_load_path', None),
            'sigma_y': getattr(args, 'sigma_y', None),
            'time': t_inference,
            'test_results': test_results,
        }
        
        torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))