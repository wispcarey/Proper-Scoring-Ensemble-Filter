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

def get_benchmarks(args):
    """
    This function processes benchmark data from a CSV file, splitting it by `sigma_y` values 1 and 0.7,
    and extracting specific columns for each `method`, then combining them into a 2*N*5 numpy array.
    
    Args:
        args: An object or namespace with a `dataset` attribute specifying the dataset name.
        
    Returns:
        result arrays for sigma_y=1 and sigma_y=0.7 (each as numpy arrays of selected columns).
    """
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
    df = pd.read_csv(file_path, usecols=['method', 'N', 'sigma_y', 'best_loc_rad','best_infl','rmse', 'rrmse_mean'])

    # if args.v == "LETKF" or (args.v.startswith('iEnKS') and args.dataset != 'lorenz63'):
    #     method = "LETKF"
    # elif args.v == "EnKF" or args.v.startswith('iEnKS'):
    #     method = "EnKF_PertObs"
    # elif args.v == "ESRF":
    #     method = "EnKF_Sqrt"
    # else:
    #     raise NotImplementedError
    method = "LETKF"
    method_data = df[(df['method'] == method) & (df['N'] == args.N)]
    
    # Filter rows where sigma_y == 1 and 0.7
    sigma_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    
    # Convert to numpy arrays
    sigma_y_1_array = sigma_y_1.to_numpy()
    sigma_y_0_7_array = sigma_y_0_7.to_numpy()

    return sigma_y_1_array, sigma_y_0_7_array

# def get_benchmarks(args):
#     folder_name = os.path.join("save", f"{args.dataset}_benchmarks", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
#     data_name = os.path.join(folder_name, f"grid_search_results_{args.N}.pt")
#     data = torch.load(data_name, weights_only=False)
#     # data['best_params']: 'infl': 1.0, 'loc_radius': None, 'mean_crps': 0.7652797698974609
#     return data['best_params']['infl'], data['best_params']['loc_radius'], data['best_params']['mean_crps']


def write_benchmark_row(args, loc_radius, infl, rmse, rrmse, rmse_std=None, rrmse_std=None):
    """
    Append or upsert a row into the benchmark CSV with method name f"{args.v}_loc".
    If a row with the same (method, N, sigma_y) exists, print a message and overwrite
    that row in place (and deduplicate if multiple exist). Otherwise, append a new row.
    """
    # Build file path consistently with get_benchmarks
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'

    # Create a base dataframe if the file does not exist
    if not os.path.isfile(file_path):
        # Define a minimal schema so all later writes have aligned columns
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

    # Prepare the candidate row using the fields we can supply
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

    # Ensure every existing CSV column exists in candidate; fill missing with NaN
    for col in df_full.columns:
        if col not in candidate:
            candidate[col] = np.nan

    # If candidate has new columns, add them into df_full
    extra_cols = [k for k in candidate.keys() if k not in df_full.columns]
    if extra_cols:
        for c in extra_cols:
            df_full[c] = np.nan

    # Upsert by (method, N, sigma_y)
    has_keys = all(k in df_full.columns for k in ['method', 'N', 'sigma_y'])
    if has_keys and len(df_full) > 0:
        # Build a boolean mask for the key
        mask = (
            (df_full['method'] == candidate['method']) &
            (df_full['N'] == candidate['N']) &
            (df_full['sigma_y'] == candidate['sigma_y'])
        )
        if mask.any():
            # Print a clear message and overwrite the first matching row
            idxs = np.flatnonzero(mask)
            idx = idxs[0]
            print(f"[write_benchmark_row] Existing row found for "
                  f"(method={candidate['method']}, N={candidate['N']}, sigma_y={candidate['sigma_y']}). Overwriting.")

            # Overwrite in place
            for k, v in candidate.items():
                df_full.at[idx, k] = v

            # If multiple duplicates exist, drop extras and keep only the first
            if len(idxs) > 1:
                dup_idx = idxs[1:]
                df_full = df_full.drop(index=dup_idx).reset_index(drop=True)
        else:
            # No existing row => append
            df_full = pd.concat([df_full, pd.DataFrame([candidate])], ignore_index=True)
    else:
        # No key columns or empty dataframe => append
        df_full = pd.concat([df_full, pd.DataFrame([candidate])], ignore_index=True)

    # Sort for readability (ignore errors if columns missing)
    try:
        df_full = df_full.sort_values(by=['method', 'N', 'sigma_y']).reset_index(drop=True)
    except Exception:
        pass

    # Persist to disk
    df_full.to_csv(file_path, index=False)



if __name__ == "__main__":
    args = get_parameters()
    
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
        
        # print test information
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.\n"
            f"Method: {args.v}")
    
        # get optimal parameters
        sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(args)
        if args.sigma_y == 1:
            dapper_array = sigma_y_1_array
        elif args.sigma_y == 0.7:
            dapper_array = sigma_y_0_7_array
        else:
            raise NotImplementedError
        print(dapper_array.shape)
        loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
        print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
        print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
        print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
        
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

        # >>> New: write/update a row back into the benchmark CSV for method "{args.v}_loc"
        # Use test_results for mean and std metrics, and use infl/loc_radius from the selected benchmark.
        # write_benchmark_row(
        #     args,
        #     loc_radius=loc_radius,
        #     infl=infl,
        #     rmse=test_results.get('mean_rmse', float('nan')),
        #     rrmse=test_results.get('mean_rrmse', float('nan')),
        #     rmse_std=test_results.get('std_rmse', float('nan')),
        #     rrmse_std=test_results.get('std_rrmse', float('nan')),
        # )
            
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
        
        # print(torch.mean((ens_tensor_enkf.mean(dim=2) - ens_tensor_nn.mean(dim=2))**2, dim=(1,2))[:100])
        
        torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))
