import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output

from train_test_utils import test_ClassicFilter

def get_benchmarks(args):
    """
    This function processes benchmark data from a CSV file, splitting it by `sigma_y` values 1 and 0.7,
    and extracting specific columns for each `method`, then combining them into a 2*N*5 numpy array.
    
    Args:
        args: An object or namespace with a `dataset` attribute specifying the dataset name.
        
    Returns:
        result_dict: A dictionary where each key is a method, and the value is a 2*N*5 numpy array.
    """
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
    df = pd.read_csv(file_path, usecols=['method', 'N', 'sigma_y', 'best_loc_rad','best_infl','rmse', 'rrmse_mean'])

    method = "LETKF"
    method_data = df[(df['method'] == method) & (df['N'] == args.N)]
    
    # Filter rows where sigma_y == 1 and 0.7
    sigma_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    
    # Convert to numpy arrays
    sigma_y_1_array = sigma_y_1.to_numpy()
    sigma_y_0_7_array = sigma_y_0_7.to_numpy()

    return sigma_y_1_array, sigma_y_0_7_array

if __name__ == "__main__":
    args = get_parameters()
    
    folder_name = os.path.join("save",f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=args.save_folder, filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # modify test_batch_size
        if args.N == 100:
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        # print test information
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.")
    
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
        print(f"Test {args.v} Results")
        loss_list_nn = []
        mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn = \
            test_ClassicFilter(test_loader, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/test_{args.N}', infl=infl, loc_radius=loc_radius, save_pdf=True)
        print(f"RMSE: {mean_rmse_nn:.3f} ± {std_rmse_nn:.3f}")
        print(f"RRMSE: {mean_rrmse_nn:.3f} ± {std_rrmse_nn:.3f}")
        print(f"RMV: {mean_rmv_nn:.3f} ± {std_rmv_nn:.3f}")
        print(f"CRPS: {mean_crps_nn:.3f} ± {std_crps_nn:.3f}")
        print(f'No NAN Percentage: {no_nan_percent_nn * 100: .2f}%')
        
            
        # save results
        tensor_dict = {
            'nn':{
                'mean_rmse':mean_rmse_nn,
                'std_rmse':std_rmse_nn,
                'mean_rrmse':mean_rrmse_nn,
                'std_rrmse':std_rrmse_nn,
                'mean_rmv':mean_rmv_nn,
                'std_rmv':std_rmv_nn,
                'mean_crps':mean_crps_nn,
                'std_crps':std_crps_nn,
                'valid_percent':no_nan_percent_nn,
                'loc_diff_dist':args.diff_dist,
            },
            'cp_load_path': args.cp_load_path,
            'sigma_y': args.sigma_y,
        }
        
        # print(torch.mean((ens_tensor_enkf.mean(dim=2) - ens_tensor_nn.mean(dim=2))**2, dim=(1,2))[:100])
        
        if args.cp_load_path != "no":
            if args.zero_infl:
                torch.save(tensor_dict, os.path.join(folder_name, f"output_records_zero_infl_{args.N}.pt"))
            else:
                torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))


