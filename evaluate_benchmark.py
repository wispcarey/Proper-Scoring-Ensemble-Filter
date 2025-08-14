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

# def get_benchmarks(args):
#     """
#     This function processes benchmark data from a CSV file, splitting it by `sigma_y` values 1 and 0.7,
#     and extracting specific columns for each `method`, then combining them into a 2*N*5 numpy array.
    
#     Args:
#         args: An object or namespace with a `dataset` attribute specifying the dataset name.
        
#     Returns:
#         result_dict: A dictionary where each key is a method, and the value is a 2*N*5 numpy array.
#     """
#     file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
#     df = pd.read_csv(file_path, usecols=['method', 'N', 'sigma_y', 'best_loc_rad','best_infl','rmse', 'rrmse_mean'])

#     if args.v == "LETKF" or (args.v.startswith('iEnKS') and args.dataset != 'lorenz63'):
#         method = "LETKF"
#     elif args.v == "EnKF" or args.v.startswith('iEnKS'):
#         method = "EnKF_PertObs"
#     elif args.v == "ESRF":
#         method = "EnKF_Sqrt"
#     else:
#         raise NotImplementedError
#     method_data = df[(df['method'] == method) & (df['N'] == args.N)]
    
#     # Filter rows where sigma_y == 1 and 0.7
#     sigma_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
#     sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    
#     # Convert to numpy arrays
#     sigma_y_1_array = sigma_y_1.to_numpy()
#     sigma_y_0_7_array = sigma_y_0_7.to_numpy()

#     return sigma_y_1_array, sigma_y_0_7_array

def get_benchmarks(args):
    folder_name = os.path.join("save", f"{args.dataset}_benchmarks", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
    data_name = os.path.join(folder_name, f"grid_search_results_{args.N}.pt")
    data = torch.load(data_name)
    # data['best_params']: 'infl': 1.0, 'loc_radius': None, 'mean_crps': 0.7652797698974609
    return data['best_params']['infl'], data['best_params']['loc_radius'], data['best_params']['mean_crps']

if __name__ == "__main__":
    args = get_parameters()
    
    folder_name = os.path.join("save", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=args.save_folder, filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # modify test_batch_size
        if args.N == 100 and args.dataset == 'ks':
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        # print test information
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.")
    
        # get optimal parameters
        # sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(args)
        # if args.sigma_y == 1:
        #     dapper_array = sigma_y_1_array
        # elif args.sigma_y == 0.7:
        #     dapper_array = sigma_y_0_7_array
        # else:
        #     raise NotImplementedError
        # print(dapper_array.shape)
        # loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
        # print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
        # print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
        # print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
        infl, loc_radius, mean_crps = get_benchmarks(args)
        print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
        print(f"Recorded CRPS from Grid Search: {mean_crps:.3f}.")
        
        # test
        print(f"Test {args.v} Results")
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
                'mean_rmv': test_results.get('mean_rmv', float('nan')),
                'std_rmv': test_results.get('std_rmv', float('nan')),  
                'mean_crps': test_results.get('mean_crps', float('nan')),
                'std_crps': test_results.get('std_crps', float('nan')),
                'mean_rcrps': test_results.get('mean_rcrps', float('nan')),
                'std_rcrps': test_results.get('std_rcrps', float('nan')),
                'valid_percent': test_results.get('no_nan_percent', 0.0),
                'cov_diff': test_results.get('mean_cov_diff', float('nan')),
                'pf_rmse': test_results.get('mean_pf_rmse', float('nan')),
                'loc_diff_dist': getattr(args, 'diff_dist', None),
            },
            'cp_load_path': getattr(args, 'cp_load_path', None),
            'sigma_y': getattr(args, 'sigma_y', None),
            'time': t_inference,
        }
        
        # print(torch.mean((ens_tensor_enkf.mean(dim=2) - ens_tensor_nn.mean(dim=2))**2, dim=(1,2))[:100])
        
        torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))


