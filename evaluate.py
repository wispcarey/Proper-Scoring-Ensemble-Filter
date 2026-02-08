import os

import torch
import torch.nn as nn
import time

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import build_observation_operator, get_dataloader, redirect_output

from train_test_utils import test_model, set_models, print_test_results


if __name__ == "__main__":
    args = get_parameters()
    
    if args.cp_load_path == "no":
        # raise ValueError("The parameter cp_load_path is invalid.")
        folder_name = os.path.join("save", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}_{args.device}")
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
    else:
        folder_name = os.path.dirname(args.cp_load_path)
    
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=folder_name, filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = build_observation_operator(args)

        # modify test_batch_size
        if args.N in [60, 100] and args.device =='cpu':
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        # print test information
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.")

        # set model
        model_list = set_models(args)
        model, infl_model, local_model, st_model1, st_model2 = model_list

        # optimizer
        optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)

        # load checkpoint
        if args.cp_load_path != "no":
            load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)

        # test
        print("Test NN Results")
        t_start = time.time()
        test_results = test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/test_{args.N}_0', save_pdf=False)
        print_test_results(test_results)
        t_inference = time.time() - t_start
        print(f"Inference finished with time {t_inference: .2f}s.")
        
        # loc_tensor= test_results['loc_tensor']
        # if args.no_localization:
        #     loc_mean = loc_tensor
        #     loc_std = loc_tensor
        # else:
        #     loc_mean = torch.mean(loc_tensor, dim=(0,1)) 
        #     loc_std = torch.std(loc_tensor, dim=(0,1)) 
        
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
        
        # Remove '_tuned' suffix if present
        if args.suffix.endswith('_tuned'):
            save_suffix = args.suffix[:-6]
        else:
            save_suffix = args.suffix
        if args.zero_infl:
            torch.save(tensor_dict, os.path.join(folder_name, f"output_records_zero_infl_{args.N}{save_suffix}.pt"))
        else:
            torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}{save_suffix}.pt"))

