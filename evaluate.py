import os

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output

from train_test_utils import test_model, set_models, print_test_results


if __name__ == "__main__":
    args = get_parameters()
    
    if args.cp_load_path == "no":
        raise ValueError("The parameter cp_load_path is invalid.")
    folder_name = os.path.dirname(args.cp_load_path)
    
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
        test_results = test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/ft_test_{args.N}_0', save_pdf=False)
        print_test_results(test_results)
        
        loc_tensor= test_results['loc_tensor']
        if args.no_localization:
            loc_mean = loc_tensor
            loc_std = loc_tensor
        else:
            loc_mean = torch.mean(loc_tensor, dim=(0,1)) 
            loc_std = torch.std(loc_tensor, dim=(0,1)) 
        
        # save results
        tensor_dict = {
            'nn':{
                'mean_rmse':test_results['mean_rmse'],
                'std_rmse':test_results['std_rmse'],
                'mean_rrmse':test_results['mean_rrmse'],
                'std_rrmse':test_results['std_rrmse'],
                'mean_rmv':test_results['mean_rmv'],
                'std_rmv':test_results['mean_rmv'],
                'mean_crps':test_results['mean_crps'],
                'std_crps':test_results['std_crps'],
                'mean_rcrps':test_results['mean_rcrps'],
                'std_rcrps':test_results['std_rcrps'],
                'valid_percent':test_results['no_nan_percent'],
                'cov_diff':test_results['mean_cov_diff'],
                'pf_rmse':test_results['mean_pf_rmse'],
                'loc_mean':loc_mean,
                'loc_std':loc_std,
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


