import os
import torch
import torch.nn as nn
import time
import numpy as np

from config.cli import get_parameters
from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output
from train_test_utils import test_model_v2, set_models, print_test_results_v2

# Function to run grid search for EnKF/LETKF
# Input: args, test_loader, model_list, infl_candidates (list), loc_candidates (list)
# Output: best_results (dict), best_params (tuple), grid_history (list)
def run_grid_search(args, test_loader, model_list, infl_candidates, loc_candidates):
    if args.v not in ["EnKF", "LETKF"]:
        raise ValueError("Grid search is only implemented for EnKF and LETKF.")

    best_w2 = float('inf')
    best_results = None
    best_params = (None, None)
    grid_history = []
    
    print(f"Starting Grid Search for {args.v}...")
    print(f"Inflation candidates: {infl_candidates}")
    print(f"Localization candidates: {loc_candidates}\n")

    for infl in infl_candidates:
        for loc in loc_candidates:
            print(f"Testing [infl={infl}, loc={loc}]...")
            
            # Run test
            results = test_model_v2(
                test_loader, model_list, args, 
                plot_figures=False, fig_name=None, save_pdf=False, 
                infl=infl, loc_radius=loc
            )
            
            # Extract key metrics for history
            current_w2 = results.get('mean_w2_diff', float('inf'))
            current_metrics = {
                'params': {'infl': infl, 'loc': loc},
                'w2_diff': current_w2,
                'rmse': results.get('mean_rmse', float('nan')),
                'rrmse': results.get('mean_rrmse', float('nan')),
                'es1': results.get('mean_es1', float('nan')),
            }
            grid_history.append(current_metrics)

            print(f"  -> W2: {current_w2:.4f}")

            # Update best
            if current_w2 < best_w2:
                best_w2 = current_w2
                best_results = results
                best_params = (infl, loc)
    
    print(f"\nGrid Search Finished. Best W2: {best_w2:.4f} with infl={best_params[0]}, loc={best_params[1]}")
    return best_results, best_params, grid_history

if __name__ == "__main__":
    args = get_parameters()
    
    # Define grid search space here
    infl_list = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    loc_list = [1, 2, 3, 5, 7, 9]

    if args.v in ["EnKF", "LETKF"]:
        folder_name = os.path.join("save", f"grid_search_{args.dataset}_{args.sigma_y}_{args.v}")
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
    else:
        raise ValueError("This script is modified for EnKF/LETKF grid search only.")
    
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=folder_name, filename="grid_search_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        train_loader, test_loader = get_dataloader(args)
        
        print(f"Test on {args.test_traj_num} trajectories with length {args.test_steps}, ensemble size {args.N}.")
        
        # set model
        model_list = set_models(args)
        
        # Run Grid Search
        t_start = time.time()
        best_full_results, best_params, history = run_grid_search(args, test_loader, model_list, infl_list, loc_list)
        t_inference = time.time() - t_start
        
        print("\n=== Best Test Results ===")
        print_test_results_v2(best_full_results)
        
        # save results
        tensor_dict = {
            'best_params': {
                'infl': best_params[0],
                'loc_radius': best_params[1]
            },
            # Stats for the best run (replacing 'nn')
            'best_metrics': {
                'mean_rmse': best_full_results.get('mean_rmse', float('nan')),
                'std_rmse': best_full_results.get('std_rmse', float('nan')),
                'mean_rrmse': best_full_results.get('mean_rrmse', float('nan')),
                'std_rrmse': best_full_results.get('std_rrmse', float('nan')),
                'mean_w2_diff': best_full_results.get('mean_w2_diff', float('nan')),
                'std_w2_diff': best_full_results.get('std_w2_diff', float('nan')),
                'valid_percent': best_full_results.get('no_nan_percent', 0.0),
            },
            'grid_history': history, # Complete search logs
            'sigma_y': getattr(args, 'sigma_y', None),
            'time': t_inference,
            'test_results': best_full_results, # Full result object for the best run
        }
        
        torch.save(tensor_dict, os.path.join(folder_name, f"best_output_{args.N}.pt"))
