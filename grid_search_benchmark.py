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

if __name__ == "__main__":
    args = get_parameters()
    
    folder_name = os.path.join("save", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    
    # Redirect output
    with redirect_output(save_output=not args.normal_output, save_folder=folder_name, filename=f"grid_search_output_{args.N}.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # Modify test_batch_size if necessary
        if args.N == 100 and args.dataset == 'ks':
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        # --- Grid Search Setup ---
        # Define search ranges for inflation and localization radius.
        # These can be adjusted for a finer or coarser search.
        infl_range = np.linspace(1.0, 1.15, 16)  
        
        # Determine if localization is needed based on the filter method
        needs_localization = args.v == "LETKF" or ((args.v.startswith('iEnKS') or args.v == "EnKF") and args.dataset != 'lorenz63' and args.dataset != 'rossler')
        if needs_localization:
            loc_radius_range = np.array([0.001, 1.0, 2.0, 3.0])
        else:
            loc_radius_range = [None] # For methods without localization like EnKF

        print("--- Starting Grid Search ---")
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}.")
        print(f"Observation noise sigma_y = {args.sigma_y}.")
        print(f"Inflation (infl) search range: {infl_range}")
        print(f"Localization Radius (loc_radius) search range: {loc_radius_range}\n")

        # --- Initialize storage for grid search results ---
        num_infl = len(infl_range)
        num_loc = len(loc_radius_range)
        
        # Define metrics to track during the search
        metrics_to_track = ['mean_crps', 'mean_rmse', 'mean_rrmse', 'mean_rmv', 'valid_percent']
        results_grid = {metric: np.full((num_infl, num_loc), np.nan) for metric in metrics_to_track}

        t_start = time.time()
        # --- Perform Grid Search ---
        for i, infl in enumerate(infl_range):
            for j, loc_radius in enumerate(loc_radius_range):
                print(f"\n[Grid Search {i*num_loc + j + 1}/{num_infl*num_loc}] Testing infl={infl:.4f}, loc_radius={loc_radius}")
                
                # Test the classic filter with the current set of parameters
                test_results = test_ClassicFilter(
                    test_loader, 
                    args, 
                    H_info=H_info, 
                    plot_figures=False, # Disable plotting for individual runs to speed up search
                    infl=infl, 
                    loc_radius=loc_radius, 
                )
                
                # Store results for each metric in its respective grid
                for metric in metrics_to_track:
                    results_grid[metric][i, j] = test_results.get(metric, np.nan)
                
                print(f" > Results: mean_crps={results_grid['mean_crps'][i, j]:.4f}, mean_rmse={results_grid['mean_rmse'][i, j]:.4f}")

        t_grid_search = time.time() - t_start
        print(f"Grid search finished with time {t_grid_search: .2f}s.")
        # --- Find and Print Optimal Parameters based on Mean CRPS ---
        best_crps_val = np.nanmin(results_grid['mean_crps'])
        
        if np.isnan(best_crps_val):
            print("\n--- Grid Search Failed ---")
            print("Could not find any valid results. Please check filter stability and parameter ranges.")
            best_params_dict = {'infl': np.nan, 'loc_radius': np.nan, 'mean_crps': np.nan}
        else:
            min_idx = np.unravel_index(np.nanargmin(results_grid['mean_crps']), results_grid['mean_crps'].shape)
            
            best_infl = infl_range[min_idx[0]]
            best_loc_radius = loc_radius_range[min_idx[1]]
            
            print("\n--- Grid Search Complete ---")
            print(f"Best Mean CRPS: {best_crps_val:.4f}")
            print(f"Optimal Inflation: {best_infl:.4f}")
            print(f"Optimal Localization Radius: {best_loc_radius}")
            
            print("\n--- Metrics for Optimal Parameters ---")
            for metric, grid in results_grid.items():
                print(f"  {metric}: {grid[min_idx]:.4f}")

            best_params_dict = {
                'infl': best_infl,
                'loc_radius': best_loc_radius,
                'mean_crps': best_crps_val
            }
            
            # Optional: Run one last time with best parameters to generate plots
            print("\nRunning final test with optimal parameters to generate plots...")
            test_ClassicFilter(
                test_loader, args, H_info=H_info, plot_figures=True, 
                fig_name=f'{folder_name}/optimal_test_{args.N}', 
                infl=best_infl, loc_radius=best_loc_radius, save_pdf=True
            )


        # --- Save Grid Search Results ---
        grid_search_output = {
            'infl_range': infl_range,
            'loc_radius_range': loc_radius_range,
            'results_grid': results_grid,
            'best_params': best_params_dict,
            'args': vars(args),
            'time': t_grid_search,
        }
        
        # Save results as a PyTorch tensor file
        torch.save(grid_search_output, os.path.join(folder_name, f"grid_search_results_{args.N}.pt"))

        # Save results to a human-readable CSV file
        results_list = []
        for i, infl in enumerate(infl_range):
            for j, loc_radius in enumerate(loc_radius_range):
                row = {'inflation': infl, 'loc_radius': loc_radius}
                for metric, grid in results_grid.items():
                    row[metric] = grid[i, j]
                results_list.append(row)
        
        df_results = pd.DataFrame(results_list)
        df_results.sort_values(by='mean_crps', inplace=True)
        df_results.to_csv(os.path.join(folder_name, f"grid_search_results_{args.N}.csv"), index=False)

        print(f"\nGrid search results saved to folder: {folder_name}")