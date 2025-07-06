import os
import time

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import get_dataloader, redirect_output

from train_test_utils import test_linear_sampling_error, print_test_results_v2


if __name__ == "__main__":
    args = get_parameters()
    
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder='save', filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        test_loader = get_dataloader(args, test_only=True)
        num_resamples = 100
        
        # print test information
        print(
            f"Test **Inherent Sampling Error** on {args.test_traj_num} trajectories "
            f"with length {args.test_steps} and ensemble size {args.N}. "
            f"Averaging over {num_resamples} resamples per step. "
            f"Observation noise sigma_y={args.sigma_y}."
        )

        # test
        print("Test NN Results")
        test_results = test_linear_sampling_error(test_loader, args, num_resamples)
        print_test_results_v2(test_results)
        
        torch.save(test_results, os.path.join("save", "benchmark", f"linear_inherent_unc_N{args.N}_resample{num_resamples}_len{args.test_steps}.pt"))


