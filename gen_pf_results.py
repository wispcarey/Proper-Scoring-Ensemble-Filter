import os
import time

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output

from train_test_utils import generate_and_cache_pf_results, print_test_results


if __name__ == "__main__":
    args = get_parameters()
    
    t = time.time()
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder='save', filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # modify test_batch_size
        if args.N == 100:
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        # print test information
        print(
            f"Test **Bootstrap Particle Filter** on {args.test_traj_num} trajectories "
            f"with the length {args.test_steps} and {args.pf_N} particles. "
            f"Observation noise sigma_y={args.sigma_y}."
        )

        # test
        print("Test NN Results")
        test_results = generate_and_cache_pf_results(test_loader, args, H_info, check_disk=False, calculate_crps=False)
        print_test_results(test_results)
        print(f"PF Time: {time.time() - t:.2f}s")


