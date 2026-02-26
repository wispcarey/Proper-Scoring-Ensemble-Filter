import os
import time

import torch
import torch.nn as nn

from config.cli import get_parameters
from config.dataset_info import DATASET_INFO

from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import build_observation_operator, get_dataloader, redirect_output

from train_test_utils import generate_and_cache_pf_results, print_test_results


if __name__ == "__main__":
    args = get_parameters()
    dataset_key = str(getattr(args, "dataset", "")).lower()
    default_obs_fn = str(DATASET_INFO.get(dataset_key, {}).get("obs_fn", "identity") or "identity").lower()
    effective_obs_fn = str(getattr(args, "obs_fn", "default") or "default").lower()
    if effective_obs_fn == "default":
        effective_obs_fn = default_obs_fn
    obs_suffix = "" if effective_obs_fn == default_obs_fn else f"_{effective_obs_fn}"
    
    t = time.time()
    # redirect output
    with redirect_output(save_output=not args.normal_output, save_folder='save', filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = build_observation_operator(args)

        test_loader = get_dataloader(args, test_only=True)
        
        # print test information
        print(
            f"Test **Bootstrap Particle Filter** on {args.test_traj_num} trajectories "
            f"with the length {args.test_steps} and {args.pf_N} particles. "
            f"Observation noise sigma_y={args.sigma_y}."
        )
        print(
            f"[PF naming] obs_fn={effective_obs_fn}, default_obs_fn={default_obs_fn}, "
            f"file_suffix='{obs_suffix}', vis_dir='save/{args.dataset}_pf_vis{obs_suffix}'"
        )

        # test
        print("Test NN Results")
        test_results = generate_and_cache_pf_results(test_loader, args, H_info, check_disk=False, calculate_es1=False, save_figure=args.pf_save_figure)
        print_test_results(test_results)
        print(f"PF Time: {time.time() - t:.2f}s")
