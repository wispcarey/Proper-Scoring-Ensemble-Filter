import os
import time

import pandas as pd
import torch

from config.cli import get_parameters
from utils import build_observation_operator, get_dataloader, redirect_output
from train_test_utils import test_ClassicFilter


def _tensor_range_to_csv_string(values: torch.Tensor) -> str:
    """Serialize a 1D tensor range into a stable, dedup-friendly string."""
    items = []
    for val in values.detach().cpu():
        if torch.isnan(val):
            items.append("None")
        else:
            items.append(f"{float(val.item()):.6f}")
    return ";".join(items)


def _append_unique_summary_row(csv_path: str, row: dict, unique_cols: list) -> None:
    """Append row into csv_path if no existing row shares all unique_cols values."""
    row_df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        row_df.to_csv(csv_path, index=False)
        print(f"[Summary CSV] Created and wrote: {csv_path}")
        return

    existing_df = pd.read_csv(csv_path)
    if existing_df.empty:
        row_df.to_csv(csv_path, index=False)
        print(f"[Summary CSV] Existing file was empty; wrote: {csv_path}")
        return

    for col in unique_cols:
        if col not in existing_df.columns:
            existing_df[col] = ""

    duplicate_mask = pd.Series(True, index=existing_df.index)
    for col in unique_cols:
        duplicate_mask &= existing_df[col].astype(str) == str(row[col])

    if duplicate_mask.any():
        print(f"[Summary CSV] Duplicate entry detected. Skip append: {csv_path}")
        return

    merged_df = pd.concat([existing_df, row_df], ignore_index=True)
    merged_df.to_csv(csv_path, index=False)
    print(f"[Summary CSV] Appended new row: {csv_path}")


if __name__ == "__main__":
    args = get_parameters()

    folder_name = os.path.join("save", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    # Redirect output
    with redirect_output(
        save_output=not args.normal_output,
        save_folder=folder_name,
        filename=f"grid_search_output_{args.N}.txt",
    ):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        device = args.device if isinstance(args.device, torch.device) else torch.device(args.device)

        # H_info
        H_info = build_observation_operator(args)

        # Modify test_batch_size if necessary
        if args.N == 100 and args.dataset == "ks":
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)

        # --- Grid Search Setup ---
        infl_range = torch.linspace(1.0, 1.15, 11, device=device, dtype=torch.float32)

        # Only EnKF/LETKF use localization search, and only for lorenz96/ks.
        needs_localization = (args.v in {"EnKF", "LETKF"}) and (args.dataset in {"lorenz96", "ks"})
        if needs_localization:
            if args.dataset == "lorenz96":
                loc_radius_range = torch.cat(
                    (
                        torch.tensor([0.001], device=device, dtype=torch.float32),
                        torch.arange(1, 11, device=device, dtype=torch.float32),
                    )
                )
            elif args.dataset == "ks":
                loc_radius_range = torch.cat(
                    (
                        torch.tensor([0.01], device=device, dtype=torch.float32),
                        torch.arange(1, 20, 2, device=device, dtype=torch.float32),
                    )
                )
            else:
                loc_radius_range = torch.tensor([float("nan")], device=device, dtype=torch.float32)
        else:
            loc_radius_range = torch.tensor([float("nan")], device=device, dtype=torch.float32)

        infl_print = [float(x) for x in infl_range.detach().cpu().tolist()]
        loc_radius_print = [None if torch.isnan(x) else float(x.item()) for x in loc_radius_range.detach().cpu()]

        print("--- Starting Grid Search ---")
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}.")
        print(f"Observation noise sigma_y = {args.sigma_y}.")
        print(f"Inflation (infl) search range: {infl_print}")
        print(f"Localization Radius (loc_radius) search range: {loc_radius_print}\\n")

        # --- Initialize storage for grid search results ---
        num_infl = int(infl_range.numel())
        num_loc = int(loc_radius_range.numel())

        # Define metrics to track during the search
        metrics_to_track = ["mean_pf_crps", "mean_rmse", "mean_rrmse", "mean_rmv", "valid_percent"]
        results_grid = {
            metric: torch.full((num_infl, num_loc), float("nan"), device=device, dtype=torch.float32)
            for metric in metrics_to_track
        }

        t_start = time.time()
        # --- Perform Grid Search ---
        for i in range(num_infl):
            infl = infl_range[i]
            infl_val = float(infl.item())
            for j in range(num_loc):
                loc_radius_raw = loc_radius_range[j]
                loc_radius_val = None if torch.isnan(loc_radius_raw) else float(loc_radius_raw.item())
                print(
                    f"\\n[Grid Search {i * num_loc + j + 1}/{num_infl * num_loc}] "
                    f"Testing infl={infl_val:.4f}, loc_radius={loc_radius_val}"
                )

                # Test the classic filter with the current set of parameters
                test_results = test_ClassicFilter(
                    test_loader,
                    args,
                    H_info=H_info,
                    plot_figures=False,
                    infl=infl_val,
                    loc_radius=loc_radius_val,
                )

                # Store results for each metric in its respective grid
                for metric in metrics_to_track:
                    metric_val = test_results.get(metric, float("nan"))
                    metric_tensor = torch.as_tensor(metric_val, device=device, dtype=torch.float32).reshape(-1)
                    if metric_tensor.numel() == 0:
                        results_grid[metric][i, j] = float("nan")
                    else:
                        results_grid[metric][i, j] = metric_tensor[0]

                crps_ij = results_grid["mean_pf_crps"][i, j].item()
                rmse_ij = results_grid["mean_rmse"][i, j].item()
                print(f" > Results: mean_pf_crps={crps_ij:.4f}, mean_rmse={rmse_ij:.4f}")

        t_grid_search = time.time() - t_start
        print(f"Grid search finished with time {t_grid_search: .2f}s.")

        # --- Find and Print Optimal Parameters based on Mean PF-CRPS ---
        crps_grid = results_grid["mean_pf_crps"]
        finite_mask = torch.isfinite(crps_grid)

        if not finite_mask.any():
            print("\\n--- Grid Search Failed ---")
            print("Could not find any valid results. Please check filter stability and parameter ranges.")
            best_params_dict = {"infl": None, "loc_radius": None, "mean_pf_crps": float("nan")}
            best_infl = None
            best_loc_radius = None
        else:
            inf_grid = torch.full_like(crps_grid, float("inf"))
            best_flat_idx = int(torch.argmin(torch.where(finite_mask, crps_grid, inf_grid)).item())
            min_idx = divmod(best_flat_idx, num_loc)

            best_infl = float(infl_range[min_idx[0]].item())
            best_loc_raw = loc_radius_range[min_idx[1]]
            best_loc_radius = None if torch.isnan(best_loc_raw) else float(best_loc_raw.item())
            best_crps_val = float(crps_grid[min_idx].item())

            print("\\n--- Grid Search Complete ---")
            print(f"Best Mean PF-CRPS: {best_crps_val:.4f}")
            print(f"Optimal Inflation: {best_infl:.4f}")
            print(f"Optimal Localization Radius: {best_loc_radius}")

            print("\\n--- Metrics for Optimal Parameters ---")
            for metric, grid in results_grid.items():
                print(f"  {metric}: {float(grid[min_idx].item()):.4f}")

            best_params_dict = {
                "infl": best_infl,
                "loc_radius": best_loc_radius,
                "mean_pf_crps": best_crps_val,
            }

            # Optional: Run one last time with best parameters to generate plots
            print("\\nRunning final test with optimal parameters to generate plots...")
            test_ClassicFilter(
                test_loader,
                args,
                H_info=H_info,
                plot_figures=True,
                fig_name=f"{folder_name}/optimal_test_{args.N}",
                infl=best_infl,
                loc_radius=best_loc_radius,
                save_pdf=True,
            )

        # --- Save Grid Search Results ---
        grid_search_output = {
            "infl_range": infl_range.detach().cpu(),
            "loc_radius_range": loc_radius_range.detach().cpu(),
            "results_grid": {k: v.detach().cpu() for k, v in results_grid.items()},
            "best_params": best_params_dict,
            "args": vars(args),
            "time": t_grid_search,
        }

        # Save results as a PyTorch tensor file
        torch.save(grid_search_output, os.path.join(folder_name, f"grid_search_results_{args.N}.pt"))

        # Save results to a human-readable CSV file
        results_list = []
        for i in range(num_infl):
            for j in range(num_loc):
                infl_ij = float(infl_range[i].item())
                loc_raw = loc_radius_range[j]
                loc_ij = None if torch.isnan(loc_raw) else float(loc_raw.item())
                row = {"inflation": infl_ij, "loc_radius": loc_ij}
                for metric, grid in results_grid.items():
                    row[metric] = float(grid[i, j].item())
                results_list.append(row)

        df_results = pd.DataFrame(results_list)
        df_results.sort_values(by="mean_pf_crps", inplace=True, na_position="last")
        per_run_csv = os.path.join(folder_name, f"grid_search_results_{args.N}.csv")
        df_results.to_csv(per_run_csv, index=False)

        summary_csv = os.path.join("save", "grid_search_benchmark_summary.csv")
        summary_row = {
            "dataset": args.dataset,
            "sigma_y": args.sigma_y,
            "method": args.v,
            "N": args.N,
            "seed": args.seed,
            "obs_fn": getattr(args, "obs_fn", "identity"),
            "infl_range": _tensor_range_to_csv_string(infl_range),
            "loc_radius_range": _tensor_range_to_csv_string(loc_radius_range),
            "best_infl": best_params_dict["infl"],
            "best_loc_radius": best_params_dict["loc_radius"],
            "best_mean_pf_crps": best_params_dict["mean_pf_crps"],
            "time_sec": t_grid_search,
            "per_run_csv": per_run_csv,
            "save_folder": folder_name,
        }
        _append_unique_summary_row(
            csv_path=summary_csv,
            row=summary_row,
            unique_cols=[
                "dataset",
                "sigma_y",
                "method",
                "N",
                "seed",
                "obs_fn",
                "infl_range",
                "loc_radius_range",
            ],
        )

        print(f"\\nGrid search results saved to folder: {folder_name}")
