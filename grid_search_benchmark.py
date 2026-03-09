import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout

# Force a non-GUI matplotlib backend for batch grid-search runs.
# This avoids Qt/Wayland crashes during the final plotting pass on headless CPUs.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch

from config.benchmark_gridsearch_info import get_benchmark_gridsearch_config
from config.cli import get_parameters
from train_test_utils import print_test_results, test_ClassicFilter
from utils import build_observation_operator, get_dataloader, redirect_output


SAVE_ROOT = os.path.join("save", "torch_grid_search")
CPU_WORKER_STATE = {}
LEGACY_ALIAS_METRIC_KEYS = {
    "mean_rrmse",
    "std_rrmse",
    "mean_res1",
    "std_res1",
    "mean_pf_crps",
    "std_pf_crps",
    "mean_pf_crps_state_avg",
    "std_pf_crps_state_avg",
    "mean_pf_crps_pca_avg",
    "std_pf_crps_pca_avg",
    "mean_pf_crps_state_dim1",
    "std_pf_crps_state_dim1",
    "mean_pf_crps_state_dim2",
    "std_pf_crps_state_dim2",
    "mean_pf_crps_state_dim3",
    "std_pf_crps_state_dim3",
    "mean_pf_crps_pca_dim1",
    "std_pf_crps_pca_dim1",
    "mean_pf_crps_pca_dim2",
    "std_pf_crps_pca_dim2",
    "mean_pf_crps_pca_dim3",
    "std_pf_crps_pca_dim3",
    "mean_ser",
    "std_ser",
    "mean_ser_minus_1",
    "std_ser_minus_1",
    "valid_percent",
}
NON_SCALAR_RESULT_KEYS = {
    "rank_hist_counts",
    "rank_hist_probs",
}
NON_CSV_NUMERIC_RESULT_KEYS = {
    "valid_traj_count",
    "total_traj_count",
    "rank_total_samples",
    "rank_num_projections",
}
PREFERRED_METRIC_ORDER = [
    "mean_rmse",
    "std_rmse",
    "mean_rrmse_step",
    "std_rrmse_step",
    "mean_rrmse_traj",
    "std_rrmse_traj",
    "mean_rmv",
    "std_rmv",
    "mean_spread_error_ratio",
    "std_spread_error_ratio",
    "mean_spread_error_ratio_minus_1",
    "std_spread_error_ratio_minus_1",
    "mean_es1",
    "std_es1",
    "mean_res1_step",
    "std_res1_step",
    "mean_res1_traj",
    "std_res1_traj",
    "mean_pf_sed",
    "std_pf_sed",
    "mean_pf_sed_state_avg",
    "std_pf_sed_state_avg",
    "mean_pf_sed_pca_avg",
    "std_pf_sed_pca_avg",
    "mean_pf_sed_state_dim1",
    "std_pf_sed_state_dim1",
    "mean_pf_sed_state_dim2",
    "std_pf_sed_state_dim2",
    "mean_pf_sed_state_dim3",
    "std_pf_sed_state_dim3",
    "mean_pf_sed_pca_dim1",
    "std_pf_sed_pca_dim1",
    "mean_pf_sed_pca_dim2",
    "std_pf_sed_pca_dim2",
    "mean_pf_sed_pca_dim3",
    "std_pf_sed_pca_dim3",
    "mean_pf_cov_diff",
    "std_pf_cov_diff",
    "mean_pf_rcov_diff",
    "std_pf_rcov_diff",
    "mean_pf_rmse",
    "std_pf_rmse",
    "mean_pf_rrmse",
    "std_pf_rrmse",
    "mean_snr_var",
    "std_snr_var",
    "no_nan_percent",
    "rank_freq_var",
    "assim_step_time_mean",
    "assim_step_time_std",
    "assim_step_time_mean_weighted",
    "assim_step_time_std_weighted",
]
CSV_SETTING_COLUMNS = [
    "dataset",
    "method",
    "N",
    "sigma_y",
    "obs_fn",
    "obs_inds",
    "obs_dim",
    "obs_fn_out_dim",
    "obs_fn_seed",
    "obs_custom_fn_path",
    "adaptive_sigma_y",
    "no_localization",
    "localization_fn",
    "pf_verification",
    "pf_verification_seed",
    "pf_N",
    "search_metric",
    "grid_search_num_seeds",
    "grid_search_seed_values",
    "test_steps",
    "test_traj_num",
    "seed",
    "test_random_seed",
    "seed_obs",
    "setting_id",
    "results_pt_path",
    "log_path",
    "best_infl",
    "best_loc_radius",
    "best_search_metric",
]


def _safe_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
        return int(stripped)
    return int(value)


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_scalar_metric_value(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.floating, np.integer)):
        return True
    return bool(torch.is_tensor(value) and value.numel() == 1)


def _format_numeric_token(value):
    if value is None:
        return "none"
    value = _safe_float(value)
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _safe_filename_token(value):
    text = str(value)
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        elif ch == ".":
            safe_chars.append("p")
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe or "value"


def _serialize_obs_inds(obs_inds):
    if obs_inds is None:
        return ""
    obs_inds_tensor = torch.as_tensor(obs_inds, dtype=torch.long).reshape(-1)
    return ";".join(str(int(v.item())) for v in obs_inds_tensor)


def _ordered_metric_keys(metric_keys):
    preferred = [key for key in PREFERRED_METRIC_ORDER if key in metric_keys]
    remaining = sorted(key for key in metric_keys if key not in preferred)
    return preferred + remaining


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_search_metric_info(args):
    if getattr(args, "pf_verification", False):
        return "mean_pf_sed", "PF-SED"
    return "mean_res1_traj", "RES1-traj"


def _build_grid_seed_values(args):
    base_seed = _safe_int_or_none(getattr(args, "seed", None))
    if base_seed is None:
        base_seed = 0
    return [base_seed + idx for idx in range(int(args.grid_search_num_seeds))]


def _is_cpu_device(device):
    if isinstance(device, torch.device):
        return device.type == "cpu"
    return str(device) == "cpu"


def _adjust_eval_args(args):
    args_copy = copy.deepcopy(args)
    if args_copy.N == 100 and str(args_copy.dataset).lower() == "ks":
        args_copy.test_batch_size = args_copy.test_batch_size // 2
    if _is_cpu_device(args_copy.device):
        args_copy.device = torch.device("cpu")
        args_copy.num_loader_workers = 0
    return args_copy


def _get_mp_context():
    available_methods = mp.get_all_start_methods()
    if "fork" in available_methods:
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _build_setting_signature(args, localization_fn, search_metric_label, effective_no_localization):
    return {
        "dataset": str(args.dataset),
        "method": str(args.v),
        "N": int(args.N),
        "sigma_y": None if args.sigma_y is None else float(args.sigma_y),
        "obs_fn": str(getattr(args, "obs_fn", "identity")),
        "obs_inds": [] if getattr(args, "obs_inds", None) is None else [
            int(v) for v in torch.as_tensor(args.obs_inds, dtype=torch.long).reshape(-1).tolist()
        ],
        "obs_dim": None if getattr(args, "obs_dim", None) is None else int(args.obs_dim),
        "obs_fn_out_dim": None
        if getattr(args, "obs_fn_out_dim", None) is None
        else int(args.obs_fn_out_dim),
        "obs_fn_seed": _safe_int_or_none(getattr(args, "obs_fn_seed", None)),
        "obs_custom_fn_path": getattr(args, "obs_custom_fn_path", None),
        "adaptive_sigma_y": bool(getattr(args, "adaptive_sigma_y", False)),
        "no_localization": bool(effective_no_localization),
        "localization_fn": localization_fn,
        "pf_verification": bool(getattr(args, "pf_verification", False)),
        "pf_verification_seed": _safe_int_or_none(getattr(args, "pf_verification_seed", None)),
        "pf_N": None if getattr(args, "pf_N", None) is None else int(args.pf_N),
        "search_metric": search_metric_label,
        "grid_search_num_seeds": int(args.grid_search_num_seeds),
        "test_steps": None if getattr(args, "test_steps", None) is None else int(args.test_steps),
        "test_traj_num": None if getattr(args, "test_traj_num", None) is None else int(args.test_traj_num),
        "seed": _safe_int_or_none(getattr(args, "seed", None)),
        "test_random_seed": _safe_int_or_none(getattr(args, "test_random_seed", None)),
        "seed_obs": _safe_int_or_none(getattr(args, "seed_obs", None)),
    }


def _make_setting_id(signature):
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(signature_json.encode("utf-8")).hexdigest()[:12]


def _build_run_paths(signature, setting_id):
    dataset_dir = os.path.join(SAVE_ROOT, str(signature["dataset"]))
    os.makedirs(dataset_dir, exist_ok=True)
    run_stem = (
        f"{_safe_filename_token(signature['method'])}"
        f"_N{signature['N']}"
        f"_sigma{_format_numeric_token(signature['sigma_y'])}"
        f"_obs_fn_{_safe_filename_token(signature['obs_fn'])}"
    )
    return {
        "dataset_dir": dataset_dir,
        "log_filename": f"{run_stem}.log",
        "log_path": os.path.join(dataset_dir, f"{run_stem}.log"),
        "results_pt_path": os.path.join(dataset_dir, f"{run_stem}.pt"),
        "plot_prefix": os.path.join(dataset_dir, f"{run_stem}_optimal"),
    }


def _extract_scalar_metrics(results, include_non_csv_metrics=False):
    scalar_metrics = {}
    for key, value in results.items():
        if key in LEGACY_ALIAS_METRIC_KEYS or key in NON_SCALAR_RESULT_KEYS:
            continue
        if not include_non_csv_metrics and key in NON_CSV_NUMERIC_RESULT_KEYS:
            continue
        if _is_scalar_metric_value(value):
            scalar_metrics[key] = _safe_float(value)
    return scalar_metrics


def _aggregate_seed_results(seed_run_entries):
    valid_entries = [entry for entry in seed_run_entries if entry.get("metrics") is not None]
    if not valid_entries:
        return {
            "aggregated_metrics": {},
            "csv_metrics": {},
            "metric_keys": [],
            "num_valid_seed_runs": 0,
        }

    scalar_metric_keys = set()
    csv_metric_keys = set()
    for entry in valid_entries:
        scalar_metric_keys.update(_extract_scalar_metrics(entry["metrics"], include_non_csv_metrics=True).keys())
        csv_metric_keys.update(_extract_scalar_metrics(entry["metrics"], include_non_csv_metrics=False).keys())

    aggregated_metrics = {}
    for key in scalar_metric_keys:
        values = []
        for entry in valid_entries:
            metric_dict = _extract_scalar_metrics(entry["metrics"], include_non_csv_metrics=True)
            if key in metric_dict:
                values.append(metric_dict[key])
        finite_values = [val for val in values if val is not None and math.isfinite(val)]
        if len(finite_values) == 0:
            aggregated_metrics[key] = float("nan")
        else:
            aggregated_metrics[key] = float(np.mean(finite_values))

    csv_metrics = {
        key: aggregated_metrics[key]
        for key in csv_metric_keys
        if key in aggregated_metrics
    }
    return {
        "aggregated_metrics": aggregated_metrics,
        "csv_metrics": csv_metrics,
        "metric_keys": _ordered_metric_keys(csv_metric_keys),
        "num_valid_seed_runs": len(valid_entries),
    }


def _combo_sort_key(loc_radius):
    if loc_radius is None:
        return (1, 0.0)
    return (0, float(loc_radius))


def _make_combo_key(infl, loc_radius):
    return (float(infl), None if loc_radius is None else float(loc_radius))


def _evaluate_single_seed_job_with_loader(test_loader, args, H_info, infl, loc_radius, seed_index, grid_seed):
    job_start = time.time()
    try:
        _seed_everything(grid_seed)
        metrics = test_ClassicFilter(
            test_loader,
            args,
            H_info=H_info,
            plot_figures=False,
            infl=infl,
            loc_radius=loc_radius,
        )
        error = None
    except Exception as exc:
        metrics = None
        error = str(exc)

    return {
        "infl": float(infl),
        "loc_radius": None if loc_radius is None else float(loc_radius),
        "seed_index": int(seed_index),
        "grid_seed": int(grid_seed),
        "metrics": metrics,
        "error": error,
        "job_time_sec": time.time() - job_start,
    }


def _aggregate_jobs_to_combo_results(job_results, infl_values, loc_values):
    grouped_entries = {}
    for job_result in job_results:
        combo_key = _make_combo_key(job_result["infl"], job_result["loc_radius"])
        grouped_entries.setdefault(combo_key, []).append(
            {
                "seed_index": job_result["seed_index"],
                "grid_seed": job_result["grid_seed"],
                "metrics": job_result["metrics"],
                "error": job_result["error"],
                "job_time_sec": job_result["job_time_sec"],
            }
        )

    combo_results = []
    for infl in infl_values:
        for loc_radius in loc_values:
            combo_key = _make_combo_key(infl, loc_radius)
            seed_run_entries = sorted(
                grouped_entries.get(combo_key, []),
                key=lambda entry: (entry["seed_index"], entry["grid_seed"]),
            )
            aggregated = _aggregate_seed_results(seed_run_entries)
            combo_results.append(
                {
                    "infl": float(infl),
                    "loc_radius": None if loc_radius is None else float(loc_radius),
                    "seed_runs": seed_run_entries,
                    "aggregated_metrics": aggregated["aggregated_metrics"],
                    "csv_metrics": aggregated["csv_metrics"],
                    "metric_keys": aggregated["metric_keys"],
                    "num_valid_seed_runs": aggregated["num_valid_seed_runs"],
                    "combo_time_sec": float(sum(entry.get("job_time_sec", 0.0) for entry in seed_run_entries)),
                }
            )
    return combo_results


def _init_cpu_gridsearch_worker(args_dict):
    worker_args = Namespace(**copy.deepcopy(args_dict))
    if not isinstance(worker_args.device, torch.device):
        worker_args.device = torch.device(str(worker_args.device))
    worker_args = _adjust_eval_args(worker_args)

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        CPU_WORKER_STATE["args"] = worker_args
        CPU_WORKER_STATE["H_info"] = build_observation_operator(worker_args)
        CPU_WORKER_STATE["test_loader"] = get_dataloader(worker_args, test_only=True)


def _cpu_worker_eval_single_job(infl, loc_radius, seed_index, grid_seed):
    worker_args = CPU_WORKER_STATE["args"]
    H_info = CPU_WORKER_STATE["H_info"]
    test_loader = CPU_WORKER_STATE["test_loader"]
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        return _evaluate_single_seed_job_with_loader(
            test_loader=test_loader,
            args=worker_args,
            H_info=H_info,
            infl=infl,
            loc_radius=loc_radius,
            seed_index=seed_index,
            grid_seed=grid_seed,
        )


def _run_grid_search_parallel(args, infl_values, loc_values, grid_seed_values):
    job_args = []
    for infl in infl_values:
        for loc_radius in loc_values:
            for seed_index, grid_seed in enumerate(grid_seed_values):
                job_args.append(
                    (
                        float(infl),
                        None if loc_radius is None else float(loc_radius),
                        int(seed_index),
                        int(grid_seed),
                    )
                )

    max_workers = min(int(args.grid_search_cpu_workers), max(1, len(job_args)))
    print(f"Running grid search on CPU with {max_workers} worker processes over {len(job_args)} jobs.")

    job_results = []
    args_dict = vars(args)
    mp_context = _get_mp_context()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_init_cpu_gridsearch_worker,
        initargs=(args_dict,),
    ) as executor:
        futures = {
            executor.submit(
                _cpu_worker_eval_single_job,
                infl,
                loc_radius,
                seed_index,
                grid_seed,
            ): (infl, loc_radius, seed_index, grid_seed)
            for infl, loc_radius, seed_index, grid_seed in job_args
        }
        for completed_idx, future in enumerate(as_completed(futures), start=1):
            infl, loc_radius, seed_index, grid_seed = futures[future]
            job_result = future.result()
            job_results.append(job_result)
            metrics = job_result.get("metrics") or {}
            res1_val = metrics.get("mean_res1_traj", float("nan"))
            pf_sed_val = metrics.get("mean_pf_sed", float("nan"))
            print(
                f"[Grid Search Job {completed_idx}/{len(job_args)}] "
                f"Completed infl={infl:.4f}, loc_radius={loc_radius}, seed={grid_seed} "
                f"RES1-traj={res1_val:.4f}, PF-SED={pf_sed_val:.4f}"
            )
    combo_results = _aggregate_jobs_to_combo_results(job_results, infl_values, loc_values)
    return combo_results, job_results


def _run_grid_search_sequential(test_loader, args, H_info, infl_values, loc_values, grid_seed_values):
    job_results = []
    total_jobs = len(infl_values) * len(loc_values) * len(grid_seed_values)
    job_index = 0
    for infl in infl_values:
        for loc_radius in loc_values:
            for seed_index, grid_seed in enumerate(grid_seed_values):
                job_index += 1
                print(
                    f"\n[Grid Search Job {job_index}/{total_jobs}] "
                    f"Testing infl={float(infl):.4f}, loc_radius={loc_radius}, seed={grid_seed}"
                )
                job_result = _evaluate_single_seed_job_with_loader(
                    test_loader=test_loader,
                    args=args,
                    H_info=H_info,
                    infl=float(infl),
                    loc_radius=None if loc_radius is None else float(loc_radius),
                    seed_index=seed_index,
                    grid_seed=grid_seed,
                )
                job_results.append(job_result)
                metrics = job_result.get("metrics") or {}
                res1_val = metrics.get("mean_res1_traj", float("nan"))
                pf_sed_val = metrics.get("mean_pf_sed", float("nan"))
                print(f" > Job RES1-traj={res1_val:.4f}, PF-SED={pf_sed_val:.4f}")
    combo_results = _aggregate_jobs_to_combo_results(job_results, infl_values, loc_values)
    return combo_results, job_results


def _select_best_combo(combo_results, search_metric_key):
    best_combo = None
    best_score = float("inf")
    for combo in combo_results:
        score = combo.get("aggregated_metrics", {}).get(search_metric_key, float("nan"))
        if score is None or not math.isfinite(score):
            continue
        if best_combo is None or score < best_score:
            best_combo = combo
            best_score = float(score)
    return best_combo, best_score


def _build_metric_grids(combo_results, infl_values, loc_values):
    infl_to_idx = {float(infl): idx for idx, infl in enumerate(infl_values)}
    loc_to_idx = {loc_radius: idx for idx, loc_radius in enumerate(loc_values)}
    metric_keys = set()
    for combo in combo_results:
        metric_keys.update(combo.get("aggregated_metrics", {}).keys())

    metric_grids = {
        key: torch.full((len(infl_values), len(loc_values)), float("nan"), dtype=torch.float32)
        for key in metric_keys
    }
    for combo in combo_results:
        i = infl_to_idx[float(combo["infl"])]
        j = loc_to_idx[combo["loc_radius"]]
        for key, value in combo.get("aggregated_metrics", {}).items():
            numeric_value = _safe_float(value)
            if numeric_value is not None:
                metric_grids[key][i, j] = float(numeric_value)
    return metric_grids


def _collect_csv_metric_keys(combo_results):
    metric_keys = set()
    for combo in combo_results:
        metric_keys.update(combo.get("csv_metrics", {}).keys())
    return _ordered_metric_keys(metric_keys)


def _build_csv_row(signature, setting_id, run_paths, best_combo, best_score, metric_keys, grid_seed_values):
    row = {
        "dataset": signature["dataset"],
        "method": signature["method"],
        "N": signature["N"],
        "sigma_y": signature["sigma_y"],
        "obs_fn": signature["obs_fn"],
        "obs_inds": _serialize_obs_inds(signature["obs_inds"]),
        "obs_dim": signature["obs_dim"],
        "obs_fn_out_dim": signature["obs_fn_out_dim"],
        "obs_fn_seed": signature["obs_fn_seed"],
        "obs_custom_fn_path": signature["obs_custom_fn_path"],
        "adaptive_sigma_y": signature["adaptive_sigma_y"],
        "no_localization": signature["no_localization"],
        "localization_fn": signature["localization_fn"],
        "pf_verification": signature["pf_verification"],
        "pf_verification_seed": signature["pf_verification_seed"],
        "pf_N": signature["pf_N"],
        "search_metric": signature["search_metric"],
        "grid_search_num_seeds": signature["grid_search_num_seeds"],
        "grid_search_seed_values": ";".join(str(seed) for seed in grid_seed_values),
        "test_steps": signature["test_steps"],
        "test_traj_num": signature["test_traj_num"],
        "seed": signature["seed"],
        "test_random_seed": signature["test_random_seed"],
        "seed_obs": signature["seed_obs"],
        "setting_id": setting_id,
        "results_pt_path": run_paths["results_pt_path"],
        "log_path": run_paths["log_path"],
        "best_infl": None if best_combo is None else best_combo["infl"],
        "best_loc_radius": None if best_combo is None else best_combo["loc_radius"],
        "best_search_metric": best_score,
    }

    if best_combo is None:
        for key in metric_keys:
            row[key] = float("nan")
        return row

    best_metrics = best_combo.get("csv_metrics", {})
    for key in metric_keys:
        row[key] = best_metrics.get(key, float("nan"))
    return row


def _upsert_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row_df = pd.DataFrame([row])
    ordered_columns = [col for col in CSV_SETTING_COLUMNS if col in row_df.columns] + [
        col for col in row_df.columns if col not in CSV_SETTING_COLUMNS
    ]
    row_df = row_df[ordered_columns]
    if not os.path.exists(csv_path):
        row_df.to_csv(csv_path, index=False)
        print(f"[CSV] Created {csv_path}")
        return

    existing_df = pd.read_csv(csv_path)
    if "setting_id" not in existing_df.columns:
        existing_df["setting_id"] = ""

    for column in row_df.columns:
        if column not in existing_df.columns:
            existing_df[column] = np.nan
    for column in existing_df.columns:
        if column not in row_df.columns:
            row_df[column] = np.nan

    existing_df = existing_df[row_df.columns]
    mask = existing_df["setting_id"].astype(str) == str(row["setting_id"])
    if mask.any():
        existing_df.loc[mask, :] = row_df.iloc[0].values
        output_df = existing_df
        print(f"[CSV] Updated existing row in {csv_path}")
    else:
        output_df = pd.concat([existing_df, row_df], ignore_index=True)
        print(f"[CSV] Appended new row to {csv_path}")

    output_df.to_csv(csv_path, index=False)


def _csv_has_setting_id(csv_path, setting_id):
    if not os.path.exists(csv_path):
        return False
    existing_df = pd.read_csv(csv_path)
    if "setting_id" not in existing_df.columns:
        return False
    return bool((existing_df["setting_id"].astype(str) == str(setting_id)).any())


if __name__ == "__main__":
    args = get_parameters()
    eval_args = _adjust_eval_args(args)
    search_metric_key, search_metric_label = _get_search_metric_info(eval_args)

    grid_cfg = get_benchmark_gridsearch_config(
        dataset=eval_args.dataset,
        method=eval_args.v,
        ensemble_size=eval_args.N,
        force_no_localization=bool(getattr(eval_args, "no_localization", False)),
    )
    infl_values = [float(val) for val in grid_cfg["infl_range"]]
    loc_values = [None if val is None else float(val) for val in grid_cfg["loc_radius_range"]]
    signature = _build_setting_signature(
        args=eval_args,
        localization_fn=grid_cfg.get("localization_fn"),
        search_metric_label=search_metric_label,
        effective_no_localization=(grid_cfg.get("localization_fn") is None),
    )
    setting_id = _make_setting_id(signature)
    run_paths = _build_run_paths(signature, setting_id)
    dataset_csv_path = os.path.join(SAVE_ROOT, f"{eval_args.dataset}.csv")
    grid_seed_values = _build_grid_seed_values(eval_args)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    setting_exists = _csv_has_setting_id(dataset_csv_path, setting_id)
    if setting_exists and not bool(getattr(eval_args, "grid_search_overwrite", False)):
        print(
            f"Skip grid search because the identical setting already exists in "
            f"{dataset_csv_path} (setting_id={setting_id})."
        )
        raise SystemExit(0)
    if setting_exists and bool(getattr(eval_args, "grid_search_overwrite", False)):
        print(
            f"Overwrite existing grid search result for setting_id={setting_id} in "
            f"{dataset_csv_path}."
        )

    with redirect_output(
        save_output=not eval_args.normal_output,
        save_folder=run_paths["dataset_dir"],
        filename=run_paths["log_filename"],
    ):
        if _safe_int_or_none(getattr(eval_args, "seed", None)) is not None:
            _seed_everything(_safe_int_or_none(eval_args.seed))

        print("--- Starting Benchmark Grid Search ---")
        print(f"Dataset: {eval_args.dataset}")
        print(f"Method: {eval_args.v}")
        print(f"Ensemble size N: {eval_args.N}")
        print(f"Observation noise sigma_y: {eval_args.sigma_y}")
        print(f"Observation function: {getattr(eval_args, 'obs_fn', 'identity')}")
        print(f"Search metric: {search_metric_label} ({search_metric_key})")
        print(f"Inflation search range: {infl_values}")
        print(f"Localization search range: {loc_values}")
        print(
            f"Search-range config N: requested={grid_cfg.get('requested_search_N')} "
            f"resolved={grid_cfg.get('resolved_search_N')}"
        )
        print(
            f"Grid-search random seeds (derived from base seed={_safe_int_or_none(getattr(eval_args, 'seed', None))}): "
            f"{grid_seed_values}"
        )
        print(f"Localization kernel: {grid_cfg.get('localization_fn')}")
        print(f"PF verification seed (unchanged): {getattr(eval_args, 'pf_verification_seed', None)}")

        H_info = build_observation_operator(eval_args)
        test_loader = get_dataloader(eval_args, test_only=True)

        t_start = time.time()
        use_cpu_parallel = (
            _is_cpu_device(eval_args.device)
            and int(eval_args.grid_search_cpu_workers) > 1
            and len(infl_values) * len(loc_values) > 1
        )
        if use_cpu_parallel:
            print(f"Execution mode: CPU parallel grid search with {int(eval_args.grid_search_cpu_workers)} requested workers.")
            combo_results, job_results = _run_grid_search_parallel(
                args=eval_args,
                infl_values=infl_values,
                loc_values=loc_values,
                grid_seed_values=grid_seed_values,
            )
        else:
            print("Execution mode: sequential grid search.")
            combo_results, job_results = _run_grid_search_sequential(
                test_loader=test_loader,
                args=eval_args,
                H_info=H_info,
                infl_values=infl_values,
                loc_values=loc_values,
                grid_seed_values=grid_seed_values,
            )
        total_grid_time = time.time() - t_start
        print(f"Grid search finished in {total_grid_time:.2f}s.")

        for combo in combo_results:
            metrics = combo.get("aggregated_metrics", {})
            res1_val = metrics.get("mean_res1_traj", float("nan"))
            pf_sed_val = metrics.get("mean_pf_sed", float("nan"))
            print(
                f"[Aggregated Combo] infl={combo['infl']:.4f}, loc_radius={combo['loc_radius']}, "
                f"valid_seed_runs={combo.get('num_valid_seed_runs', 0)}/{len(grid_seed_values)}, "
                f"RES1-traj={res1_val:.4f}, PF-SED={pf_sed_val:.4f}"
            )

        metric_keys = _collect_csv_metric_keys(combo_results)
        best_combo, best_score = _select_best_combo(combo_results, search_metric_key)
        if best_combo is None:
            print("\n--- Grid Search Failed ---")
            print("No finite search metric was found across the tested parameter combinations.")
        else:
            print("\n--- Grid Search Complete ---")
            print(f"Best search metric ({search_metric_label}): {best_score:.6f}")
            print(f"Best inflation: {best_combo['infl']}")
            print(f"Best localization radius: {best_combo['loc_radius']}")
            print("\n--- Aggregated Metrics For Best Parameters ---")
            print_test_results(best_combo["aggregated_metrics"])

        metric_grids = _build_metric_grids(combo_results, infl_values, loc_values)
        grid_search_output = {
            "signature": signature,
            "setting_id": setting_id,
            "search_metric_key": search_metric_key,
            "search_metric_label": search_metric_label,
            "infl_values": infl_values,
            "loc_radius_values": loc_values,
            "grid_seed_values": grid_seed_values,
            "job_results": job_results,
            "combo_results": combo_results,
            "metric_grids": metric_grids,
            "best_combo": best_combo,
            "best_score": best_score,
            "total_grid_time_sec": total_grid_time,
            "args": vars(eval_args),
        }
        torch.save(grid_search_output, run_paths["results_pt_path"])
        print(f"\nSaved grid-search tensor results to: {run_paths['results_pt_path']}")

        csv_row = _build_csv_row(
            signature=signature,
            setting_id=setting_id,
            run_paths=run_paths,
            best_combo=best_combo,
            best_score=best_score,
            metric_keys=metric_keys,
            grid_seed_values=grid_seed_values,
        )
        _upsert_csv_row(dataset_csv_path, csv_row)
        print(f"Updated dataset summary CSV: {dataset_csv_path}")

        if best_combo is not None:
            print("\nRunning final plotting pass with the best parameters...")
            try:
                _seed_everything(grid_seed_values[0])
                test_ClassicFilter(
                    test_loader,
                    eval_args,
                    H_info=H_info,
                    plot_figures=True,
                    fig_name=run_paths["plot_prefix"],
                    infl=best_combo["infl"],
                    loc_radius=best_combo["loc_radius"],
                    save_pdf=True,
                )
            except Exception as exc:
                print(f"[Plotting Warning] Final plotting pass failed after results were saved: {exc}")
