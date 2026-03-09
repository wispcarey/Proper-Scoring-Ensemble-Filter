import os
import math
import torch
import pandas as pd
import numpy as np
import time

from config.benchmark_gridsearch_info import get_benchmark_gridsearch_config
from config.cli import get_parameters
from grid_search_benchmark import (
    _build_setting_signature,
    _get_search_metric_info,
    _make_setting_id,
    _serialize_obs_inds,
)
from utils import build_observation_operator, get_dataloader, redirect_output, should_redirect_output
from train_test_utils import test_ClassicFilter, print_test_results


def _safe_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
        return int(stripped)
    return int(value)


TORCH_GRID_LOOKUP_COLUMNS = [
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
    "test_steps",
    "test_traj_num",
    "seed",
    "test_random_seed",
    "seed_obs",
]
TORCH_GRID_BOOL_COLUMNS = {
    "adaptive_sigma_y",
    "no_localization",
    "pf_verification",
}
TORCH_GRID_INT_COLUMNS = {
    "N",
    "obs_dim",
    "obs_fn_out_dim",
    "obs_fn_seed",
    "pf_verification_seed",
    "pf_N",
    "grid_search_num_seeds",
    "test_steps",
    "test_traj_num",
    "seed",
    "test_random_seed",
    "seed_obs",
}
TORCH_GRID_FLOAT_COLUMNS = {"sigma_y"}


def _build_torch_grid_signature(args):
    grid_cfg = get_benchmark_gridsearch_config(
        dataset=args.dataset,
        method=args.v,
        ensemble_size=args.N,
        force_no_localization=bool(getattr(args, "no_localization", False)),
    )
    _, search_metric_label = _get_search_metric_info(args)
    return _build_setting_signature(
        args=args,
        localization_fn=grid_cfg.get("localization_fn"),
        search_metric_label=search_metric_label,
        effective_no_localization=(grid_cfg.get("localization_fn") is None),
    )


def _csv_cell_is_missing(value):
    if value is None:
        return True
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return True
    return False


def _parse_csv_bool(value):
    if _csv_cell_is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value in torch-grid CSV: {value}")


def _normalize_lookup_value(column, value):
    if column == "obs_inds":
        return _serialize_obs_inds(value)
    if column in TORCH_GRID_BOOL_COLUMNS:
        return None if value is None else bool(value)
    if column in TORCH_GRID_INT_COLUMNS:
        return _safe_int_or_none(value)
    if column in TORCH_GRID_FLOAT_COLUMNS:
        return None if value is None else float(value)
    return None if value is None else str(value)


def _row_matches_lookup_signature(row, lookup_values):
    for column, expected in lookup_values.items():
        if column not in row.index:
            return False
        actual = row[column]
        if column == "obs_inds":
            actual_norm = "" if _csv_cell_is_missing(actual) else str(actual).strip()
            if actual_norm != expected:
                return False
            continue
        if column in TORCH_GRID_BOOL_COLUMNS:
            if _parse_csv_bool(actual) != expected:
                return False
            continue
        if column in TORCH_GRID_INT_COLUMNS:
            actual_int = None if _csv_cell_is_missing(actual) else _safe_int_or_none(actual)
            if actual_int != expected:
                return False
            continue
        if column in TORCH_GRID_FLOAT_COLUMNS:
            actual_float = None if _csv_cell_is_missing(actual) else float(actual)
            if actual_float is None or expected is None:
                if actual_float != expected:
                    return False
            elif not math.isclose(actual_float, expected, rel_tol=0.0, abs_tol=1e-12):
                return False
            continue
        actual_text = None if _csv_cell_is_missing(actual) else str(actual)
        if actual_text != expected:
            return False
    return True


def _find_matching_torch_grid_row(df, signature, setting_id):
    missing_columns = [column for column in TORCH_GRID_LOOKUP_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Torch-grid CSV is missing required setting columns: "
            + ", ".join(missing_columns)
        )

    lookup_values = {
        column: _normalize_lookup_value(column, signature.get(column))
        for column in TORCH_GRID_LOOKUP_COLUMNS
    }
    mask = df.apply(lambda row: _row_matches_lookup_signature(row, lookup_values), axis=1)
    matches = df.loc[mask].copy()
    if matches.empty:
        if "setting_id" in df.columns:
            setting_id_matches = df[df["setting_id"].astype(str) == str(setting_id)].copy()
            if not setting_id_matches.empty:
                return setting_id_matches.iloc[0], "setting_id"
        lookup_summary = ", ".join(
            f"{column}={lookup_values[column]!r}" for column in TORCH_GRID_LOOKUP_COLUMNS
        )
        raise KeyError(
            f"No torch grid-search row matched the current setting in CSV. "
            f"Expected {{{lookup_summary}}} with setting_id={setting_id}."
        )

    if "setting_id" in matches.columns:
        exact_matches = matches[matches["setting_id"].astype(str) == str(setting_id)]
        if not exact_matches.empty:
            return exact_matches.iloc[0], "settings+setting_id"

    return matches.iloc[0], "settings"

def get_benchmarks(args, source='dapper'):
    """
    Retrieve benchmark parameters and metrics.
    
    Args:
        args: Argument namespace.
        source (str): 'dapper' to read from CSV, 'torch' to read from grid search .pt file.
        
    Returns:
        If source is 'dapper': Returns (sigma_y_1_array, sigma_y_0_7_array)
        If source is 'torch': Returns (infl, loc_radius)
    """
    if source == 'dapper':
        # --- Original CSV Reading Logic ---
        file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
        df = pd.read_csv(file_path, usecols=['method', 'N', 'sigma_y', 'best_loc_rad','best_infl','rmse', 'rrmse_mean'])

        # Hardcoded method selection from original snippet (can be adjusted if needed)
        method = "LETKF" 
        method_data = df[(df['method'] == method) & (df['N'] == args.N)]
        
        # Filter rows for specific sigma_y
        sigma_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
        sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
        
        return sigma_y_1.to_numpy(), sigma_y_0_7.to_numpy()

    elif source == 'torch':
        dataset_csv_path = os.path.join("save", "torch_grid_search", f"{args.dataset}.csv")
        if not os.path.exists(dataset_csv_path):
            raise FileNotFoundError(f"Benchmark CSV not found: {dataset_csv_path}")

        signature = _build_torch_grid_signature(args)
        setting_id = _make_setting_id(signature)
        df = pd.read_csv(dataset_csv_path)
        row, match_source = _find_matching_torch_grid_row(df=df, signature=signature, setting_id=setting_id)
        infl = row.get("best_infl", np.nan)
        loc_radius = row.get("best_loc_radius", np.nan)

        print(
            f"Loading benchmarks from: {dataset_csv_path} "
            f"(match={match_source}, setting_id={setting_id}, sigma_y={signature['sigma_y']})"
        )
        return infl, loc_radius

    else:
        raise ValueError(f"Unknown benchmark source: {source}")


def write_benchmark_row(args, loc_radius, infl, rmse, rrmse, rmse_std=None, rrmse_std=None):
    """
    Append or upsert a row into the benchmark CSV.
    """
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'

    if not os.path.isfile(file_path):
        base_cols = [
            'method', 'N', 'sigma_y',
            'best_loc_rad', 'best_infl',
            'nan_exist',
            'rmse', 'rmse_dstd', 'rmse_std', 'rmse_std_dstd',
            'rmv_mean', 'rmv_mean_dstd', 'rmv_std', 'rmv_std_dstd',
            'rrmse_mean', 'rrmse_mean_dstd', 'rrmse_std', 'rrmse_std_dstd'
        ]
        df_full = pd.DataFrame(columns=base_cols)
    else:
        df_full = pd.read_csv(file_path)

    candidate = {
        'method': f'{args.v}_loc',
        'N': int(args.N),
        'sigma_y': float(args.sigma_y),
        'best_loc_rad': float(loc_radius) if loc_radius is not None else np.nan,
        'best_infl': float(infl) if infl is not None else np.nan,
        'rmse': float(rmse) if rmse is not None else np.nan,
        'rrmse_mean': float(rrmse) if rrmse is not None else np.nan,
        'rmse_std': float(rmse_std) if rmse_std is not None else np.nan,
        'rrmse_std': float(rrmse_std) if rrmse_std is not None else np.nan,
    }

    for col in df_full.columns:
        if col not in candidate:
            candidate[col] = np.nan

    extra_cols = [k for k in candidate.keys() if k not in df_full.columns]
    if extra_cols:
        for c in extra_cols:
            df_full[c] = np.nan

    has_keys = all(k in df_full.columns for k in ['method', 'N', 'sigma_y'])
    if has_keys and len(df_full) > 0:
        mask = (
            (df_full['method'] == candidate['method']) &
            (df_full['N'] == candidate['N']) &
            (df_full['sigma_y'] == candidate['sigma_y'])
        )
        if mask.any():
            idxs = np.flatnonzero(mask)
            idx = idxs[0]
            print(f"[write_benchmark_row] Overwriting existing row for {candidate['method']}, N={candidate['N']}.")
            for k, v in candidate.items():
                df_full.at[idx, k] = v
            if len(idxs) > 1:
                df_full = df_full.drop(index=idxs[1:]).reset_index(drop=True)
        else:
            df_full = pd.concat([df_full, pd.DataFrame([candidate])], ignore_index=True)
    else:
        df_full = pd.concat([df_full, pd.DataFrame([candidate])], ignore_index=True)

    try:
        df_full = df_full.sort_values(by=['method', 'N', 'sigma_y']).reset_index(drop=True)
    except Exception:
        pass

    df_full.to_csv(file_path, index=False)


if __name__ == "__main__":
    args = get_parameters()
    
    # === Configuration ===
    # Set the benchmark source here: 'dapper' (CSV) or 'torch' (Grid Search .pt)
    BENCHMARK_SOURCE = 'torch' 
    # =====================

    folder_name = os.path.join("save", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    
    # redirect output
    with redirect_output(save_output=should_redirect_output(args), save_folder=folder_name, filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = build_observation_operator(args)

        # modify test_batch_size
        if args.N == 100 and args.dataset == 'ks':
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.\n"
              f"Method: {args.v}")
    
        # --- Retrieve Optimal Parameters ---
        default_infl = 1.0
        default_loc_radius = None
        infl = default_infl
        loc_radius = default_loc_radius

        if BENCHMARK_SOURCE == 'dapper':
            try:
                sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(args, source='dapper')
                
                # Select appropriate array based on sigma_y
                if np.isclose(args.sigma_y, 1.0):
                    dapper_array = sigma_y_1_array
                elif np.isclose(args.sigma_y, 0.7):
                    dapper_array = sigma_y_0_7_array
                else:
                    dapper_array = np.empty((0, 4))
                    print(f"Warning: DAPPER benchmark not configured for sigma_y={args.sigma_y}. "
                          f"Falling back to default infl={default_infl}, loc={default_loc_radius}.")
                
                # Unpack array [best_loc_rad, best_infl, rmse, rrmse_mean]
                if dapper_array.shape[0] > 0:
                    loc_radius_candidate = dapper_array[0, 0]
                    infl_candidate = dapper_array[0, 1]
                    rmse_baseline = dapper_array[0, 2]
                    rrmse_baseline = dapper_array[0, 3]

                    if pd.notna(infl_candidate):
                        infl = float(infl_candidate)
                    if pd.notna(loc_radius_candidate):
                        loc_radius = float(loc_radius_candidate)

                    if pd.notna(rmse_baseline):
                        print(f"RMSE from DAPPER: {rmse_baseline:.3f}.")
                    if pd.notna(rrmse_baseline):
                        print(f"RRMSE from DAPPER: {rrmse_baseline:.3f}.")
                else:
                    print(f"Warning: No DAPPER benchmark found for dataset={args.dataset}, "
                          f"N={args.N}, sigma_y={args.sigma_y}. "
                          f"Using default infl={default_infl}, loc={default_loc_radius}.")
            except FileNotFoundError:
                print(f"Warning: benchmark file for dataset={args.dataset} is missing. "
                      f"Using default infl={default_infl}, loc={default_loc_radius}.")
            except Exception as e:
                print(f"Warning: failed to load DAPPER benchmark ({e}). "
                      f"Using default infl={default_infl}, loc={default_loc_radius}.")

        elif BENCHMARK_SOURCE == 'torch':
            try:
                infl_loaded, loc_radius_loaded = get_benchmarks(args, source='torch')
                if infl_loaded is not None and pd.notna(infl_loaded):
                    infl = float(infl_loaded)
                if loc_radius_loaded is not None and pd.notna(loc_radius_loaded):
                    loc_radius = float(loc_radius_loaded)
                print(f"Loaded from Torch Grid Search -> Inflation: {infl}; Localization Radius: {loc_radius}")
            except FileNotFoundError:
                print(f"Warning: torch benchmark file for dataset={args.dataset} is missing. "
                      f"Using default infl={default_infl}, loc={default_loc_radius}.")
            except Exception as e:
                print(f"Warning: failed to load torch benchmark ({e}). "
                      f"Using default infl={default_infl}, loc={default_loc_radius}.")

        print(f"Using Inflation: {infl}; Localization Radius: {loc_radius}")
        
        # test
        t_start = time.time()
        loss_list_nn = []
        test_results = \
            test_ClassicFilter(test_loader, 
                            args, 
                            H_info=H_info, 
                            plot_figures=args.save_test_figures,
                            fig_name=f'{folder_name}/test_{args.N}_0',
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
                'mean_spread_error_ratio': test_results.get('mean_spread_error_ratio', float('nan')),
                'std_spread_error_ratio': test_results.get('std_spread_error_ratio', float('nan')),
                'mean_spread_error_ratio_minus_1': test_results.get('mean_spread_error_ratio_minus_1', float('nan')),
                'std_spread_error_ratio_minus_1': test_results.get('std_spread_error_ratio_minus_1', float('nan')),
                'mean_ser': test_results.get('mean_ser', float('nan')),
                'std_ser': test_results.get('std_ser', float('nan')),
                'mean_ser_minus_1': test_results.get('mean_ser_minus_1', float('nan')),
                'std_ser_minus_1': test_results.get('std_ser_minus_1', float('nan')),
                'rank_freq_range': test_results.get('rank_freq_range', float('nan')),
                'rank_uniform_l1': test_results.get('rank_uniform_l1', float('nan')),
                'rank_uniform_l2': test_results.get('rank_uniform_l2', float('nan')),
                'rank_chi2': test_results.get('rank_chi2', float('nan')),
                'rank_total_samples': test_results.get('rank_total_samples', 0),
                'mean_es1': test_results.get('mean_es1', float('nan')),
                'std_es1': test_results.get('std_es1', float('nan')),
                'mean_res1': test_results.get('mean_res1', float('nan')),
                'std_res1': test_results.get('std_res1', float('nan')),
                'mean_pf_crps': test_results.get('mean_pf_crps', float('nan')),
                'std_pf_crps': test_results.get('std_pf_crps', float('nan')),
                'mean_pf_crps_state_avg': test_results.get('mean_pf_crps_state_avg', float('nan')),
                'std_pf_crps_state_avg': test_results.get('std_pf_crps_state_avg', float('nan')),
                'mean_pf_crps_pca_avg': test_results.get('mean_pf_crps_pca_avg', float('nan')),
                'std_pf_crps_pca_avg': test_results.get('std_pf_crps_pca_avg', float('nan')),
                'mean_pf_crps_state_dim1': test_results.get('mean_pf_crps_state_dim1', float('nan')),
                'std_pf_crps_state_dim1': test_results.get('std_pf_crps_state_dim1', float('nan')),
                'mean_pf_crps_state_dim2': test_results.get('mean_pf_crps_state_dim2', float('nan')),
                'std_pf_crps_state_dim2': test_results.get('std_pf_crps_state_dim2', float('nan')),
                'mean_pf_crps_state_dim3': test_results.get('mean_pf_crps_state_dim3', float('nan')),
                'std_pf_crps_state_dim3': test_results.get('std_pf_crps_state_dim3', float('nan')),
                'mean_pf_crps_pca_dim1': test_results.get('mean_pf_crps_pca_dim1', float('nan')),
                'std_pf_crps_pca_dim1': test_results.get('std_pf_crps_pca_dim1', float('nan')),
                'mean_pf_crps_pca_dim2': test_results.get('mean_pf_crps_pca_dim2', float('nan')),
                'std_pf_crps_pca_dim2': test_results.get('std_pf_crps_pca_dim2', float('nan')),
                'mean_pf_crps_pca_dim3': test_results.get('mean_pf_crps_pca_dim3', float('nan')),
                'std_pf_crps_pca_dim3': test_results.get('std_pf_crps_pca_dim3', float('nan')),
                'mean_pf_rcrps': test_results.get('mean_pf_rcrps', float('nan')),
                'std_pf_rcrps': test_results.get('std_pf_rcrps', float('nan')),
                'valid_percent': test_results.get('no_nan_percent', 0.0),
                'pf_cov_diff': test_results.get('mean_pf_cov_diff', float('nan')),
                'pf_rcov_diff': test_results.get('mean_pf_rcov_diff', float('nan')),
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
        
        torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))
