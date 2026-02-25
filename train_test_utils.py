import numpy as np
import time
import os
import csv
import re
from contextlib import nullcontext

import torch
import torch.nn as nn
import math
from config.dataset_info import DATASET_INFO

from utils import L63, L96, Rossler, rk4, etd_rk4_wrapper, CircleODE, DoubleWellODE, DoublingMap1D, ComplexSquareMap2D
from utils import project_to_unit_circle
from utils import AverageMeter, mystery_operator, get_mean_std, get_test_noise_generator
from utils import post_process, mean0
from visualization import (
    plot_particle_trajectories_with_histograms,
    plot_particle_trajectories,
    plot_and_test_point_clouds,
    plot_and_test_point_clouds_ring,
    plot_linear_kalman_vs_method_2d,
    _compute_axis_limits,
    _compute_zoomed_ranges_from_fixed_3d,
)
from localization import dist2coeff, create_loc_mat, pairwise_distances
from loss import (
    compute_loss,
    compute_es,
    compute_root_mean_variance,
    compute_spread_error_ratio,
    compute_ensemble_rank_histogram,
    sample_projection_directions,
    wasserstein2_multivariate_gaussian,
)
from networks import NaiveNetwork, SetTransformer, Simple_MLP, ConditionTransformerNetwork
from benchmark_analysis import ensemble_kalman_filter_analysis, bootstrap_particle_filter_analysis
from typing import Optional, List, Tuple, Dict, Any

from tqdm.auto import tqdm

L63_TEST_SNAPSHOT_STEPS = [100, 200, 300, 400, 500]
L63_FIXED_LIMITS_3D = {
    "xlim": (-22.0, 22.0),
    "ylim": (-30.0, 30.0),
    "zlim": (0.0, 52.0),
}
DYNAMIC_FIXED_LIMITS_3D = {
    "lorenz63": L63_FIXED_LIMITS_3D,
    "lorenz96": {"xlim": (-15.0, 15.0), "ylim": (-15.0, 15.0), "zlim": (-15.0, 15.0)},
    "ks": {"xlim": (-6.0, 6.0), "ylim": (-6.0, 6.0), "zlim": (-6.0, 6.0)},
    "rossler": {"xlim": (-15.0, 15.0), "ylim": (-15.0, 15.0), "zlim": (0.0, 30.0)},
}


def _sample_true_obs_noise_like(ref_tensor: torch.Tensor, test_noise_gen: torch.Generator) -> torch.Tensor:
    """Sample noise for true-observation generation from the dedicated test generator."""
    noise = torch.randn(
        ref_tensor.shape,
        generator=test_noise_gen,
        device='cpu',
        dtype=torch.float32,
    )
    return noise.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def _get_dataset_default_obs_fn(dataset: str) -> str:
    """Get default obs_fn for dataset; fallback to identity."""
    dataset_key = str(dataset or "").lower()
    dataset_cfg = DATASET_INFO.get(dataset_key, {})
    return str(dataset_cfg.get("obs_fn", "identity") or "identity").lower()


def _resolve_obs_fn_for_pf_paths(args) -> str:
    """Resolve effective obs_fn used by PF path naming."""
    default_obs_fn = _get_dataset_default_obs_fn(getattr(args, "dataset", ""))
    obs_fn = str(getattr(args, "obs_fn", "default") or "default").lower()
    if obs_fn == "default":
        return default_obs_fn
    return obs_fn


def _pf_obs_fn_suffix(args) -> str:
    """
    Return obs_fn suffix for PF files/folders.
    - Default obs_fn for dataset => empty suffix.
    - Non-default obs_fn => _{obs_fn}.
    """
    default_obs_fn = _get_dataset_default_obs_fn(getattr(args, "dataset", ""))
    obs_fn = _resolve_obs_fn_for_pf_paths(args)
    if obs_fn == default_obs_fn:
        return ""
    safe_obs_fn = re.sub(r"[^0-9a-zA-Z._-]+", "-", obs_fn).strip("-")
    if safe_obs_fn == "":
        return ""
    return f"_{safe_obs_fn}"


def _build_pf_cache_filename(args, batch_size: int, traj_len: int, avg: bool = False) -> str:
    """Build PF cache filename with obs_fn suffix policy."""
    obs_suffix = _pf_obs_fn_suffix(args)
    base = (
        f"pf_results_sigma_y_{args.sigma_y}_batch_{batch_size}_len_{traj_len}_pfN_{args.pf_N}"
    )
    if avg:
        return f"{base}_avg{obs_suffix}.pt"
    return f"{base}_{args.seed}{obs_suffix}.pt"


def _safe_pf_obs_fn_name(args) -> str:
    obs_fn = _resolve_obs_fn_for_pf_paths(args)
    safe_obs_fn = re.sub(r"[^0-9a-zA-Z._-]+", "-", str(obs_fn)).strip("-")
    return safe_obs_fn if safe_obs_fn else "unknown"


def _build_pf_cache_dir(args) -> str:
    return os.path.join("data", str(args.dataset), f"pf_{_safe_pf_obs_fn_name(args)}")


def _setup_mixed_precision(args):
    """
    Configure mixed precision runtime once per process.

    Default is fp32 (disabled). Low precision modes are enabled only on CUDA.
    """
    if getattr(args, "_precision_runtime_configured", False):
        return

    precision = str(getattr(args, "precision", "fp32")).lower()
    device_is_cuda = isinstance(args.device, torch.device) and args.device.type == "cuda"

    args._amp_enabled = False
    args._amp_dtype = None
    args._grad_scaler = None

    if precision == "fp32":
        pass
    elif precision == "bf16":
        if device_is_cuda:
            args._amp_enabled = True
            args._amp_dtype = torch.bfloat16
            print("[INFO] AMP enabled with bf16 on CUDA.")
        else:
            print("[WARN] bf16 mixed precision is only enabled for CUDA in this project. Falling back to fp32.")
    elif precision == "fp16":
        if device_is_cuda:
            args._amp_enabled = True
            args._amp_dtype = torch.float16
            args._grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
            print("[INFO] AMP enabled with fp16 on CUDA (GradScaler active).")
        else:
            print("[WARN] fp16 mixed precision requires CUDA. Falling back to fp32.")
    else:
        print(f"[WARN] Unknown precision '{precision}'. Falling back to fp32.")

    args._precision_runtime_configured = True


def _autocast_context(args):
    if getattr(args, "_amp_enabled", False):
        return torch.autocast(device_type="cuda", dtype=args._amp_dtype, enabled=True)
    return nullcontext()


def _backward_and_step(loss, optimizer, all_trainable_params, args):
    scaler = getattr(args, "_grad_scaler", None)
    if scaler is not None:
        scaler.scale(loss).backward()
        if all_trainable_params:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_trainable_params, max_norm=getattr(args, 'grad_clip_norm', 1.0))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if all_trainable_params:
            nn.utils.clip_grad_norm_(all_trainable_params, max_norm=getattr(args, 'grad_clip_norm', 1.0))
        optimizer.step()


def _build_observation_plot_tensor(args, batch_v, obs_y_list):
    """Map observation-space tensors to state-space layout for plotting when possible."""
    obs_tensor = torch.stack(obs_y_list).squeeze(2)
    observations = torch.full_like(batch_v, float('nan'), device=batch_v.device)

    obs_coord_inds = getattr(args, 'obs_coord_inds', None)
    obs_inds = getattr(args, 'obs_inds', None)
    target_inds = obs_coord_inds if obs_coord_inds is not None else obs_inds

    if target_inds is not None:
        target_inds = torch.as_tensor(target_inds, device=batch_v.device, dtype=torch.long)
        if target_inds.numel() == obs_tensor.shape[-1]:
            observations[:, :, target_inds] = obs_tensor
        else:
            print(
                f"Warning: cannot scatter observations for plotting because "
                f"obs_dim={obs_tensor.shape[-1]} != number of coordinate indices={target_inds.numel()}."
            )
    else:
        print("Warning: no observation coordinate indices available; plotting observations as NaN.")

    return observations


def _sigma_y_sq_per_traj(sigma_y, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalize sigma_y into a per-trajectory variance vector with shape [B]."""
    sigma_y_tensor = torch.as_tensor(sigma_y, device=device, dtype=dtype)

    if sigma_y_tensor.ndim >= 1 and sigma_y_tensor.shape[0] == batch_size:
        sigma_y_vec = sigma_y_tensor.reshape(batch_size, -1)[:, 0]
        return sigma_y_vec * sigma_y_vec

    flat = sigma_y_tensor.reshape(-1)
    if flat.numel() == 1:
        return (flat[0] * flat[0]).expand(batch_size)
    if flat.numel() == batch_size:
        return flat * flat

    print(
        f"Warning: sigma_y shape {tuple(sigma_y_tensor.shape)} is not aligned with batch size {batch_size}. "
        "Falling back to sigma_y[0] for all trajectories."
    )
    return (flat[0] * flat[0]).expand(batch_size)


def _compute_traj_snr_var_from_hvalues(h_values: torch.Tensor, sigma_y) -> torch.Tensor:
    """
    Compute per-trajectory SNR_var from clean observation-space trajectories.

    Args:
        h_values: [T, B, d_obs], where h_values[t, b] = h(v_t^b).
        sigma_y: Scalar or per-trajectory observation noise std.

    Returns:
        Tensor with shape [B], each entry is SNR_var for one trajectory.
    """
    if h_values.ndim != 3:
        raise ValueError(f"h_values must have shape [T, B, d_obs], but got {tuple(h_values.shape)}.")

    _, batch_size, d_obs = h_values.shape
    mean_h = torch.nanmean(h_values, dim=0, keepdim=True)
    signal_var = torch.nanmean(torch.sum((h_values - mean_h) ** 2, dim=-1), dim=0)

    sigma_y_sq = _sigma_y_sq_per_traj(
        sigma_y=sigma_y,
        batch_size=batch_size,
        device=h_values.device,
        dtype=h_values.dtype,
    )
    denom = float(d_obs) * sigma_y_sq

    eps = torch.finfo(h_values.dtype).eps if torch.is_floating_point(h_values) else 1e-12
    snr_var = signal_var / torch.clamp(denom, min=eps)
    invalid = ~torch.isfinite(signal_var) | ~torch.isfinite(denom)
    snr_var = snr_var.masked_fill(invalid, float('nan'))
    return snr_var


def _select_plot_batch_index(ens_tensor: torch.Tensor) -> int:
    """Pick a trajectory index with the fewest NaN time steps for visualization."""
    if ens_tensor.ndim != 4 or ens_tensor.shape[1] == 0:
        return 0
    nan_per_step = torch.isnan(ens_tensor).any(dim=(2, 3))  # [T, B]
    nan_counts = nan_per_step.sum(dim=0)
    return int(torch.argmin(nan_counts).item())


def _get_pf_cached_post_means(entry: Dict[str, Any]) -> Optional[torch.Tensor]:
    if "post_means" in entry:
        return entry["post_means"]
    return entry.get("means", None)


def _get_pf_cached_post_covs(entry: Dict[str, Any]) -> Optional[torch.Tensor]:
    if "post_covs" in entry:
        return entry["post_covs"]
    return entry.get("covs", None)


def _get_pf_cached_range_int(entry: Dict[str, Any], mode: str, range_idx: int, bidx: int, pad_int: int = 5) -> Optional[torch.Tensor]:
    key_int = f"{mode}_range_int"
    if key_int in entry:
        tensor = entry[key_int]
        if tensor.ndim == 4 and 0 <= range_idx < tensor.shape[0] and 0 <= bidx < tensor.shape[1]:
            return tensor[range_idx, bidx]

    key_q = f"{mode}_range_q01_q99"
    if key_q in entry:
        q_tensor = entry[key_q]
        if q_tensor.ndim == 4 and 0 <= range_idx < q_tensor.shape[0] and 0 <= bidx < q_tensor.shape[1]:
            qvals = q_tensor[range_idx, bidx].to(torch.float32)
            low = torch.floor(qvals[:, 0]) - int(pad_int)
            high = torch.ceil(qvals[:, 1]) + int(pad_int)
            return torch.stack((low, high), dim=-1).to(torch.int32)
    return None


def _get_pf_cached_quantile_block(
    entry: Dict[str, Any],
    key: str,
    range_idx: int,
    bidx: int,
) -> Optional[torch.Tensor]:
    if key not in entry:
        return None
    tensor = entry[key]
    if tensor.ndim < 4:
        return None
    if not (0 <= range_idx < tensor.shape[0] and 0 <= bidx < tensor.shape[1]):
        return None
    return tensor[range_idx, bidx]


def _resolve_snapshot_steps_for_plot(args, total_steps: int) -> List[int]:
    if total_steps <= 1:
        return []
    snapshot_steps = getattr(args, "test_snapshot_steps", None)
    if snapshot_steps is None and str(getattr(args, "dataset", "")).lower() == "lorenz63":
        snapshot_steps = list(L63_TEST_SNAPSHOT_STEPS)
    if snapshot_steps is None:
        snapshot_steps = [max(1, total_steps - 1)]
    valid = []
    for step in snapshot_steps:
        s = int(step)
        if 1 <= s < total_steps and s not in valid:
            valid.append(s)
    return valid


def _plot_pf_style_projections_from_cache(
    args,
    ens_post_tensor: torch.Tensor,
    ens_prior_tensor: Optional[torch.Tensor],
    true_tensor: torch.Tensor,
    pf_entry: Dict[str, Any],
    fig_name: str,
    plot_batch_indices: List[int],
    batch_start_index: int = 0,
    global_index_naming: bool = False,
):
    if ens_post_tensor.ndim != 4 or ens_post_tensor.shape[0] <= 1 or ens_post_tensor.shape[1] == 0:
        return
    if ens_prior_tensor is None or ens_prior_tensor.ndim != 4 or ens_prior_tensor.shape[0] <= 1:
        return

    plot_steps = _resolve_snapshot_steps_for_plot(args, total_steps=ens_post_tensor.shape[0])
    if len(plot_steps) == 0:
        return

    for bidx in plot_batch_indices:
        if not (0 <= int(bidx) < ens_post_tensor.shape[1]):
            continue
        b = int(bidx)
        idx_tag = f"g{batch_start_index + b}" if global_index_naming else f"b{b}"
        for step_label in plot_steps:
            prior_cloud = ens_prior_tensor[step_label, b:b + 1, :, :]
            post_cloud = ens_post_tensor[step_label, b:b + 1, :, :]
            true_state = true_tensor[step_label, b, :].detach().cpu()
            range_idx = step_label - 1
            prior_range_int = _get_pf_cached_range_int(
                pf_entry, mode="prior", range_idx=range_idx, bidx=b,
                pad_int=int(getattr(args, "pf_range_pad_int", 5))
            )
            post_range_int = _get_pf_cached_range_int(
                pf_entry, mode="post", range_idx=range_idx, bidx=b,
                pad_int=int(getattr(args, "pf_range_pad_int", 5))
            )
            q_probs = pf_entry.get("quantile_probs", None)
            prior_quantiles = _get_pf_cached_quantile_block(pf_entry, "prior_quantiles", range_idx=range_idx, bidx=b)
            post_quantiles = _get_pf_cached_quantile_block(pf_entry, "post_quantiles", range_idx=range_idx, bidx=b)
            prior_pca_quantiles = _get_pf_cached_quantile_block(pf_entry, "prior_pca_quantiles", range_idx=range_idx, bidx=b)
            post_pca_quantiles = _get_pf_cached_quantile_block(pf_entry, "post_pca_quantiles", range_idx=range_idx, bidx=b)
            prefix = f"{fig_name}_{idx_tag}_step{step_label}"
            plot_and_test_point_clouds(
                args=args,
                prior_tensor=prior_cloud.detach().cpu(),
                posterior_tensor=post_cloud.detach().cpu(),
                num_samples_plot=min(1000000, int(getattr(args, "pf_N", post_cloud.shape[2]))),
                prefix=prefix,
                num_swd_reference_samples=getattr(args, "pf_plot_swd_samples", 1000000),
                num_swd_directions=getattr(args, "pf_plot_swd_directions", 50),
                plot_indices=[0],
                history_traj=true_tensor[1:step_label + 1, b:b + 1, :].detach().cpu(),
                true_state=true_state,
                legend_in_figure=getattr(args, "legend_in_figure", False),
                prior_range_int=None if prior_range_int is None else prior_range_int.detach().cpu(),
                post_range_int=None if post_range_int is None else post_range_int.detach().cpu(),
                quantile_probs=None if q_probs is None else torch.as_tensor(q_probs).detach().cpu(),
                prior_quantiles=None if prior_quantiles is None else prior_quantiles.detach().cpu()[:3],
                post_quantiles=None if post_quantiles is None else post_quantiles.detach().cpu()[:3],
                prior_pca_quantiles=None if prior_pca_quantiles is None else prior_pca_quantiles.detach().cpu(),
                post_pca_quantiles=None if post_pca_quantiles is None else post_pca_quantiles.detach().cpu(),
            )


def _safe_int_seed(seed_like, default: int = 0) -> int:
    """Convert seed-like value to int with a stable fallback."""
    if seed_like is None:
        return int(default)
    if isinstance(seed_like, str):
        sval = seed_like.strip().lower()
        if sval in {"", "none", "null"}:
            return int(default)
    try:
        return int(seed_like)
    except Exception:
        return int(default)


def _resolve_plot_batch_indices(args, batch_size: int, nan_counts: Optional[torch.Tensor] = None) -> List[int]:
    """
    Resolve batch indices for plotting from args.test_plot_index.

    Supported forms:
    - "adaptive": choose one index with fewest NaN steps
    - int/list/tuple of ints: use those indices (after bounds filtering)
    """
    if batch_size <= 0:
        return []

    cfg = getattr(args, "test_plot_index", [0])
    if isinstance(cfg, str) and cfg.lower() == "adaptive":
        if nan_counts is not None and nan_counts.numel() == batch_size:
            return [int(torch.argmin(nan_counts).item())]
        return [0]

    if isinstance(cfg, int):
        requested = [int(cfg)]
    elif isinstance(cfg, (list, tuple)):
        requested = [int(x) for x in cfg]
    else:
        requested = [0]

    valid = []
    for idx in requested:
        if 0 <= idx < batch_size and idx not in valid:
            valid.append(idx)

    if len(valid) == 0:
        print(
            f"Warning: no valid plot indices in test_plot_index={cfg} for batch_size={batch_size}. "
            "Falling back to [0]."
        )
        return [0]
    return valid


def _normalize_global_plot_indices(args) -> Tuple[str, List[int]]:
    """
    Normalize args.test_plot_index into either:
    - ("adaptive", [])
    - ("global", [global_idx0, global_idx1, ...])
    """
    cfg = getattr(args, "test_plot_index", [0])
    if isinstance(cfg, str) and cfg.lower() == "adaptive":
        return "adaptive", []

    if isinstance(cfg, int):
        requested = [int(cfg)]
    elif isinstance(cfg, (list, tuple)):
        requested = []
        for x in cfg:
            try:
                requested.append(int(x))
            except Exception:
                continue
    else:
        requested = [0]

    valid = []
    for idx in requested:
        if idx >= 0 and idx not in valid:
            valid.append(idx)

    if len(valid) == 0:
        print(f"Warning: invalid global test_plot_index={cfg}. Falling back to [0].")
        valid = [0]

    return "global", valid


def _resolve_global_plot_indices_for_batch(
    global_indices: List[int],
    batch_start_index: int,
    batch_size: int,
) -> List[int]:
    """Map global trajectory indices to local batch indices for current batch."""
    if batch_size <= 0:
        return []
    local = []
    batch_end = batch_start_index + batch_size
    for gidx in global_indices:
        if batch_start_index <= gidx < batch_end:
            lidx = int(gidx - batch_start_index)
            if lidx not in local:
                local.append(lidx)
    return local


def _get_lowdim_snapshot_indices(
    total_steps: int,
    start_step: int = 100,
    num_slices: int = 3,
    step_offset: int = 0,
    explicit_step_labels: Optional[List[int]] = None,
):
    """Select snapshot indices using 1-based step labels."""
    if total_steps <= 0:
        return []

    min_label = max(1 + step_offset, int(start_step))
    max_label = total_steps + step_offset
    if min_label > max_label:
        min_label = max_label

    if explicit_step_labels is not None and len(explicit_step_labels) > 0:
        labels = []
        for lbl in explicit_step_labels:
            try:
                lbl_i = int(lbl)
            except (TypeError, ValueError):
                continue
            if min_label <= lbl_i <= max_label:
                labels.append(lbl_i)
        if len(labels) == 0:
            labels = [max_label]
    else:
        if max_label - min_label + 1 >= num_slices:
            labels = np.rint(np.linspace(min_label, max_label, num_slices)).astype(int).tolist()
        else:
            labels = list(range(min_label, max_label + 1))

    # De-duplicate while preserving order.
    dedup_labels = []
    for lbl in labels:
        if lbl not in dedup_labels:
            dedup_labels.append(lbl)

    selected = []
    for lbl in dedup_labels:
        idx = lbl - 1 - step_offset
        if 0 <= idx < total_steps:
            selected.append((idx, lbl))
    return selected


def _plot_last_three_steps_ring(
    args,
    ens_traj,
    true_traj,
    observations,
    fig_name,
    step_offset: int = 0,
    plot_history: bool = False,
    legend_in_figure: bool = True,
    point_color: str = "red",
    snapshot_steps: Optional[List[int]] = None,
):
    """Use ring-mapped visualization for selected 3 low-dim slices."""
    T = ens_traj.shape[0]
    selected_steps = _get_lowdim_snapshot_indices(
        T,
        start_step=100,
        num_slices=3,
        step_offset=step_offset,
        explicit_step_labels=snapshot_steps,
    )
    for t_idx, step_label in selected_steps:
        obs_t = observations[t_idx]
        obs_input = obs_t if torch.isfinite(obs_t.reshape(-1)[:1]).all() else None
        history_traj = true_traj[: t_idx + 1].unsqueeze(1)  # [t+1, 1, d]

        plot_and_test_point_clouds_ring(
            args=args,
            tensor=ens_traj[t_idx : t_idx + 1],   # [1, N, d]
            num_samples_plot=min(ens_traj.shape[1], 10000),
            prefix=f"{fig_name}_slice_step{step_label}",
            point_color=point_color,
            observation=obs_input,
            true_state=true_traj[t_idx],
            plot_history=plot_history,
            plot_indices=[0],
            history_traj=history_traj,
            legend_in_figure=legend_in_figure,
        )


def _plot_last_three_steps_lowdim_generic(
    ens_traj,
    true_traj,
    observations,
    fig_name,
    save_pdf=False,
    step_offset: int = 0,
    legend_in_figure: bool = True,
    ens_color: str = 'tab:red',
    dataset: Optional[str] = None,
    snapshot_steps: Optional[List[int]] = None,
):
    """Plot ensemble/true/obs snapshots for selected steps (dim <= 3, non-ring datasets)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    T, _, D = ens_traj.shape
    ens_color = str(ens_color or 'tab:red')
    ens_color_lower = ens_color.lower()
    density_cmap = 'Blues' if 'blue' in ens_color_lower else 'Reds'
    density_threshold = 1000
    selected_steps = _get_lowdim_snapshot_indices(
        T,
        start_step=100,
        num_slices=3,
        step_offset=step_offset,
        explicit_step_labels=snapshot_steps,
    )
    if len(selected_steps) == 0:
        return
    legend_labels = []

    def _add_label(label: str) -> None:
        if label not in legend_labels:
            legend_labels.append(label)

    if D == 3:
        dataset_key = str(dataset or "").lower()
        projection_specs = [
            ("xy", 0, 1, "x", "y"),
            ("yz", 1, 2, "y", "z"),
            ("xz", 0, 2, "x", "z"),
        ]
        if dataset_key in DYNAMIC_FIXED_LIMITS_3D:
            fixed_limits_xyz = DYNAMIC_FIXED_LIMITS_3D[dataset_key]
        else:
            ens_all = ens_traj.reshape(-1, D)
            finite_all = torch.isfinite(ens_all).all(dim=1)
            ens_all_np = ens_all[finite_all].numpy() if finite_all.any() else None
            if ens_all_np is not None and ens_all_np.shape[0] > 0:
                fixed_limits_xyz = {
                    "xlim": _compute_axis_limits(ens_all_np[:, 0]),
                    "ylim": _compute_axis_limits(ens_all_np[:, 1]),
                    "zlim": _compute_axis_limits(ens_all_np[:, 2]),
                }
            else:
                fixed_limits_xyz = {
                    "xlim": (-1.0, 1.0),
                    "ylim": (-1.0, 1.0),
                    "zlim": (-1.0, 1.0),
                }
        for t_idx, step_label in selected_steps:
            ens_t = ens_traj[t_idx]
            valid = torch.isfinite(ens_t).all(dim=1)
            ens_t = ens_t[valid]
            ens_np = ens_t.numpy() if ens_t.shape[0] > 0 else None

            has_true = torch.isfinite(true_traj[t_idx, :3]).all()
            has_obs = torch.isfinite(observations[t_idx, :3]).all()
            true_np = true_traj[t_idx, :3].numpy() if has_true else None
            obs_np = observations[t_idx, :3].numpy() if has_obs else None

            adaptive_limits_xyz = fixed_limits_xyz
            if ens_np is not None and ens_np.shape[0] > 0:
                adaptive_limits_xyz = _compute_zoomed_ranges_from_fixed_3d(
                    points_xyz=ens_np,
                    fixed_limits=fixed_limits_xyz,
                    num_splits=10,
                )

            for range_mode in ["fixed", "adaptive"]:
                fig, axes = plt.subplots(1, len(projection_specs), figsize=(12.0, 3.8), squeeze=False)
                axes = axes[0]
                curr_limits_xyz = fixed_limits_xyz if range_mode == "fixed" else adaptive_limits_xyz

                for pidx, (plane_tag, dim_x, dim_y, x_label, y_label) in enumerate(projection_specs):
                    ax = axes[pidx]

                    if plane_tag == "xy":
                        xlim = curr_limits_xyz["xlim"]
                        ylim = curr_limits_xyz["ylim"]
                    elif plane_tag == "yz":
                        xlim = curr_limits_xyz["ylim"]
                        ylim = curr_limits_xyz["zlim"]
                    else:
                        xlim = curr_limits_xyz["xlim"]
                        ylim = curr_limits_xyz["zlim"]

                    if ens_np is not None:
                        xy = ens_np[:, [dim_x, dim_y]]

                        ax.scatter(
                            xy[:, 0],
                            xy[:, 1],
                            s=9,
                            alpha=0.35,
                            c=ens_color,
                            edgecolors='none',
                            zorder=3,
                            label='Ensemble',
                        )
                        _add_label('Ensemble')

                    if true_np is not None:
                        ax.scatter(
                            true_np[dim_x],
                            true_np[dim_y],
                            marker='*',
                            s=220,
                            c='orange',
                            edgecolors='black',
                            linewidth=0.6,
                            zorder=2,
                            label='True state',
                        )
                        _add_label('True state')

                    if obs_np is not None:
                        ax.scatter(
                            obs_np[dim_x],
                            obs_np[dim_y],
                            marker='*',
                            s=120,
                            c='orange',
                            edgecolors='black',
                            linewidth=0.5,
                            label='Obs',
                        )
                        _add_label('Obs')

                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    if legend_in_figure:
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_title(f"{plane_tag} | step={step_label} | {range_mode}")

                if legend_in_figure:
                    axes[0].legend(loc='best', fontsize=8)
                fig.tight_layout()
                fig.savefig(f"{fig_name}_slice_step{step_label}_{range_mode}.png", dpi=150, bbox_inches='tight')
                if save_pdf:
                    fig.savefig(f"{fig_name}_slice_step{step_label}_{range_mode}.pdf", bbox_inches='tight')
                plt.close(fig)
    else:
        fig, axes = plt.subplots(1, len(selected_steps), figsize=(5 * len(selected_steps), 4))
        if len(selected_steps) == 1:
            axes = [axes]

        for ax, (t_idx, step_label) in zip(axes, selected_steps):
            ens_t = ens_traj[t_idx]
            valid = torch.isfinite(ens_t).all(dim=1)
            ens_t = ens_t[valid]

            if D == 2:
                if ens_t.shape[0] > 0:
                    if ens_t.shape[0] < density_threshold:
                        ax.scatter(ens_t[:, 0], ens_t[:, 1], s=12, alpha=0.45, c=ens_color, label='Ensemble')
                        _add_label('Ensemble')
                    else:
                        ax.hexbin(
                            ens_t[:, 0].numpy(),
                            ens_t[:, 1].numpy(),
                            gridsize=45,
                            bins='log',
                            mincnt=1,
                            cmap=density_cmap,
                            linewidths=0.0,
                            alpha=0.9,
                        )
                        _add_label('Ensemble density')
                if torch.isfinite(true_traj[t_idx, :2]).all():
                    ax.scatter(true_traj[t_idx, 0], true_traj[t_idx, 1],
                               marker='*', s=220, c='orange', edgecolors='black', linewidth=0.6, zorder=2, label='True state')
                    _add_label('True state')
                if torch.isfinite(observations[t_idx, :2]).all():
                    ax.scatter(observations[t_idx, 0], observations[t_idx, 1],
                               marker='*', s=120, c='orange', edgecolors='black', linewidth=0.5, label='Obs')
                    _add_label('Obs')
                if legend_in_figure:
                    ax.set_xlabel("dim0")
                    ax.set_ylabel("dim1")
                ax.set_aspect('equal', adjustable='box')
            else:  # D == 1
                if ens_t.shape[0] > 0:
                    ax.scatter(ens_t[:, 0], torch.zeros_like(ens_t[:, 0]), s=8, alpha=0.35, c=ens_color, label='Ensemble')
                    _add_label('Ensemble')
                if torch.isfinite(true_traj[t_idx, 0]).all():
                    ax.scatter(true_traj[t_idx, 0], 0.0, marker='*', s=220, c='orange', edgecolors='black', linewidth=0.6, zorder=2, label='True state')
                    _add_label('True state')
                if torch.isfinite(observations[t_idx, 0]).all():
                    ax.scatter(observations[t_idx, 0], 0.0, marker='*', s=120, c='orange', edgecolors='black', linewidth=0.5, label='Obs')
                    _add_label('Obs')
                if legend_in_figure:
                    ax.set_xlabel("dim0")
                    ax.set_yticks([])

            if legend_in_figure:
                ax.set_title(f"step={step_label}")
            if legend_in_figure and ax is axes[0]:
                ax.legend(loc='best', fontsize=8)
        fig.tight_layout()

    if D != 3:
        fig.savefig(f"{fig_name}_slice_steps.png", dpi=150, bbox_inches='tight')
        if save_pdf:
            fig.savefig(f"{fig_name}_slice_steps.pdf", bbox_inches='tight')
        plt.close(fig)

    if not legend_in_figure and len(legend_labels) > 0:
        handles = []
        labels = []
        for label in legend_labels:
            if label == 'Ensemble':
                handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                      markerfacecolor=ens_color, markeredgecolor='none', markersize=7))
            elif label == 'Ensemble density':
                handles.append(mpatches.Patch(facecolor=ens_color, alpha=0.9, edgecolor='none'))
            elif label == 'True state':
                handles.append(Line2D([0], [0], marker='*', linestyle='None',
                                      markerfacecolor='orange', markeredgecolor='black', markersize=11))
            elif label == 'Obs':
                handles.append(Line2D([0], [0], marker='*', linestyle='None',
                                      markerfacecolor='orange', markeredgecolor='black', markersize=11))
            else:
                continue
            labels.append(label)

        if len(handles) > 0:
            fig_leg, ax_leg = plt.subplots(figsize=(max(4.5, 2.2 * len(handles)), 1.35))
            ax_leg.axis('off')
            ax_leg.legend(handles, labels, loc='center', ncol=len(handles), frameon=True, fontsize=10)
            fig_leg.tight_layout(pad=0.1)
            fig_leg.savefig(f"{fig_name}_slice_steps_legend.png", dpi=150, bbox_inches='tight')
            if save_pdf:
                fig_leg.savefig(f"{fig_name}_slice_steps_legend.pdf", bbox_inches='tight')
            plt.close(fig_leg)


def _plot_lowdim_traj_summary(
    ens_traj,
    ref_traj,
    observations,
    fig_name,
    save_pdf=False,
    legend_in_figure: bool = True,
    ens_color: str = 'tab:red',
):
    """Plot trajectory-level summaries for dim <= 3 against a reference trajectory."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    D = ref_traj.shape[-1]
    dim_indices = list(range(D))

    plot_particle_trajectories_with_histograms(
        particles=ens_traj,
        true_traj=ref_traj,
        observation=observations,
        dim_indices=dim_indices,
        start_time=0,
        end_time=ens_traj.shape[0],
        mode='quantile',
        save_fig=True,
        save_pdf=save_pdf,
        save_name=fig_name + "_trajdims",
        hist_step=1,
        fontsize=16,
        legend_in_figure=legend_in_figure,
        ensemble_color=ens_color,
    )

    ens_mean = ens_traj.mean(dim=1)
    legend_labels = ['Reference', 'Ensemble mean']
    if D == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], c='black', linewidth=1.6, label='Reference')
        ax.plot(ens_mean[:, 0], ens_mean[:, 1], ens_mean[:, 2], c=ens_color, linewidth=1.2, label='Ensemble mean')
        valid_obs = torch.isfinite(observations[:, :3]).all(dim=1)
        if valid_obs.any():
            obs = observations[valid_obs, :3]
            ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], c='orange', s=10, alpha=0.8, label='Obs')
            legend_labels.append('Obs')
        if legend_in_figure:
            ax.set_title("Trajectory 3D View")
            ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{fig_name}_traj3d.png", dpi=150, bbox_inches='tight')
        if save_pdf:
            fig.savefig(f"{fig_name}_traj3d.pdf", bbox_inches='tight')
        plt.close(fig)
    elif D == 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], c='black', linewidth=1.6, label='Reference')
        ax.plot(ens_mean[:, 0], ens_mean[:, 1], c=ens_color, linewidth=1.2, label='Ensemble mean')
        valid_obs = torch.isfinite(observations[:, :2]).all(dim=1)
        if valid_obs.any():
            obs = observations[valid_obs, :2]
            ax.scatter(obs[:, 0], obs[:, 1], c='orange', s=12, alpha=0.8, label='Obs')
            legend_labels.append('Obs')
        if legend_in_figure:
            ax.set_xlabel("dim0")
            ax.set_ylabel("dim1")
            ax.set_title("Trajectory 2D View")
            ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{fig_name}_traj2d.png", dpi=150, bbox_inches='tight')
        if save_pdf:
            fig.savefig(f"{fig_name}_traj2d.pdf", bbox_inches='tight')
        plt.close(fig)

    if not legend_in_figure and D in [2, 3]:
        handles = []
        labels = []
        seen = set()
        for label in legend_labels:
            if label in seen:
                continue
            seen.add(label)
            if label == 'Reference':
                handles.append(Line2D([0], [0], color='black', linewidth=1.6))
            elif label == 'Ensemble mean':
                handles.append(Line2D([0], [0], color=ens_color, linewidth=1.2))
            elif label == 'Obs':
                handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                      markerfacecolor='orange', markeredgecolor='none', markersize=7))
            else:
                continue
            labels.append(label)
        if len(handles) > 0:
            fig_leg, ax_leg = plt.subplots(figsize=(max(4.5, 2.2 * len(handles)), 1.35))
            ax_leg.axis('off')
            ax_leg.legend(handles, labels, loc='center', ncol=len(handles), frameon=True, fontsize=10)
            fig_leg.tight_layout(pad=0.1)
            fig_leg.savefig(f"{fig_name}_traj_summary_legend.png", dpi=150, bbox_inches='tight')
            if save_pdf:
                fig_leg.savefig(f"{fig_name}_traj_summary_legend.pdf", bbox_inches='tight')
            plt.close(fig_leg)


def _plot_test_visualizations(
    args,
    ens_tensor,
    true_tensor,
    observations,
    fig_name,
    save_pdf=False,
    comparison_tensor=None,
    step_offset: int = 0,
    plot_batch_indices: Optional[List[int]] = None,
    batch_start_index: int = 0,
    global_index_naming: bool = False,
    ens_color: str = 'tab:red',
    ring_plot_history: bool = False,
    enable_highdim_slices: bool = False,
):
    """Dispatch visualization strategy by state dimension.

    Args:
        true_tensor: ground-truth trajectory used for per-step snapshots (e.g., black X).
        comparison_tensor: reference trajectory used for ensemble-mean trajectory comparison.
            If None, falls back to true_tensor.
    """
    if ens_tensor.ndim != 4 or ens_tensor.shape[1] == 0:
        return
    if comparison_tensor is None:
        comparison_tensor = true_tensor

    if plot_batch_indices is None:
        nan_per_step = torch.isnan(ens_tensor).any(dim=(2, 3))  # [T, B]
        nan_counts = nan_per_step.sum(dim=0)
        plot_bidx_list = _resolve_plot_batch_indices(
            args=args,
            batch_size=ens_tensor.shape[1],
            nan_counts=nan_counts,
        )
    else:
        plot_bidx_list = []
        for idx in plot_batch_indices:
            i = int(idx)
            if 0 <= i < ens_tensor.shape[1] and i not in plot_bidx_list:
                plot_bidx_list.append(i)
        if len(plot_bidx_list) == 0:
            return

    if args.ori_dim <= 3:
        multi_bidx = len(plot_bidx_list) > 1 or global_index_naming
        ring_point_color = "blue" if "blue" in str(ens_color).lower() else "red"
        dataset_key = str(getattr(args, "dataset", "")).lower()
        snapshot_steps = getattr(args, "test_snapshot_steps", None)
        if snapshot_steps is None and dataset_key == "lorenz63":
            snapshot_steps = list(L63_TEST_SNAPSHOT_STEPS)
        for bidx in plot_bidx_list:
            ens_traj = ens_tensor[:, bidx, :, :].detach().cpu()
            true_traj = true_tensor[:, bidx, :].detach().cpu()
            compare_traj = comparison_tensor[:, bidx, :].detach().cpu()
            obs_traj = observations[:, bidx, :].detach().cpu()
            idx_tag = f"g{batch_start_index + bidx}" if global_index_naming else f"b{bidx}"
            fig_name_b = f"{fig_name}_{idx_tag}" if multi_bidx else fig_name

            if args.dataset in {"doubling1d", "complex2d"}:
                _plot_last_three_steps_ring(
                    args,
                    ens_traj,
                    true_traj,
                    obs_traj,
                    fig_name=fig_name_b,
                    step_offset=step_offset,
                    plot_history=ring_plot_history,
                    legend_in_figure=getattr(args, "legend_in_figure", False),
                    point_color=ring_point_color,
                    snapshot_steps=snapshot_steps,
                )
            else:
                _plot_last_three_steps_lowdim_generic(
                    ens_traj=ens_traj,
                    true_traj=true_traj,
                    observations=obs_traj,
                    fig_name=fig_name_b,
                    save_pdf=save_pdf,
                    step_offset=step_offset,
                    legend_in_figure=getattr(args, "legend_in_figure", False),
                    ens_color=ens_color,
                    dataset=dataset_key,
                    snapshot_steps=snapshot_steps,
                )

            _plot_lowdim_traj_summary(
                ens_traj=ens_traj,
                ref_traj=compare_traj,
                observations=obs_traj,
                fig_name=fig_name_b,
                save_pdf=save_pdf,
                legend_in_figure=getattr(args, "legend_in_figure", False),
                ens_color=ens_color,
            )
        return

    # High-dimensional fallback: use selected batch index/indices.
    num_dims_plot = 4
    dim_indices_plot = list(range(min(args.ori_dim, num_dims_plot)))
    multi_bidx = len(plot_bidx_list) > 1 or global_index_naming
    for bidx in plot_bidx_list:
        idx_tag = f"g{batch_start_index + bidx}" if global_index_naming else f"b{bidx}"
        fig_name_b = f"{fig_name}_{idx_tag}" if multi_bidx else fig_name

        if enable_highdim_slices and int(args.ori_dim) >= 3:
            dataset_key = str(getattr(args, "dataset", "")).lower()
            snapshot_steps = getattr(args, "test_snapshot_steps", None)
            if snapshot_steps is None and dataset_key == "lorenz63":
                snapshot_steps = list(L63_TEST_SNAPSHOT_STEPS)
            _plot_last_three_steps_lowdim_generic(
                ens_traj=ens_tensor[:, bidx, :, :3],
                true_traj=true_tensor[:, bidx, :3],
                observations=observations[:, bidx, :3],
                fig_name=fig_name_b + "_slice3d",
                save_pdf=save_pdf,
                step_offset=step_offset,
                legend_in_figure=getattr(args, "legend_in_figure", False),
                ens_color=ens_color,
                dataset=dataset_key,
                snapshot_steps=snapshot_steps,
            )

        plot_particle_trajectories_with_histograms(
            particles=ens_tensor[:, bidx, :, :],
            true_traj=comparison_tensor[:, bidx, :],
            observation=None,
            dim_indices=dim_indices_plot,
            start_time=0,
            end_time=ens_tensor.shape[0],
            mode='quantile',
            save_fig=True,
            save_pdf=save_pdf,
            save_name=fig_name_b + "_hist",
            hist_step=1,
            fontsize=None,
            legend_in_figure=getattr(args, "legend_in_figure", False),
            ensemble_color=ens_color,
        )
        plot_particle_trajectories(
            particles=ens_tensor[:, bidx, :, :],
            true_traj=comparison_tensor[:, bidx, :],
            observation=observations[:, bidx, :],
            cmap_name='bwr',
            start_time=0,
            end_time=ens_tensor.shape[0],
            main_fig_size=(5, 2),
            save_fig=True,
            save_pdf=save_pdf,
            save_name=fig_name_b + "_traj",
            colorbar_range=args.colorbar_range if hasattr(args, 'colorbar_range') else None,
            plot_vertical_colorbar=False,
            plot_horizontal_colorbar=True,
            legend_in_figure=getattr(args, "legend_in_figure", False),
        )


def _get_forward_fun(args):
    """Selects the model propagator RHS/map for the current dataset."""
    if args.dataset == "lorenz63":
        return L63.forward
    if args.dataset == "rossler":
        return Rossler.forward
    if args.dataset == "lorenz96":
        return L96.forward
    if args.dataset == "circle":
        return CircleODE.forward
    if args.dataset == "Hdoublewell":
        return DoubleWellODE.forward
    if args.dataset == "doubling1d":
        return DoublingMap1D.forward
    if args.dataset == "complex2d":
        return ComplexSquareMap2D.forward
    if args.dataset == "ks":
        if args.dt_iter <= 0:
            raise ValueError("args.dt_iter must be positive for KS.")
        return etd_rk4_wrapper(device=args.device, dt=args.dt / args.dt_iter)
    raise NotImplementedError(f"Dataset {args.dataset} not implemented.")


def _forecast_ensemble(args, ens_v_a, step_idx, forward_fun):
    """
    Forecast one assimilation step for an ensemble tensor.

    Args:
        ens_v_a: analysis ensemble, shape [B, N, D].
        step_idx: current time index i.
        forward_fun: dynamics RHS/map selected by _get_forward_fun.
    """
    ens_size = ens_v_a.shape[1]
    ens_v_f_in = ens_v_a.reshape(-1, args.ori_dim)
    sigma_v = 0.0 if args.sigma_v is None else args.sigma_v

    if args.dataset == "doubling1d":
        ens_v_f_in = forward_fun(ens_v_f_in)
        ens_v_f_in = torch.remainder(
            ens_v_f_in + sigma_v * torch.randn_like(ens_v_f_in, device=args.device),
            1.0,
        )
    elif args.dataset == "complex2d":
        ens_v_f_in = forward_fun(ens_v_f_in)
        ens_v_f_in = ens_v_f_in + sigma_v * torch.randn_like(ens_v_f_in, device=args.device)
        ens_v_f_in = project_to_unit_circle(ens_v_f_in)
    elif args.dataset == "ks":
        for _ in range(args.dt_iter):
            ens_v_f_in = forward_fun(ens_v_f_in, None, args.dt / args.dt_iter)
        ens_v_f_in = ens_v_f_in + sigma_v * torch.randn_like(ens_v_f_in, device=args.device)
    else:
        for j_iter in range(args.dt_iter):
            t_curr = step_idx * args.dt + j_iter * (args.dt / args.dt_iter)
            ens_v_f_in = rk4(forward_fun, ens_v_f_in, t_curr, args.dt / args.dt_iter)
        ens_v_f_in = ens_v_f_in + sigma_v * torch.randn_like(ens_v_f_in, device=args.device)

    return ens_v_f_in.view(-1, ens_size, args.ori_dim)


def _build_ienks_model_args(args, forward_fun):
    """
    Build model_args for iEnKS with dataset-specific propagator behavior.

    KS is special: the propagator is ETD-RK4-based and ignores the rhs argument.
    """
    if args.dataset in ['doubling1d', 'complex2d']:
        raise NotImplementedError("iEnKS is currently not supported for discrete-map datasets.")

    if args.dataset == "ks":
        propagator = lambda func, u, t, dt: etd_rk4_wrapper(device=args.device, dt=dt)(u, None, dt)
    else:
        propagator = rk4

    return {
        "propagator": propagator,
        "rhs": forward_fun,
        "dt": args.dt / args.dt_iter,
        "steps_between_analyses": args.dt_iter,
    }


def set_models(args):
    has_loc_geometry = (
        getattr(args, 'num_dist', 0) > 0
        and getattr(args, 'Lvy', None) is not None
        and getattr(args, 'Lyy', None) is not None
    )
    # set models
    if args.v == 'CorrTerms':
        model = Simple_MLP(d_input=args.input_dim, d_output=args.obs_dim + args.ori_dim, num_hidden_layers=3).to(args.device)
    elif args.v == 'EtE' or args.v == 'EtE-LRes':
        model = Simple_MLP(d_input=args.input_dim, d_output=args.ori_dim, num_hidden_layers=4).to(args.device)
    elif args.v == 'EtE2':
        model = ConditionTransformerNetwork(u_dim=args.ori_dim + args.obs_dim, y_dim=args.obs_dim, output_dim=args.ori_dim, hidden_dim=args.hidden_dim, 
                                            num_blocks=4, num_heads=8, ff_expansion=2).to(args.device)
    else:
        model = NaiveNetwork(1)
    if args.no_localization or args.v.startswith('EtE') or not has_loc_geometry:
        local_model = NaiveNetwork(1)
    else:
        local_model = Simple_MLP(d_input=args.local_input_dim, d_output=args.num_dist, num_hidden_layers=2).to(args.device)
    if args.v == 'EtE2':
        st_model1 = NaiveNetwork(1)
        st_model2 = NaiveNetwork(1)
        
        infl_model = NaiveNetwork(1)
    else:
        if args.st_type == 'separate':
            st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = SetTransformer(input_dim=args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            if args.v.startswith('EtE'):
                infl_model = NaiveNetwork(1)
            else:
                infl_model = Simple_MLP(d_input=args.ori_dim + args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
        elif args.st_type == 'state_only':
            st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = NaiveNetwork(1)
            if args.v.startswith('EtE'):
                infl_model = NaiveNetwork(1)
            else:
                infl_model = Simple_MLP(d_input=args.ori_dim + args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
        elif args.st_type == 'joint':
            st_model1 = SetTransformer(input_dim=args.ori_dim + args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim * 2, 
                                        hidden_dim=args.hidden_dim, num_layers=3, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = NaiveNetwork(1)
            if args.v.startswith('EtE'):
                infl_model = NaiveNetwork(1)
            else:
                infl_model = Simple_MLP(d_input=args.ori_dim + 2 * args.st_output_dim, d_output=args.ori_dim, num_hidden_layers=2).to(args.device)
        else:
            st_model1, st_model2, infl_model, local_model = NaiveNetwork(1), NaiveNetwork(1), NaiveNetwork(1), NaiveNetwork(1)
    if args.use_data_parallel:
        model, infl_model, local_model, st_model1, st_model2 = \
            nn.DataParallel(model), nn.DataParallel(infl_model), nn.DataParallel(local_model), nn.DataParallel(st_model1), nn.DataParallel(st_model2)
    model_list = [model, infl_model, local_model, st_model1, st_model2]
    total_params = sum(sum(p.numel() for p in model.parameters()) for model in model_list)
    print(f'Total number of parameters: {total_params}')
    
    return model_list

def _get_model_inputs(args, st_model1, st_model2, ens_v_f, hv, obs_y):
    """Prepare input tensors for the models based on args.st_type."""
    N_ens = ens_v_f.shape[1]
    
    if args.noise_st_input:
        st_input_2 = hv + args.sigma_y * torch.randn_like(obs_y, device=args.device)
    else:
        st_input_2 = hv
    
    if args.mlp_y_type == 'obs':
        mlp_y_input = obs_y.expand(-1, N_ens, -1)
    elif args.mlp_y_type == 'innov':
        mlp_y_input = obs_y.expand(-1, N_ens, -1) - hv
    elif args.mlp_y_type == 'noise_innov':
        mlp_y_input = obs_y.expand(-1, N_ens, -1) - hv - args.sigma_y * torch.randn_like(obs_y, device=args.device)
    else:
        raise ValueError("args.mlp_y_type must be one within 'obs', 'innov', 'noise_innov'.")
    
    
    if args.v == 'EtE2':
        nn_input_cat_list = [ens_v_f, st_input_2]
        nn_input = torch.cat(nn_input_cat_list, dim=-1)
    else:
        # Initialize lists to avoid NameError
        local_nn_input_cat_list = []
        infl_nn_input_cat_list = []

        if args.st_type == 'state_only':
            ens_nn_output = st_model1(ens_v_f)
            nn_input_cat_list = [ens_v_f, hv, mlp_y_input, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            if args.v == 'CorrTerms':
                local_nn_input_cat_list = [obs_y.squeeze(1), ens_nn_output] if args.obs_in_loc else [ens_nn_output]
                infl_nn_input_cat_list = [ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            st_output_dim_actual = args.st_output_dim
            
        elif args.st_type == 'separate':
            ens_nn_output = st_model1(ens_v_f)
            ens_o_nn_output = st_model2(st_input_2)
            nn_input_cat_list = [ens_v_f, hv, obs_y.expand(-1, N_ens, -1), ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            if args.v == 'CorrTerms':
                local_nn_input_cat_list = [obs_y.squeeze(1), ens_nn_output, ens_o_nn_output] if args.obs_in_loc else [ens_nn_output, ens_o_nn_output]
                infl_nn_input_cat_list = [ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            st_output_dim_actual = args.st_output_dim
            
        elif args.st_type == 'joint':
            ens_nn_output = st_model1(torch.cat([ens_v_f, st_input_2], dim=-1))
            nn_input_cat_list = [ens_v_f, hv, mlp_y_input, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            if args.v == 'CorrTerms':
                local_nn_input_cat_list = [obs_y.squeeze(1), ens_nn_output] if args.obs_in_loc else [ens_nn_output]
                infl_nn_input_cat_list = [ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N_ens, -1)]
            st_output_dim_actual = args.st_output_dim * 2
            
        else:
            raise ValueError(f"Unknown args.st_type: {args.st_type}")
        nn_input = torch.cat(nn_input_cat_list, dim=-1).view(-1, args.input_dim)
    
    if args.v == 'CorrTerms':
        # These are only computed and returned if needed
        local_nn_input = torch.cat(local_nn_input_cat_list, dim=-1)
        infl_nn_input = torch.cat(infl_nn_input_cat_list, dim=-1).view(-1, args.ori_dim + st_output_dim_actual)
        return nn_input, local_nn_input, infl_nn_input
    elif args.v == 'EtE' or args.v == 'EtE-LRes':
        return nn_input, None, None # Return None for unused values
    elif args.v == 'EtE2':
        return nn_input, obs_y.squeeze(1), None
    else:
        raise NotImplementedError(f"args.v = {args.v} is not implemented.")

def _process_analysis_step(args, model_list, ens_v_f, hv, obs_y, ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=None, infl=1.0, loc_radius=None):
    """
    Process the analysis ensemble step.
    
    Inputs:
        args: Argument parser with config (must contain Lvy/Lyy for EnKF/LETKF loc).
        model_list: List of neural networks.
        ens_v_f: Forecast ensemble (B, N_ens, D_state).
        hv: Observation of forecast ensemble (B, N_ens, d_obs).
        obs_y: Observations (B, d_obs).
        ens_i_innov: Innovation (obs_y - hv) + noise (B, N_ens, d_obs). [Used for EnKF]
        mean_ens_v_f: Mean of forecast ensemble (B, 1, D_state).
        mean_hv: Mean of observation forecast (B, 1, d_obs).
        sigma_y: Observation noise std.
        infl: Multiplicative inflation factor (applied post-analysis).
        loc_radius: Localization radius.

    Outputs:
        ens_v_a: Analyzed ensemble (B, N_ens, D_state).
        loc_nn_output: Localization mask from NN (if applicable, else None).
    """
    if sigma_y is None:
        sigma_y = args.sigma_y
        
    if isinstance(sigma_y, torch.Tensor):
        sigma_y = sigma_y.detach().to(args.device)
    else:
        sigma_y = torch.tensor(sigma_y, dtype=torch.float32).to(args.device)
    
    model, infl_model, local_model, st_model1, st_model2 = model_list
    B, N_ens, D_state = ens_v_f.shape
    d_obs = hv.shape[2]
    
    # Handle sigma_y shape
    if sigma_y.ndim == 0:
        sigma_y = sigma_y.expand(B).view(B, 1, 1)
    else:
        sigma_y = sigma_y.view(B, 1, 1)
    
    loc_nn_output = None 

    if args.v == 'CorrTerms':
        nn_input, local_nn_input, infl_nn_input = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        
        nn_output = model(nn_input).view(B, N_ens, -1)
        infl_output = infl_model(infl_nn_input).view(B, N_ens, -1)
        
        if hasattr(args, 'zero_infl') and args.zero_infl:
            Vnn1 = ens_v_f
        else:
            Vnn1 = ens_v_f + infl_output
        Vnn2 = ens_v_f - mean_ens_v_f + nn_output[:, :, :D_state]
        Ynn = hv - mean_hv + nn_output[:, :, D_state:]

        R_cov = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(d_obs, device=args.device).unsqueeze(0).repeat(B, 1, 1)
        
        can_use_loc = (
            (not args.no_localization)
            and args.diff_dist is not None
            and args.Lvy is not None
            and args.Lyy is not None
            and getattr(args, 'num_dist', 0) > 0
        )

        if not can_use_loc:
            K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) 
            K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R_cov * (N_ens - 1)
        else:
            loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * args.loc_max_val
            loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
            loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
            
            K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
            K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R_cov * (N_ens - 1)
        
        K = torch.bmm(K1, torch.inverse(K2))
        current_analyzed_ens_v_a = Vnn1 + torch.bmm(ens_i_innov, K.transpose(1, 2))
    
    elif args.v == 'EtE':
        nn_input, _, _ = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        nn_output = model(nn_input).view(B, N_ens, -1)
        current_analyzed_ens_v_a = nn_output
        
    elif args.v == 'EtE-LRes':
        nn_input, _, _ = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        nn_output = model(nn_input).view(B, N_ens, -1)
        current_analyzed_ens_v_a = ens_v_f + nn_output
        
    elif args.v == 'EtE2':
        nn_input, y, _ = _get_model_inputs(
            args, st_model1, st_model2, ens_v_f, hv, obs_y
        )
        nn_output = model(nn_input, y).view(B, N_ens, -1)
        current_analyzed_ens_v_a = nn_output
        
    elif args.v == 'EnKF':
        Vnn1 = ens_v_f
        Vnn2 = ens_v_f - mean_ens_v_f
        Ynn = hv - mean_hv

        R_cov = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(d_obs, device=args.device).unsqueeze(0).repeat(B, 1, 1)
        
        K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) 
        K2_ens = torch.bmm(Ynn.transpose(1, 2), Ynn)
        
        # Apply Covariance Localization
        if loc_radius is not None and loc_radius > 0:
            if args.Lvy is None or args.Lyy is None:
                raise ValueError("EnKF localization requires pre-computed args.Lvy and args.Lyy")

            rho_vy = dist2coeff(args.Lvy, loc_radius, tag='GC')
            rho_yy = dist2coeff(args.Lyy, loc_radius, tag='GC')
            
            K1 = K1 * rho_vy.unsqueeze(0)
            K2_ens = K2_ens * rho_yy.unsqueeze(0)

        K2 = K2_ens + R_cov * (N_ens - 1)
        K = torch.bmm(K1, torch.inverse(K2))
        current_analyzed_ens_v_a = Vnn1 + torch.bmm(ens_i_innov, K.transpose(1, 2))
        
        # Apply Post-Analysis Inflation
        if infl != 1.0:
            ens_mean = current_analyzed_ens_v_a.mean(dim=1, keepdim=True)
            current_analyzed_ens_v_a = ens_mean + infl * (current_analyzed_ens_v_a - ens_mean)

    elif args.v == 'LETKF':
        if args.Lvy is None:
            raise ValueError("LETKF requires args.Lvy for localization distances.")

        # Prepare Anomalies (B, N_ens, D)
        X_b = ens_v_f - mean_ens_v_f
        Y_b = hv - mean_hv
        
        # Innovation (Mean only, no perturbations for LETKF usually)
        # d_mean: (B, 1, D_obs)
        if mean_hv.shape[1] != 1:
            mean_hv_for_innov = mean_hv.mean(dim=1, keepdim=True)
        else:
            mean_hv_for_innov = mean_hv
        d_mean = obs_y.view(B, 1, d_obs) - mean_hv_for_innov

        # To store the analyzed state for each grid point
        analyzed_cols = []
        
        # Pre-compute Identity matrix
        eye_N = torch.eye(N_ens, device=args.device).unsqueeze(0) # (1, N, N)

        # Iterate over each state dimension (Grid Point)
        # Using args.ori_dim or D_state (they should be consistent here)
        for k in range(D_state):
            # 1. Identify Local Observations
            # args.Lvy[k] is distance from state k to all obs
            dists = args.Lvy[k] # Shape (d_obs,)
            local_obs_idx = torch.where(dists <= loc_radius)[0]
            
            # If no local observations, analysis = forecast
            if len(local_obs_idx) == 0:
                analyzed_cols.append(ens_v_f[:, :, k:k+1])
                continue
            
            # 2. Extract Local Quantities
            # Y_loc: (B, N_ens, n_loc)
            Y_loc = Y_b[:, :, local_obs_idx]
            
            # d_loc: (B, 1, n_loc)
            d_loc = d_mean[:, :, local_obs_idx]
            
            # Normalize by R (assuming diagonal sigma)
            # sigma_y is (B, 1, 1), broadcast to (B, 1, n_loc)
            sigma_loc = sigma_y.view(B, 1, 1)
            Y_loc_norm = Y_loc / sigma_loc
            d_loc_norm = d_loc / sigma_loc
            
            # 3. Compute Transform Matrix T
            # Hessian H = (N-1)I + Y_loc_norm @ Y_loc_norm.T
            # Shape: (B, N_ens, N_ens)
            YRY = torch.bmm(Y_loc_norm, Y_loc_norm.transpose(1, 2))
            H = (N_ens - 1) * eye_N + YRY
            
            # Pa = H^-1 (Analysis Covariance in Ensemble Space)
            # Use cholesky or standard inverse. Inverse is safer for general batch.
            Pa = torch.inverse(H)
            
            # 4. Compute Weights
            # Mean weights w_bar: (B, N_ens, 1)
            # w_bar = Pa @ Y_loc_norm @ d_loc_norm.T
            w_bar = torch.bmm(torch.bmm(Pa, Y_loc_norm), d_loc_norm.transpose(1, 2))
            
            # Perturbation weights W_a: (B, N_ens, N_ens)
            # W_a = sqrt(N-1) * Pa^(1/2)
            # Using Eigendecomposition for square root: Pa = V L V^T -> Pa^0.5 = V L^0.5 V^T
            L, V = torch.linalg.eigh(Pa)
            L = torch.clamp(L, min=1e-10) # Numerical stability
            Pa_sqrt = torch.bmm(V, torch.diag_embed(torch.sqrt(L)))
            Pa_sqrt = torch.bmm(Pa_sqrt, V.transpose(1, 2))
            
            W_a = math.sqrt(N_ens - 1) * Pa_sqrt
            
            # 5. Update State
            # T = w_bar + W_a (Broadcasting w_bar to all columns)
            # x_a = x_f_mean + X_b @ T
            # Note: X_b is (B, N_ens, D_state). We need local X_b_k: (B, N_ens, 1)
            # In ensemble space logic:
            # X_a_k = X_b_k.T @ (w_bar + W_a) -> Result (B, 1, N_ens) then transpose back?
            # Let's write explicitly:
            #   Analyzed Ensemble = Mean + Perturbation
            #   Ens_A = (mean_f + X_b^T w_bar) + (X_b^T W_a)
            #   X_b_k: (B, N_ens, 1) -> X_b_k^T: (B, 1, N_ens)
            
            X_b_k_T = X_b[:, :, k:k+1].transpose(1, 2) # (B, 1, N_ens)
            
            # Transform matrix T_total: (B, N_ens, N_ens)
            # Adding w_bar to every column of W_a effectively adds the mean update to every member
            T_total = W_a + w_bar 
            
            # Apply transform
            # (B, 1, N_ens) @ (B, N_ens, N_ens) -> (B, 1, N_ens) -> transpose -> (B, N_ens, 1)
            update_k = torch.bmm(X_b_k_T, T_total).transpose(1, 2)
            
            analyzed_cols.append(mean_ens_v_f[:, :, k:k+1] + update_k)
            
        current_analyzed_ens_v_a = torch.cat(analyzed_cols, dim=2)
        
        # Apply Post-Analysis Inflation (Consistent with EnKF branch)
        if infl != 1.0:
            ens_mean = current_analyzed_ens_v_a.mean(dim=1, keepdim=True)
            current_analyzed_ens_v_a = ens_mean + infl * (current_analyzed_ens_v_a - ens_mean)

    else:
        raise NotImplementedError(f"args.v = {args.v} is not implemented.")

    if args.clamp is not None:
        ens_v_a = torch.clamp(current_analyzed_ens_v_a, min=-args.clamp, max=args.clamp)
    else:
        ens_v_a = current_analyzed_ens_v_a
    
    return ens_v_a, loc_nn_output

### compute likelihoods for weight particle filter training
def compute_likelihood_weights(hv, obs_y, sigma_y):
    """
    Computes normalized importance weights w ~ p(y|x) for the forecast ensemble.
    
    Args:
        hv (torch.Tensor): H(x_forecast). Shape: [B, N, D_obs]
        obs_y (torch.Tensor): Observation y. Shape: [B, D_obs]
        sigma_y (float): Observation noise standard deviation.
        
    Returns:
        torch.Tensor: Normalized weights. Shape: [B, N]
    """
    # Ensure obs_y matches particle dimension for broadcasting: [B, 1, D_obs]
    # obs_y_expanded = obs_y.unsqueeze(1)
    
    # Compute squared error summed over observation dimensions: ||y - H(x)||^2
    # Shape: [B, N]
    sq_error = torch.sum((hv - obs_y)**2, dim=2)
    
    # Calculate Log-Weights: -0.5 * error / sigma^2
    # Adding a small epsilon to sigma is good practice but args.sigma_y implies valid std.
    log_w = -0.5 * sq_error / (sigma_y ** 2)
    
    # Stable Softmax over the particle dimension (dim=1) to get normalized weights
    w = torch.softmax(log_w, dim=1)
    
    return w

###############################################################

def train_model(epoch, loader, model_list, optimizer, scheduler, args, H_info=None):
    """
    Function to train the model for one epoch with support for WPF, NLL, and Pre/Post analysis losses.
    """
    model, infl_model, local_model, st_model1, st_model2 = model_list
    m = args.N
    losses = AverageMeter()
    batch_time = AverageMeter()

    # --- Loss Pre-processing & WPF Guard ---
    pre_losses, post_losses = [], []
    wpf_names = ['wpf_ed', 'wpf_fmmd', 'wpf_ammd', 'wpf_st_ed', 'wpf_st_fmmd', 'wpf_st_ammd']
    for lt, lw in zip(args.loss_type, args.loss_weights):
        if lt.startswith('pre_'):
            base = lt[4:]
            if any(wn in base for wn in wpf_names):
                raise ValueError(f"WPF losses ({base}) are not supported in pre-analysis mode.")
            pre_losses.append({'name': base, 'weight': lw})
        else:
            base = lt[5:] if lt.startswith('post_') else lt
            post_losses.append({'name': base, 'weight': lw})

    need_pre, need_post = len(pre_losses) > 0, len(post_losses) > 0
    is_wpf_mode = any(any(wn in l['name'] for wn in wpf_names) for l in pre_losses + post_losses)
    is_nll_mode = any('nll' in l['name'] for l in pre_losses + post_losses)

    # --- Forward Function Selection ---
    forward_fun = _get_forward_fun(args)
    
    if H_info is None: H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else: H_fun, H = H_info
    _setup_mixed_precision(args)

    # --- Set Models to Train Mode ---
    model.train()
    if hasattr(infl_model, 'train'): infl_model.train()
    if hasattr(local_model, 'train'): local_model.train()
    if hasattr(st_model1, 'train'): st_model1.train()
    if hasattr(st_model2, 'train'): st_model2.train()

    all_trainable_params = []
    for m_ in [model, infl_model, local_model, st_model1, st_model2]:
        if hasattr(m_, 'parameters') and not isinstance(m_, NaiveNetwork):
            all_trainable_params.extend(list(filter(lambda p: p.requires_grad, m_.parameters())))

    num_batches_all_nan = 0
    
    for batch_ind, batch_v_trajectory in enumerate(loader):
        t_start = time.time()
        batch_v_trajectory = batch_v_trajectory.to(device=args.device)
        current_actual_batch_size = batch_v_trajectory.shape[1]
        optimizer.zero_grad()
        
        with _autocast_context(args):
            ens_v_a = batch_v_trajectory[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens
            
            end_ind_t = min(epoch + 1, len(batch_v_trajectory) - 1) if args.loss_warm_up else len(batch_v_trajectory) - 1
            if end_ind_t <= 0:
                if (batch_ind + 1) % args.print_batch == 0:
                    print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\tSkipped batch due to end_ind_t <=0')
                batch_time.update(time.time() - t_start)
                continue
            
            accumulated_loss_for_batch_load = 0.0
            num_valid_loss_contributions = 0
            running_valid_count_t = torch.zeros((), device=args.device, dtype=torch.long) if args.running_loss else None
            
            # Trajectory storage
            traj_cache = { 'ens_f': [], 'ens_a': [], 'target_ens': [], 'target_weights': [], 'true_obs': [] }

            for i in range(end_ind_t):
                # --- Forecast Step ---
                obs_y = H_fun(batch_v_trajectory[i + 1].unsqueeze(1))
                obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)
                
                ens_v_f = _forecast_ensemble(args, ens_v_a, i, forward_fun)
                hv = H_fun(ens_v_f)
                
                curr_lik_w = compute_likelihood_weights(hv, obs_y, args.sigma_y) if is_wpf_mode else None

                # --- Analysis Step ---
                r_noise = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
                curr_v_a, _ = _process_analysis_step(
                    args, model_list, ens_v_f, hv, obs_y, 
                    obs_y - hv - r_noise, torch.mean(ens_v_f, dim=1, keepdim=True), torch.mean(hv, dim=1, keepdim=True)
                )

                if args.running_loss:
                    if (i + 1) > args.ignore_first:
                        add_in = {}
                        if is_wpf_mode: add_in.update({'target_ens': ens_v_f.unsqueeze(0), 'target_weights': curr_lik_w.unsqueeze(0), 'sigma': getattr(args, 'kes_sigma', 1.0)})
                        if is_nll_mode: add_in.update({'obs_map': H_fun, 'sigma_y': args.sigma_y, 'true_obs': obs_y.unsqueeze(0)})

                        for mode_cfg, ens_curr in [(pre_losses, ens_v_f), (post_losses, curr_v_a)]:
                            if not mode_cfg: continue
                            ens_ts = ens_curr.unsqueeze(0)
                            mask = ~torch.isnan(ens_ts).any(dim=(0, 2, 3)).squeeze(0)
                            if mask.any():
                                step_loss_sum = sum(l['weight'] * compute_loss(
                                    ens_tensor=ens_ts, batch_v=batch_v_trajectory[i+1].unsqueeze(0),
                                    loss_type=l['name'], ignore_first=0, end_ind=None,
                                    valid_B_mask=mask.unsqueeze(0), norm_p=args.es_p, 
                                    kes_sigma=args.kes_sigma, return_sum=True, additional_inputs=add_in
                                ) for l in mode_cfg)
                                accumulated_loss_for_batch_load += step_loss_sum
                                running_valid_count_t += torch.sum(mask)
                else:
                    if need_pre: traj_cache['ens_f'].append(ens_v_f)
                    if need_post: traj_cache['ens_a'].append(curr_v_a)
                    if is_wpf_mode:
                        traj_cache['target_ens'].append(ens_v_f)
                        traj_cache['target_weights'].append(curr_lik_w)
                    if is_nll_mode: traj_cache['true_obs'].append(obs_y.squeeze(1))

                ens_v_a = curr_v_a
                if epoch <= args.detach_training_epoch and args.detach_steps > 0 and (i + 1) % args.detach_steps == 0:
                    ens_v_a = ens_v_a.detach()

            # --- Trajectory Backprop ---
            if not args.running_loss and len(traj_cache['ens_f'] if need_pre else traj_cache['ens_a']) > args.ignore_first:
                batch_v = batch_v_trajectory[1:end_ind_t + 1][args.ignore_first:]
                add_in_traj = {}
                if is_wpf_mode: add_in_traj.update({'target_ens': torch.stack(traj_cache['target_ens'], dim=0)[args.ignore_first:], 'target_weights': torch.stack(traj_cache['target_weights'], dim=0)[args.ignore_first:], 'sigma': getattr(args, 'kes_sigma', 1.0)})
                if is_nll_mode: add_in_traj.update({'obs_map': H_fun, 'sigma_y': args.sigma_y, 'true_obs': torch.stack(traj_cache['true_obs'], dim=0)[args.ignore_first:]})

                for mode_cfg, key in [(pre_losses, 'ens_f'), (post_losses, 'ens_a')]:
                    if not mode_cfg: continue
                    ens_ts = torch.stack(traj_cache[key], dim=0)[args.ignore_first:]
                    mask = ~torch.isnan(ens_ts).any(dim=(2, 3))
                    if mask.any():
                        t_loss = sum(l['weight'] * compute_loss(ens_tensor=ens_ts, batch_v=batch_v, loss_type=l['name'], ignore_first=0, end_ind=None, valid_B_mask=mask, norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True, additional_inputs=add_in_traj) for l in mode_cfg)
                        accumulated_loss_for_batch_load += t_loss
                        num_v = torch.sum(mask).item()
                        num_valid_loss_contributions += num_v
                        losses.update(t_loss.item() / num_v, num_v)
            elif args.running_loss:
                num_valid_loss_contributions = int(running_valid_count_t.item())
                if num_valid_loss_contributions > 0:
                    running_avg_loss = (
                        accumulated_loss_for_batch_load / running_valid_count_t.to(accumulated_loss_for_batch_load.dtype)
                    ).detach()
                    losses.update(running_avg_loss.item(), num_valid_loss_contributions)

        if num_valid_loss_contributions > 0:
            _backward_and_step(
                accumulated_loss_for_batch_load / num_valid_loss_contributions,
                optimizer,
                all_trainable_params,
                args
            )
        else: num_batches_all_nan += 1

        batch_time.update(time.time() - t_start)
        total_possible = current_actual_batch_size * max(0, end_ind_t - args.ignore_first)
        no_nan_perc = (num_valid_loss_contributions / total_possible * 100) if total_possible > 0 else 100.0

        if (batch_ind + 1) % args.print_batch == 0:
            print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                  f'Batch time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} (Avg: {losses.avg:.3f})\t'
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}\tNo NAN %: {no_nan_perc:.2f}%')

    if num_batches_all_nan == len(loader) and len(loader) > 0:
        print(f"Warning: All batches in epoch {epoch} resulted in NaN.")
    scheduler.step()
    return losses.avg if losses.count > 0 else float('nan')

def generate_and_cache_pf_results(
    loader,
    args,
    H_info,
    check_disk: bool = True,
    calculate_crps: bool = True,
    save_figure: bool = False,
):
    """
    Runs a particle filter, saves results (means, covariances) to a cache file,
    and computes performance metrics (RMSE, optional CRPS).

    NEW IN THIS VERSION:
    - Records BOTH prior (forecast) and posterior (analysis) ensemble statistics in the cache:
        {'prior_means', 'prior_covs', 'post_means', 'post_covs'}
    - For 3D PF datasets, saves combined prior/posterior projection plots
      (adaptive + fixed ranges, including a fixed-range 3D scatter).
    - For ring datasets ('doubling1d', 'complex2d'), it also saves no-history counterparts.

    If calculate_crps is False, CRPS keys are omitted from the output.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing dataset batches (shape: T x B x D).
        args (argparse.Namespace): Script arguments (expects fields used below).
        H_info (tuple): (H_fun, H) observation operator function and matrix; if None, uses mystery_operator.
        check_disk (bool): If True, checks cache file existence and returns NaNs (metrics) if present.
        calculate_crps (bool): If True, calculates CRPS and RCRPS metrics.
        save_figure (bool): If True, saves prior/posterior figures at selected steps.

    Returns:
        dict: A dictionary containing performance metrics. CRPS-related keys exist only if calculate_crps=True.
    """

    # --- Model and Observation Initialization ---
    forward_fun = _get_forward_fun(args)
    test_noise_gen = get_test_noise_generator(args)

    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info

    # --- Cache Filepath Generation ---
    first_batch_for_shape = next(iter(loader))
    traj_len = first_batch_for_shape.shape[0]
    batch_size = first_batch_for_shape.shape[1]
    cache_dir = _build_pf_cache_dir(args)
    obs_suffix = _pf_obs_fn_suffix(args)
    cache_filename = _build_pf_cache_filename(args, batch_size=batch_size, traj_len=traj_len, avg=False)
    cache_filepath = os.path.join(cache_dir, cache_filename)
    user_suffix = str(getattr(args, "suffix", "") or "").strip()
    if user_suffix and not user_suffix.startswith("_"):
        user_suffix = f"_{user_suffix}"
    pf_vis_folder = f"save/{args.dataset}_pf_vis{obs_suffix}{user_suffix}"
    if obs_suffix:
        print(
            f"[PF cache naming] dataset={args.dataset}, obs_fn={_resolve_obs_fn_for_pf_paths(args)}, "
            f"default_obs_fn={_get_dataset_default_obs_fn(args.dataset)}, suffix='{obs_suffix}'"
        )

    # --- Check for Existing Cache ---
    if check_disk and os.path.exists(cache_filepath):
        print(f"Particle filter results already exist at: {cache_filepath}")
        metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rrmse', 'std_rrmse']
        if calculate_crps:
            metrics_keys.extend(['mean_crps', 'std_crps', 'mean_rcrps', 'std_rcrps'])
        return {key: float('nan') for key in metrics_keys}

    print(f"Generating particle filter results and saving to: {cache_filepath}")
    all_pf_results_to_cache = []
    pf_non_gaussian_records = []
    distance_dir = pf_vis_folder
    distance_prefix = (
        f"{distance_dir}/sigma_y{args.sigma_y}_batch{batch_size}_len{traj_len}_pfN{args.pf_N}_{args.seed}"
        "_non_gaussian_distance"
    )
    detailed_pt = f"{distance_prefix}_detail.pt"
    detailed_csv = f"{distance_prefix}_detail.csv"
    summary_csv = f"{distance_prefix}_per_step_mean.csv"

    # --- Initialize Metric Dictionaries ---
    all_pf_metrics = {
        'rmse': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
    }
    if calculate_crps:
        all_pf_metrics['crps'] = torch.empty(0, device=args.device)
        all_pf_metrics['rcrps'] = torch.empty(0, device=args.device)

    # --- Distribution summaries to cache ---
    num_quantiles = int(getattr(args, "pf_num_quantiles", 257))
    if num_quantiles < 2:
        raise ValueError(f"pf_num_quantiles must be >= 2, got {num_quantiles}.")
    q_lo = float(getattr(args, "pf_range_q_lo", 0.01))
    q_hi = float(getattr(args, "pf_range_q_hi", 0.99))
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError(f"Invalid PF range quantiles: q_lo={q_lo}, q_hi={q_hi}.")
    range_pad_int = int(getattr(args, "pf_range_pad_int", 5))
    quantile_probs = torch.linspace(
        0.0, 1.0, steps=num_quantiles, device=args.device, dtype=torch.float32
    )
    range_probs = torch.tensor([q_lo, q_hi], device=args.device, dtype=torch.float32)

    def _compute_pf_distribution_summary(ens_tensor: torch.Tensor):
        """Compute per-step per-trajectory per-dimension summary statistics for PF clouds."""
        ens32 = ens_tensor.to(torch.float32)
        B, Np, D = ens32.shape
        qvals = torch.quantile(ens32, quantile_probs, dim=1).permute(1, 2, 0).contiguous()  # [B, D, K]
        q01_q99 = torch.quantile(ens32, range_probs, dim=1).permute(1, 2, 0).contiguous()   # [B, D, 2]
        min_vals = ens32.amin(dim=1)
        max_vals = ens32.amax(dim=1)
        minmax = torch.stack((min_vals, max_vals), dim=-1)  # [B, D, 2]

        low_int = torch.floor(q01_q99[..., 0]) - range_pad_int
        high_int = torch.ceil(q01_q99[..., 1]) + range_pad_int
        range_int = torch.stack((low_int, high_int), dim=-1).to(torch.int32)  # [B, D, 2]

        mean_vals = ens32.mean(dim=1, keepdim=True)
        centered = ens32 - mean_vals
        std_vals = ens32.std(dim=1, unbiased=False)
        std_safe = torch.clamp(std_vals, min=1e-8)
        m3 = torch.mean(centered ** 3, dim=1)
        m4 = torch.mean(centered ** 4, dim=1)
        skewness = m3 / (std_safe ** 3)
        kurtosis_excess = m4 / (std_safe ** 4) - 3.0

        cov = torch.bmm(centered.transpose(1, 2), centered) / max(1, Np - 1)  # [B, D, D]
        eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float32))  # ascending
        eigvals_desc = torch.flip(eigvals, dims=(-1,))
        eigvecs_desc = torch.flip(eigvecs, dims=(-1,))
        pcs = torch.bmm(centered, eigvecs_desc)  # [B, N, D]
        pca3 = pcs[:, :, :min(3, D)]
        pca_quantiles = torch.full(
            (B, 3, quantile_probs.shape[0]),
            float("nan"),
            device=ens32.device,
            dtype=torch.float32,
        )
        if pca3.shape[-1] > 0:
            pca_q = torch.quantile(pca3, quantile_probs, dim=1).permute(1, 2, 0).contiguous()  # [B, k, K]
            pca_quantiles[:, :pca3.shape[-1], :] = pca_q

        return {
            "quantiles": qvals,
            "range_q01_q99": q01_q99,
            "minmax": minmax,
            "range_int": range_int,
            "skewness": skewness,
            "kurtosis_excess": kurtosis_excess,
            "pca_eigvals": eigvals_desc,
            "pca_quantiles": pca_quantiles,
        }

    # --- Which batch indices to visualize when saving figures ---
    # Controlled by args.test_plot_index (default [0], also supports "adaptive" and lists like [0,1]).
    can_plot_3d = int(getattr(args, 'ori_dim', 0)) >= 3
    can_plot_ring = int(getattr(args, 'ori_dim', 0)) in [1, 2]
    track_pf_non_gaussian = (
        str(getattr(args, "dataset", "")).lower() == "lorenz63"
        and bool(getattr(args, "pf_swd", False))
    )
    plot_mode, requested_global_indices = _normalize_global_plot_indices(args)
    unresolved_global_indices = set(requested_global_indices)
    batch_start_index = 0
    pf_ring_cdf_datasets = {"doubling1d", "complex2d"}
    pf_plot_ring_cdf = getattr(args, "dataset", None) in pf_ring_cdf_datasets
    pf_plot_ring_dual_history = getattr(args, "dataset", None) in pf_ring_cdf_datasets
    if save_figure and can_plot_ring:
        print(f"[INFO] Using ring-mapped PF visualization for dataset='{args.dataset}' (ori_dim={args.ori_dim}).")
    if save_figure and (not can_plot_3d and not can_plot_ring):
        print(
            f"[INFO] Skipping PF visualization for dataset='{args.dataset}' "
            f"because ori_dim={args.ori_dim} is unsupported for current plotters."
        )

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)  # shape: (T, B, D)
            batch_nan_counts = torch.isnan(batch_v).any(dim=2).sum(dim=0)  # [B]
            if plot_mode == "adaptive":
                vis_indices = _resolve_plot_batch_indices(
                    args=args,
                    batch_size=batch_v.shape[1],
                    nan_counts=batch_nan_counts,
                )
            else:
                vis_indices = _resolve_global_plot_indices_for_batch(
                    global_indices=requested_global_indices,
                    batch_start_index=batch_start_index,
                    batch_size=batch_v.shape[1],
                )
                for lidx in vis_indices:
                    unresolved_global_indices.discard(batch_start_index + lidx)

            # --- Particle Filter Initialization ---
            pf_ens_v_a = batch_v[0].unsqueeze(1).repeat(1, args.pf_N, 1)  # (B, Np, D)
            pf_ens_v_a += torch.randn_like(pf_ens_v_a, device=args.device) * args.sigma_ens
            if args.dataset == "complex2d":
                pf_ens_v_a = project_to_unit_circle(pf_ens_v_a)

            # These will be stacked and cached per-batch
            batch_prior_means_to_cache, batch_prior_covs_to_cache = [], []
            batch_post_means_to_cache,  batch_post_covs_to_cache  = [], []
            batch_prior_quantiles_to_cache, batch_post_quantiles_to_cache = [], []
            batch_prior_q01_q99_to_cache, batch_post_q01_q99_to_cache = [], []
            batch_prior_minmax_to_cache, batch_post_minmax_to_cache = [], []
            batch_prior_range_int_to_cache, batch_post_range_int_to_cache = [], []
            batch_prior_skew_to_cache, batch_post_skew_to_cache = [], []
            batch_prior_kurt_to_cache, batch_post_kurt_to_cache = [], []
            batch_prior_pca_eigvals_to_cache, batch_post_pca_eigvals_to_cache = [], []
            batch_prior_pca_quantiles_to_cache, batch_post_pca_quantiles_to_cache = [], []
            batch_post_ess_to_cache = []
            batch_post_weight_entropy_to_cache = []
            batch_post_weight_abundance_to_cache = []

            batch_rmse_steps = []
            if calculate_crps:
                batch_crps_steps = []

            # --- Metrics at t=0 (posterior at initialization) ---
            true_state_t0 = batch_v[0]  # (B, D)
            rmse_t0 = torch.sqrt(torch.mean((pf_ens_v_a.mean(dim=1) - true_state_t0) ** 2, dim=1))
            batch_rmse_steps.append(rmse_t0)
            if calculate_crps:
                crps_t0 = compute_es(pf_ens_v_a.unsqueeze(0), true_state_t0.unsqueeze(0), norm_p=1)
                batch_crps_steps.append(crps_t0)

            # --- Generate Observations y_t ---
            obs_y_list = []
            for i in range(len(batch_v)):
                clean_h_step = H_fun(batch_v[i].unsqueeze(1))
                obs_noise_step = _sample_true_obs_noise_like(clean_h_step, test_noise_gen)
                obs_y_step = clean_h_step + args.sigma_y * obs_noise_step
                obs_y_list.append(obs_y_step)

            # --- Main PF Assimilation Loop ---
            for i in range(len(batch_v) - 1):
                # -------- Forecast (PRIOR) step --------
                pf_ens_v_f = _forecast_ensemble(args, pf_ens_v_a, i, forward_fun)

                # Cache PRIOR stats (means/covs)
                prior_mean = torch.mean(pf_ens_v_f, dim=1)                     # (B, D)
                prior_cov  = get_ens_cov(pf_ens_v_f)                           # (B, D, D)
                batch_prior_means_to_cache.append(prior_mean)
                batch_prior_covs_to_cache.append(prior_cov)
                prior_summary = _compute_pf_distribution_summary(pf_ens_v_f)
                batch_prior_quantiles_to_cache.append(prior_summary["quantiles"])
                batch_prior_q01_q99_to_cache.append(prior_summary["range_q01_q99"])
                batch_prior_minmax_to_cache.append(prior_summary["minmax"])
                batch_prior_range_int_to_cache.append(prior_summary["range_int"])
                batch_prior_skew_to_cache.append(prior_summary["skewness"])
                batch_prior_kurt_to_cache.append(prior_summary["kurtosis_excess"])
                batch_prior_pca_eigvals_to_cache.append(prior_summary["pca_eigvals"])
                batch_prior_pca_quantiles_to_cache.append(prior_summary["pca_quantiles"])

                # Prepare true state for metrics/visualization.
                true_state_ti1 = batch_v[i + 1]  # (B, D)

                # -------- Analysis (POSTERIOR) step --------
                pf_ens_v_a, pf_diag = bootstrap_particle_filter_analysis(
                    pf_ens_v_f,
                    obs_y_list[i + 1].squeeze(1),  # (B, obs_dim)
                    H_fun if not isinstance(H, torch.Tensor) else H.transpose(1, 0),
                    args.sigma_y,
                    resampling_method="multinomial",
                    sigma_reg=args.sigma_reg,
                    max_chunk_size=1000000,
                    resample_on_cpu=False,
                    return_diagnostics=True,
                )
                pf_ens_v_a = torch.clamp(pf_ens_v_a, min=-args.clamp, max=args.clamp)

                # Cache POSTERIOR stats (means/covs)
                post_mean = torch.mean(pf_ens_v_a, dim=1)                      # (B, D)
                post_cov  = get_ens_cov(pf_ens_v_a)                            # (B, D, D)
                batch_post_means_to_cache.append(post_mean)
                batch_post_covs_to_cache.append(post_cov)
                post_summary = _compute_pf_distribution_summary(pf_ens_v_a)
                batch_post_quantiles_to_cache.append(post_summary["quantiles"])
                batch_post_q01_q99_to_cache.append(post_summary["range_q01_q99"])
                batch_post_minmax_to_cache.append(post_summary["minmax"])
                batch_post_range_int_to_cache.append(post_summary["range_int"])
                batch_post_skew_to_cache.append(post_summary["skewness"])
                batch_post_kurt_to_cache.append(post_summary["kurtosis_excess"])
                batch_post_pca_eigvals_to_cache.append(post_summary["pca_eigvals"])
                batch_post_pca_quantiles_to_cache.append(post_summary["pca_quantiles"])
                batch_post_ess_to_cache.append(pf_diag["ess"])
                batch_post_weight_entropy_to_cache.append(pf_diag["weight_entropy"])
                batch_post_weight_abundance_to_cache.append(pf_diag["weight_abundance"])

                # Metrics at t=i+1 using POSTERIOR mean
                rmse_ti = torch.sqrt(torch.mean((post_mean - true_state_ti1) ** 2, dim=1))
                batch_rmse_steps.append(rmse_ti)
                if calculate_crps:
                    crps_ti = compute_es(pf_ens_v_a.unsqueeze(0), true_state_ti1.unsqueeze(0), norm_p=1)
                    batch_crps_steps.append(crps_ti)

                # -------- Visualization (PRIOR + POSTERIOR), both include observation --------
                if i < 600 and save_figure and (can_plot_3d or can_plot_ring):
                    save_folder = pf_vis_folder
                    os.makedirs(save_folder, exist_ok=True)

                    # Use the same time step prefix, but suffix per batch index and type
                    base_prefix = (
                        f'{save_folder}/sigma_y{args.sigma_y}_batch{batch_size}_len{traj_len}_pfN{args.pf_N}'
                        f'_timestep{i+1}_{args.seed}'
                    )

                    # We plot per selected batch index so each figure gets the correct observation vector
                    for bidx in vis_indices:
                        gidx = batch_start_index + bidx
                        if can_plot_3d:
                            # Prepare per-item tensors for the plotting helper (shape: (1, Np, D>=3))
                            prior_cloud_for_plot     = pf_ens_v_f[bidx:bidx+1, :, :].detach().cpu()
                            posterior_cloud_for_plot = pf_ens_v_a[bidx:bidx+1, :, :].detach().cpu()

                            # History trajectory for this item up to current step (shape: steps x 1 x D>=3)
                            hist_traj = batch_v[1:i+2, bidx:bidx+1, :].detach().cpu()

                            # Prior + posterior in one set of projection plots.
                            prefix_pair = f"{base_prefix}_g{gidx}"
                            swd_records = plot_and_test_point_clouds(
                                args,
                                prior_tensor=prior_cloud_for_plot,
                                posterior_tensor=posterior_cloud_for_plot,
                                num_samples_plot=1000000,
                                prefix=prefix_pair,
                                num_swd_reference_samples=getattr(args, "pf_plot_swd_samples", 1000000),
                                num_swd_directions=getattr(args, "pf_plot_swd_directions", 50),
                                plot_indices=[0],
                                history_traj=hist_traj,
                                true_state=true_state_ti1[bidx].detach().cpu(),
                                legend_in_figure=getattr(args, "legend_in_figure", False),
                                prior_range_int=prior_summary["range_int"][bidx].detach().cpu(),
                                post_range_int=post_summary["range_int"][bidx].detach().cpu(),
                                quantile_probs=quantile_probs.detach().cpu(),
                                prior_quantiles=prior_summary["quantiles"][bidx].detach().cpu()[:3],
                                post_quantiles=post_summary["quantiles"][bidx].detach().cpu()[:3],
                                prior_pca_quantiles=prior_summary["pca_quantiles"][bidx].detach().cpu(),
                                post_pca_quantiles=post_summary["pca_quantiles"][bidx].detach().cpu(),
                            )
                            if track_pf_non_gaussian:
                                for record in swd_records:
                                    pf_non_gaussian_records.append(
                                        {
                                            "batch_index": int(batch_ind),
                                            "global_index": int(gidx),
                                            "local_index": int(bidx),
                                            "step": int(i + 1),
                                            "prefix": str(prefix_pair),
                                            "cloud_index": int(record.get("cloud_index", 0)),
                                            "prior_swd_ratio": float(record.get("prior_swd_ratio", float("nan"))),
                                            "prior_swd_data": float(record.get("prior_swd_data", float("nan"))),
                                            "prior_swd_baseline": float(record.get("prior_swd_baseline", float("nan"))),
                                            "post_swd_ratio": float(record.get("post_swd_ratio", float("nan"))),
                                            "post_swd_data": float(record.get("post_swd_data", float("nan"))),
                                            "post_swd_baseline": float(record.get("post_swd_baseline", float("nan"))),
                                        }
                                    )
                        elif can_plot_ring:
                            # Use full state for ring mapping (1D/2D).
                            prior_cloud_for_plot     = pf_ens_v_f[bidx:bidx+1, :, :].detach().cpu()
                            posterior_cloud_for_plot = pf_ens_v_a[bidx:bidx+1, :, :].detach().cpu()
                            hist_traj = batch_v[1:i+2, bidx:bidx+1, :].detach().cpu()
                            obs_x = obs_y_list[i + 1][bidx, 0].detach().cpu()

                            prefix_prior = f"{base_prefix}_g{gidx}_PRIOR"
                            plot_and_test_point_clouds_ring(
                                args,
                                prior_cloud_for_plot,
                                num_samples_plot=100000,
                                prefix=prefix_prior,
                                point_color="blue",
                                observation=obs_x,
                                true_state=true_state_ti1[bidx].detach().cpu(),
                                plot_history=True,
                                plot_indices=[0],
                                history_traj=hist_traj,
                                plot_cdf=pf_plot_ring_cdf,
                                legend_in_figure=getattr(args, "legend_in_figure", False),
                            )
                            if pf_plot_ring_dual_history:
                                plot_and_test_point_clouds_ring(
                                    args,
                                    prior_cloud_for_plot,
                                    num_samples_plot=100000,
                                    prefix=f"{prefix_prior}_nohist",
                                    point_color="blue",
                                    observation=obs_x,
                                    true_state=true_state_ti1[bidx].detach().cpu(),
                                    plot_history=False,
                                    plot_indices=[0],
                                    history_traj=hist_traj,
                                    plot_cdf=pf_plot_ring_cdf,
                                    legend_in_figure=getattr(args, "legend_in_figure", False),
                                )

                            prefix_post = f"{base_prefix}_g{gidx}_POST"
                            plot_and_test_point_clouds_ring(
                                args,
                                posterior_cloud_for_plot,
                                num_samples_plot=100000,
                                prefix=prefix_post,
                                point_color="red",
                                observation=obs_x,
                                true_state=true_state_ti1[bidx].detach().cpu(),
                                plot_history=True,
                                plot_indices=[0],
                                history_traj=hist_traj,
                                plot_cdf=pf_plot_ring_cdf,
                                legend_in_figure=getattr(args, "legend_in_figure", False),
                            )
                            if pf_plot_ring_dual_history:
                                plot_and_test_point_clouds_ring(
                                    args,
                                    posterior_cloud_for_plot,
                                    num_samples_plot=100000,
                                    prefix=f"{prefix_post}_nohist",
                                    point_color="red",
                                    observation=obs_x,
                                    true_state=true_state_ti1[bidx].detach().cpu(),
                                    plot_history=False,
                                    plot_indices=[0],
                                    history_traj=hist_traj,
                                    plot_cdf=pf_plot_ring_cdf,
                                    legend_in_figure=getattr(args, "legend_in_figure", False),
                                )

                    # Periodically checkpoint the accumulated SWD list every 100 steps.
                    if track_pf_non_gaussian and len(pf_non_gaussian_records) > 0 and ((i + 1) % 100 == 0):
                        os.makedirs(distance_dir, exist_ok=True)
                        torch.save(pf_non_gaussian_records, detailed_pt)
                        print(
                            "[PF non-Gaussian] Periodic list checkpoint: "
                            f"step={i + 1}, records={len(pf_non_gaussian_records)}, file={detailed_pt}"
                        )

            # --- Aggregate and Cache Batch Results ---
            all_pf_results_to_cache.append({
                'prior_means': torch.stack(batch_prior_means_to_cache),  # (T-1, B, D)
                'prior_covs':  torch.stack(batch_prior_covs_to_cache),   # (T-1, B, D, D)
                'post_means':  torch.stack(batch_post_means_to_cache),   # (T-1, B, D)
                'post_covs':   torch.stack(batch_post_covs_to_cache),    # (T-1, B, D, D)
                'quantile_probs': quantile_probs.detach().cpu(),          # (K,)
                'prior_quantiles': torch.stack(batch_prior_quantiles_to_cache),  # (T-1, B, D, K)
                'post_quantiles': torch.stack(batch_post_quantiles_to_cache),    # (T-1, B, D, K)
                'prior_range_q01_q99': torch.stack(batch_prior_q01_q99_to_cache),# (T-1, B, D, 2)
                'post_range_q01_q99': torch.stack(batch_post_q01_q99_to_cache),  # (T-1, B, D, 2)
                'prior_minmax': torch.stack(batch_prior_minmax_to_cache),         # (T-1, B, D, 2)
                'post_minmax': torch.stack(batch_post_minmax_to_cache),           # (T-1, B, D, 2)
                'prior_range_int': torch.stack(batch_prior_range_int_to_cache),   # (T-1, B, D, 2) int32
                'post_range_int': torch.stack(batch_post_range_int_to_cache),     # (T-1, B, D, 2) int32
                'prior_skewness': torch.stack(batch_prior_skew_to_cache),         # (T-1, B, D)
                'post_skewness': torch.stack(batch_post_skew_to_cache),           # (T-1, B, D)
                'prior_kurtosis_excess': torch.stack(batch_prior_kurt_to_cache),  # (T-1, B, D)
                'post_kurtosis_excess': torch.stack(batch_post_kurt_to_cache),    # (T-1, B, D)
                'prior_pca_eigvals': torch.stack(batch_prior_pca_eigvals_to_cache), # (T-1, B, D)
                'post_pca_eigvals': torch.stack(batch_post_pca_eigvals_to_cache),   # (T-1, B, D)
                'prior_pca_quantiles': torch.stack(batch_prior_pca_quantiles_to_cache), # (T-1, B, 3, K)
                'post_pca_quantiles': torch.stack(batch_post_pca_quantiles_to_cache),   # (T-1, B, 3, K)
                'post_ess': torch.stack(batch_post_ess_to_cache),                 # (T-1, B)
                'post_weight_entropy': torch.stack(batch_post_weight_entropy_to_cache), # (T-1, B)
                'post_weight_abundance': torch.stack(batch_post_weight_abundance_to_cache), # (T-1, B)
            })

            # --- Calculate and Aggregate Average Metrics for the Batch (posterior-based) ---
            rmse_val = torch.mean(torch.stack(batch_rmse_steps), dim=0)                 # (B,)
            rms_val  = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0) # (B,)
            all_pf_metrics['rmse']  = torch.cat((all_pf_metrics['rmse'],  rmse_val))
            all_pf_metrics['rrmse'] = torch.cat((all_pf_metrics['rrmse'], rmse_val / rms_val))
            if calculate_crps:
                crps_val  = torch.mean(torch.stack(batch_crps_steps), dim=0)            # (B,)
                rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
                all_pf_metrics['crps']  = torch.cat((all_pf_metrics['crps'],  crps_val))
                all_pf_metrics['rcrps'] = torch.cat((all_pf_metrics['rcrps'], rcrps_val))

            print("update results")
            batch_start_index += batch_v.shape[1]

    if save_figure and plot_mode == "global" and len(unresolved_global_indices) > 0:
        unresolved_sorted = sorted(list(unresolved_global_indices))
        print(
            f"Warning: some global test_plot_index values were not found in loader batches: "
            f"{unresolved_sorted[:10]}{'...' if len(unresolved_sorted) > 10 else ''}"
        )

    # --- Save All Results to Cache File ---
    print(f"Saving PF results to: {cache_filepath}")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(all_pf_results_to_cache, cache_filepath)

    if len(pf_non_gaussian_records) > 0:
        os.makedirs(distance_dir, exist_ok=True)

        torch.save(pf_non_gaussian_records, detailed_pt)
        csv_fields = [
            "batch_index", "global_index", "local_index", "step", "prefix", "cloud_index",
            "prior_swd_ratio", "prior_swd_data", "prior_swd_baseline",
            "post_swd_ratio", "post_swd_data", "post_swd_baseline",
        ]
        with open(detailed_csv, "w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
            writer.writeheader()
            for row in pf_non_gaussian_records:
                writer.writerow(row)

        step_groups: Dict[int, Dict[str, List[float]]] = {}
        for row in pf_non_gaussian_records:
            step = int(row["step"])
            if step not in step_groups:
                step_groups[step] = {
                    "prior_swd_ratio": [],
                    "prior_swd_data": [],
                    "prior_swd_baseline": [],
                    "post_swd_ratio": [],
                    "post_swd_data": [],
                    "post_swd_baseline": [],
                }
            for key in step_groups[step].keys():
                val = float(row.get(key, float("nan")))
                if np.isfinite(val):
                    step_groups[step][key].append(val)

        with open(summary_csv, "w", newline="") as f_summary:
            writer = csv.DictWriter(
                f_summary,
                fieldnames=[
                    "step",
                    "prior_swd_ratio_mean",
                    "prior_swd_data_mean",
                    "prior_swd_baseline_mean",
                    "post_swd_ratio_mean",
                    "post_swd_data_mean",
                    "post_swd_baseline_mean",
                    "num_records",
                ],
            )
            writer.writeheader()
            for step in sorted(step_groups.keys()):
                group = step_groups[step]
                writer.writerow(
                    {
                        "step": int(step),
                        "prior_swd_ratio_mean": float(np.mean(group["prior_swd_ratio"])) if len(group["prior_swd_ratio"]) > 0 else float("nan"),
                        "prior_swd_data_mean": float(np.mean(group["prior_swd_data"])) if len(group["prior_swd_data"]) > 0 else float("nan"),
                        "prior_swd_baseline_mean": float(np.mean(group["prior_swd_baseline"])) if len(group["prior_swd_baseline"]) > 0 else float("nan"),
                        "post_swd_ratio_mean": float(np.mean(group["post_swd_ratio"])) if len(group["post_swd_ratio"]) > 0 else float("nan"),
                        "post_swd_data_mean": float(np.mean(group["post_swd_data"])) if len(group["post_swd_data"]) > 0 else float("nan"),
                        "post_swd_baseline_mean": float(np.mean(group["post_swd_baseline"])) if len(group["post_swd_baseline"]) > 0 else float("nan"),
                        "num_records": int(len(group["post_swd_ratio"])),
                    }
                )

        print(
            "[PF non-Gaussian] Saved SWD distance records: "
            f"{detailed_pt}, {detailed_csv}, {summary_csv}"
        )

    # --- Final Metrics Calculation (posterior-based) ---
    if all_pf_metrics['rrmse'].numel() == 0:
        return {}

    valid_B_mask = ~torch.isnan(all_pf_metrics['rrmse'])
    if not valid_B_mask.any():
        return {}

    mean_rrmse, std_rrmse = get_mean_std(all_pf_metrics['rrmse'][valid_B_mask])
    mean_rmse,  std_rmse  = get_mean_std(all_pf_metrics['rmse'][valid_B_mask])

    final_metrics = {
        'mean_rrmse': mean_rrmse,
        'std_rrmse':  std_rrmse,
        'mean_rmse':  mean_rmse,
        'std_rmse':   std_rmse,
    }

    if calculate_crps:
        mean_crps,  std_crps  = get_mean_std(all_pf_metrics['crps'][valid_B_mask])
        mean_rcrps, std_rcrps = get_mean_std(all_pf_metrics['rcrps'][valid_B_mask])
        final_metrics.update({
            'mean_crps':  mean_crps,
            'std_crps':   std_crps,
            'mean_rcrps': mean_rcrps,
            'std_rcrps':  std_rcrps,
        })

    final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_pf_metrics['rrmse'].numel() * 100.0

    return final_metrics


def test_model(loader, model_list, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', save_pdf=False):
    """
    Runs the learned model-based assimilation and (optionally) compares with PF.

    NEW (timing): Records wall-clock time ONLY for the analysis step per i
    (on active trajectories), across all batches. Also records a trajectory-
    weighted variant that replicates each step duration by the number of
    active trajectories during that step. Appends mean/std (in seconds) to
    final_metrics with keys:
        - 'assim_step_time_mean', 'assim_step_time_std'
        - 'assim_step_time_mean_weighted', 'assim_step_time_std_weighted'
    """
    import os
    import time  # <-- NEW: timing
    import torch

    model, infl_model, local_model, st_model1, st_model2 = model_list
    m = args.N
    test_noise_gen = get_test_noise_generator(args)
    
    # Select forward function
    forward_fun = _get_forward_fun(args)

    # Observation operator
    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info
    
    model.eval()
    if hasattr(infl_model, 'eval'): infl_model.eval()
    if hasattr(local_model, 'eval'): local_model.eval()
    if hasattr(st_model1, 'eval'): st_model1.eval()
    if hasattr(st_model2, 'eval'): st_model2.eval()
    
    # Optional PF cache
    cached_pf_data = None
    if args.pf_verification:
        first_batch_for_shape = next(iter(loader))
        traj_len = first_batch_for_shape.shape[0]
        batch_size = first_batch_for_shape.shape[1]
        
        cache_dir = _build_pf_cache_dir(args)
        cache_filename = _build_pf_cache_filename(args, batch_size=batch_size, traj_len=traj_len, avg=True)
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"Loading cached PF results from: {cache_filepath}")
            cached_pf_data = torch.load(cache_filepath, map_location=args.device, weights_only=True)
        else:
            legacy_cache_dir = os.path.join('data', args.dataset)
            legacy_cache_filepath = os.path.join(legacy_cache_dir, cache_filename)
            if os.path.exists(legacy_cache_filepath):
                print(f"[PF cache fallback] Loading legacy cache from: {legacy_cache_filepath}")
                cached_pf_data = torch.load(legacy_cache_filepath, map_location=args.device, weights_only=True)
            else:
                raise FileNotFoundError(
                    f"Required particle filter cache file not found at: {cache_filepath} "
                    f"(legacy checked: {legacy_cache_filepath}). "
                    f"(obs_fn={_resolve_obs_fn_for_pf_paths(args)}, default_obs_fn={_get_dataset_default_obs_fn(args.dataset)}) "
                    f"Please run generate_and_cache_pf_results() first."
                )

    # Aggregated results
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
        'rmv': torch.empty(0, device=args.device),
        'spread_error_ratio': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'snr_var': torch.empty(0, device=args.device),
        'cov_diff': torch.empty(0, device=args.device),
        'rcov_diff': torch.empty(0, device=args.device),
        'pf_rmse': torch.empty(0, device=args.device),
        'pf_rrmse': torch.empty(0, device=args.device),
    }
    loc_tensor_all_batches = None

    # NEW: analysis-step timing collectors
    assim_step_times = []              # per analysis step across batches
    assim_step_times_weighted = []     # replicated by #active trajectories
    plot_mode, requested_global_indices = _normalize_global_plot_indices(args)
    unresolved_global_indices = set(requested_global_indices)
    batch_start_index = 0
    pf_style_projection_enabled = bool(
        args.pf_verification
        and str(getattr(args, "dataset", "")).lower() not in {"doubling1d", "complex2d"}
        and int(getattr(args, "ori_dim", 0)) >= 3
    )
    plot_prior_enabled = bool(
        plot_figures and (
            args.dataset in {"lorenz63", "doubling1d", "complex2d"} or pf_style_projection_enabled
        )
    )

    # Rank-hist configuration: fixed projection directions for the whole test run.
    rank_num_projections = max(1, int(getattr(args, "rank_num_projections", 8)))
    rank_seed_default = _safe_int_seed(getattr(args, "seed", None), default=0)
    rank_projection_seed = _safe_int_seed(
        getattr(args, "rank_projection_seed", None),
        default=rank_seed_default,
    )
    rank_tie_break = str(getattr(args, "rank_tie_break", "random")).lower()
    rank_projection_dirs = sample_projection_directions(
        state_dim=int(args.ori_dim),
        num_projections=rank_num_projections,
        device=args.device,
        dtype=torch.float32,
        seed=rank_projection_seed,
    )
    rank_counts_total = torch.zeros(int(args.N) + 1, dtype=torch.int64)
    rank_total_samples = 0

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)  # [T, B, d]
            B = batch_v.shape[1]

            # ---- Active mask per-trajectory (B,) ----
            active_mask = torch.ones(B, dtype=torch.bool, device=args.device)

            # Initialize ensemble analysis at t0: [B, N, d]
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a += torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Deactivate any trajectories that already contain NaNs after initialization
            init_nan = torch.isnan(ens_v_a).any(dim=(1, 2))
            if init_nan.any():
                active_mask = active_mask & (~init_nan)
                ens_v_a[~active_mask] = torch.nan  # keep shapes; mark inactive with NaN

            cov_diff_list, rcov_diff_list, pf_rmse_list = [], [], []
            ens_list = [ens_v_a]
            if plot_prior_enabled:
                ens_f_list = [torch.full_like(ens_v_a, float('nan'))]  # no prior at t=0
            loc_records = []
            
            # Precompute noisy observations for all times (independent of active_mask)
            obs_y_list = []
            clean_h_list = []
            for i in range(len(batch_v)):
                clean_h_step = H_fun(batch_v[i].unsqueeze(1))
                obs_noise_step = _sample_true_obs_noise_like(clean_h_step, test_noise_gen)
                obs_y_step = clean_h_step + args.sigma_y * obs_noise_step
                obs_y_list.append(obs_y_step)
                clean_h_list.append(clean_h_step.squeeze(1))

            d_state_batch = int(batch_v.shape[-1])
            d_obs_batch = int(obs_y_list[0].shape[-1]) if len(obs_y_list) > 0 else int(args.obs_dim)
            (
                coords_state_runtime,
                coords_obs_runtime,
                loc_domain_runtime,
                loc_lvy_runtime,
                loc_lyy_runtime,
            ) = _build_runtime_localization_geometry(
                args=args,
                d_state=d_state_batch,
                d_obs=d_obs_batch,
                dtype=batch_v.dtype,
            )

            h_tensor = torch.stack(clean_h_list, dim=0)
            snr_var_batch = _compute_traj_snr_var_from_hvalues(h_tensor, args.sigma_y)
            all_results['snr_var'] = torch.cat((all_results['snr_var'], snr_var_batch))

            # Time loop
            for i in range(len(batch_v) - 1):
                # Early bail: if no active trajectories remain, append a NaN step to keep time length and continue
                if not active_mask.any():
                    ens_list.append(torch.full_like(ens_v_a, float('nan')))
                    if plot_prior_enabled:
                        ens_f_list.append(torch.full_like(ens_v_a, float('nan')))
                    continue  # keep T length for plotting/metrics

                # -------- Forecast step (not timed here) --------
                ens_v_f = _forecast_ensemble(args, ens_v_a, i, forward_fun)

                # Deactivate offending trajectories that became NaN after forecast
                nan_now = torch.isnan(ens_v_f).any(dim=(1, 2))
                if nan_now.any():
                    active_mask = active_mask & (~nan_now)
                    ens_v_f[~active_mask] = torch.nan
                if plot_prior_enabled:
                    ens_f_list.append(ens_v_f)

                # Active indices
                idx_active = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)

                # Prepare container for next analysis; default NaNs for inactive
                ens_v_a_next = torch.full_like(ens_v_f, float('nan'))

                if idx_active.numel() > 0:
                    # Slice active trajectories only
                    ens_v_f_active = ens_v_f.index_select(0, idx_active)                 # [B_active, N, d]
                    obs_y = obs_y_list[i + 1]                                            # [T,B,?] -> [B,1,d_obs]
                    obs_y_active = obs_y.index_select(0, idx_active)                     # [B_active, 1, d_obs]

                    # ------ BEGIN TIMED ANALYSIS STEP ------
                    t0 = time.perf_counter()

                    # Compute H(ens) and innovation on active subset
                    hv_active = H_fun(ens_v_f_active)                                    # [B_active, N, d_obs]
                    r_noise_active = mean0(args.sigma_y * torch.randn_like(hv_active, device=args.device))
                    ens_i_innov_active = obs_y_active - hv_active - r_noise_active

                    # Means on active subset
                    mean_hv_active = torch.mean(hv_active, dim=1, keepdim=True)
                    mean_ens_v_f_active = torch.mean(ens_v_f_active, dim=1, keepdim=True)

                    # Analysis step (delegated to learned components)
                    ens_v_a_active, loc_nn_output = _process_analysis_step(
                        args, model_list, ens_v_f_active, hv_active, obs_y_active, ens_i_innov_active,
                        mean_ens_v_f_active, mean_hv_active
                    )

                    # Record localization-related outputs if any
                    if loc_nn_output is not None:
                        loc_records.append(loc_nn_output)

                    # Inflation / post-processing on active subset only
                    ens_v_a_active = post_process(ens_v_a_active, infl=infl)

                    # ------ END TIMED ANALYSIS STEP ------
                    t1 = time.perf_counter()
                    duration = t1 - t0
                    assim_step_times.append(duration)
                    num_active = int(idx_active.numel())
                    if num_active > 0:
                        assim_step_times_weighted.extend([duration] * num_active)

                    # Scatter active results back; inactive remain NaN
                    ens_v_a_next.index_copy_(0, idx_active, ens_v_a_active)

                # Move to next analysis state
                ens_v_a = ens_v_a_next
                ens_list.append(ens_v_a)

                # PF verification: compute only for current active subset, scatter back
                if args.pf_verification and idx_active.numel() > 0:
                    pf_entry = cached_pf_data[batch_ind]
                    pf_post_means = _get_pf_cached_post_means(pf_entry)
                    pf_post_covs = _get_pf_cached_post_covs(pf_entry)
                    if pf_post_means is None or pf_post_covs is None:
                        raise KeyError(
                            "PF cache entry must contain either (post_means, post_covs) or legacy (means, covs)."
                        )
                    pf_mean_a_full = pf_post_means[i]     # [B, d]
                    pf_cov_ens_a_full = pf_post_covs[i]   # [B, d, d]

                    pf_mean_a = pf_mean_a_full.index_select(0, idx_active)
                    pf_cov_ens_a = pf_cov_ens_a_full.index_select(0, idx_active)

                    our_method_mean_a = torch.mean(ens_v_a.index_select(0, idx_active), dim=1)  # [B_active, d]
                    pf_rmse_active = torch.sqrt(torch.mean((our_method_mean_a - pf_mean_a)**2, dim=-1))  # [B_active]

                    # Optional dataset-specific masking (rossler)
                    if args.dataset == 'rossler':
                        cond = batch_v[i + 1, :, 2] <= 5
                        cond_active = cond.index_select(0, idx_active)
                        pf_rmse_active = pf_rmse_active.masked_fill(cond_active, float('nan'))

                    # Scatter PF RMSE back to full B vector
                    pf_rmse_full = torch.full((B,), float('nan'), device=args.device)
                    pf_rmse_full.index_copy_(0, idx_active, pf_rmse_active)
                    pf_rmse_list.append(pf_rmse_full)

                    # Covariance diffs on active subset and scatter back
                    cov_ens_a_active = get_ens_cov(ens_v_a.index_select(0, idx_active))     # [B_active, d, d]
                    cov_diff_active = torch.norm(cov_ens_a_active - pf_cov_ens_a, p='fro', dim=(-2, -1))
                    rcov_diff_active = cov_diff_active / torch.norm(pf_cov_ens_a, p='fro', dim=(-2, -1))

                    if args.dataset == 'rossler':
                        cond_active = cond.index_select(0, idx_active)
                        cov_diff_active = cov_diff_active.masked_fill(cond_active, float('nan'))
                        rcov_diff_active = rcov_diff_active.masked_fill(cond_active, float('nan'))

                    cov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    rcov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    cov_diff_full.index_copy_(0, idx_active, cov_diff_active)
                    rcov_diff_full.index_copy_(0, idx_active, rcov_diff_active)
                    cov_diff_list.append(cov_diff_full)
                    rcov_diff_list.append(rcov_diff_full)

            # Stack ensembles over time: [T, B, N, d]
            ens_tensor = torch.stack(ens_list)
            ens_prior_tensor = torch.stack(ens_f_list) if plot_prior_enabled else None
            
            # Metrics (NaNs are handled later by masks)
            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0)
            rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
            rmse_val = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_val = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_val = rmse_val / rms_val
            rmv_val = torch.nanmean(compute_root_mean_variance(ens_tensor), dim=0)
            spread_error_ratio_val = torch.nanmean(compute_spread_error_ratio(ens_tensor, batch_v), dim=0)

            rank_stats_batch = compute_ensemble_rank_histogram(
                ens_states=ens_tensor,
                true_states=batch_v,
                projection_directions=rank_projection_dirs.to(device=ens_tensor.device, dtype=ens_tensor.dtype),
                num_projections=rank_num_projections,
                tie_break=rank_tie_break,
                seed=rank_projection_seed + batch_ind,
            )
            rank_counts_total += rank_stats_batch["counts"].to(dtype=torch.int64)
            rank_total_samples += int(rank_stats_batch["total_samples"])
            
            # Aggregate metrics
            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['rmv'] = torch.cat((all_results['rmv'], rmv_val))
            all_results['spread_error_ratio'] = torch.cat((all_results['spread_error_ratio'], spread_error_ratio_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            if args.pf_verification and len(pf_rmse_list) > 0:
                # Use nanmean over time for PF-based metrics
                all_results['cov_diff'] = torch.cat((all_results['cov_diff'], torch.stack(cov_diff_list).nanmean(0)))
                all_results['rcov_diff'] = torch.cat((all_results['rcov_diff'], torch.stack(rcov_diff_list).nanmean(0)))
                all_results['pf_rmse'] = torch.cat((all_results['pf_rmse'], torch.stack(pf_rmse_list).nanmean(0)))
                all_results['pf_rrmse'] = torch.cat((all_results['pf_rrmse'], torch.stack(pf_rmse_list).nanmean(0) / rms_val))
            
            # Localization tensor collection (unchanged)
            if args.v != "EtE" and not args.no_localization and len(loc_records) > 0:
                current_loc_tensor = torch.stack(loc_records).unsqueeze(0)
                if loc_tensor_all_batches is None:
                    loc_tensor_all_batches = current_loc_tensor
                else:
                    try:
                        loc_tensor_all_batches = torch.cat((loc_tensor_all_batches, current_loc_tensor), dim=0)
                    except Exception as e:
                        print(f"Warning: Could not concatenate loc_tensors due to shape mismatch or other error: {e}")

            # Build observation tensor for plotting (unchanged)
            observations = _build_observation_plot_tensor(args, batch_v, obs_y_list)

            if plot_figures and plot_mode == "global":
                local_plot_indices = _resolve_global_plot_indices_for_batch(
                    global_indices=requested_global_indices,
                    batch_start_index=batch_start_index,
                    batch_size=B,
                )
                for lidx in local_plot_indices:
                    unresolved_global_indices.discard(batch_start_index + lidx)

                if len(local_plot_indices) > 0:
                    if pf_style_projection_enabled:
                        _plot_pf_style_projections_from_cache(
                            args=args,
                            ens_post_tensor=ens_tensor,
                            ens_prior_tensor=ens_prior_tensor,
                            true_tensor=batch_v,
                            pf_entry=cached_pf_data[batch_ind],
                            fig_name=fig_name,
                            plot_batch_indices=local_plot_indices,
                            batch_start_index=batch_start_index,
                            global_index_naming=True,
                        )
                    else:
                        ens_plot = ens_tensor
                        true_plot = batch_v
                        compare_plot = batch_v
                        obs_plot = observations

                        if args.pf_verification:
                            pf_post_means = _get_pf_cached_post_means(cached_pf_data[batch_ind])
                            if pf_post_means is None:
                                raise KeyError("PF cache entry missing post means.")
                            true_plot = true_plot[1:]
                            compare_plot = pf_post_means
                            ens_plot = ens_plot[1:]
                            obs_plot = obs_plot[1:]
                            step_offset = 0
                        else:
                            step_offset = 0

                        _plot_test_visualizations(
                            args=args,
                            ens_tensor=ens_plot,
                            true_tensor=true_plot,
                            comparison_tensor=compare_plot,
                            observations=obs_plot,
                            fig_name=fig_name,
                            save_pdf=save_pdf,
                            step_offset=step_offset,
                            plot_batch_indices=local_plot_indices,
                            batch_start_index=batch_start_index,
                            global_index_naming=True,
                            ens_color='tab:red',
                            ring_plot_history=False,
                            enable_highdim_slices=True,
                        )
                        if plot_prior_enabled and ens_prior_tensor is not None and ens_prior_tensor.shape[0] > 1:
                            _plot_test_visualizations(
                                args=args,
                                ens_tensor=ens_prior_tensor[1:],
                                true_tensor=batch_v[1:],
                                comparison_tensor=batch_v[1:],
                                observations=observations[1:],
                                fig_name=fig_name + "_prior",
                                save_pdf=save_pdf,
                                step_offset=1,
                                plot_batch_indices=local_plot_indices,
                                batch_start_index=batch_start_index,
                                global_index_naming=True,
                                ens_color='tab:blue',
                                ring_plot_history=False,
                                enable_highdim_slices=True,
                            )
            batch_start_index += B
    
    if plot_figures and plot_mode == "adaptive":
        if pf_style_projection_enabled:
            nan_per_step = torch.isnan(ens_tensor).any(dim=(2, 3))
            nan_counts = nan_per_step.sum(dim=0)
            adaptive_indices = _resolve_plot_batch_indices(
                args=args,
                batch_size=ens_tensor.shape[1],
                nan_counts=nan_counts,
            )
            _plot_pf_style_projections_from_cache(
                args=args,
                ens_post_tensor=ens_tensor,
                ens_prior_tensor=ens_prior_tensor,
                true_tensor=batch_v,
                pf_entry=cached_pf_data[-1],
                fig_name=fig_name,
                plot_batch_indices=adaptive_indices,
                batch_start_index=0,
                global_index_naming=False,
            )
        else:
            ens_plot = ens_tensor
            true_plot = batch_v
            compare_plot = batch_v
            obs_plot = observations

            # With PF verification, compare ensemble means against PF posterior means,
            # while keeping per-step snapshot markers on true states.
            if args.pf_verification:
                pf_post_means = _get_pf_cached_post_means(cached_pf_data[-1])
                if pf_post_means is None:
                    raise KeyError("PF cache entry missing post means.")
                true_plot = true_plot[1:]
                compare_plot = pf_post_means
                ens_plot = ens_plot[1:]
                obs_plot = obs_plot[1:]
                step_offset = 0
            else:
                step_offset = 0

            _plot_test_visualizations(
                args=args,
                ens_tensor=ens_plot,
                true_tensor=true_plot,
                comparison_tensor=compare_plot,
                observations=obs_plot,
                fig_name=fig_name,
                save_pdf=save_pdf,
                step_offset=step_offset,
                ens_color='tab:red',
                ring_plot_history=False,
                enable_highdim_slices=True,
            )
            if plot_prior_enabled and ens_prior_tensor is not None and ens_prior_tensor.shape[0] > 1:
                _plot_test_visualizations(
                    args=args,
                    ens_tensor=ens_prior_tensor[1:],
                    true_tensor=batch_v[1:],
                    comparison_tensor=batch_v[1:],
                    observations=observations[1:],
                    fig_name=fig_name + "_prior",
                    save_pdf=save_pdf,
                    step_offset=0,
                    ens_color='tab:blue',
                    ring_plot_history=False,
                    enable_highdim_slices=True,
                )
    elif plot_figures and len(unresolved_global_indices) > 0:
        unresolved_sorted = sorted(list(unresolved_global_indices))
        print(
            f"Warning: some global test_plot_index values were not found in loader batches: "
            f"{unresolved_sorted[:10]}{'...' if len(unresolved_sorted) > 10 else ''}"
        )

    # Final metrics
    final_metrics = {}
    
    if all_results['rrmse'].numel() == 0:
        metrics_keys = ['mean_rmse', 'std_rmse',
                        'mean_rrmse', 'std_rrmse', 'mean_rmv', 'std_rmv',
                        'mean_spread_error_ratio', 'std_spread_error_ratio',
                        'mean_crps', 'std_crps',
                        'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                        'mean_rcov_diff', 'std_rcov_diff', 'mean_pf_rmse', 'std_pf_rmse',
                        'mean_pf_rrmse', 'std_pf_rrmse']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        nan_mask = torch.isnan(all_results['rrmse'])
        valid_B_mask = ~nan_mask
        
        if not valid_B_mask.any():
            metrics_keys = ['mean_rmse', 'std_rmse',
                            'mean_rrmse', 'std_rrmse', 'mean_rmv', 'std_rmv',
                            'mean_spread_error_ratio', 'std_spread_error_ratio',
                            'mean_crps', 'std_crps',
                            'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                            'mean_rcov_diff', 'std_rcov_diff', 'mean_pf_rmse', 'std_pf_rmse',
                            'mean_pf_rrmse', 'std_pf_rrmse']
            final_metrics = {key: float('nan') for key in metrics_keys}
            final_metrics['no_nan_percent'] = 0.0
        else:
            final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
            final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
            final_metrics['mean_rmv'], final_metrics['std_rmv'] = get_mean_std(all_results['rmv'][valid_B_mask])
            final_metrics['mean_spread_error_ratio'], final_metrics['std_spread_error_ratio'] = get_mean_std(
                all_results['spread_error_ratio'][valid_B_mask]
            )
            final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
            final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
            final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0
            if args.pf_verification:
                final_metrics['mean_cov_diff'], final_metrics['std_cov_diff'] = get_mean_std(all_results['cov_diff'][valid_B_mask])
                final_metrics['mean_rcov_diff'], final_metrics['std_rcov_diff'] = get_mean_std(all_results['rcov_diff'][valid_B_mask])
                final_metrics['mean_pf_rmse'], final_metrics['std_pf_rmse'] = get_mean_std(all_results['pf_rmse'][valid_B_mask])
                final_metrics['mean_pf_rrmse'], final_metrics['std_pf_rrmse'] = get_mean_std(all_results['pf_rrmse'][valid_B_mask])

    snr_values = all_results['snr_var']
    finite_snr_mask = torch.isfinite(snr_values)
    if finite_snr_mask.any():
        final_metrics['mean_snr_var'], final_metrics['std_snr_var'] = get_mean_std(snr_values[finite_snr_mask])
    else:
        final_metrics['mean_snr_var'] = float('nan')
        final_metrics['std_snr_var'] = float('nan')

    if rank_total_samples > 0:
        rank_probs = rank_counts_total.to(torch.float64) / float(rank_total_samples)
        rank_uniform = torch.full_like(rank_probs, 1.0 / float(rank_counts_total.numel()))
        rank_l1 = torch.sum(torch.abs(rank_probs - rank_uniform))
        rank_l2 = torch.sqrt(torch.mean((rank_probs - rank_uniform) ** 2))
        rank_freq_range = torch.max(rank_probs) - torch.min(rank_probs)
        rank_expected = float(rank_total_samples) / float(rank_counts_total.numel())
        rank_chi2 = torch.sum((rank_counts_total.to(torch.float64) - rank_expected) ** 2 / max(rank_expected, 1e-12))

        final_metrics['rank_hist_counts'] = rank_counts_total.tolist()
        final_metrics['rank_hist_probs'] = rank_probs.to(torch.float32).tolist()
        final_metrics['rank_total_samples'] = int(rank_total_samples)
        final_metrics['rank_num_projections'] = int(rank_num_projections)
        final_metrics['rank_freq_range'] = float(rank_freq_range.item())
        final_metrics['rank_uniform_l1'] = float(rank_l1.item())
        final_metrics['rank_uniform_l2'] = float(rank_l2.item())
        final_metrics['rank_chi2'] = float(rank_chi2.item())
    else:
        final_metrics['rank_hist_counts'] = [0 for _ in range(int(args.N) + 1)]
        final_metrics['rank_hist_probs'] = [float('nan') for _ in range(int(args.N) + 1)]
        final_metrics['rank_total_samples'] = 0
        final_metrics['rank_num_projections'] = int(rank_num_projections)
        final_metrics['rank_freq_range'] = float('nan')
        final_metrics['rank_uniform_l1'] = float('nan')
        final_metrics['rank_uniform_l2'] = float('nan')
        final_metrics['rank_chi2'] = float('nan')

    # ---- NEW: compute analysis-step timing stats and attach to final_metrics ----
    if len(assim_step_times) > 0:
        t_tensor = torch.tensor(assim_step_times, dtype=torch.float64)
        final_metrics['assim_step_time_mean'] = float(t_tensor.mean().item())
        final_metrics['assim_step_time_std']  = float(t_tensor.std(unbiased=False).item())
    else:
        final_metrics['assim_step_time_mean'] = float('nan')
        final_metrics['assim_step_time_std']  = float('nan')

    if len(assim_step_times_weighted) > 0:
        tw_tensor = torch.tensor(assim_step_times_weighted, dtype=torch.float64)
        final_metrics['assim_step_time_mean_weighted'] = float(tw_tensor.mean().item())
        final_metrics['assim_step_time_std_weighted']  = float(tw_tensor.std(unbiased=False).item())
    else:
        final_metrics['assim_step_time_mean_weighted'] = float('nan')
        final_metrics['assim_step_time_std_weighted']  = float('nan')

    # Return one localization tensor (unchanged)
    final_loc_tensor_to_return = loc_tensor_all_batches[0] if loc_tensor_all_batches is not None and loc_tensor_all_batches.shape[0] > 0 else torch.empty(1, device=args.device)
    final_metrics['loc_tensor'] = final_loc_tensor_to_return

    return final_metrics




def print_test_results(results):
    """Pretty-print test metrics including optional timing stats (times in ms)."""
    # Core metrics
    if 'mean_rmse' in results and 'std_rmse' in results:
        print(f"RMSE: {results['mean_rmse']:.3f} ± {results['std_rmse']:.3f}")
    if 'mean_rrmse' in results and 'std_rrmse' in results:
        print(f"RRMSE: {results['mean_rrmse']:.3f} ± {results['std_rrmse']:.3f}")
    if 'mean_rmv' in results and 'std_rmv' in results:
        print(f"RMV: {results['mean_rmv']:.3f} ± {results['std_rmv']:.3f}")
    if 'mean_spread_error_ratio' in results and 'std_spread_error_ratio' in results:
        print(f"Spread-Error Ratio: {results['mean_spread_error_ratio']:.3f} ± {results['std_spread_error_ratio']:.3f}")
    if 'mean_crps' in results and 'std_crps' in results:
        print(f"CRPS: {results['mean_crps']:.3f} ± {results['std_crps']:.3f}")
    if 'mean_rcrps' in results and 'std_rcrps' in results:
        print(f"RCRPS: {results['mean_rcrps']:.3f} ± {results['std_rcrps']:.3f}")

    # Optional PF verification metrics
    if 'mean_cov_diff' in results and 'std_cov_diff' in results:
        print(f"Cov-Diff: {results['mean_cov_diff']:.3f} ± {results['std_cov_diff']:.3f}")
    if 'mean_rcov_diff' in results and 'std_rcov_diff' in results:
        print(f"RCov-Diff: {results['mean_rcov_diff']:.3f} ± {results['std_rcov_diff']:.3f}")
    if 'mean_pf_rmse' in results and 'std_pf_rmse' in results:
        print(f"PF-RMSE: {results['mean_pf_rmse']:.3f} ± {results['std_pf_rmse']:.3f}")
    if 'mean_pf_rrmse' in results and 'std_pf_rrmse' in results:
        print(f"PF-RRMSE: {results['mean_pf_rrmse']:.3f} ± {results['std_pf_rrmse']:.3f}")

    # Percentage of non-NaN trajectories
    if 'no_nan_percent' in results:
        print(f"No NAN Percentage: {results['no_nan_percent']:.2f}%")

    if 'mean_snr_var' in results and 'std_snr_var' in results:
        print(f"SNR_var: {results['mean_snr_var']:.3f} ± {results['std_snr_var']:.3f}")

    if 'rank_freq_range' in results:
        rank_total = results.get('rank_total_samples', 0)
        rank_proj = results.get('rank_num_projections', None)
        rank_proj_str = f", projections={rank_proj}" if rank_proj is not None else ""
        print(f"Rank-Hist FreqRange(max-min): {results['rank_freq_range']:.4f}{rank_proj_str}, samples={rank_total}")

    if 'rank_uniform_l1' in results:
        rank_total = results.get('rank_total_samples', 0)
        rank_proj = results.get('rank_num_projections', None)
        rank_proj_str = f", projections={rank_proj}" if rank_proj is not None else ""
        print(f"Rank-Hist Uniform L1: {results['rank_uniform_l1']:.4f}{rank_proj_str}, samples={rank_total}")
    if 'rank_uniform_l2' in results:
        print(f"Rank-Hist Uniform L2: {results['rank_uniform_l2']:.4f}")
    if 'rank_chi2' in results:
        print(f"Rank-Hist Chi2: {results['rank_chi2']:.3f}")
    if 'rank_hist_probs' in results:
        probs = results['rank_hist_probs']
        if isinstance(probs, (list, tuple)) and len(probs) <= 32:
            probs_str = ", ".join(f"{float(p):.3f}" for p in probs)
            print(f"Rank-Hist probs: [{probs_str}]")

    # --- Timing outputs in milliseconds (ms) ---

    # Unweighted per-step assimilation time (originally in seconds)
    if 'assim_step_time_mean' in results and 'assim_step_time_std' in results:
        mean_ms = results['assim_step_time_mean'] * 1000.0
        std_ms  = results['assim_step_time_std'] * 1000.0
        print(f"Assim-step time (ms): {mean_ms:.3f} ± {std_ms:.3f}")

    # Trajectory-weighted per-step assimilation time
    if 'assim_step_time_mean_weighted' in results and 'assim_step_time_std_weighted' in results:
        mean_ms_w = results['assim_step_time_mean_weighted'] * 1000.0
        std_ms_w  = results['assim_step_time_std_weighted'] * 1000.0
        print(f"Assim-step time (ms, weighted): {mean_ms_w:.3f} ± {std_ms_w:.3f}")


def _build_runtime_localization_geometry(args, d_state: int, d_obs: int, dtype: torch.dtype):
    """
    Build localization geometry on-the-fly.

    This supports nonlinear observations where args.obs_coord_inds/args.Lvy/args.Lyy
    may be unavailable by falling back to evenly spaced observation coordinates.
    """
    if d_obs <= 0:
        raise ValueError(f"d_obs must be positive, but got {d_obs}.")

    device = args.device
    coords_state = torch.arange(d_state, device=device, dtype=dtype).unsqueeze(1)

    obs_coord_inds = getattr(args, "obs_coord_inds", None)
    obs_inds = getattr(args, "obs_inds", None)
    if obs_coord_inds is not None and len(obs_coord_inds) == d_obs:
        coords_obs = torch.as_tensor(obs_coord_inds, device=device, dtype=dtype).reshape(-1, 1)
    elif obs_inds is not None and len(obs_inds) == d_obs:
        coords_obs = torch.as_tensor(obs_inds, device=device, dtype=dtype).reshape(-1, 1)
    else:
        if d_obs == 1:
            coords_obs = torch.zeros((1, 1), device=device, dtype=dtype)
        else:
            coords_obs = torch.linspace(0.0, float(d_state - 1), steps=d_obs, device=device, dtype=dtype).unsqueeze(1)

    if args.dataset in {"lorenz96", "ks"}:
        domain = torch.tensor([float(d_state)], device=device, dtype=dtype)
    else:
        domain = None

    lvy = pairwise_distances(coords_state, coords_obs, domain=domain).to(device=device, dtype=dtype)
    lyy = pairwise_distances(coords_obs, coords_obs, domain=domain).to(device=device, dtype=dtype)

    return coords_state, coords_obs, domain, lvy, lyy





def test_ClassicFilter(loader, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig', loc_radius=None, save_pdf=False):
    """
    Tests a classic data assimilation filter (e.g., EnKF, ESRF, LETKF) and
    optionally compares results against a particle filter baseline.

    NOTE (added): If any trajectory (a specific b in the batch dimension) ever
    produces NaNs at any step, that trajectory is marked inactive and skipped
    for all subsequent steps. We keep shape consistency by writing NaNs for its
    outputs so that downstream metrics (which already NaN-mask) ignore it.

    CHANGE: Includes spread metrics (RMV/SER) and rank-hist calibration stats.

    NEW (timing): Records wall-clock time for each assimilation step (per i),
    aggregates across all batches. Also records a trajectory-weighted variant
    that replicates each step duration by the number of active trajectories at
    that step. Returns mean/std for both.
    """
    import os
    import time  # <-- NEW: timing
    import torch
    from tqdm import tqdm

    # Keep figure-saving behavior aligned with test_model callers:
    # allow CLI flag --save_test_figures to enable plotting even if caller forgot.
    if getattr(args, "save_test_figures", False):
        plot_figures = True

    # If plotting is enabled but fig_name is left as default, create a stable
    # benchmark-style output path so generated figures are saved under save/.
    if plot_figures and fig_name == 'example_fig':
        auto_fig_dir = os.path.join("save", f"benchmark_{args.dataset}_{args.sigma_y}_{args.v}")
        os.makedirs(auto_fig_dir, exist_ok=True)
        fig_name = os.path.join(auto_fig_dir, f"test_{args.N}_0")

    m = args.N
    test_noise_gen = get_test_noise_generator(args)

    # Select forward function
    forward_fun = _get_forward_fun(args)
    ienks_model_args = _build_ienks_model_args(args, forward_fun) if args.v.startswith('iEnKS') else None

    # Observation operator
    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info

    if args.no_localization:
        print("Do not use localization")
        loc_radius = None

    # Optional PF cache
    cached_pf_data = None
    if args.pf_verification:
        first_batch_for_shape = next(iter(loader))
        traj_len = first_batch_for_shape.shape[0]
        batch_size = first_batch_for_shape.shape[1]

        cache_dir = _build_pf_cache_dir(args)
        cache_filename = _build_pf_cache_filename(args, batch_size=batch_size, traj_len=traj_len, avg=True)
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"Loading cached PF results from: {cache_filepath}")
            cached_pf_data = torch.load(cache_filepath, map_location=args.device, weights_only=True)
        else:
            legacy_cache_dir = os.path.join('data', args.dataset)
            legacy_cache_filepath = os.path.join(legacy_cache_dir, cache_filename)
            if os.path.exists(legacy_cache_filepath):
                print(f"[PF cache fallback] Loading legacy cache from: {legacy_cache_filepath}")
                cached_pf_data = torch.load(legacy_cache_filepath, map_location=args.device, weights_only=True)
            else:
                raise FileNotFoundError(
                    f"Required particle filter cache file not found at: {cache_filepath} "
                    f"(legacy checked: {legacy_cache_filepath}). "
                    f"(obs_fn={_resolve_obs_fn_for_pf_paths(args)}, default_obs_fn={_get_dataset_default_obs_fn(args.dataset)}) "
                    f"Please run generate_and_cache_pf_results() first."
                )

    # Aggregated results
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
        'rmv': torch.empty(0, device=args.device),
        'spread_error_ratio': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'snr_var': torch.empty(0, device=args.device),
        'cov_diff': torch.empty(0, device=args.device),
        'rcov_diff': torch.empty(0, device=args.device),
        'pf_rmse': torch.empty(0, device=args.device),
        'pf_rrmse': torch.empty(0, device=args.device),
    }

    # NEW: timing collectors
    assim_step_times = []              # per-step durations across all batches
    assim_step_times_weighted = []     # per-trajectory-weighted durations
    plot_mode, requested_global_indices = _normalize_global_plot_indices(args)
    unresolved_global_indices = set(requested_global_indices)
    batch_start_index = 0
    plot_prior_enabled = bool(plot_figures and args.dataset in {"lorenz63", "doubling1d", "complex2d"})
    rank_num_projections = max(1, int(getattr(args, "rank_num_projections", 8)))
    rank_seed_default = _safe_int_seed(getattr(args, "seed", None), default=0)
    rank_projection_seed = _safe_int_seed(
        getattr(args, "rank_projection_seed", None),
        default=rank_seed_default,
    )
    rank_tie_break = str(getattr(args, "rank_tie_break", "random")).lower()
    rank_projection_dirs = sample_projection_directions(
        state_dim=int(args.ori_dim),
        num_projections=rank_num_projections,
        device=args.device,
        dtype=torch.float32,
        seed=rank_projection_seed,
    )
    rank_counts_total = torch.zeros(int(args.N) + 1, dtype=torch.int64)
    rank_total_samples = 0

    with torch.no_grad():
        for batch_ind, batch_v in enumerate(loader):
            batch_v = batch_v.to(device=args.device)  # [T, B, d]

            # ---- NEW: active mask per-trajectory (B,) ----
            B = batch_v.shape[1]
            active_mask = torch.ones(B, dtype=torch.bool, device=args.device)

            # Initialize ensemble analysis at t0: [B, N, d]
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a += torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Detect NaN at initialization and deactivate those trajectories
            init_nan = torch.isnan(ens_v_a).any(dim=(1, 2))
            if init_nan.any():
                active_mask = active_mask & (~init_nan)
                # Fill NaNs for the deactivated ones to keep shapes consistent
                ens_v_a[~active_mask] = torch.nan

            ens_list = [ens_v_a]
            if plot_prior_enabled:
                ens_f_list = [torch.full_like(ens_v_a, float('nan'))]  # no prior at t=0
            cov_diff_list, rcov_diff_list, pf_rmse_list = [], [], []

            # Precompute noisy observations y_t for all t (shape aligned to H_fun outputs)
            obs_y_list = []
            clean_h_list = []
            for i in range(len(batch_v)):
                clean_h_step = H_fun(batch_v[i].unsqueeze(1))
                obs_noise_step = _sample_true_obs_noise_like(clean_h_step, test_noise_gen)
                obs_y_step = clean_h_step + args.sigma_y * obs_noise_step
                obs_y_list.append(obs_y_step)
                clean_h_list.append(clean_h_step.squeeze(1))

            d_state_batch = int(batch_v.shape[-1])
            d_obs_batch = int(obs_y_list[0].shape[-1]) if len(obs_y_list) > 0 else int(args.obs_dim)
            (
                coords_state_runtime,
                coords_obs_runtime,
                loc_domain_runtime,
                loc_lvy_runtime,
                loc_lyy_runtime,
            ) = _build_runtime_localization_geometry(
                args=args,
                d_state=d_state_batch,
                d_obs=d_obs_batch,
                dtype=batch_v.dtype,
            )

            h_tensor = torch.stack(clean_h_list, dim=0)
            snr_var_batch = _compute_traj_snr_var_from_hvalues(h_tensor, args.sigma_y)
            all_results['snr_var'] = torch.cat((all_results['snr_var'], snr_var_batch))

            # Time loop
            for i in tqdm(range(len(batch_v) - 1), desc="Processing", unit="item"):
                # Early exit if no active trajectories remain
                if not active_mask.any():
                    ens_list.append(torch.full_like(ens_v_a, float('nan')))
                    if plot_prior_enabled:
                        ens_f_list.append(torch.full_like(ens_v_a, float('nan')))
                    # Record a zero-duration placeholder for clarity? No: skip to avoid bias
                    continue

                # ---- NEW: start timing the entire assimilation step (forecast + analysis + book-keeping) ----
                t0 = time.perf_counter()

                obs_y = obs_y_list[i + 1]  # [B, 1, d_obs]

                # Forecast step
                if args.v.startswith('iEnKS'):
                    ens_v_f = ens_v_a  # smoothing will handle propagation
                else:
                    ens_v_f = _forecast_ensemble(args, ens_v_a, i, forward_fun)

                # ---- NEW: detect NaNs in forecast and deactivate offending trajectories ----
                nan_now = torch.isnan(ens_v_f).any(dim=(1, 2))
                if nan_now.any():
                    active_mask = active_mask & (~nan_now)
                    ens_v_f[~active_mask] = torch.nan
                if plot_prior_enabled:
                    ens_f_list.append(ens_v_f)

                # Common args
                common_enkf_args = {
                    "observation_y": None,  # will be filled with subset
                    "observation_operator_ens": H_fun,
                    "sigma_y": args.sigma_y,
                    "sigma_v": args.sigma_v,
                    "inflation_factor": infl
                }

                # ---- operate only on active indices for the analysis step ----
                idx_active = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                ens_v_a_next = torch.full_like(ens_v_f, float('nan'))  # default NaN for inactive

                if idx_active.numel() > 0:
                    ens_v_f_active = ens_v_f.index_select(0, idx_active)
                    obs_y_active = obs_y.squeeze(1).index_select(0, idx_active)
                    common_enkf_args["observation_y"] = obs_y_active

                    if args.v == 'EnKF':
                        use_loc = loc_radius is not None
                        loc_vy = dist2coeff(loc_lvy_runtime, radius=loc_radius).unsqueeze(0) if use_loc else None
                        loc_yy = dist2coeff(loc_lyy_runtime, radius=loc_radius).unsqueeze(0) if use_loc else None
                        ens_v_a_active, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active, **common_enkf_args, method='EnKF-PertObs',
                            localization_matrix_Lxy=loc_vy, localization_matrix_Lyy=loc_yy)

                    elif args.v == 'ESRF':
                        ens_v_a_active, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active, **common_enkf_args, method='ESRF')

                    elif args.v == 'LETKF':
                        letkf_radius = float(loc_radius) if loc_radius is not None else float("inf")
                        ens_v_a_active, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active, **common_enkf_args, method='LETKF',
                            localization_radius=letkf_radius, coords_state=coords_state_runtime,
                            coords_obs=coords_obs_runtime, localization_domain=loc_domain_runtime)

                    elif args.v.startswith('iEnKS'):
                        E_smoothed_at_start, _ = ensemble_kalman_filter_analysis(
                            ens_v_f_active,
                            **common_enkf_args,
                            method=args.v,
                            localization_radius=None,
                            coords_state=coords_state_runtime,
                            coords_obs=coords_obs_runtime, 
                            localization_domain=loc_domain_runtime,
                            ienks_lag=1,
                            ienks_niter=10,
                            ienks_wtol=1e-5,
                            model_args=ienks_model_args
                        )

                        # Forecast from smoothed start to analysis time
                        ens_v_a_active = _forecast_ensemble(args, E_smoothed_at_start, i, forward_fun)

                    else:
                        raise NotImplementedError(f"The filter {args.v} is not implemented")

                    # Clamp only on active subset
                    ens_v_a_active = torch.clamp(ens_v_a_active, min=-args.clamp, max=args.clamp)

                    # Scatter back active results into full tensor; inactive stay NaN
                    ens_v_a_next.index_copy_(0, idx_active, ens_v_a_active)

                # Move to next analysis state (NaN for inactive, valid values for active)
                ens_v_a = ens_v_a_next

                # Append for metrics (inactive remain NaN)
                ens_list.append(ens_v_a)

                # PF verification (compute only on currently active trajectories)
                if args.pf_verification and idx_active.numel() > 0:
                    pf_entry = cached_pf_data[batch_ind]
                    pf_post_means = _get_pf_cached_post_means(pf_entry)
                    pf_post_covs = _get_pf_cached_post_covs(pf_entry)
                    if pf_post_means is None or pf_post_covs is None:
                        raise KeyError(
                            "PF cache entry must contain either (post_means, post_covs) or legacy (means, covs)."
                        )
                    pf_mean_a_full = pf_post_means[i]         # [B, d]
                    pf_cov_ens_a_full = pf_post_covs[i]      # [B, d, d]

                    # Subset PF data to active ones
                    pf_mean_a = pf_mean_a_full.index_select(0, idx_active)
                    pf_cov_ens_a = pf_cov_ens_a_full.index_select(0, idx_active)

                    our_method_mean_a = torch.mean(ens_v_a.index_select(0, idx_active), dim=1)  # [B_active, d]
                    pf_rmse = torch.sqrt(torch.mean((our_method_mean_a - pf_mean_a) ** 2, dim=-1))  # [B_active]
                    pf_rmse_full = torch.full((B,), float('nan'), device=args.device)
                    pf_rmse_full.index_copy_(0, idx_active, pf_rmse)
                    pf_rmse_list.append(pf_rmse_full)

                    cov_ens_a = get_ens_cov(ens_v_a.index_select(0, idx_active))  # [B_active, d, d]
                    cov_diff_active = torch.norm(cov_ens_a - pf_cov_ens_a, p='fro', dim=(-2, -1))
                    rcov_diff_active = cov_diff_active / torch.norm(pf_cov_ens_a, p='fro', dim=(-2, -1))
                    cov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    rcov_diff_full = torch.full((B,), float('nan'), device=args.device)
                    cov_diff_full.index_copy_(0, idx_active, cov_diff_active)
                    rcov_diff_full.index_copy_(0, idx_active, rcov_diff_active)
                    cov_diff_list.append(cov_diff_full)
                    rcov_diff_list.append(rcov_diff_full)

                # ---- NEW: stop timing and record ----
                t1 = time.perf_counter()
                duration = t1 - t0  # seconds
                assim_step_times.append(duration)
                # Weight by #active trajectories at this step
                num_active = int(idx_active.numel()) if 'idx_active' in locals() else 0
                if num_active > 0:
                    assim_step_times_weighted.extend([duration] * num_active)
                # If no active, we skip weighting to avoid bias

            # Stack ensembles over time: [T, B, N, d]
            ens_tensor = torch.stack(ens_list)
            ens_prior_tensor = torch.stack(ens_f_list) if plot_prior_enabled else None

            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0)
            rcrps_val = crps_val / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
            rmse_val = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
            rms_val = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
            rrmse_val = rmse_val / rms_val
            rmv_val = torch.nanmean(compute_root_mean_variance(ens_tensor), dim=0)
            spread_error_ratio_val = torch.nanmean(compute_spread_error_ratio(ens_tensor, batch_v), dim=0)

            rank_stats_batch = compute_ensemble_rank_histogram(
                ens_states=ens_tensor,
                true_states=batch_v,
                projection_directions=rank_projection_dirs.to(device=ens_tensor.device, dtype=ens_tensor.dtype),
                num_projections=rank_num_projections,
                tie_break=rank_tie_break,
                seed=rank_projection_seed + batch_ind,
            )
            rank_counts_total += rank_stats_batch["counts"].to(dtype=torch.int64)
            rank_total_samples += int(rank_stats_batch["total_samples"])

            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['rmv'] = torch.cat((all_results['rmv'], rmv_val))
            all_results['spread_error_ratio'] = torch.cat((all_results['spread_error_ratio'], spread_error_ratio_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            if args.pf_verification and len(pf_rmse_list) > 0:
                all_results['cov_diff'] = torch.cat((all_results['cov_diff'], torch.stack(cov_diff_list).mean(0)))
                all_results['rcov_diff'] = torch.cat((all_results['rcov_diff'], torch.stack(rcov_diff_list).mean(0)))
                all_results['pf_rmse'] = torch.cat((all_results['pf_rmse'], torch.stack(pf_rmse_list).mean(0)))
                all_results['pf_rrmse'] = torch.cat((all_results['pf_rrmse'], torch.stack(pf_rmse_list).nanmean(0) / rms_val))
            
            # Build observation tensor for plotting (unchanged)
            observations = _build_observation_plot_tensor(args, batch_v, obs_y_list)
            if plot_figures and plot_mode == "global":
                local_plot_indices = _resolve_global_plot_indices_for_batch(
                    global_indices=requested_global_indices,
                    batch_start_index=batch_start_index,
                    batch_size=B,
                )
                for lidx in local_plot_indices:
                    unresolved_global_indices.discard(batch_start_index + lidx)

                if len(local_plot_indices) > 0:
                    ens_plot = ens_tensor
                    true_plot = batch_v
                    compare_plot = batch_v
                    obs_plot = observations

                    if args.pf_verification:
                        pf_post_means = _get_pf_cached_post_means(cached_pf_data[batch_ind])
                        if pf_post_means is None:
                            raise KeyError("PF cache entry missing post means.")
                        true_plot = true_plot[1:]
                        compare_plot = pf_post_means
                        ens_plot = ens_plot[1:]
                        obs_plot = obs_plot[1:]
                        step_offset = 1
                    else:
                        step_offset = 0

                    _plot_test_visualizations(
                        args=args,
                        ens_tensor=ens_plot,
                        true_tensor=true_plot,
                        comparison_tensor=compare_plot,
                        observations=obs_plot,
                        fig_name=fig_name + "_classic",
                        save_pdf=save_pdf,
                        step_offset=step_offset,
                        plot_batch_indices=local_plot_indices,
                        batch_start_index=batch_start_index,
                        global_index_naming=True,
                        ens_color='tab:red',
                        ring_plot_history=False,
                    )
                    if plot_prior_enabled and ens_prior_tensor is not None and ens_prior_tensor.shape[0] > 1:
                        _plot_test_visualizations(
                            args=args,
                            ens_tensor=ens_prior_tensor[1:],
                            true_tensor=batch_v[1:],
                            comparison_tensor=batch_v[1:],
                            observations=observations[1:],
                            fig_name=fig_name + "_classic_prior",
                            save_pdf=save_pdf,
                            step_offset=1,
                            plot_batch_indices=local_plot_indices,
                            batch_start_index=batch_start_index,
                            global_index_naming=True,
                            ens_color='tab:blue',
                            ring_plot_history=False,
                        )
            batch_start_index += B

    if plot_figures and plot_mode == "adaptive":
        ens_plot = ens_tensor
        true_plot = batch_v
        compare_plot = batch_v
        obs_plot = observations

        if args.pf_verification:
            pf_post_means = _get_pf_cached_post_means(cached_pf_data[-1])
            if pf_post_means is None:
                raise KeyError("PF cache entry missing post means.")
            true_plot = true_plot[1:]
            compare_plot = pf_post_means
            ens_plot = ens_plot[1:]
            obs_plot = obs_plot[1:]
            step_offset = 1
        else:
            step_offset = 0

        _plot_test_visualizations(
            args=args,
            ens_tensor=ens_plot,
            true_tensor=true_plot,
            comparison_tensor=compare_plot,
            observations=obs_plot,
            fig_name=fig_name + "_classic",
            save_pdf=save_pdf,
            step_offset=step_offset,
            ens_color='tab:red',
            ring_plot_history=False,
        )
        if plot_prior_enabled and ens_prior_tensor is not None and ens_prior_tensor.shape[0] > 1:
            _plot_test_visualizations(
                args=args,
                ens_tensor=ens_prior_tensor[1:],
                true_tensor=batch_v[1:],
                comparison_tensor=batch_v[1:],
                observations=observations[1:],
                fig_name=fig_name + "_classic_prior",
                save_pdf=save_pdf,
                step_offset=1,
                ens_color='tab:blue',
                ring_plot_history=False,
            )
    elif plot_figures and len(unresolved_global_indices) > 0:
        unresolved_sorted = sorted(list(unresolved_global_indices))
        print(
            f"Warning: some global test_plot_index values were not found in loader batches: "
            f"{unresolved_sorted[:10]}{'...' if len(unresolved_sorted) > 10 else ''}"
        )

    final_metrics = {}
    if all_results['rrmse'].numel() == 0:
        metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rrmse', 'std_rrmse',
                        'mean_rmv', 'std_rmv', 'mean_spread_error_ratio', 'std_spread_error_ratio',
                        'mean_crps', 'std_crps', 'mean_rcrps', 'std_rcrps',
                        'mean_cov_diff', 'std_cov_diff', 'mean_rcov_diff', 'std_rcov_diff',
                        'mean_pf_rmse', 'std_pf_rmse', 'mean_pf_rrmse', 'std_pf_rrmse']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        nan_mask = torch.isnan(all_results['rrmse'])
        valid_B_mask = ~nan_mask

        if not valid_B_mask.any():
            metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rrmse', 'std_rrmse',
                            'mean_rmv', 'std_rmv', 'mean_spread_error_ratio', 'std_spread_error_ratio',
                            'mean_crps', 'std_crps', 'mean_rcrps', 'std_rcrps',
                            'mean_cov_diff', 'std_cov_diff', 'mean_rcov_diff', 'std_rcov_diff',
                            'mean_pf_rmse', 'std_pf_rmse', 'mean_pf_rrmse', 'std_pf_rrmse']
            final_metrics = {key: float('nan') for key in metrics_keys}
            final_metrics['no_nan_percent'] = 0.0
        else:
            final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
            final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
            final_metrics['mean_rmv'], final_metrics['std_rmv'] = get_mean_std(all_results['rmv'][valid_B_mask])
            final_metrics['mean_spread_error_ratio'], final_metrics['std_spread_error_ratio'] = get_mean_std(
                all_results['spread_error_ratio'][valid_B_mask]
            )
            final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
            final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
            final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0
            if args.pf_verification:
                final_metrics['mean_cov_diff'], final_metrics['std_cov_diff'] = get_mean_std(all_results['cov_diff'][valid_B_mask])
                final_metrics['mean_rcov_diff'], final_metrics['std_rcov_diff'] = get_mean_std(all_results['rcov_diff'][valid_B_mask])
                final_metrics['mean_pf_rmse'], final_metrics['std_pf_rmse'] = get_mean_std(all_results['pf_rmse'][valid_B_mask])
                final_metrics['mean_pf_rrmse'], final_metrics['std_pf_rrmse'] = get_mean_std(all_results['pf_rrmse'][valid_B_mask])

    snr_values = all_results['snr_var']
    finite_snr_mask = torch.isfinite(snr_values)
    if finite_snr_mask.any():
        final_metrics['mean_snr_var'], final_metrics['std_snr_var'] = get_mean_std(snr_values[finite_snr_mask])
    else:
        final_metrics['mean_snr_var'] = float('nan')
        final_metrics['std_snr_var'] = float('nan')

    if rank_total_samples > 0:
        rank_probs = rank_counts_total.to(torch.float64) / float(rank_total_samples)
        rank_uniform = torch.full_like(rank_probs, 1.0 / float(rank_counts_total.numel()))
        rank_l1 = torch.sum(torch.abs(rank_probs - rank_uniform))
        rank_l2 = torch.sqrt(torch.mean((rank_probs - rank_uniform) ** 2))
        rank_freq_range = torch.max(rank_probs) - torch.min(rank_probs)
        rank_expected = float(rank_total_samples) / float(rank_counts_total.numel())
        rank_chi2 = torch.sum((rank_counts_total.to(torch.float64) - rank_expected) ** 2 / max(rank_expected, 1e-12))

        final_metrics['rank_hist_counts'] = rank_counts_total.tolist()
        final_metrics['rank_hist_probs'] = rank_probs.to(torch.float32).tolist()
        final_metrics['rank_total_samples'] = int(rank_total_samples)
        final_metrics['rank_num_projections'] = int(rank_num_projections)
        final_metrics['rank_freq_range'] = float(rank_freq_range.item())
        final_metrics['rank_uniform_l1'] = float(rank_l1.item())
        final_metrics['rank_uniform_l2'] = float(rank_l2.item())
        final_metrics['rank_chi2'] = float(rank_chi2.item())
    else:
        final_metrics['rank_hist_counts'] = [0 for _ in range(int(args.N) + 1)]
        final_metrics['rank_hist_probs'] = [float('nan') for _ in range(int(args.N) + 1)]
        final_metrics['rank_total_samples'] = 0
        final_metrics['rank_num_projections'] = int(rank_num_projections)
        final_metrics['rank_freq_range'] = float('nan')
        final_metrics['rank_uniform_l1'] = float('nan')
        final_metrics['rank_uniform_l2'] = float('nan')
        final_metrics['rank_chi2'] = float('nan')

    # ---- NEW: compute timing stats and attach to final_metrics ----
    # We compute (1) per-step mean/std and (2) trajectory-weighted mean/std.
    if len(assim_step_times) > 0:
        t_tensor = torch.tensor(assim_step_times, dtype=torch.float64)
        final_metrics['assim_step_time_mean'] = float(t_tensor.mean().item())               # seconds
        final_metrics['assim_step_time_std']  = float(t_tensor.std(unbiased=False).item())  # population std
    else:
        final_metrics['assim_step_time_mean'] = float('nan')
        final_metrics['assim_step_time_std']  = float('nan')

    if len(assim_step_times_weighted) > 0:
        tw_tensor = torch.tensor(assim_step_times_weighted, dtype=torch.float64)
        final_metrics['assim_step_time_mean_weighted'] = float(tw_tensor.mean().item())
        final_metrics['assim_step_time_std_weighted']  = float(tw_tensor.std(unbiased=False).item())
    else:
        final_metrics['assim_step_time_mean_weighted'] = float('nan')
        final_metrics['assim_step_time_std_weighted']  = float('nan')

    return final_metrics





def train_model_v2(epoch, loader, model_list, optimizer, scheduler, args):
    """
    Function to train the model for one epoch (V2 with linear dynamics).

    Args:
        epoch (int): The current epoch number.
        loader (DataLoader): The data loader for training data.
        model_list (list): A list of models to be trained.
        optimizer (Optimizer): The optimizer for updating model weights.
        scheduler (Scheduler): The learning rate scheduler.
        args (Namespace): A namespace containing all arguments/hyperparameters.

    Returns:
        float: The average loss for the epoch.
    """
    model, infl_model, local_model, st_model1, st_model2 = model_list
    N = args.N
    losses = AverageMeter()
    batch_time = AverageMeter()
    
    # --- Model and Observation Operator ---
    if args.dataset == "linear":
        forward_fun = lambda x, A: torch.bmm(x, A.transpose(-1,-2))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    H_fun = lambda x, H: torch.bmm(x, H.transpose(-1,-2))
    _setup_mixed_precision(args)
    
    # --- Set Models to Train Mode ---
    model.train()
    if hasattr(infl_model, 'train'): infl_model.train()
    if hasattr(local_model, 'train'): local_model.train()
    if hasattr(st_model1, 'train'): st_model1.train()
    if hasattr(st_model2, 'train'): st_model2.train()

    all_trainable_params = []
    for m_ in [model, infl_model, local_model, st_model1, st_model2]:
        if hasattr(m_, 'parameters') and not isinstance(m_, NaiveNetwork):
            all_trainable_params.extend(list(filter(lambda p: p.requires_grad, m_.parameters())))

    num_batches_all_nan = 0

    # --- Batch Loop ---
    for batch_ind, batch_info in enumerate(loader):
        t_start = time.time()
        
        m, A, C, H, sigma_v, sigma_y = (batch_info['m'], batch_info['A'], batch_info['C'], 
                                        batch_info['H'], batch_info['sigma_v'].squeeze(), 
                                        batch_info['sigma_y'].squeeze())
        m, A, C, H, sigma_v, sigma_y = (m.to(args.device), A.to(args.device), C.to(args.device), 
                                        H.to(args.device), sigma_v.to(args.device), sigma_y.to(args.device))
        
        current_actual_batch_size = m.shape[0]
        D, D_obs = args.ori_dim, args.obs_dim
        optimizer.zero_grad()
        loss_for_backward = None
        
        with _autocast_context(args):
            # --- Initialize States and Noise Covariances ---
            ens_v_a = m.unsqueeze(1).repeat(1, N, 1) + torch.bmm(torch.randn_like(m.unsqueeze(1).repeat(1, N, 1)), C.transpose(-1,-2))
            gt_v_a = m.unsqueeze(1)
            Q = (sigma_v.view(current_actual_batch_size, 1, 1) ** 2) * torch.eye(D, device=args.device).unsqueeze(0)
            R = (sigma_y.view(current_actual_batch_size, 1, 1) ** 2) * torch.eye(D_obs, device=args.device).unsqueeze(0)

            # --- Initialize variables for loss calculation ---
            accumulated_loss_for_batch_load = 0.0
            num_valid_loss_contributions = 0
            running_valid_count_t = torch.zeros((), device=args.device, dtype=torch.long) if args.running_loss else None
            collected_ens_v_a = [] if not args.running_loss else None
            collected_gt_v_a = [] if not args.running_loss else None

            end_ind_t = min(epoch + 1, args.train_steps - 1) if args.loss_warm_up else args.train_steps - 1
            
            if end_ind_t <= 0:
                if (batch_ind + 1) % args.print_batch == 0:
                    print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                          f'Skipped batch due to end_ind_t <=0 (Warm-up)')
                batch_time.update(time.time() - t_start)
                continue

            # ======================= [REFACTORED UNIFIED TIME-STEP LOOP] =======================
            for i in range(end_ind_t):
                # --- Common Forecast and Analysis Step ---
                gt_v_a = forward_fun(gt_v_a, A) + torch.bmm(torch.randn_like(gt_v_a), Q.sqrt())
                obs_y = H_fun(gt_v_a, H) + torch.bmm(torch.randn_like(H_fun(gt_v_a, H)), R.sqrt())
                ens_v_f = forward_fun(ens_v_a, A) + torch.bmm(torch.randn_like(ens_v_a), Q.sqrt())
                hv = H_fun(ens_v_f, H)
                r_noise = mean0(torch.bmm(torch.randn_like(hv), R.sqrt()))
                ens_i_innov = obs_y - hv - r_noise
                mean_hv = torch.mean(hv, dim=1, keepdim=True)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)

                current_analyzed_ens_v_a, _ = _process_analysis_step(
                    args, model_list, ens_v_f, hv, obs_y,
                    ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=sigma_y,
                )
                
                # --- Strategy-Dependent Action inside the loop ---
                if args.running_loss:
                    if (i + 1) > args.ignore_first:
                        ens_tensor_step = current_analyzed_ens_v_a.unsqueeze(0)
                        batch_v_step = gt_v_a.squeeze(1).unsqueeze(0)
                        valid_B_mask_this_step = ~torch.isnan(ens_tensor_step).any(dim=(0, 2, 3)).squeeze(0)

                        if valid_B_mask_this_step.any():
                            step_loss = sum(compute_loss(
                                ens_tensor=ens_tensor_step, batch_v=batch_v_step, loss_type=lt, ignore_first=0, end_ind=None,
                                valid_B_mask=valid_B_mask_this_step.unsqueeze(0), norm_p=args.es_p,
                                kes_sigma=args.kes_sigma, return_sum=True) for lt in args.loss_type)
                            
                            accumulated_loss_for_batch_load += step_loss
                            running_valid_count_t += torch.sum(valid_B_mask_this_step)
                else: # Trajectory loss: collect tensors
                    collected_ens_v_a.append(current_analyzed_ens_v_a)
                    collected_gt_v_a.append(gt_v_a)
                
                # --- Common State Update and Detach Logic ---
                ens_v_a = current_analyzed_ens_v_a
                if epoch <= args.detach_training_epoch and args.detach_steps > 0 and (i + 1) % args.detach_steps == 0 and (i + 1) < end_ind_t:
                    ens_v_a = ens_v_a.detach()
            
            # ======================= [POST-LOOP BACKPROPAGATION] =======================
            if args.running_loss:
                num_valid_loss_contributions = int(running_valid_count_t.item())
                if num_valid_loss_contributions > 0:
                    average_loss = accumulated_loss_for_batch_load / running_valid_count_t.to(accumulated_loss_for_batch_load.dtype)
                    loss_for_backward = average_loss
                    losses.update(average_loss.detach().item(), num_valid_loss_contributions)
                else:
                    num_batches_all_nan += 1
            else: # Trajectory loss
                if len(collected_ens_v_a) > args.ignore_first:
                    ens_tensor = torch.stack(collected_ens_v_a, dim=0)[args.ignore_first:]
                    batch_v = torch.stack(collected_gt_v_a, dim=0).squeeze(2)[args.ignore_first:]
                    
                    if ens_tensor.shape[0] > 0:
                        valid_B_mask = ~torch.isnan(ens_tensor).any(dim=(2, 3))
                        num_valid_loss_contributions = torch.sum(valid_B_mask).item()

                        if num_valid_loss_contributions > 0:
                            normalize_val = None
                            total_loss = sum(compute_loss(
                                ens_tensor=ens_tensor, batch_v=batch_v, loss_type=lt, ignore_first=0, end_ind=None,
                                valid_B_mask=valid_B_mask, norm_p=args.es_p, kes_sigma=args.kes_sigma, return_sum=True,
                                normalize_val=normalize_val,
                            ) for lt in args.loss_type)
                            
                            average_loss = total_loss / num_valid_loss_contributions
                            loss_for_backward = average_loss
                            losses.update(average_loss.item(), num_valid_loss_contributions)
                        else:
                            num_batches_all_nan += 1
                    else:
                        num_batches_all_nan += 1
                else:
                    num_batches_all_nan += 1
        if loss_for_backward is not None:
            _backward_and_step(loss_for_backward, optimizer, all_trainable_params, args)

        # --- Common Logging and Timing ---
        batch_time.update(time.time() - t_start)
        total_possible_contributions = current_actual_batch_size * max(0, end_ind_t - args.ignore_first)
        current_no_nan_percentage = (num_valid_loss_contributions / total_possible_contributions * 100) if total_possible_contributions > 0 else 100.0

        if (batch_ind + 1) % args.print_batch == 0:
            print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                  f'Batch time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} (Avg: {losses.avg:.3f})\t'
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}\t'
                  f'No NAN % (batch): {current_no_nan_percentage:.2f}%')

    if num_batches_all_nan == len(loader) and len(loader) > 0:
        print(f"Warning: All {len(loader)} batches in epoch {epoch} resulted in NaN or no valid loss.")
        if losses.count == 0:
            return float('nan')

    scheduler.step()
    return losses.avg

# Helper function to compute batch-wise covariance
def get_ens_cov(ens_v):
    """
    Computes the sample covariance matrix for an ensemble of states.
    Input:
        ens_v: Ensemble of states with shape (B, N, D)
               B: batch size, N: ensemble members, D: state dimension
    Output:
        Covariance matrix with shape (B, D, D)
    """
    mean = torch.mean(ens_v, dim=1, keepdim=True)
    ens_v_minus_mean = ens_v - mean
    cov = torch.bmm(ens_v_minus_mean.transpose(1, 2), ens_v_minus_mean) / (ens_v.shape[1] - 1)
    return cov

def test_model_v2(loader, model_list, args, plot_figures=True, fig_name='example_fig_v2', save_pdf=False, infl=1, loc_radius=None):
    """
    Tests the data assimilation models on a linear dataset.
    This version computes and compares against the ground truth error covariance
    updated via Kalman Filter equations.
    """
    model, infl_model, local_model, st_model1, st_model2 = model_list
    N = args.N

    # --- Forward and Observation Function Initialization for Linear Model ---
    forward_fun = lambda x, A: torch.bmm(x, A.transpose(-1, -2))
    H_fun = lambda x, H: torch.bmm(x, H.transpose(-1, -2))
    test_noise_gen = get_test_noise_generator(args)

    # --- Set Models to Evaluation Mode ---
    model.eval()
    if hasattr(infl_model, 'eval'): infl_model.eval()
    if hasattr(local_model, 'eval'): local_model.eval()
    if hasattr(st_model1, 'eval'): st_model1.eval()
    if hasattr(st_model2, 'eval'): st_model2.eval()

    # --- Initialize Result Tensors for All Batches---
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rmv': torch.empty(0, device=args.device), # Using RMV for spread
        'rrmse': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'snr_var': torch.empty(0, device=args.device),
        'cov_diff': torch.empty(0, device=args.device),
        'rcov_diff': torch.empty(0, device=args.device),
        'w2_diff': torch.empty(0, device=args.device),
    }
    loc_tensor_all_batches = None

    with torch.no_grad():
        min_m_norm, max_m_norm = float('inf'), float('-inf')
        for batch_ind, batch_info in enumerate(loader):
            # --- Unpack Batch Data ---
            m, A, C, H, sigma_v, sigma_y = (batch_info['m'], batch_info['A'], batch_info['C'], 
                                            batch_info['H'], batch_info['sigma_v'].squeeze(), 
                                            batch_info['sigma_y'].squeeze())
            m, A, C, H, sigma_v, sigma_y = (m.to(args.device), A.to(args.device), C.to(args.device), 
                                            H.to(args.device), sigma_v.to(args.device), sigma_y.to(args.device))
            
            batch_norms = torch.linalg.norm(m, dim=1)
            current_min_norm = batch_norms.min().item()
            current_max_norm = batch_norms.max().item()
            min_m_norm = min(min_m_norm, current_min_norm)
            max_m_norm = max(max_m_norm, current_max_norm)
            
            B, D, D_obs = m.shape[0], args.ori_dim, args.obs_dim
            
            # --- Initialize Ground Truth and Ensemble States ---
            gt_v_a = m.unsqueeze(1)
            kalman_m_a = m.unsqueeze(1)
            # Initial analysis error covariance is the provided C
            P_a = C @ C.transpose(-1, -2) 
            
            # Initial ensemble
            ens_v_a = m.unsqueeze(1).repeat(1, N, 1)
            ens_v_a += torch.bmm(torch.randn_like(ens_v_a, device=args.device), C.transpose(-1,-2))
            init_P_ens_a = get_ens_cov(ens_v_a)

            # --- Lists to Store Trajectory Data for the Current Batch ---
            ens_list = [ens_v_a]
            gt_list = [gt_v_a.squeeze(1)]
            loc_records = []
            cov_diff_list = []
            rcov_diff_list = []
            w2_diff_list = []
            kalman_mean_list = [kalman_m_a.squeeze(1)]
            kalman_cov_list = [P_a]
            method_mean_list = [ens_v_a.mean(dim=1)]
            method_cov_list = [init_P_ens_a]

            # --- Main Assimilation Loop ---
            for i in range(args.test_steps -1):
                # --- Ground Truth Evolution ---
                Q = (sigma_v.view(B, 1, 1) ** 2) * torch.eye(D, device=args.device).unsqueeze(0).repeat(B, 1, 1)
                R = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(D_obs, device=args.device).unsqueeze(0).repeat(B, 1, 1)
                gt_v_f = forward_fun(gt_v_a, A)
                gt_v_a = gt_v_f + torch.bmm(torch.randn_like(gt_v_f), Q.sqrt())
                clean_h = H_fun(gt_v_a, H)
                obs_noise = _sample_true_obs_noise_like(clean_h, test_noise_gen)
                obs_y = clean_h + torch.bmm(obs_noise, R.sqrt())

                # --- Kalman Filter Mean/Covariance Update (Ground Truth Reference) ---
                kalman_m_f = forward_fun(kalman_m_a, A)
                P_f = A @ P_a @ A.transpose(-1, -2) + Q
                S = H @ P_f @ H.transpose(-1, -2) + R
                K = P_f @ H.transpose(-1, -2) @ torch.inverse(S)
                kalman_m_a = kalman_m_f + torch.bmm(
                    (obs_y - H_fun(kalman_m_f, H)),
                    K.transpose(-1, -2),
                )
                P_a = (torch.eye(D, device=args.device).unsqueeze(0).repeat(B,1,1) - K @ H) @ P_f

                # --- Ensemble Forecast ---
                ens_v_f = forward_fun(ens_v_a, A)
                # Add model error (process noise)
                ens_v_f += torch.bmm(torch.randn_like(ens_v_f), Q.sqrt())
                
                # --- Prepare for Analysis ---
                hv = H_fun(ens_v_f, H)
                r_noise = mean0(torch.bmm(torch.randn_like(hv), R.sqrt()))
                ens_i_innov = obs_y - hv - r_noise
                mean_hv = torch.mean(hv, dim=1, keepdim=True)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True)
                
                # --- Analysis Step (Refactored) ---
                ens_v_a, loc_nn_output = _process_analysis_step(
                    args, model_list, ens_v_f, hv, obs_y, ens_i_innov, mean_ens_v_f, mean_hv, sigma_y=sigma_y,
                    infl=infl, loc_radius=loc_radius, 
                )

                # --- Calculate and Store Covariance Difference ---
                P_ens_a = get_ens_cov(ens_v_a)
                cov_diff = torch.norm(P_ens_a - P_a, p='fro', dim=(-2, -1))
                rcov_diff =  cov_diff / torch.norm(P_a, p='fro', dim=(-2, -1))
                w2_diff = wasserstein2_multivariate_gaussian(mean_true=gt_v_a, cov_true=P_a, mean_sample=ens_v_a.mean(dim=1), cov_sample=P_ens_a)
                cov_diff_list.append(cov_diff)
                rcov_diff_list.append(rcov_diff)
                w2_diff_list.append(w2_diff)
                
                if loc_nn_output is not None:
                    loc_records.append(loc_nn_output)
                
                # --- Store results for this step ---
                ens_list.append(ens_v_a)
                gt_list.append(gt_v_a.squeeze(1))
                kalman_mean_list.append(kalman_m_a.squeeze(1))
                kalman_cov_list.append(P_a)
                method_mean_list.append(ens_v_a.mean(dim=1))
                method_cov_list.append(P_ens_a)
            
            # --- Process and Store Batch Results ---
            ens_tensor = torch.stack(ens_list, dim=0)
            gt_tensor = torch.stack(gt_list, dim=0)
            kalman_mean_tensor = torch.stack(kalman_mean_list, dim=0)
            kalman_cov_tensor = torch.stack(kalman_cov_list, dim=0)
            method_mean_tensor = torch.stack(method_mean_list, dim=0)
            method_cov_tensor = torch.stack(method_cov_list, dim=0)

            # SNR_var per trajectory using clean observations h(v_j) = H v_j.
            h_tensor = torch.einsum('tbd,bod->tbo', gt_tensor, H)
            snr_var_batch = _compute_traj_snr_var_from_hvalues(h_tensor, sigma_y)
            
            # --- Metric Calculation for the Batch ---
            # Note: We compute mean over time steps (dim=0) first, then over batch items
            rmse_val = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - gt_tensor) ** 2, dim=-1)).mean(dim=0)
            rms_val = torch.sqrt(torch.mean(gt_tensor ** 2, dim=-1)).mean(dim=0)
            rrmse_val = rmse_val / rms_val
            rmv_val = torch.sqrt(torch.mean((ens_tensor - gt_tensor.unsqueeze(2))**2, dim=(2,-1))).mean(dim=0)
            crps_val = torch.mean(compute_es(ens_states=ens_tensor, true_states=gt_tensor, norm_p=1), dim=0) 
            rcrps_val = crps_val / torch.mean(torch.norm(gt_tensor, p=2, dim=-1), dim=0)
            
            cov_diff_tensor = torch.stack(cov_diff_list, dim=0)
            rcov_diff_tensor = torch.stack(rcov_diff_list, dim=0)
            w2_diff_tensor = torch.stack(w2_diff_list, dim=0)
            cov_diff_val = cov_diff_tensor.mean(dim=0)
            rcov_diff_val = rcov_diff_tensor.mean(dim=0)
            w2_diff_val = w2_diff_tensor.mean(dim=0)

            # --- Aggregate Results ---
            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rmv'] = torch.cat((all_results['rmv'], rmv_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            all_results['snr_var'] = torch.cat((all_results['snr_var'], snr_var_batch))
            all_results['cov_diff'] = torch.cat((all_results['cov_diff'], cov_diff_val))
            all_results['rcov_diff'] = torch.cat((all_results['rcov_diff'], rcov_diff_val))
            all_results['w2_diff'] = torch.cat((all_results['w2_diff'], w2_diff_val))
            
            # Handle localization tensor if it exists
            if args.v != "EtE" and not args.no_localization and loc_records:
                current_loc_tensor = torch.stack(loc_records).unsqueeze(0) # Add batch dim
                if loc_tensor_all_batches is None:
                    loc_tensor_all_batches = current_loc_tensor
                else:
                    loc_tensor_all_batches = torch.cat((loc_tensor_all_batches, current_loc_tensor), dim=0)
                    
        if plot_figures: 
            # Reuse test_plot_index policy to choose one trajectory.
            nan_per_step = torch.isnan(ens_tensor).any(dim=(2, 3))
            nan_counts = nan_per_step.sum(dim=0)
            vis_indices = _resolve_plot_batch_indices(args, batch_size=ens_tensor.shape[1], nan_counts=nan_counts)
            vis_bidx = int(vis_indices[0]) if len(vis_indices) > 0 else 0
            vis_bidx = max(0, min(vis_bidx, ens_tensor.shape[1] - 1))

            # Default to the last 20 steps for the 2D projection.
            tail_steps = int(getattr(args, "linear_vis_tail_steps", 20))
            if args.ori_dim >= 2:
                plot_linear_kalman_vs_method_2d(
                    kalman_means=kalman_mean_tensor[:, vis_bidx, :],
                    kalman_covs=kalman_cov_tensor[:, vis_bidx, :, :],
                    method_means=method_mean_tensor[:, vis_bidx, :],
                    method_covs=method_cov_tensor[:, vis_bidx, :, :],
                    dim_indices=(0, 1),
                    tail_steps=tail_steps,
                    save_fig=True,
                    save_pdf=save_pdf,
                    save_name=f"{fig_name}_kalman2d_b{vis_bidx}",
                    legend_in_figure=getattr(args, "legend_in_figure", False),
                )
            else:
                dim_indices_plot = list(range(min(args.ori_dim, 1)))
                plot_particle_trajectories_with_histograms(
                    particles=ens_tensor[:, vis_bidx, :, :],
                    true_traj=gt_tensor[:, vis_bidx, :],
                    observation=None,
                    dim_indices=dim_indices_plot,
                    start_time=max(0, ens_tensor.shape[0] - tail_steps),
                    end_time=ens_tensor.shape[0],
                    mode='quantile',
                    save_fig=True,
                    save_pdf=save_pdf,
                    save_name=fig_name + "_hist",
                    hist_step=1,
                    fontsize=None,
                    legend_in_figure=getattr(args, "legend_in_figure", False),
                )

    # --- Final Metrics Calculation ---
    final_metrics = {}
    
    # Check for NaN values in a reference metric (e.g., rrmse)
    nan_mask = torch.isnan(all_results['rrmse'])
    valid_B_mask = ~nan_mask
    
    if not valid_B_mask.any():
         # Return NaNs if all results are invalid
        metrics_keys = ['mean_rmse', 'std_rmse', 'mean_rmv', 'std_rmv', 
                        'mean_rrmse', 'std_rrmse', 'mean_crps', 'std_crps',
                        'mean_rcrps', 'std_rcrps', 'mean_cov_diff', 'std_cov_diff',
                        'mean_rcov_diff', 'std_rcov_diff', 'mean_w2_diff', 'std_w2_diff',
                        'min_m_norm', 'max_m_norm']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        # Calculate mean and std for valid results
        final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
        final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
        final_metrics['mean_rmv'], final_metrics['std_rmv'] = get_mean_std(all_results['rmv'][valid_B_mask])
        final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
        final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
        final_metrics['mean_cov_diff'], final_metrics['std_cov_diff'] = get_mean_std(all_results['cov_diff'][valid_B_mask])
        final_metrics['mean_rcov_diff'], final_metrics['std_rcov_diff'] = get_mean_std(all_results['rcov_diff'][valid_B_mask])
        final_metrics['mean_w2_diff'], final_metrics['std_w2_diff'] = get_mean_std(all_results['w2_diff'][valid_B_mask])
        final_metrics['min_m_norm'], final_metrics['max_m_norm'] = min_m_norm, max_m_norm
        final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0

    snr_values = all_results['snr_var']
    finite_snr_mask = torch.isfinite(snr_values)
    if finite_snr_mask.any():
        final_metrics['mean_snr_var'], final_metrics['std_snr_var'] = get_mean_std(snr_values[finite_snr_mask])
    else:
        final_metrics['mean_snr_var'] = float('nan')
        final_metrics['std_snr_var'] = float('nan')

    final_loc_tensor = loc_tensor_all_batches[0] if loc_tensor_all_batches is not None else torch.empty(1, device=args.device)

    final_metrics['loc_tensor'] = final_loc_tensor
    return final_metrics

def print_test_results_v2(results):
    """
    Print formatted test results from a dictionary.

    Args:
        results (dict): A dictionary containing metric names as keys
                        and their calculated values. It checks for the
                        existence of keys before printing.
    """
    if 'mean_rmse' in results and 'std_rmse' in results:
        print(f"RMSE: {results['mean_rmse']:.3f} ± {results['std_rmse']:.3f}")
        
    if 'mean_rrmse' in results and 'std_rrmse' in results:
        print(f"RRMSE: {results['mean_rrmse']:.3f} ± {results['std_rrmse']:.3f}")
        
    if 'mean_rmv' in results and 'std_rmv' in results:
        print(f"RMV: {results['mean_rmv']:.3f} ± {results['std_rmv']:.3f}")
        
    if 'mean_crps' in results and 'std_crps' in results:
        print(f"CRPS: {results['mean_crps']:.3f} ± {results['std_crps']:.3f}")
        
    if 'mean_rcrps' in results and 'std_rcrps' in results:
        print(f"RCRPS: {results['mean_rcrps']:.3f} ± {results['std_rcrps']:.3f}")
        
    if 'mean_w2_diff' in results and 'std_w2_diff' in results:
        print(f"W2: {results['mean_w2_diff']:.3f} ± {results['std_w2_diff']:.3f}")
        
    if 'no_nan_percent' in results:
        print(f"No NAN Percentage: {results['no_nan_percent']:.2f}%")
        
    if 'min_m_norm' in results and 'max_m_norm' in results:
        print(f"Min initial norm: {results['min_m_norm']:.2f}, Max initial norm: {results['max_m_norm']:.2f}")

    if 'mean_snr_var' in results and 'std_snr_var' in results:
        print(f"SNR_var: {results['mean_snr_var']:.3f} ± {results['std_snr_var']:.3f}")
    

### test gt uncertainty 
def test_linear_sampling_error(loader, args, num_resamples):
    """
    Computes the sampling error of an ensemble in a linear-Gaussian system.

    This function runs a Kalman Filter to get the true posterior mean and
    covariance. At each step, it repeatedly samples from this true posterior
    to measure the statistical error (RMSE, CRPS, W2) introduced by a
    finite ensemble size N.

    Args:
        loader (DataLoader): DataLoader for initial conditions.
        args (Namespace): Experiment parameters (N, device, test_steps, etc.).
        num_resamples (int): Number of times to resample at each step for averaging.

    Returns:
        dict: A dictionary of final, averaged metrics and their standard deviation.
    """
    N = args.N
    D = args.ori_dim
    D_obs = args.obs_dim

    # --- Forward and Observation Function Initialization ---
    forward_fun = lambda x, A: torch.bmm(x, A.transpose(-1, -2))
    H_fun = lambda x, H: torch.bmm(x, H.transpose(-1, -2))
    test_noise_gen = get_test_noise_generator(args)

    # --- Initialize Result Tensors for All Batches---
    all_results = {
        'rmse': torch.empty(0, device=args.device),
        'rmv': torch.empty(0, device=args.device),
        'rrmse': torch.empty(0, device=args.device),
        'crps': torch.empty(0, device=args.device),
        'rcrps': torch.empty(0, device=args.device),
        'w2_diff': torch.empty(0, device=args.device),
    }

    with torch.no_grad():
        for batch_ind, batch_info in enumerate(loader):
            # --- Unpack Batch Data ---
            m, A, C, H, sigma_v, sigma_y = (batch_info['m'], batch_info['A'], batch_info['C'], 
                                             batch_info['H'], batch_info['sigma_v'].squeeze(), 
                                             batch_info['sigma_y'].squeeze())
            m, A, C, H, sigma_v, sigma_y = (m.to(args.device), A.to(args.device), C.to(args.device), 
                                             H.to(args.device), sigma_v.to(args.device), sigma_y.to(args.device))
            
            B = m.shape[0]
            
            # --- Initialize Ground Truth State and Covariance ---
            gt_v_a = m.unsqueeze(1) # Shape: (B, 1, D)
            P_a = C @ C.transpose(-1, -2) 
            
            # --- Lists to Store Trajectory-Averaged Data ---
            gt_list = [gt_v_a.squeeze(1)]
            rmse_list, rmv_list, crps_list, w2_diff_list = [], [], [], []

            # --- Main Assimilation Loop ---
            for i in range(args.test_steps - 1):
                print(i, args.test_steps-1)
                # --- Ground Truth Evolution (Kalman Filter) ---
                Q = (sigma_v.view(B, 1, 1) ** 2) * torch.eye(D, device=args.device)
                R = (sigma_y.view(B, 1, 1) ** 2) * torch.eye(D_obs, device=args.device)
                
                gt_v_f = forward_fun(gt_v_a, A)
                gt_v_a = gt_v_f + torch.bmm(torch.randn_like(gt_v_f), Q.sqrt())
                clean_h = H_fun(gt_v_a, H)
                obs_noise = _sample_true_obs_noise_like(clean_h, test_noise_gen)
                obs_y = clean_h + torch.bmm(obs_noise, R.sqrt())

                # --- Kalman Filter Update (Ground Truth) ---
                P_f = A @ P_a @ A.transpose(-1, -2) + Q
                S = H @ P_f @ H.transpose(-1, -2) + R
                K = P_f @ H.transpose(-1, -2) @ torch.inverse(S)
                P_a = (torch.eye(D, device=args.device) - K @ H) @ P_f
                gt_v_a = (gt_v_f.transpose(-1,-2) + K @ (obs_y - H_fun(gt_v_f, H)).transpose(-1,-2)).transpose(-1,-2)

                # --- Resampling and Metric Calculation Loop ---
                step_rmses, step_rmvs, step_crpses, step_w2s = [], [], [], []
                L_a = torch.linalg.cholesky(P_a)
                
                for _ in range(num_resamples):
                    noise = torch.randn(B, N, D, device=args.device)
                    ens_v_a = gt_v_a + torch.bmm(noise, L_a.transpose(-1,-2))
                    
                    ens_mean = ens_v_a.mean(dim=1)
                    P_ens_a = get_ens_cov(ens_v_a) 

                    # --- Calculate Metrics for this Sample Set ---
                    rmse = torch.sqrt(torch.mean((ens_mean - gt_v_a.squeeze(1)) ** 2, dim=-1))
                    step_rmses.append(rmse)

                    rmv = torch.sqrt(torch.mean((ens_v_a - gt_v_a)**2, dim=(1,-1)))
                    step_rmvs.append(rmv)
                    
                    crps = compute_es(ens_v_a.unsqueeze(0), gt_v_a.squeeze(1).unsqueeze(0)).squeeze(0)
                    step_crpses.append(crps)

                    w2 = wasserstein2_multivariate_gaussian(
                        mean_true=gt_v_a.squeeze(1), cov_true=P_a, 
                        mean_sample=ens_mean, cov_sample=P_ens_a
                    )
                    step_w2s.append(w2)

                # --- Average metrics over the resamples for this time step ---
                rmse_list.append(torch.stack(step_rmses).mean(dim=0))
                rmv_list.append(torch.stack(step_rmvs).mean(dim=0))
                crps_list.append(torch.stack(step_crpses).mean(dim=0))
                w2_diff_list.append(torch.stack(step_w2s).mean(dim=0))
                
                gt_list.append(gt_v_a.squeeze(1))

            # --- Process and Store Batch Results (Averaged over trajectory) ---
            gt_tensor = torch.stack(gt_list, dim=0)
            
            rmse_val = torch.stack(rmse_list).mean(dim=0)
            rmv_val = torch.stack(rmv_list).mean(dim=0)
            crps_val = torch.stack(crps_list).mean(dim=0)
            w2_diff_val = torch.stack(w2_diff_list).mean(dim=0)

            rms_val = torch.sqrt(torch.mean(gt_tensor ** 2, dim=-1)).mean(dim=0)
            rrmse_val = rmse_val / rms_val
            rcrps_val = crps_val / torch.mean(torch.norm(gt_tensor, p=2, dim=-1), dim=0)

            # --- Aggregate Results ---
            all_results['rmse'] = torch.cat((all_results['rmse'], rmse_val))
            all_results['rmv'] = torch.cat((all_results['rmv'], rmv_val))
            all_results['rrmse'] = torch.cat((all_results['rrmse'], rrmse_val))
            all_results['crps'] = torch.cat((all_results['crps'], crps_val))
            all_results['rcrps'] = torch.cat((all_results['rcrps'], rcrps_val))
            all_results['w2_diff'] = torch.cat((all_results['w2_diff'], w2_diff_val))

    # --- Final Metrics Calculation ---
    final_metrics = {}
    nan_mask = torch.isnan(all_results['rrmse'])
    valid_B_mask = ~nan_mask
    
    if not valid_B_mask.any():
        metrics_keys = ['mean_rrmse', 'std_rrmse', 'mean_rmse', 'std_rmse', 
                        'mean_rmv', 'std_rmv', 'mean_crps', 'std_crps', 
                        'mean_rcrps', 'std_rcrps', 'mean_w2_diff', 'std_w2_diff']
        final_metrics = {key: float('nan') for key in metrics_keys}
        final_metrics['no_nan_percent'] = 0.0
    else:
        final_metrics['mean_rrmse'], final_metrics['std_rrmse'] = get_mean_std(all_results['rrmse'][valid_B_mask])
        final_metrics['mean_rmse'], final_metrics['std_rmse'] = get_mean_std(all_results['rmse'][valid_B_mask])
        final_metrics['mean_rmv'], final_metrics['std_rmv'] = get_mean_std(all_results['rmv'][valid_B_mask])
        final_metrics['mean_crps'], final_metrics['std_crps'] = get_mean_std(all_results['crps'][valid_B_mask])
        final_metrics['mean_rcrps'], final_metrics['std_rcrps'] = get_mean_std(all_results['rcrps'][valid_B_mask])
        final_metrics['mean_w2_diff'], final_metrics['std_w2_diff'] = get_mean_std(all_results['w2_diff'][valid_B_mask])
        final_metrics['no_nan_percent'] = torch.sum(valid_B_mask).float() / all_results['rrmse'].numel() * 100.0

    return final_metrics
