import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

from config.cli import get_parameters
from config.dataset_info import DATASET_INFO


# Built-in PF particle counts (do not use args.N).
PF_N_LIST: List[int] = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

PF_FILE_PATTERN = re.compile(
    r"^pf_results_sigma_y_(?P<sigma>[^_]+)_batch_(?P<batch>\d+)_len_(?P<seq_len>\d+)_pfN_(?P<pf_N>\d+)_(?P<seed>\d+)(?P<obs_suffix>_[^.]+)?\.pt$"
)

AVERAGE_KEYS = {
    "means",
    "covs",
    "quantiles",
    "pca_quantiles",
    "ess",
    "weight_entropy",
    "weight_abundance",
    "skewness",
    "kurtosis_excess",
    "post_means",
    "post_covs",
    "post_quantiles",
    "post_pca_quantiles",
    "post_ess",
    "post_weight_entropy",
    "post_weight_abundance",
    "post_skewness",
    "post_kurtosis_excess",
    "prior_means",
    "prior_covs",
    "prior_quantiles",
    "prior_pca_quantiles",
    "prior_skewness",
    "prior_kurtosis_excess",
}


@dataclass
class PFFileMeta:
    path: Path
    sigma: str
    batch: int
    seq_len: int
    pf_N: int
    seed: int
    obs_suffix: str


def _safe_obs_name(obs_fn: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z._-]+", "-", str(obs_fn)).strip("-")
    return safe if safe else "unknown"


def _default_obs_fn(dataset: str) -> str:
    cfg = DATASET_INFO.get(str(dataset).lower(), {})
    return str(cfg.get("obs_fn", "identity") or "identity").lower()


def _effective_obs_fn(dataset: str, obs_fn: str) -> str:
    dflt = _default_obs_fn(dataset)
    user = str(obs_fn or "default").lower()
    return dflt if user == "default" else user


def _accepted_obs_suffixes(dataset: str, effective_obs_fn: str) -> List[str]:
    dflt = _default_obs_fn(dataset)
    safe = _safe_obs_name(effective_obs_fn)
    if effective_obs_fn == dflt:
        # Keep compatibility for potential legacy naming that appends default obs suffix.
        return ["", f"_{safe}"]
    return [f"_{safe}"]


def _parse_pf_file(path: Path) -> Optional[PFFileMeta]:
    m = PF_FILE_PATTERN.match(path.name)
    if m is None:
        return None
    return PFFileMeta(
        path=path,
        sigma=str(m.group("sigma")),
        batch=int(m.group("batch")),
        seq_len=int(m.group("seq_len")),
        pf_N=int(m.group("pf_N")),
        seed=int(m.group("seed")),
        obs_suffix=str(m.group("obs_suffix") or ""),
    )


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _clone_value(v: Any) -> Any:
    if torch.is_tensor(v):
        return v.clone()
    if isinstance(v, dict):
        return {k: _clone_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clone_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_clone_value(x) for x in v)
    return copy.deepcopy(v)


def _all_same_shape(tensors: Sequence[torch.Tensor]) -> bool:
    if len(tensors) <= 1:
        return True
    ref = tuple(tensors[0].shape)
    return all(tuple(t.shape) == ref for t in tensors[1:])


def _mean_tensors_keep_dtype(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    ref = tensors[0]
    stack = torch.stack([t.to(torch.float32) for t in tensors], dim=0)
    mean = stack.mean(dim=0)
    if torch.is_floating_point(ref):
        return mean.to(dtype=ref.dtype)
    return mean.round().to(dtype=ref.dtype)


def _build_avg_payload(seed_payloads: List[List[Dict[str, Any]]], seeds: List[int]) -> List[Dict[str, Any]]:
    if len(seed_payloads) == 0:
        raise ValueError("No payloads provided for averaging.")
    if len(seed_payloads) != len(seeds):
        raise ValueError("seed_payloads and seeds size mismatch.")

    num_batches = len(seed_payloads[0])
    for payload in seed_payloads:
        if len(payload) != num_batches:
            raise ValueError("Inconsistent number of batch entries across seeds.")

    max_seed = max(seeds)
    max_seed_idx = seeds.index(max_seed)
    avg_payload: List[Dict[str, Any]] = []

    for bidx in range(num_batches):
        entries = [payload[bidx] for payload in seed_payloads]
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("Each PF payload entry must be a dict.")

        max_entry = entries[max_seed_idx]
        keys_union = sorted(set().union(*(entry.keys() for entry in entries)))
        out: Dict[str, Any] = {}

        for key in keys_union:
            source_value = None
            for entry in entries:
                if key in entry:
                    source_value = entry[key]
                    break

            if key in AVERAGE_KEYS:
                vals: List[torch.Tensor] = []
                all_present = True
                for entry in entries:
                    if key not in entry or not torch.is_tensor(entry[key]):
                        all_present = False
                        break
                    vals.append(entry[key])

                if all_present and len(vals) == len(entries) and _all_same_shape(vals):
                    out[key] = _mean_tensors_keep_dtype(vals)
                else:
                    out[key] = _clone_value(source_value if key not in max_entry else max_entry[key])
            else:
                out[key] = _clone_value(source_value if key not in max_entry else max_entry[key])

        # Backward-compatible aliases for downstream code.
        if "post_means" in out and "means" not in out and torch.is_tensor(out["post_means"]):
            out["means"] = out["post_means"].clone()
        if "post_covs" in out and "covs" not in out and torch.is_tensor(out["post_covs"]):
            out["covs"] = out["post_covs"].clone()

        avg_payload.append(out)

    return avg_payload


def _pick_key(entry: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in entry:
            return k
    return None


def _concat_batches(payload: List[Dict[str, Any]], key: str) -> Optional[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for entry in payload:
        if key not in entry or not torch.is_tensor(entry[key]):
            return None
        tensors.append(entry[key].to(torch.float32))
    if len(tensors) == 0:
        return None

    if tensors[0].ndim >= 2:
        ref_t = tensors[0].shape[0]
        ref_tail = tensors[0].shape[2:]
        for t in tensors[1:]:
            if t.shape[0] != ref_t or tuple(t.shape[2:]) != tuple(ref_tail):
                return None
        return torch.cat(tensors, dim=1)
    return torch.cat(tensors, dim=0)


def _rmse_mean_error(seed_payload: List[Dict[str, Any]], avg_payload: List[Dict[str, Any]]) -> Optional[float]:
    k = _pick_key(avg_payload[0], ["post_means", "means"])
    if k is None:
        return None
    seed = _concat_batches(seed_payload, k)
    avg = _concat_batches(avg_payload, k)
    if seed is None or avg is None or seed.shape != avg.shape:
        return None
    diff = seed - avg
    rmse_tb = torch.sqrt(torch.mean(diff * diff, dim=-1))
    return float(torch.nanmean(rmse_tb).item())


def _fnorm_cov_error(seed_payload: List[Dict[str, Any]], avg_payload: List[Dict[str, Any]]) -> Optional[float]:
    k = _pick_key(avg_payload[0], ["post_covs", "covs"])
    if k is None:
        return None
    seed = _concat_batches(seed_payload, k)
    avg = _concat_batches(avg_payload, k)
    if seed is None or avg is None or seed.shape != avg.shape:
        return None
    diff = seed - avg
    fnorm_tb = torch.linalg.norm(diff, ord="fro", dim=(-2, -1))
    return float(torch.nanmean(fnorm_tb).item())


def _get_quantile_probs(payload: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    for entry in payload:
        q = entry.get("quantile_probs")
        if torch.is_tensor(q):
            return q.to(torch.float32)
    return None


def _build_quantile_6d(payload: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    state_key = _pick_key(payload[0], ["post_quantiles", "quantiles"])
    pca_key = _pick_key(payload[0], ["post_pca_quantiles", "pca_quantiles"])

    q_state = _concat_batches(payload, state_key) if state_key is not None else None
    q_pca = _concat_batches(payload, pca_key) if pca_key is not None else None
    if q_state is None and q_pca is None:
        return None

    if q_state is not None:
        t_dim, b_dim, _, k_dim = q_state.shape
        device = q_state.device
    else:
        t_dim, b_dim, _, k_dim = q_pca.shape
        device = q_pca.device

    out = torch.full((t_dim, b_dim, 6, k_dim), float("nan"), device=device, dtype=torch.float32)
    if q_state is not None:
        d = min(3, q_state.shape[2])
        out[:, :, :d, :] = q_state[:, :, :d, :]
    if q_pca is not None:
        d = min(3, q_pca.shape[2])
        out[:, :, 3:3 + d, :] = q_pca[:, :, :d, :]
    return out


def _quantile_l2_error_by_dim(
    seed_payload: List[Dict[str, Any]],
    avg_payload: List[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    q_seed = _build_quantile_6d(seed_payload)
    q_avg = _build_quantile_6d(avg_payload)
    q_probs = _get_quantile_probs(avg_payload)

    if q_seed is None or q_avg is None or q_probs is None:
        return None
    if q_seed.shape != q_avg.shape:
        return None
    if q_probs.ndim != 1 or q_probs.shape[0] != q_seed.shape[-1]:
        return None

    # L2 over quantile levels using actual (possibly non-uniform) quantile points.
    diff_sq = (q_seed - q_avg) ** 2
    l2_tb_dim = torch.sqrt(torch.trapz(diff_sq, x=q_probs.to(diff_sq.device), dim=-1))
    return torch.nanmean(l2_tb_dim, dim=(0, 1))


def _scalar_rmse_error(
    seed_payload: List[Dict[str, Any]],
    avg_payload: List[Dict[str, Any]],
    key_candidates: Sequence[str],
) -> Optional[float]:
    k = _pick_key(avg_payload[0], key_candidates)
    if k is None:
        return None
    seed = _concat_batches(seed_payload, k)
    avg = _concat_batches(avg_payload, k)
    if seed is None or avg is None or seed.shape != avg.shape:
        return None
    rmse = torch.sqrt(torch.nanmean((seed - avg) ** 2))
    return float(rmse.item())


def _scalar_mean_over_tb(
    payload: List[Dict[str, Any]],
    key_candidates: Sequence[str],
) -> Optional[float]:
    k = _pick_key(payload[0], key_candidates)
    if k is None:
        return None
    x = _concat_batches(payload, k)
    if x is None:
        return None
    return float(torch.nanmean(x).item())


def _skew_rmse_by_dim(
    seed_payload: List[Dict[str, Any]],
    avg_payload: List[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    k = _pick_key(avg_payload[0], ["post_skewness", "skewness"])
    if k is None:
        return None
    seed = _concat_batches(seed_payload, k)
    avg = _concat_batches(avg_payload, k)
    if seed is None or avg is None or seed.shape != avg.shape:
        return None
    if seed.ndim != 3:
        return None
    return torch.sqrt(torch.nanmean((seed - avg) ** 2, dim=(0, 1)))


def _skew_mean_by_dim(payload: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    k = _pick_key(payload[0], ["post_skewness", "skewness"])
    if k is None:
        return None
    x = _concat_batches(payload, k)
    if x is None or x.ndim != 3:
        return None
    return torch.nanmean(x, dim=(0, 1))


def _kurtosis_rmse_by_dim(
    seed_payload: List[Dict[str, Any]],
    avg_payload: List[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    k = _pick_key(avg_payload[0], ["post_kurtosis_excess", "kurtosis_excess"])
    if k is None:
        return None
    seed = _concat_batches(seed_payload, k)
    avg = _concat_batches(avg_payload, k)
    if seed is None or avg is None or seed.shape != avg.shape:
        return None
    if seed.ndim != 3:
        return None
    return torch.sqrt(torch.nanmean((seed - avg) ** 2, dim=(0, 1)))


def _kurtosis_mean_by_dim(payload: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    k = _pick_key(payload[0], ["post_kurtosis_excess", "kurtosis_excess"])
    if k is None:
        return None
    x = _concat_batches(payload, k)
    if x is None or x.ndim != 3:
        return None
    return torch.nanmean(x, dim=(0, 1))


def _standard_error(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return float("nan")
    x = torch.tensor(vals, dtype=torch.float32)
    return float((torch.std(x, unbiased=True) / math.sqrt(len(vals))).item())


def _standard_error_by_dim(values: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    arr = [v for v in values if v is not None]
    if len(arr) < 2:
        return None
    if not _all_same_shape(arr):
        return None
    x = torch.stack([v.to(torch.float32) for v in arr], dim=0)
    return torch.std(x, dim=0, unbiased=True) / math.sqrt(x.shape[0])


def _range(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) == 0:
        return float("nan")
    return float(max(vals) - min(vals))


def _range_by_dim(values: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    arr = [v for v in values if v is not None]
    if len(arr) == 0:
        return None
    if not _all_same_shape(arr):
        return None
    x = torch.stack([v.to(torch.float32) for v in arr], dim=0)
    return torch.max(x, dim=0).values - torch.min(x, dim=0).values


def _mean_min_max(values: Sequence[float]) -> Tuple[float, float, float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    x = torch.tensor(vals, dtype=torch.float32)
    return float(torch.mean(x).item()), float(torch.min(x).item()), float(torch.max(x).item())


def _mean_min_max_by_dim(values: Sequence[torch.Tensor]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    arr = [v for v in values if v is not None]
    if len(arr) == 0:
        return None
    if not _all_same_shape(arr):
        return None
    x = torch.stack([v.to(torch.float32) for v in arr], dim=0)
    return torch.mean(x, dim=0), torch.min(x, dim=0).values, torch.max(x, dim=0).values


def _plot_line(
    x: List[int],
    y: List[float],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x, y, marker="o", linewidth=3.0, markersize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of particles)", fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.tick_params(axis="both", labelsize=22, width=2.5, length=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_multi_line_with_band(
    x: List[int],
    centers: Dict[str, List[float]],
    lowers: Dict[str, List[float]],
    uppers: Dict[str, List[float]],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, y in centers.items():
        y_low = lowers[name]
        y_high = uppers[name]
        ax.plot(x, y, marker="o", label=name, linewidth=3.0, markersize=10)
        ax.fill_between(x, y_low, y_high, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of particles)", fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.tick_params(axis="both", labelsize=22, width=2.5, length=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=20, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_multi_line(
    x: List[int],
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, y in series.items():
        ax.plot(x, y, marker="o", label=name, linewidth=3.0, markersize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of particles)", fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.tick_params(axis="both", labelsize=22, width=2.5, length=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=20, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _maybe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if not math.isfinite(fv):
        return None
    return fv


def _default_from_dataset(dataset: str, key: str, fallback: int) -> int:
    cfg = DATASET_INFO.get(str(dataset).lower(), {})
    try:
        return int(cfg.get(key, fallback))
    except Exception:
        return int(fallback)


def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _sigma_matches(token: str, sigma_arg: Any, tol: float = 1e-8) -> bool:
    token_s = str(token).strip()
    arg_s = str(sigma_arg).strip()
    if token_s == arg_s:
        return True
    tf = _try_float(token_s)
    af = _try_float(sigma_arg)
    if tf is not None and af is not None:
        return abs(tf - af) <= tol
    return False


def main() -> None:
    args = get_parameters()

    dataset = str(args.dataset)
    effective_obs = _effective_obs_fn(dataset, args.obs_fn)
    safe_obs = _safe_obs_name(effective_obs)
    default_obs = _default_obs_fn(dataset)
    canonical_suffix = "" if effective_obs == default_obs else f"_{safe_obs}"
    accepted_suffixes = _accepted_obs_suffixes(dataset, effective_obs)

    test_steps = int(args.test_steps) if args.test_steps is not None else _default_from_dataset(dataset, "test_steps", 500)
    test_traj_num = int(args.test_traj_num) if args.test_traj_num is not None else _default_from_dataset(dataset, "test_traj_num", 64)
    target_sigma = getattr(args, "sigma_y", None)
    adaptive_enabled = bool(getattr(args, "adaptive_sigma_y", False))

    pf_dir = Path("data") / dataset / f"pf_{safe_obs}"
    if not pf_dir.exists():
        raise FileNotFoundError(f"PF directory not found: {pf_dir}")

    print(f"[INFO] dataset={dataset}, obs_fn={effective_obs}, pf_dir={pf_dir}")
    print(f"[INFO] filter: batch={test_traj_num}, len={test_steps}, accepted_suffixes={accepted_suffixes}")
    print(f"[INFO] target sigma_y from args={target_sigma} (adaptive_sigma_y={adaptive_enabled})")
    print(f"[INFO] built-in PF_N_LIST={PF_N_LIST}")

    grouped: Dict[int, List[PFFileMeta]] = {n: [] for n in PF_N_LIST}
    for path in sorted(pf_dir.glob("pf_results_sigma_y_*.pt")):
        meta = _parse_pf_file(path)
        if meta is None:
            continue
        if meta.batch != test_traj_num or meta.seq_len != test_steps:
            continue
        if meta.obs_suffix not in accepted_suffixes:
            continue
        if meta.pf_N not in grouped:
            continue
        grouped[meta.pf_N].append(meta)

    analysis_rows: Dict[int, Dict[str, Any]] = {}
    sigma_conflicts: Dict[int, List[str]] = {}

    for n in PF_N_LIST:
        records = grouped[n]
        if len(records) == 0:
            # N not found in list -> skip directly.
            continue

        raw_sigma_values = sorted({r.sigma for r in records})
        seeds = sorted({r.seed for r in records})
        print(f"[N={n}] found seeds={seeds}, sigma_y={raw_sigma_values}")

        if target_sigma is not None:
            matched_records = [r for r in records if _sigma_matches(r.sigma, target_sigma)]
            if len(matched_records) > 0:
                records = matched_records
                used_sigma_values = sorted({r.sigma for r in records})
                if used_sigma_values != raw_sigma_values:
                    print(f"[N={n}] sigma_y filtered by args to: {used_sigma_values}")

        sigma_values = sorted({r.sigma for r in records})
        if len(sigma_values) == 0:
            # Current N has no files with args.sigma_y -> skip.
            continue
        if len(sigma_values) > 1:
            sigma_conflicts[n] = sigma_values
            raise RuntimeError(
                f"[ALARM] N={n} has multiple sigma_y values after filtering: {sigma_values}. "
                f"Please specify a unique --sigma_y (current args.sigma_y={target_sigma}, "
                f"adaptive_sigma_y={adaptive_enabled})."
            )

        sigma = sigma_values[0]
        records = sorted([r for r in records if r.sigma == sigma], key=lambda x: x.seed)
        # Deduplicate by seed if both canonical and legacy suffix files coexist.
        seed_to_rec: Dict[int, PFFileMeta] = {}
        for rec in records:
            if rec.seed not in seed_to_rec:
                seed_to_rec[rec.seed] = rec
                continue
            prev = seed_to_rec[rec.seed]
            if rec.obs_suffix == canonical_suffix and prev.obs_suffix != canonical_suffix:
                seed_to_rec[rec.seed] = rec
        records = [seed_to_rec[s] for s in sorted(seed_to_rec.keys())]
        seeds = [r.seed for r in records]
        if len(records) == 0:
            continue

        seed_payloads: List[List[Dict[str, Any]]] = []
        valid_seeds: List[int] = []
        for rec in records:
            try:
                payload = _torch_load(rec.path)
                if not isinstance(payload, list) or len(payload) == 0:
                    print(f"[WARN] skip invalid payload: {rec.path}")
                    continue
                seed_payloads.append(payload)
                valid_seeds.append(rec.seed)
            except Exception as exc:
                print(f"[WARN] failed to load {rec.path}: {exc}")

        if len(seed_payloads) == 0:
            print(f"[N={n}] no loadable seed payload.")
            continue

        avg_payload = _build_avg_payload(seed_payloads, valid_seeds)
        avg_name = f"pf_results_sigma_y_{sigma}_batch_{test_traj_num}_len_{test_steps}_pfN_{n}_avg"
        avg_path = pf_dir / f"{avg_name}{canonical_suffix}.pt"
        torch.save(avg_payload, avg_path)
        print(f"[N={n}] saved avg file: {avg_path}")

        rmse_mean_errors: List[float] = []
        fnorm_cov_errors: List[float] = []
        quantile_l2_dim_errors: List[torch.Tensor] = []
        ess_rmse_errors: List[float] = []
        entropy_rmse_errors: List[float] = []
        abundance_rmse_errors: List[float] = []
        skew_rmse_dim_errors: List[torch.Tensor] = []
        kurt_rmse_dim_errors: List[torch.Tensor] = []

        ess_seed_means: List[float] = []
        entropy_seed_means: List[float] = []
        abundance_seed_means: List[float] = []
        skew_seed_mean_dims: List[torch.Tensor] = []
        kurt_seed_mean_dims: List[torch.Tensor] = []

        for payload in seed_payloads:
            m_err = _rmse_mean_error(payload, avg_payload)
            c_err = _fnorm_cov_error(payload, avg_payload)
            q_err = _quantile_l2_error_by_dim(payload, avg_payload)
            e_err = _scalar_rmse_error(payload, avg_payload, ["post_ess", "ess"])
            ent_err = _scalar_rmse_error(payload, avg_payload, ["post_weight_entropy", "weight_entropy"])
            ab_err = _scalar_rmse_error(payload, avg_payload, ["post_weight_abundance", "weight_abundance"])
            s_err = _skew_rmse_by_dim(payload, avg_payload)
            k_err = _kurtosis_rmse_by_dim(payload, avg_payload)

            if m_err is not None:
                rmse_mean_errors.append(m_err)
            if c_err is not None:
                fnorm_cov_errors.append(c_err)
            if q_err is not None:
                quantile_l2_dim_errors.append(q_err)
            if e_err is not None:
                ess_rmse_errors.append(e_err)
            if ent_err is not None:
                entropy_rmse_errors.append(ent_err)
            if ab_err is not None:
                abundance_rmse_errors.append(ab_err)
            if s_err is not None:
                skew_rmse_dim_errors.append(s_err)
            if k_err is not None:
                kurt_rmse_dim_errors.append(k_err)

            ess_mean = _scalar_mean_over_tb(payload, ["post_ess", "ess"])
            entropy_mean = _scalar_mean_over_tb(payload, ["post_weight_entropy", "weight_entropy"])
            abundance_mean = _scalar_mean_over_tb(payload, ["post_weight_abundance", "weight_abundance"])
            skew_mean = _skew_mean_by_dim(payload)
            kurt_mean = _kurtosis_mean_by_dim(payload)

            if ess_mean is not None:
                ess_seed_means.append(ess_mean)
            if entropy_mean is not None:
                entropy_seed_means.append(entropy_mean)
            if abundance_mean is not None:
                abundance_seed_means.append(abundance_mean)
            if skew_mean is not None:
                skew_seed_mean_dims.append(skew_mean)
            if kurt_mean is not None:
                kurt_seed_mean_dims.append(kurt_mean)

        quantile_se = _standard_error_by_dim(quantile_l2_dim_errors)
        skew_se = _standard_error_by_dim(skew_rmse_dim_errors)
        skew_range = _range_by_dim(skew_seed_mean_dims)
        kurt_se = _standard_error_by_dim(kurt_rmse_dim_errors)
        kurt_range = _range_by_dim(kurt_seed_mean_dims)
        ess_mean_center, ess_mean_min, ess_mean_max = _mean_min_max(ess_seed_means)
        entropy_mean_center, entropy_mean_min, entropy_mean_max = _mean_min_max(entropy_seed_means)
        abundance_mean_center, abundance_mean_min, abundance_mean_max = _mean_min_max(abundance_seed_means)
        skew_mean_minmax = _mean_min_max_by_dim(skew_seed_mean_dims)
        kurt_mean_minmax = _mean_min_max_by_dim(kurt_seed_mean_dims)

        row: Dict[str, Any] = {
            "sigma_y": sigma,
            "n_seeds": len(valid_seeds),
            "seeds": valid_seeds,
            "avg_file": str(avg_path),
            "se_mean_rmse": _standard_error(rmse_mean_errors),
            "se_cov_fnorm": _standard_error(fnorm_cov_errors),
            "se_ess_rmse": _standard_error(ess_rmse_errors),
            "se_weight_entropy_rmse": _standard_error(entropy_rmse_errors),
            "se_weight_abundance_rmse": _standard_error(abundance_rmse_errors),
            "range_ess_mean": _range(ess_seed_means),
            "range_weight_entropy_mean": _range(entropy_seed_means),
            "range_weight_abundance_mean": _range(abundance_seed_means),
            "mean_ess_mean": ess_mean_center,
            "min_ess_mean": ess_mean_min,
            "max_ess_mean": ess_mean_max,
            "mean_weight_entropy_mean": entropy_mean_center,
            "min_weight_entropy_mean": entropy_mean_min,
            "max_weight_entropy_mean": entropy_mean_max,
            "mean_weight_abundance_mean": abundance_mean_center,
            "min_weight_abundance_mean": abundance_mean_min,
            "max_weight_abundance_mean": abundance_mean_max,
            "se_quantile_l2_dim": None if quantile_se is None else [float(v) for v in quantile_se.tolist()],
            "se_skew_rmse_dim": None if skew_se is None else [float(v) for v in skew_se.tolist()],
            "range_skew_mean_dim": None if skew_range is None else [float(v) for v in skew_range.tolist()],
            "mean_skew_mean_dim": None if skew_mean_minmax is None else [float(v) for v in skew_mean_minmax[0].tolist()],
            "min_skew_mean_dim": None if skew_mean_minmax is None else [float(v) for v in skew_mean_minmax[1].tolist()],
            "max_skew_mean_dim": None if skew_mean_minmax is None else [float(v) for v in skew_mean_minmax[2].tolist()],
            "se_kurtosis_rmse_dim": None if kurt_se is None else [float(v) for v in kurt_se.tolist()],
            "range_kurtosis_mean_dim": None if kurt_range is None else [float(v) for v in kurt_range.tolist()],
            "mean_kurtosis_mean_dim": None if kurt_mean_minmax is None else [float(v) for v in kurt_mean_minmax[0].tolist()],
            "min_kurtosis_mean_dim": None if kurt_mean_minmax is None else [float(v) for v in kurt_mean_minmax[1].tolist()],
            "max_kurtosis_mean_dim": None if kurt_mean_minmax is None else [float(v) for v in kurt_mean_minmax[2].tolist()],
        }
        analysis_rows[n] = row

    available_n = sorted(analysis_rows.keys())
    if len(available_n) == 0:
        print("[INFO] no N has complete analysis outputs after filtering.")
        return

    sigma_values_all = sorted({analysis_rows[n]["sigma_y"] for n in available_n})
    sigma_tag = sigma_values_all[0] if len(sigma_values_all) == 1 else "mixed"

    analysis_dir = Path("save") / f"pf_analysis_{dataset}_{safe_obs}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"sigma_{sigma_tag}_batch_{test_traj_num}_len_{test_steps}"

    se_mean_rmse = [_maybe_float(analysis_rows[n]["se_mean_rmse"]) for n in available_n]
    se_cov_fnorm = [_maybe_float(analysis_rows[n]["se_cov_fnorm"]) for n in available_n]
    _plot_multi_line(
        x=available_n,
        series={
            "mean RMSE SE": [float("nan") if v is None else v for v in se_mean_rmse],
            "cov F-norm SE": [float("nan") if v is None else v for v in se_cov_fnorm],
        },
        title="",
        ylabel="Standard Error",
        save_path=analysis_dir / f"se_mean_cov_{base_tag}.png",
    )

    state_quantile_series: Dict[str, List[float]] = {}
    pca_quantile_series: Dict[str, List[float]] = {}
    for q_dim, q_name in enumerate(["x1", "x2", "x3", "pca1", "pca2", "pca3"]):
        y = []
        for n in available_n:
            arr = analysis_rows[n].get("se_quantile_l2_dim")
            if isinstance(arr, list) and q_dim < len(arr):
                y.append(float(arr[q_dim]))
            else:
                y.append(float("nan"))
        if q_dim < 3:
            state_quantile_series[f"{q_name} quantile SE"] = y
        else:
            pca_quantile_series[f"{q_name} quantile SE"] = y

    _plot_multi_line(
        x=available_n,
        series=state_quantile_series,
        title="",
        ylabel="Quantile L2 SE",
        save_path=analysis_dir / f"se_quantile_state_{base_tag}.png",
    )
    _plot_multi_line(
        x=available_n,
        series=pca_quantile_series,
        title="",
        ylabel="Quantile L2 SE",
        save_path=analysis_dir / f"se_quantile_pca_{base_tag}.png",
    )

    _plot_multi_line(
        x=available_n,
        series={
            "ESS RMSE SE": [float("nan") if _maybe_float(analysis_rows[n]["se_ess_rmse"]) is None else float(analysis_rows[n]["se_ess_rmse"]) for n in available_n],
            "weight entropy RMSE SE": [float("nan") if _maybe_float(analysis_rows[n]["se_weight_entropy_rmse"]) is None else float(analysis_rows[n]["se_weight_entropy_rmse"]) for n in available_n],
            "weight abundance RMSE SE": [float("nan") if _maybe_float(analysis_rows[n]["se_weight_abundance_rmse"]) is None else float(analysis_rows[n]["se_weight_abundance_rmse"]) for n in available_n],
        },
        title="",
        ylabel="Standard Error",
        save_path=analysis_dir / f"se_scalar_metrics_{base_tag}.png",
    )

    _plot_multi_line_with_band(
        x=available_n,
        centers={
            "ESS mean": [float("nan") if _maybe_float(analysis_rows[n]["mean_ess_mean"]) is None else float(analysis_rows[n]["mean_ess_mean"]) for n in available_n],
        },
        lowers={
            "ESS mean": [float("nan") if _maybe_float(analysis_rows[n]["min_ess_mean"]) is None else float(analysis_rows[n]["min_ess_mean"]) for n in available_n],
        },
        uppers={
            "ESS mean": [float("nan") if _maybe_float(analysis_rows[n]["max_ess_mean"]) is None else float(analysis_rows[n]["max_ess_mean"]) for n in available_n],
        },
        title="",
        ylabel="ESS mean",
        save_path=analysis_dir / f"range_ess_{base_tag}.png",
    )

    _plot_multi_line_with_band(
        x=available_n,
        centers={
            "weight entropy mean": [float("nan") if _maybe_float(analysis_rows[n]["mean_weight_entropy_mean"]) is None else float(analysis_rows[n]["mean_weight_entropy_mean"]) for n in available_n],
        },
        lowers={
            "weight entropy mean": [float("nan") if _maybe_float(analysis_rows[n]["min_weight_entropy_mean"]) is None else float(analysis_rows[n]["min_weight_entropy_mean"]) for n in available_n],
        },
        uppers={
            "weight entropy mean": [float("nan") if _maybe_float(analysis_rows[n]["max_weight_entropy_mean"]) is None else float(analysis_rows[n]["max_weight_entropy_mean"]) for n in available_n],
        },
        title="",
        ylabel="Weight entropy mean",
        save_path=analysis_dir / f"range_weight_entropy_{base_tag}.png",
    )

    _plot_multi_line_with_band(
        x=available_n,
        centers={
            "weight abundance mean": [float("nan") if _maybe_float(analysis_rows[n]["mean_weight_abundance_mean"]) is None else float(analysis_rows[n]["mean_weight_abundance_mean"]) for n in available_n],
        },
        lowers={
            "weight abundance mean": [float("nan") if _maybe_float(analysis_rows[n]["min_weight_abundance_mean"]) is None else float(analysis_rows[n]["min_weight_abundance_mean"]) for n in available_n],
        },
        uppers={
            "weight abundance mean": [float("nan") if _maybe_float(analysis_rows[n]["max_weight_abundance_mean"]) is None else float(analysis_rows[n]["max_weight_abundance_mean"]) for n in available_n],
        },
        title="",
        ylabel="Weight abundance mean",
        save_path=analysis_dir / f"range_weight_abundance_{base_tag}.png",
    )

    # Skewness dims (if present).
    skew_dim_len = 0
    for n in available_n:
        arr = analysis_rows[n].get("se_skew_rmse_dim")
        if isinstance(arr, list):
            skew_dim_len = max(skew_dim_len, len(arr))
    if skew_dim_len > 0:
        skew_se_series = {}
        skew_mean_series = {}
        skew_min_series = {}
        skew_max_series = {}
        for d in range(skew_dim_len):
            y_se = []
            y_mean = []
            y_min = []
            y_max = []
            for n in available_n:
                se_arr = analysis_rows[n].get("se_skew_rmse_dim")
                mean_arr = analysis_rows[n].get("mean_skew_mean_dim")
                min_arr = analysis_rows[n].get("min_skew_mean_dim")
                max_arr = analysis_rows[n].get("max_skew_mean_dim")
                y_se.append(float(se_arr[d]) if isinstance(se_arr, list) and d < len(se_arr) else float("nan"))
                y_mean.append(float(mean_arr[d]) if isinstance(mean_arr, list) and d < len(mean_arr) else float("nan"))
                y_min.append(float(min_arr[d]) if isinstance(min_arr, list) and d < len(min_arr) else float("nan"))
                y_max.append(float(max_arr[d]) if isinstance(max_arr, list) and d < len(max_arr) else float("nan"))
            skew_se_series[f"skew dim {d + 1} RMSE SE"] = y_se
            key = f"skew dim {d + 1} mean"
            skew_mean_series[key] = y_mean
            skew_min_series[key] = y_min
            skew_max_series[key] = y_max

        _plot_multi_line(
            x=available_n,
            series=skew_se_series,
            title="",
            ylabel="Standard Error",
            save_path=analysis_dir / f"se_skewness_{base_tag}.png",
        )
        _plot_multi_line_with_band(
            x=available_n,
            centers=skew_mean_series,
            lowers=skew_min_series,
            uppers=skew_max_series,
            title="",
            ylabel="Value",
            save_path=analysis_dir / f"range_skewness_{base_tag}.png",
        )

    kurt_dim_len = 0
    for n in available_n:
        arr = analysis_rows[n].get("se_kurtosis_rmse_dim")
        if isinstance(arr, list):
            kurt_dim_len = max(kurt_dim_len, len(arr))
    if kurt_dim_len > 0:
        kurt_se_series = {}
        kurt_mean_series = {}
        kurt_min_series = {}
        kurt_max_series = {}
        for d in range(kurt_dim_len):
            y_se = []
            y_mean = []
            y_min = []
            y_max = []
            for n in available_n:
                se_arr = analysis_rows[n].get("se_kurtosis_rmse_dim")
                mean_arr = analysis_rows[n].get("mean_kurtosis_mean_dim")
                min_arr = analysis_rows[n].get("min_kurtosis_mean_dim")
                max_arr = analysis_rows[n].get("max_kurtosis_mean_dim")
                y_se.append(float(se_arr[d]) if isinstance(se_arr, list) and d < len(se_arr) else float("nan"))
                y_mean.append(float(mean_arr[d]) if isinstance(mean_arr, list) and d < len(mean_arr) else float("nan"))
                y_min.append(float(min_arr[d]) if isinstance(min_arr, list) and d < len(min_arr) else float("nan"))
                y_max.append(float(max_arr[d]) if isinstance(max_arr, list) and d < len(max_arr) else float("nan"))
            kurt_se_series[f"kurt dim {d + 1} RMSE SE"] = y_se
            key = f"kurt dim {d + 1} mean"
            kurt_mean_series[key] = y_mean
            kurt_min_series[key] = y_min
            kurt_max_series[key] = y_max

        _plot_multi_line(
            x=available_n,
            series=kurt_se_series,
            title="",
            ylabel="Standard Error",
            save_path=analysis_dir / f"se_kurtosis_{base_tag}.png",
        )
        _plot_multi_line_with_band(
            x=available_n,
            centers=kurt_mean_series,
            lowers=kurt_min_series,
            uppers=kurt_max_series,
            title="",
            ylabel="Value",
            save_path=analysis_dir / f"range_kurtosis_{base_tag}.png",
        )

    summary = {
        "dataset": dataset,
        "obs_fn": effective_obs,
        "pf_dir": str(pf_dir),
        "test_steps": test_steps,
        "test_traj_num": test_traj_num,
        "built_in_pf_N_list": PF_N_LIST,
        "available_N": available_n,
        "sigma_conflicts": {str(k): v for k, v in sigma_conflicts.items()},
        "analysis": {str(k): analysis_rows[k] for k in available_n},
    }
    summary_path = analysis_dir / f"summary_{base_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] saved analysis summary: {summary_path}")
    print(f"[INFO] saved plots in: {analysis_dir}")


if __name__ == "__main__":
    main()
