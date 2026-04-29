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


ANALYSIS_DEVICE = torch.device("cpu")


# Built-in PF particle counts (do not use args.N).
PF_M_LIST: List[int] = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

PF_FILE_PATTERN = re.compile(
    r"^pf_results_sigma_y_(?P<sigma>[^_]+)_batch_(?P<batch>\d+)_len_(?P<seq_len>\d+)_pfN_(?P<pf_M>\d+)_(?P<seed>\d+)(?P<obs_suffix>_[^.]+)?\.pt$"
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
    "post_pca_range_q01_q99",
    "post_pca_range_int",
    "post_ess",
    "post_weight_entropy",
    "post_weight_abundance",
    "post_skewness",
    "post_kurtosis_excess",
    "prior_means",
    "prior_covs",
    "prior_quantiles",
    "prior_pca_quantiles",
    "prior_pca_range_q01_q99",
    "prior_pca_range_int",
    "prior_skewness",
    "prior_kurtosis_excess",
}


@dataclass
class PFFileMeta:
    path: Path
    sigma: str
    batch: int
    seq_len: int
    pf_M: int
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
        pf_M=int(m.group("pf_M")),
        seed=int(m.group("seed")),
        obs_suffix=str(m.group("obs_suffix") or ""),
    )


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location=ANALYSIS_DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=ANALYSIS_DEVICE)


def _to_cpu_value(v: Any) -> Any:
    if torch.is_tensor(v):
        return v.detach().cpu()
    if isinstance(v, dict):
        return {k: _to_cpu_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_cpu_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_to_cpu_value(x) for x in v)
    return v


def _tensor_to_float_list(x: Optional[torch.Tensor]) -> Optional[List[float]]:
    if x is None:
        return None
    return [float(v) for v in x.detach().cpu().tolist()]


def _new_running_tensor_stats() -> Dict[str, Any]:
    return {"count": 0, "mean": None, "m2": None, "shape": None, "valid": True}


def _update_running_tensor_stats(state: Dict[str, Any], x: Optional[torch.Tensor]) -> bool:
    if not state["valid"] or x is None or not torch.is_tensor(x):
        state["valid"] = False
        return False

    x32 = x.to(torch.float32)
    shape = tuple(x32.shape)
    if state["mean"] is None:
        state["count"] = 1
        state["shape"] = shape
        state["mean"] = x32.clone()
        state["m2"] = torch.zeros_like(x32)
        return True

    if shape != state["shape"]:
        state["valid"] = False
        return False

    count_next = int(state["count"]) + 1
    delta = x32 - state["mean"]
    state["mean"] = state["mean"] + delta / float(count_next)
    delta2 = x32 - state["mean"]
    state["m2"] = state["m2"] + delta * delta2
    state["count"] = count_next
    return True


def _finalize_running_tensor_se(state: Dict[str, Any]) -> Optional[torch.Tensor]:
    if (not state["valid"]) or state["mean"] is None or state["m2"] is None:
        return None
    k = int(state["count"])
    if k < 2:
        return None
    return torch.sqrt(state["m2"] / float((k - 1) * k))


def _finalize_running_tensor_mean(state: Dict[str, Any]) -> Optional[torch.Tensor]:
    if (not state["valid"]) or state["mean"] is None:
        return None
    return state["mean"]


def _new_running_scalar_mean() -> Dict[str, Any]:
    return {"sum": 0.0, "count": 0}


def _update_running_scalar_mean(state: Dict[str, Any], value: Optional[float]) -> None:
    if value is None:
        return
    try:
        fv = float(value)
    except Exception:
        return
    if not math.isfinite(fv):
        return
    state["sum"] += fv
    state["count"] += 1


def _finalize_running_scalar_mean(state: Dict[str, Any]) -> float:
    if int(state["count"]) <= 0:
        return float("nan")
    return float(state["sum"] / float(state["count"]))


def _new_running_scalar_minmax() -> Dict[str, Any]:
    return {"min": None, "max": None}


def _update_running_scalar_minmax(state: Dict[str, Any], value: Optional[float]) -> None:
    if value is None:
        return
    try:
        fv = float(value)
    except Exception:
        return
    if not math.isfinite(fv):
        return
    if state["min"] is None or fv < float(state["min"]):
        state["min"] = fv
    if state["max"] is None or fv > float(state["max"]):
        state["max"] = fv


def _finalize_running_scalar_minmax(state: Dict[str, Any]) -> Tuple[float, float]:
    min_v = state["min"]
    max_v = state["max"]
    if min_v is None or max_v is None:
        return float("nan"), float("nan")
    return float(min_v), float(max_v)


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


def _build_avg_payload_stream(records: List[PFFileMeta]) -> Tuple[Optional[List[Dict[str, Any]]], List[PFFileMeta]]:
    if len(records) == 0:
        return None, []

    num_batches: Optional[int] = None
    batch_states: Optional[List[Dict[str, Dict[str, Any]]]] = None
    max_seed_payload: Optional[List[Dict[str, Any]]] = None
    valid_records: List[PFFileMeta] = []

    for rec in records:
        try:
            payload = _torch_load(rec.path)
        except Exception as exc:
            print(f"[WARN] failed to load {rec.path}: {exc}")
            continue

        if not isinstance(payload, list) or len(payload) == 0:
            print(f"[WARN] skip invalid payload: {rec.path}")
            continue
        if any(not isinstance(entry, dict) for entry in payload):
            print(f"[WARN] skip invalid payload entries (non-dict): {rec.path}")
            continue

        if num_batches is None:
            num_batches = len(payload)
            batch_states = [{} for _ in range(num_batches)]
        elif len(payload) != num_batches:
            print(f"[WARN] skip payload with inconsistent batch entries: {rec.path}")
            continue

        valid_records.append(rec)
        max_seed_payload = payload

        for bidx, entry in enumerate(payload):
            states = batch_states[bidx] if batch_states is not None else {}
            for key, value in entry.items():
                st = states.get(key)
                if st is None:
                    st = {
                        "present": 0,
                        "tensor_count": 0,
                        "shape_ok": True,
                        "sum": None,
                        "ref_shape": None,
                        "ref_dtype": None,
                        "ref_is_float": None,
                        "source_value": None,
                    }
                    states[key] = st

                st["present"] += 1
                if st["source_value"] is None:
                    st["source_value"] = _clone_value(value)

                if key not in AVERAGE_KEYS:
                    continue
                if not torch.is_tensor(value):
                    st["shape_ok"] = False
                    continue

                t = value.to(torch.float32)
                if st["sum"] is None:
                    st["sum"] = t.clone()
                    st["ref_shape"] = tuple(t.shape)
                    st["ref_dtype"] = value.dtype
                    st["ref_is_float"] = bool(torch.is_floating_point(value))
                else:
                    if tuple(t.shape) != st["ref_shape"]:
                        st["shape_ok"] = False
                    else:
                        st["sum"] += t
                st["tensor_count"] += 1

    if len(valid_records) == 0 or num_batches is None or batch_states is None or max_seed_payload is None:
        return None, []

    k = len(valid_records)
    avg_payload: List[Dict[str, Any]] = []
    for bidx in range(num_batches):
        states = batch_states[bidx]
        max_entry = max_seed_payload[bidx]
        out: Dict[str, Any] = {}

        for key in sorted(states.keys()):
            st = states[key]
            can_avg = (
                key in AVERAGE_KEYS
                and st["present"] == k
                and st["tensor_count"] == k
                and st["shape_ok"]
                and st["sum"] is not None
            )
            if can_avg:
                mean = st["sum"] / float(k)
                if bool(st["ref_is_float"]):
                    out[key] = mean.to(dtype=st["ref_dtype"])
                else:
                    out[key] = mean.round().to(dtype=st["ref_dtype"])
            else:
                if key in max_entry:
                    out[key] = _clone_value(max_entry[key])
                else:
                    out[key] = _clone_value(st["source_value"])

        # Backward-compatible aliases for downstream code.
        if "post_means" in out and "means" not in out and torch.is_tensor(out["post_means"]):
            out["means"] = out["post_means"].clone()
        if "post_covs" in out and "covs" not in out and torch.is_tensor(out["post_covs"]):
            out["covs"] = out["post_covs"].clone()
        avg_payload.append(out)

    return avg_payload, valid_records


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


def _concat_batches_slice(payload: List[Dict[str, Any]], key: str, t_start: int, t_end: int) -> Optional[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for entry in payload:
        if key not in entry or not torch.is_tensor(entry[key]):
            return None
        x = entry[key].to(torch.float32)
        if x.ndim < 1 or t_start < 0 or t_end > x.shape[0] or t_start >= t_end:
            return None
        tensors.append(x[t_start:t_end])
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


def _stack_seed_tensors(
    seed_payloads: List[List[Dict[str, Any]]],
    key_candidates: Sequence[str],
) -> Optional[torch.Tensor]:
    stacked_items: List[torch.Tensor] = []
    for payload in seed_payloads:
        if len(payload) == 0:
            return None
        k = _pick_key(payload[0], key_candidates)
        if k is None:
            return None
        x = _concat_batches(payload, k)
        if x is None:
            return None
        stacked_items.append(x.to(torch.float32))
    if len(stacked_items) == 0:
        return None
    if not _all_same_shape(stacked_items):
        return None
    return torch.stack(stacked_items, dim=0)


def _se_from_seed_stack(seed_stack: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if seed_stack is None or seed_stack.ndim == 0:
        return None
    k = int(seed_stack.shape[0])
    if k < 2:
        return None
    return torch.std(seed_stack, dim=0, unbiased=True) / math.sqrt(k)


def _se_mean_rmse(seed_payloads: List[List[Dict[str, Any]]]) -> Optional[float]:
    seed_stack = _stack_seed_tensors(seed_payloads, ["post_means", "means"])
    se = _se_from_seed_stack(seed_stack)
    if se is None or se.ndim != 3:
        return None
    rmse_tb = torch.sqrt(torch.mean(se * se, dim=-1))
    return float(torch.nanmean(rmse_tb).item())


def _se_mean_avg(seed_payloads: List[List[Dict[str, Any]]]) -> Optional[float]:
    seed_stack = _stack_seed_tensors(seed_payloads, ["post_means", "means"])
    se = _se_from_seed_stack(seed_stack)
    if se is None or se.ndim != 3:
        return None
    return float(torch.nanmean(se).item())


def _se_cov_fnorm(seed_payloads: List[List[Dict[str, Any]]]) -> Optional[float]:
    seed_stack = _stack_seed_tensors(seed_payloads, ["post_covs", "covs"])
    se = _se_from_seed_stack(seed_stack)
    if se is None or se.ndim != 4:
        return None
    fnorm_tb = torch.linalg.norm(se, ord="fro", dim=(-2, -1))
    return float(torch.nanmean(fnorm_tb).item())


def _se_cov_avg(seed_payloads: List[List[Dict[str, Any]]]) -> Optional[float]:
    seed_stack = _stack_seed_tensors(seed_payloads, ["post_covs", "covs"])
    se = _se_from_seed_stack(seed_stack)
    if se is None or se.ndim != 4:
        return None
    return float(torch.nanmean(se).item())


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


def _compact_payload_for_analysis(payload: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if len(payload) == 0:
        return None

    out: Dict[str, Any] = {}

    mean_key = _pick_key(payload[0], ["post_means", "means"])
    cov_key = _pick_key(payload[0], ["post_covs", "covs"])
    if mean_key is not None:
        mean_x = _concat_batches(payload, mean_key)
        if mean_x is not None:
            out["post_means"] = mean_x
    if cov_key is not None:
        cov_x = _concat_batches(payload, cov_key)
        if cov_x is not None:
            out["post_covs"] = cov_x

    q6d = _build_quantile_6d(payload)
    if q6d is not None:
        out["quantiles_6d"] = q6d

    q_probs = _get_quantile_probs(payload)
    if q_probs is not None:
        out["quantile_probs"] = q_probs

    scalar_key_map = [
        (["post_ess", "ess"], "post_ess"),
        (["post_weight_entropy", "weight_entropy"], "post_weight_entropy"),
        (["post_weight_abundance", "weight_abundance"], "post_weight_abundance"),
        (["post_skewness", "skewness"], "post_skewness"),
        (["post_kurtosis_excess", "kurtosis_excess"], "post_kurtosis_excess"),
    ]
    for candidates, out_key in scalar_key_map:
        k = _pick_key(payload[0], candidates)
        if k is None:
            continue
        x = _concat_batches(payload, k)
        if x is not None:
            out[out_key] = x

    if len(out) == 0:
        return None
    return [out]


def _infer_quantile_shape(payload: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int]]:
    if len(payload) == 0:
        return None

    def infer_one(key: Optional[str]) -> Optional[Tuple[int, int, int]]:
        if key is None:
            return None
        t_dim: Optional[int] = None
        b_total = 0
        q_dim: Optional[int] = None
        for entry in payload:
            if key not in entry or not torch.is_tensor(entry[key]):
                return None
            x = entry[key]
            if x.ndim != 4:
                return None
            if t_dim is None:
                t_dim = int(x.shape[0])
                q_dim = int(x.shape[3])
            elif int(x.shape[0]) != t_dim or int(x.shape[3]) != q_dim:
                return None
            b_total += int(x.shape[1])
        if t_dim is None or q_dim is None:
            return None
        return t_dim, b_total, q_dim

    compact_key = _pick_key(payload[0], ["quantiles_6d"])
    if compact_key is not None:
        return infer_one(compact_key)

    state_key = _pick_key(payload[0], ["post_quantiles", "quantiles"])
    pca_key = _pick_key(payload[0], ["post_pca_quantiles", "pca_quantiles"])
    state_shape = infer_one(state_key)
    pca_shape = infer_one(pca_key)
    if state_shape is None and pca_shape is None:
        return None
    if state_shape is None:
        return pca_shape
    if pca_shape is None:
        return state_shape
    if state_shape != pca_shape:
        return None
    return state_shape


def _build_quantile_6d_slice(payload: List[Dict[str, Any]], t_start: int, t_end: int) -> Optional[torch.Tensor]:
    if len(payload) == 0:
        return None

    compact_key = _pick_key(payload[0], ["quantiles_6d"])
    if compact_key is not None:
        return _concat_batches_slice(payload, compact_key, t_start, t_end)

    state_key = _pick_key(payload[0], ["post_quantiles", "quantiles"])
    pca_key = _pick_key(payload[0], ["post_pca_quantiles", "pca_quantiles"])
    q_state = _concat_batches_slice(payload, state_key, t_start, t_end) if state_key is not None else None
    q_pca = _concat_batches_slice(payload, pca_key, t_start, t_end) if pca_key is not None else None
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


def _se_quantile_metrics(
    records: List[PFFileMeta],
    chunk_steps: int = 16,
) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[torch.Tensor]]:
    if len(records) < 2:
        return None, None, None

    q_probs: Optional[torch.Tensor] = None
    shape0: Optional[Tuple[int, int, int]] = None
    for rec in records:
        try:
            payload = _torch_load(rec.path)
        except Exception as exc:
            print(f"[WARN] failed to load {rec.path} for quantile SE setup: {exc}")
            return None, None, None
        if not isinstance(payload, list) or len(payload) == 0:
            return None, None, None
        q_probs = _get_quantile_probs(payload)
        shape0 = _infer_quantile_shape(payload)
        if q_probs is not None and shape0 is not None:
            break

    if q_probs is None or q_probs.ndim != 1 or shape0 is None:
        return None, None, None

    t_dim, b_dim, q_dim = shape0
    if q_dim != int(q_probs.shape[0]):
        return None, None, None

    q_probs = q_probs.to(torch.float32)
    step = max(1, int(chunk_steps))
    sum_avg = 0.0
    count_avg = 0
    sum_l2 = torch.zeros(6, dtype=torch.float64)
    count_l2 = torch.zeros(6, dtype=torch.float64)
    sum_dim_avg = torch.zeros(6, dtype=torch.float64)
    count_dim_avg = torch.zeros(6, dtype=torch.float64)

    for t_start in range(0, t_dim, step):
        t_end = min(t_dim, t_start + step)
        q_state = _new_running_tensor_stats()
        chunk_b_dim: Optional[int] = None
        chunk_q_dim: Optional[int] = None

        for rec in records:
            try:
                payload = _torch_load(rec.path)
            except Exception as exc:
                print(f"[WARN] failed to load {rec.path} for quantile SE chunk: {exc}")
                return None, None, None
            if not isinstance(payload, list) or len(payload) == 0:
                return None, None, None

            q_chunk = _build_quantile_6d_slice(payload, t_start, t_end)
            if q_chunk is None:
                return None, None, None
            if chunk_b_dim is None:
                chunk_b_dim = int(q_chunk.shape[1])
                chunk_q_dim = int(q_chunk.shape[-1])
            if int(q_chunk.shape[1]) != chunk_b_dim or int(q_chunk.shape[-1]) != chunk_q_dim:
                return None, None, None
            if chunk_b_dim != b_dim or chunk_q_dim != q_dim:
                return None, None, None
            if not _update_running_tensor_stats(q_state, q_chunk):
                return None, None, None

        se_q = _finalize_running_tensor_se(q_state)
        if se_q is None:
            return None, None, None

        finite_q = torch.isfinite(se_q)
        if bool(torch.any(finite_q).item()):
            sum_avg += float(torch.where(finite_q, se_q, torch.zeros_like(se_q)).sum().item())
            count_avg += int(finite_q.sum().item())
            sum_dim_avg += torch.where(finite_q, se_q, torch.zeros_like(se_q)).sum(dim=(0, 1, 3)).to(torch.float64).cpu()
            count_dim_avg += finite_q.sum(dim=(0, 1, 3)).to(torch.float64).cpu()

        l2_tb_dim = torch.sqrt(torch.trapz(se_q ** 2, x=q_probs.to(se_q.device), dim=-1))
        finite_l2 = torch.isfinite(l2_tb_dim)
        if bool(torch.any(finite_l2).item()):
            sum_l2 += torch.where(finite_l2, l2_tb_dim, torch.zeros_like(l2_tb_dim)).sum(dim=(0, 1)).to(torch.float64).cpu()
            count_l2 += finite_l2.sum(dim=(0, 1)).to(torch.float64).cpu()

    l2_out = torch.full((6,), float("nan"), dtype=torch.float32)
    valid_l2 = count_l2 > 0
    l2_ret: Optional[torch.Tensor]
    if bool(torch.any(valid_l2).item()):
        l2_out[valid_l2] = (sum_l2[valid_l2] / count_l2[valid_l2]).to(torch.float32)
        l2_ret = l2_out
    else:
        l2_ret = None

    dim_avg_out = torch.full((6,), float("nan"), dtype=torch.float32)
    valid_dim = count_dim_avg > 0
    dim_avg_ret: Optional[torch.Tensor]
    if bool(torch.any(valid_dim).item()):
        dim_avg_out[valid_dim] = (sum_dim_avg[valid_dim] / count_dim_avg[valid_dim]).to(torch.float32)
        dim_avg_ret = dim_avg_out
    else:
        dim_avg_ret = None

    avg_ret: Optional[float] = None if count_avg == 0 else float(sum_avg / count_avg)
    return l2_ret, avg_ret, dim_avg_ret


def _se_scalar_rmse(
    seed_payloads: List[List[Dict[str, Any]]],
    key_candidates: Sequence[str],
) -> Optional[float]:
    seed_stack = _stack_seed_tensors(seed_payloads, key_candidates)
    se = _se_from_seed_stack(seed_stack)
    if se is None:
        return None
    rmse = torch.sqrt(torch.nanmean(se ** 2))
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


def _se_skew_rmse_by_dim(seed_payloads: List[List[Dict[str, Any]]]) -> Optional[torch.Tensor]:
    seed_stack = _stack_seed_tensors(seed_payloads, ["post_skewness", "skewness"])
    se = _se_from_seed_stack(seed_stack)
    if se is None or se.ndim != 3:
        return None
    return torch.sqrt(torch.nanmean(se ** 2, dim=(0, 1)))


def _skew_mean_by_dim(payload: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    k = _pick_key(payload[0], ["post_skewness", "skewness"])
    if k is None:
        return None
    x = _concat_batches(payload, k)
    if x is None or x.ndim != 3:
        return None
    return torch.nanmean(x, dim=(0, 1))


def _se_kurtosis_rmse_by_dim(seed_payloads: List[List[Dict[str, Any]]]) -> Optional[torch.Tensor]:
    seed_stack = _stack_seed_tensors(seed_payloads, ["post_kurtosis_excess", "kurtosis_excess"])
    se = _se_from_seed_stack(seed_stack)
    if se is None or se.ndim != 3:
        return None
    return torch.sqrt(torch.nanmean(se ** 2, dim=(0, 1)))


def _kurtosis_mean_by_dim(payload: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    k = _pick_key(payload[0], ["post_kurtosis_excess", "kurtosis_excess"])
    if k is None:
        return None
    x = _concat_batches(payload, k)
    if x is None or x.ndim != 3:
        return None
    return torch.nanmean(x, dim=(0, 1))


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
    force_log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, y, marker="o", linewidth=3.0, markersize=10)
    ax.set_xscale("log")
    if force_log_y:
        _set_log_y_scale(ax, [y])
    else:
        _set_adaptive_y_scale(ax, [y])
    ax.set_xlabel("M (number of particles)", fontsize=26)
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
    force_log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    all_series: List[List[float]] = []
    for name, y in centers.items():
        y_low = lowers[name]
        y_high = uppers[name]
        ax.plot(x, y, marker="o", label=name, linewidth=3.0, markersize=10)
        ax.fill_between(x, y_low, y_high, alpha=0.2)
        all_series.extend([y, y_low, y_high])
    ax.set_xscale("log")
    if force_log_y:
        _set_log_y_scale(ax, all_series)
    else:
        _set_adaptive_y_scale(ax, all_series)
    ax.set_xlabel("M (number of particles)", fontsize=26)
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
    force_log_y: bool = False,
    force_linear_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    all_series: List[List[float]] = []
    for name, y in series.items():
        ax.plot(x, y, marker="o", label=name, linewidth=3.0, markersize=10)
        all_series.append(y)
    ax.set_xscale("log")
    if force_linear_y:
        ax.set_yscale("linear")
    elif force_log_y:
        _set_log_y_scale(ax, all_series)
    else:
        _set_adaptive_y_scale(ax, all_series)
    ax.set_xlabel("M (number of particles)", fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.tick_params(axis="both", labelsize=22, width=2.5, length=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=20, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_ess_entropy_abundance(
    x: List[int],
    ess: List[float],
    entropy: List[float],
    abundance: List[float],
    save_path: Path,
) -> None:
    fig, ax_left = plt.subplots(figsize=(10, 7))
    ax_right = ax_left.twinx()

    left_lines = [
        ax_left.plot(x, ess, marker="o", label="ESS", linewidth=3.0, markersize=10)[0],
        ax_left.plot(
            x,
            abundance,
            marker="s",
            label="Weight abundance",
            linewidth=3.0,
            markersize=10,
        )[0],
    ]
    right_lines = [
        ax_right.plot(
            x,
            entropy,
            marker="^",
            label="Entropy",
            linewidth=3.0,
            markersize=10,
            color="tab:green",
        )[0],
    ]

    ax_left.set_xscale("log")
    _set_adaptive_y_scale(ax_left, [ess, abundance])
    _set_adaptive_y_scale(ax_right, [entropy])

    ax_left.set_xlabel("M (number of particles)", fontsize=26)
    ax_left.set_ylabel("ESS / Weight abundance", fontsize=26)
    ax_right.set_ylabel("Entropy", fontsize=26)
    ax_left.tick_params(axis="both", labelsize=22, width=2.5, length=8)
    ax_right.tick_params(axis="y", labelsize=22, width=2.5, length=8)
    ax_left.grid(True, which="both", linestyle="--", alpha=0.5)

    lines = left_lines + right_lines
    ax_left.legend(
        lines,
        [line.get_label() for line in lines],
        fontsize=20,
        frameon=True,
        loc="best",
    )
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


def _set_adaptive_y_scale(ax: Any, series_list: Sequence[Sequence[float]]) -> None:
    finite_vals: List[float] = []
    for series in series_list:
        for v in series:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                finite_vals.append(fv)

    if len(finite_vals) == 0:
        ax.set_yscale("linear")
        return
    if any(v <= 0.0 for v in finite_vals):
        ax.set_yscale("linear")
        return

    v_min = min(finite_vals)
    v_max = max(finite_vals)
    if v_min <= 0.0:
        ax.set_yscale("linear")
        return
    ax.set_yscale("log" if (v_max / v_min) >= 10.0 else "linear")


def _set_log_y_scale(ax: Any, series_list: Sequence[Sequence[float]]) -> None:
    finite_vals: List[float] = []
    for series in series_list:
        for v in series:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv) and fv > 0.0:
                finite_vals.append(fv)

    ax.set_yscale("log", nonpositive="mask")
    if len(finite_vals) == 0:
        return

    v_min = min(finite_vals)
    v_max = max(finite_vals)
    if v_min == v_max:
        ax.set_ylim(v_min / 2.0, v_max * 2.0)


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
    global ANALYSIS_DEVICE
    args = get_parameters()
    ANALYSIS_DEVICE = args.device if isinstance(args.device, torch.device) else torch.device(str(args.device))

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
    print(f"[INFO] analysis_device={ANALYSIS_DEVICE}")
    print(f"[INFO] filter: batch={test_traj_num}, len={test_steps}, accepted_suffixes={accepted_suffixes}")
    print(f"[INFO] target sigma_y from args={target_sigma} (adaptive_sigma_y={adaptive_enabled})")
    print(f"[INFO] built-in PF_M_LIST={PF_M_LIST}")

    grouped: Dict[int, List[PFFileMeta]] = {m: [] for m in PF_M_LIST}
    for path in sorted(pf_dir.glob("pf_results_sigma_y_*.pt")):
        meta = _parse_pf_file(path)
        if meta is None:
            continue
        if meta.batch != test_traj_num or meta.seq_len != test_steps:
            continue
        if meta.obs_suffix not in accepted_suffixes:
            continue
        if meta.pf_M not in grouped:
            continue
        grouped[meta.pf_M].append(meta)

    analysis_rows: Dict[int, Dict[str, Any]] = {}
    sigma_conflicts: Dict[int, List[str]] = {}

    for m in PF_M_LIST:
        records = grouped[m]
        if len(records) == 0:
            # M not found in list -> skip directly.
            continue

        raw_sigma_values = sorted({r.sigma for r in records})
        seeds = sorted({r.seed for r in records})
        print(f"[M={m}] found seeds={seeds}, sigma_y={raw_sigma_values}")

        if target_sigma is not None:
            matched_records = [r for r in records if _sigma_matches(r.sigma, target_sigma)]
            if len(matched_records) > 0:
                records = matched_records
                used_sigma_values = sorted({r.sigma for r in records})
                if used_sigma_values != raw_sigma_values:
                    print(f"[M={m}] sigma_y filtered by args to: {used_sigma_values}")

        sigma_values = sorted({r.sigma for r in records})
        if len(sigma_values) == 0:
            # Current M has no files with args.sigma_y -> skip.
            continue
        if len(sigma_values) > 1:
            sigma_conflicts[m] = sigma_values
            raise RuntimeError(
                f"[ALARM] M={m} has multiple sigma_y values after filtering: {sigma_values}. "
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

        avg_payload, valid_records = _build_avg_payload_stream(records)
        if avg_payload is None or len(valid_records) == 0:
            print(f"[M={m}] no loadable seed payload.")
            continue

        valid_seeds = [r.seed for r in valid_records]
        avg_name = f"pf_results_sigma_y_{sigma}_batch_{test_traj_num}_len_{test_steps}_pfN_{m}_avg"
        avg_path = pf_dir / f"{avg_name}{canonical_suffix}.pt"
        avg_payload_cpu = _to_cpu_value(avg_payload)
        torch.save(avg_payload_cpu, avg_path)
        print(f"[M={m}] saved avg file: {avg_path}")
        del avg_payload
        del avg_payload_cpu

        # Second pass: stream per-seed metrics without stacking all seeds in memory.
        mean_state = _new_running_tensor_stats()
        cov_state = _new_running_tensor_stats()
        kurt_mean_state = _new_running_tensor_stats()
        ess_mean_state = _new_running_scalar_mean()
        ess_minmax_state = _new_running_scalar_minmax()
        entropy_mean_state = _new_running_scalar_mean()
        entropy_minmax_state = _new_running_scalar_minmax()
        abundance_mean_state = _new_running_scalar_mean()
        abundance_minmax_state = _new_running_scalar_minmax()
        metric_records: List[PFFileMeta] = []

        for rec in valid_records:
            try:
                payload = _torch_load(rec.path)
            except Exception as exc:
                print(f"[WARN] failed to load {rec.path} on metrics pass: {exc}")
                continue

            if not isinstance(payload, list) or len(payload) == 0:
                print(f"[WARN] skip invalid payload on metrics pass: {rec.path}")
                continue
            if any(not isinstance(entry, dict) for entry in payload):
                print(f"[WARN] skip invalid payload entries on metrics pass: {rec.path}")
                continue

            metric_records.append(rec)

            mean_key = _pick_key(payload[0], ["post_means", "means"])
            cov_key = _pick_key(payload[0], ["post_covs", "covs"])
            kurt_key = _pick_key(payload[0], ["post_kurtosis_excess", "kurtosis_excess"])

            mean_x = _concat_batches(payload, mean_key) if mean_key is not None else None
            cov_x = _concat_batches(payload, cov_key) if cov_key is not None else None
            _update_running_tensor_stats(mean_state, mean_x)
            _update_running_tensor_stats(cov_state, cov_x)

            kurt_x = _concat_batches(payload, kurt_key) if kurt_key is not None else None
            if kurt_x is None or kurt_x.ndim != 3:
                _update_running_tensor_stats(kurt_mean_state, None)
            else:
                _update_running_tensor_stats(kurt_mean_state, torch.nanmean(kurt_x, dim=(0, 1)))

            ess_mean_value = _scalar_mean_over_tb(payload, ["post_ess", "ess"])
            entropy_mean_value = _scalar_mean_over_tb(payload, ["post_weight_entropy", "weight_entropy"])
            abundance_mean_value = _scalar_mean_over_tb(payload, ["post_weight_abundance", "weight_abundance"])
            _update_running_scalar_mean(ess_mean_state, ess_mean_value)
            _update_running_scalar_minmax(ess_minmax_state, ess_mean_value)
            _update_running_scalar_mean(entropy_mean_state, entropy_mean_value)
            _update_running_scalar_minmax(entropy_minmax_state, entropy_mean_value)
            _update_running_scalar_mean(
                abundance_mean_state,
                abundance_mean_value,
            )
            _update_running_scalar_minmax(abundance_minmax_state, abundance_mean_value)

        if len(metric_records) == 0:
            print(f"[M={m}] no loadable seed payload on metrics pass.")
            continue

        valid_seeds = [r.seed for r in metric_records]

        se_mean_tensor = _finalize_running_tensor_se(mean_state)
        se_cov_tensor = _finalize_running_tensor_se(cov_state)

        se_mean_rmse: Optional[float]
        se_mean_avg: Optional[float]
        if se_mean_tensor is None or se_mean_tensor.ndim != 3:
            se_mean_rmse = None
            se_mean_avg = None
        else:
            rmse_tb = torch.sqrt(torch.mean(se_mean_tensor * se_mean_tensor, dim=-1))
            se_mean_rmse = float(torch.nanmean(rmse_tb).item())
            # Elementwise seed SE, averaged over timestep, trajectory, and state dimensions.
            se_mean_avg = float(torch.nanmean(se_mean_tensor).item())

        se_cov_fnorm: Optional[float]
        se_cov_avg: Optional[float]
        if se_cov_tensor is None or se_cov_tensor.ndim != 4:
            se_cov_fnorm = None
            se_cov_avg = None
        else:
            fnorm_tb = torch.linalg.norm(se_cov_tensor, ord="fro", dim=(-2, -1))
            se_cov_fnorm = float(torch.nanmean(fnorm_tb).item())
            # Elementwise seed SE, averaged over timestep, trajectory, and covariance entries.
            se_cov_avg = float(torch.nanmean(se_cov_tensor).item())

        quantile_se, se_quantile_avg, quantile_se_avg_dim = _se_quantile_metrics(metric_records)
        se_quantile_l2 = None if quantile_se is None else float(torch.nanmean(quantile_se).item())

        kurt_mean_dims = _finalize_running_tensor_mean(kurt_mean_state)
        if kurt_mean_dims is not None and kurt_mean_dims.ndim != 1:
            kurt_mean_dims = None
        ess_mean_min, ess_mean_max = _finalize_running_scalar_minmax(ess_minmax_state)
        entropy_mean_min, entropy_mean_max = _finalize_running_scalar_minmax(entropy_minmax_state)
        abundance_mean_min, abundance_mean_max = _finalize_running_scalar_minmax(abundance_minmax_state)

        row: Dict[str, Any] = {
            "sigma_y": sigma,
            "n_seeds": len(valid_seeds),
            "seeds": valid_seeds,
            "avg_file": str(avg_path),
            "analysis_device": str(ANALYSIS_DEVICE),
            "se_mean_rmse": float("nan") if se_mean_rmse is None else float(se_mean_rmse),
            "se_mean_avg": float("nan") if se_mean_avg is None else float(se_mean_avg),
            "se_mean_elementwise_avg": float("nan") if se_mean_avg is None else float(se_mean_avg),
            "se_cov_fnorm": float("nan") if se_cov_fnorm is None else float(se_cov_fnorm),
            "se_cov_avg": float("nan") if se_cov_avg is None else float(se_cov_avg),
            "se_cov_elementwise_avg": float("nan") if se_cov_avg is None else float(se_cov_avg),
            "se_quantile_l2": float("nan") if se_quantile_l2 is None else float(se_quantile_l2),
            "se_quantile_avg": float("nan") if se_quantile_avg is None else float(se_quantile_avg),
            "mean_ess_mean": _finalize_running_scalar_mean(ess_mean_state),
            "min_ess_mean": ess_mean_min,
            "max_ess_mean": ess_mean_max,
            "mean_weight_entropy_mean": _finalize_running_scalar_mean(entropy_mean_state),
            "min_weight_entropy_mean": entropy_mean_min,
            "max_weight_entropy_mean": entropy_mean_max,
            "mean_weight_abundance_mean": _finalize_running_scalar_mean(abundance_mean_state),
            "min_weight_abundance_mean": abundance_mean_min,
            "max_weight_abundance_mean": abundance_mean_max,
            "se_quantile_l2_dim": _tensor_to_float_list(quantile_se),
            "se_quantile_avg_dim": _tensor_to_float_list(quantile_se_avg_dim),
            "mean_kurtosis_mean_dim": _tensor_to_float_list(kurt_mean_dims),
        }
        analysis_rows[m] = row

    available_m = sorted(analysis_rows.keys())
    if len(available_m) == 0:
        print("[INFO] no M has complete analysis outputs after filtering.")
        return

    sigma_values_all = sorted({analysis_rows[m]["sigma_y"] for m in available_m})
    sigma_tag = sigma_values_all[0] if len(sigma_values_all) == 1 else "mixed"

    analysis_dir = Path("save") / f"pf_analysis_{dataset}_{safe_obs}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"sigma_{sigma_tag}_batch_{test_traj_num}_len_{test_steps}"

    se_mean_rmse = [_maybe_float(analysis_rows[m]["se_mean_rmse"]) for m in available_m]
    se_mean_avg = [_maybe_float(analysis_rows[m]["se_mean_elementwise_avg"]) for m in available_m]
    se_cov_fnorm = [_maybe_float(analysis_rows[m]["se_cov_fnorm"]) for m in available_m]
    se_cov_avg = [_maybe_float(analysis_rows[m]["se_cov_elementwise_avg"]) for m in available_m]
    se_quantile_l2 = [_maybe_float(analysis_rows[m]["se_quantile_l2"]) for m in available_m]
    se_quantile_avg = [_maybe_float(analysis_rows[m]["se_quantile_avg"]) for m in available_m]

    _plot_multi_line(
        x=available_m,
        series={
            "mean SE": [float("nan") if v is None else v for v in se_mean_avg],
            "cov SE": [float("nan") if v is None else v for v in se_cov_avg],
        },
        title="",
        ylabel="Standard Error",
        save_path=analysis_dir / f"se_mean_cov_{base_tag}.png",
        force_linear_y=True,
    )

    _plot_multi_line(
        x=available_m,
        series={
            "mean RMSE SE": [float("nan") if v is None else v for v in se_mean_rmse],
            "mean avg SE": [float("nan") if v is None else v for v in se_mean_avg],
        },
        title="",
        ylabel="Standard Error",
        save_path=analysis_dir / f"se_mean_{base_tag}.png",
    )

    _plot_multi_line(
        x=available_m,
        series={
            "cov F-norm SE": [float("nan") if v is None else v for v in se_cov_fnorm],
            "cov avg SE": [float("nan") if v is None else v for v in se_cov_avg],
        },
        title="",
        ylabel="Standard Error",
        save_path=analysis_dir / f"se_cov_{base_tag}.png",
    )

    _plot_multi_line(
        x=available_m,
        series={
            "quantile L2[0,1] SE": [float("nan") if v is None else v for v in se_quantile_l2],
            "quantile avg SE": [float("nan") if v is None else v for v in se_quantile_avg],
        },
        title="",
        ylabel="Standard Error",
        save_path=analysis_dir / f"se_quantile_{base_tag}.png",
    )

    state_quantile_dim_series: Dict[str, List[float]] = {}
    pca_quantile_dim_series: Dict[str, List[float]] = {}
    for q_dim, q_name in enumerate(["x1", "x2", "x3", "pca1", "pca2", "pca3"]):
        y: List[float] = []
        for m in available_m:
            arr = analysis_rows[m].get("se_quantile_avg_dim")
            if isinstance(arr, list) and q_dim < len(arr):
                y.append(float(arr[q_dim]))
            else:
                y.append(float("nan"))
        if q_dim < 3:
            state_quantile_dim_series[f"{q_name} quantile SE"] = y
        else:
            pca_quantile_dim_series[f"{q_name} quantile SE"] = y

    if len(state_quantile_dim_series) > 0:
        _plot_multi_line(
            x=available_m,
            series=state_quantile_dim_series,
            title="",
            ylabel="Quantile SE",
            save_path=analysis_dir / f"se_quantile_state_dim_{base_tag}.png",
        )

    if len(pca_quantile_dim_series) > 0:
        _plot_multi_line(
            x=available_m,
            series=pca_quantile_dim_series,
            title="",
            ylabel="Quantile SE",
            save_path=analysis_dir / f"se_quantile_pca_dim_{base_tag}.png",
        )

    _plot_ess_entropy_abundance(
        x=available_m,
        ess=[
            float("nan")
            if _maybe_float(analysis_rows[m]["mean_ess_mean"]) is None
            else float(analysis_rows[m]["mean_ess_mean"])
            for m in available_m
        ],
        entropy=[
            float("nan")
            if _maybe_float(analysis_rows[m]["mean_weight_entropy_mean"]) is None
            else float(analysis_rows[m]["mean_weight_entropy_mean"])
            for m in available_m
        ],
        abundance=[
            float("nan")
            if _maybe_float(analysis_rows[m]["mean_weight_abundance_mean"]) is None
            else float(analysis_rows[m]["mean_weight_abundance_mean"])
            for m in available_m
        ],
        save_path=analysis_dir / f"mean_ess_entropy_abundance_{base_tag}.png",
    )

    _plot_multi_line_with_band(
        x=available_m,
        centers={
            "ESS mean": [
                float("nan")
                if _maybe_float(analysis_rows[m]["mean_ess_mean"]) is None
                else float(analysis_rows[m]["mean_ess_mean"])
                for m in available_m
            ],
            "weight abundance mean": [
                float("nan")
                if _maybe_float(analysis_rows[m]["mean_weight_abundance_mean"]) is None
                else float(analysis_rows[m]["mean_weight_abundance_mean"])
                for m in available_m
            ],
        },
        lowers={
            "ESS mean": [
                float("nan")
                if _maybe_float(analysis_rows[m]["min_ess_mean"]) is None
                else float(analysis_rows[m]["min_ess_mean"])
                for m in available_m
            ],
            "weight abundance mean": [
                float("nan")
                if _maybe_float(analysis_rows[m]["min_weight_abundance_mean"]) is None
                else float(analysis_rows[m]["min_weight_abundance_mean"])
                for m in available_m
            ],
        },
        uppers={
            "ESS mean": [
                float("nan")
                if _maybe_float(analysis_rows[m]["max_ess_mean"]) is None
                else float(analysis_rows[m]["max_ess_mean"])
                for m in available_m
            ],
            "weight abundance mean": [
                float("nan")
                if _maybe_float(analysis_rows[m]["max_weight_abundance_mean"]) is None
                else float(analysis_rows[m]["max_weight_abundance_mean"])
                for m in available_m
            ],
        },
        title="",
        ylabel="Mean",
        save_path=analysis_dir / f"mean_ess_abundance_{base_tag}.png",
    )

    kurt_dim_len = 0
    for m in available_m:
        arr = analysis_rows[m].get("mean_kurtosis_mean_dim")
        if isinstance(arr, list):
            kurt_dim_len = max(kurt_dim_len, min(3, len(arr)))
    if kurt_dim_len > 0:
        kurt_mean_series: Dict[str, List[float]] = {}
        for d in range(kurt_dim_len):
            y: List[float] = []
            for m in available_m:
                mean_arr = analysis_rows[m].get("mean_kurtosis_mean_dim")
                y.append(float(mean_arr[d]) if isinstance(mean_arr, list) and d < len(mean_arr) else float("nan"))
            kurt_mean_series[f"kurt dim {d + 1} mean"] = y

        _plot_multi_line(
            x=available_m,
            series=kurt_mean_series,
            title="",
            ylabel="Kurtosis mean",
            save_path=analysis_dir / f"mean_kurtosis_{base_tag}.png",
        )

    summary = {
        "dataset": dataset,
        "obs_fn": effective_obs,
        "pf_dir": str(pf_dir),
        "analysis_device": str(ANALYSIS_DEVICE),
        "test_steps": test_steps,
        "test_traj_num": test_traj_num,
        "built_in_pf_M_list": PF_M_LIST,
        "available_M": available_m,
        "sigma_conflicts": {str(k): v for k, v in sigma_conflicts.items()},
        "analysis": {str(k): analysis_rows[k] for k in available_m},
    }
    summary_path = analysis_dir / f"summary_{base_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] saved analysis summary: {summary_path}")
    print(f"[INFO] saved plots in: {analysis_dir}")


if __name__ == "__main__":
    main()
