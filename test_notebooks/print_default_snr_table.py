import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Ensure project root is on path when running from test_notebooks/.
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dataset_info import DATASET_INFO
from utils import (
    DapperKS,
    DoublingMap1D,
    ComplexSquareMap2D,
    L63,
    L96,
    etd_rk4_wrapper,
    partial_obs_operator,
    project_to_unit_circle,
    rk4,
)


DATASETS = ["doubling1d", "complex2d", "lorenz63", "lorenz96", "ks"]
OBS_FNS_FOR_CHAOTIC = ["identity", "square_root", "square", "arctan"]
SIGMA_Y_FALLBACK = {
    "lorenz63": 1.0,
    "lorenz96": 1.0,
    "ks": 1.0,
}


def get_default_sigma_y(dataset: str) -> float:
    sigma_y = DATASET_INFO[dataset].get("sigma_y", None)
    if sigma_y is None:
        if dataset not in SIGMA_Y_FALLBACK:
            raise ValueError(f"Dataset '{dataset}' has no default sigma_y and no fallback configured.")
        sigma_y = SIGMA_Y_FALLBACK[dataset]
    return float(sigma_y)


def init_state(dataset: str, batch_size: int, dim: int, dt: float, dt_iter: int) -> torch.Tensor:
    if dataset == "doubling1d":
        return torch.rand(batch_size, 1, dtype=torch.float32)
    if dataset == "complex2d":
        theta = 2.0 * math.pi * torch.rand(batch_size, 1, dtype=torch.float32)
        return torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)
    if dataset == "lorenz63":
        return torch.randn(batch_size, dim, dtype=torch.float32)
    if dataset == "lorenz96":
        return torch.randn(batch_size, dim, dtype=torch.float32) + 5.0
    if dataset == "ks":
        try:
            dapper_x0 = DapperKS(dt=dt / dt_iter).x0
            base = torch.as_tensor(dapper_x0, dtype=torch.float32).reshape(1, dim)
            return torch.randn(batch_size, dim, dtype=torch.float32) + base
        except Exception:
            return torch.randn(batch_size, dim, dtype=torch.float32)
    raise NotImplementedError(f"Unsupported dataset: {dataset}")


def advance_one_step(
    dataset: str,
    x: torch.Tensor,
    forward_fun,
    sigma_v: float,
    dt: float,
    dt_iter: int,
    step_idx: int,
) -> torch.Tensor:
    sigma_v_eff = 0.0 if sigma_v is None else float(sigma_v)

    if dataset == "doubling1d":
        x = forward_fun(x)
        x = torch.remainder(x + sigma_v_eff * torch.randn_like(x), 1.0)
        return x

    if dataset == "complex2d":
        x = forward_fun(x)
        x = x + sigma_v_eff * torch.randn_like(x)
        x = project_to_unit_circle(x)
        return x

    if dataset == "ks":
        sub_dt = dt / dt_iter
        for _ in range(dt_iter):
            x = forward_fun(x, None, sub_dt)
        x = x + sigma_v_eff * torch.randn_like(x)
        return x

    sub_dt = dt / dt_iter
    for j_iter in range(dt_iter):
        t_curr = step_idx * dt + j_iter * sub_dt
        x = rk4(forward_fun, x, t_curr, sub_dt)
    x = x + sigma_v_eff * torch.randn_like(x)
    return x


def simulate_trajectories(
    dataset: str,
) -> Tuple[torch.Tensor, float]:
    cfg = DATASET_INFO[dataset]
    dim = int(cfg["dim"])
    test_steps = int(cfg["test_steps"])
    test_traj_num = int(cfg["test_traj_num"])
    dt = float(cfg["dt"])
    dt_iter = int(cfg["dt_iter"])
    sigma_v = cfg.get("sigma_v", None)
    sigma_y = get_default_sigma_y(dataset)

    if dataset == "lorenz63":
        forward_fun = L63.forward
    elif dataset == "lorenz96":
        forward_fun = L96.forward
    elif dataset == "doubling1d":
        forward_fun = DoublingMap1D.forward
    elif dataset == "complex2d":
        forward_fun = ComplexSquareMap2D.forward
    elif dataset == "ks":
        forward_fun = etd_rk4_wrapper(device=torch.device("cpu"), dt=dt / dt_iter, Nx=dim)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    x = init_state(dataset, test_traj_num, dim, dt=dt, dt_iter=dt_iter)

    traj = []
    with torch.no_grad():
        for step_idx in range(test_steps):
            x = advance_one_step(
                dataset=dataset,
                x=x,
                forward_fun=forward_fun,
                sigma_v=sigma_v,
                dt=dt,
                dt_iter=dt_iter,
                step_idx=step_idx,
            )
            traj.append(x.clone())

    states = torch.stack(traj, dim=0)  # [T, B, D]
    return states, sigma_y


def compute_snr_stats(states: torch.Tensor, obs_inds: torch.Tensor, obs_fn: str, sigma_y: float) -> Tuple[float, float]:
    ori_dim = states.shape[-1]
    obs_fun, _ = partial_obs_operator(
        ori_dim=ori_dim,
        obs_inds=obs_inds,
        device=torch.device("cpu"),
        obs_fn=obs_fn,
        obs_fn_out_dim=None,
        obs_fn_seed=None,
        obs_custom_fn_path=None,
    )
    h_values = obs_fun(states)  # [T, B, d_y]
    mean_h = torch.mean(h_values, dim=0, keepdim=True)
    signal_term = torch.mean(torch.sum((h_values - mean_h) ** 2, dim=-1), dim=0)  # [B]
    d_y = h_values.shape[-1]
    snr = signal_term / (float(d_y) * (float(sigma_y) ** 2))
    return snr.mean().item(), snr.std(unbiased=False).item()


def save_csv(headers: List[str], rows: List[List[str]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        parts = [f" {cell:<{widths[i]}} " for i, cell in enumerate(cells)]
        return "|" + "|".join(parts) + "|"

    sep = "|" + "|".join(["-" * (w + 2) for w in widths]) + "|"
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print SNR table with dataset default settings.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory generation.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    headers = [
        "dataset",
        "sigma_y_default",
        "default",
        "square_root",
        "square",
        "arctan",
    ]
    rows: List[List[str]] = []

    for dataset in DATASETS:
        cfg = DATASET_INFO[dataset]
        obs_inds = torch.as_tensor(cfg["obs_inds"], dtype=torch.long)
        default_obs_fn = str(cfg.get("obs_fn", "identity")).lower()

        states, sigma_y = simulate_trajectories(dataset=dataset)
        row: Dict[str, str] = {
            "dataset": dataset,
            "sigma_y_default": f"{sigma_y:.6f}",
            "default": "",
            "square_root": "",
            "square": "",
            "arctan": "",
        }

        if dataset in ["doubling1d", "complex2d"]:
            mean_val, std_val = compute_snr_stats(
                states=states,
                obs_inds=obs_inds,
                obs_fn=default_obs_fn,
                sigma_y=sigma_y,
            )
            _ = std_val
            row["default"] = f"{mean_val:.6f}"
        else:
            mean_default, std_default = compute_snr_stats(
                states=states,
                obs_inds=obs_inds,
                obs_fn="identity",
                sigma_y=sigma_y,
            )
            _ = std_default
            row["default"] = f"{mean_default:.6f}"
            for obs_fn in OBS_FNS_FOR_CHAOTIC:
                if obs_fn == "identity":
                    continue
                mean_val, std_val = compute_snr_stats(
                    states=states,
                    obs_inds=obs_inds,
                    obs_fn=obs_fn,
                    sigma_y=sigma_y,
                )
                _ = std_val
                row[obs_fn] = f"{mean_val:.6f}"

        rows.append([row[h] for h in headers])

    save_path = ROOT / "save" / "default_snr" / "default_snr.csv"
    save_csv(headers=headers, rows=rows, save_path=save_path)
    print(f"Saved CSV to: {save_path}")
    print_table(headers, rows)


if __name__ == "__main__":
    main()
