#!/usr/bin/env python
"""
Create paper-style grid-search visualizations for Lorenz63 and Lorenz96.

Outputs:
    1. Lorenz63 EnKF, N=10: 1x3 SED-vs-inflation curves.
    2. Lorenz96 EnKF/LETKF, N=10: 2x3 ES heatmaps over inflation/localization.

The script reads full `.pt` grid-search outputs directly instead of using the
summary CSV files, since the `.pt` files contain the complete metric grids.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "save" / "torch_grid_search" / "visualization"
OBS_FNS = ("identity", "square", "arctan")
OBS_FN_TITLES = {
    "identity": "Partial + Identity",
    "square": "Partial + Square",
    "arctan": "Partial + Arctan",
}

LINE_COLOR = "#2c6fbb"
STAR_COLOR = "#ff0000"
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
COLUMN_LABEL_FONTSIZE = 16
METHOD_LABEL_FONTSIZE = 16


@dataclass(frozen=True)
class PanelSpec:
    dataset: str
    method: str
    obs_fn: str
    N: int
    pt_path: Path


L63_SPECS = tuple(
    PanelSpec(
        dataset="lorenz63",
        method="EnKF",
        obs_fn=obs_fn,
        N=10,
        pt_path=REPO_ROOT
        / "save"
        / "torch_grid_search"
        / "lorenz63"
        / filename,
    )
    for obs_fn, filename in (
        ("identity", "EnKF_N10_sigma2_obs_fn_identity.pt"),
        ("square", "EnKF_N10_sigma16p97_obs_fn_square.pt"),
        ("arctan", "EnKF_N10_sigma0p32_obs_fn_arctan.pt"),
    )
)

L96_SPECS = tuple(
    PanelSpec(
        dataset="lorenz96",
        method=method,
        obs_fn=obs_fn,
        N=10,
        pt_path=REPO_ROOT
        / "save"
        / "torch_grid_search"
        / "lorenz96"
        / filename,
    )
    for method, obs_fn, filename in (
        ("EnKF", "identity", "EnKF_N10_sigma1_obs_fn_identity.pt"),
        ("EnKF", "square", "EnKF_N10_sigma6p69_obs_fn_square.pt"),
        ("EnKF", "arctan", "EnKF_N10_sigma0p27_obs_fn_arctan.pt"),
        ("LETKF", "identity", "LETKF_N10_sigma1_obs_fn_identity.pt"),
        ("LETKF", "square", "LETKF_N10_sigma6p69_obs_fn_square.pt"),
        ("LETKF", "arctan", "LETKF_N10_sigma0p27_obs_fn_arctan.pt"),
    )
)


def load_grid_search(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing grid-search output: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Grid-search output is empty: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(float)
    return np.asarray(value, dtype=float)


def fmt_float(value) -> str:
    if value is None:
        return "None"
    value = float(value)
    if abs(value) >= 100 or (0 < abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def find_value_index(values: Iterable[float | None], target: float | None) -> int | None:
    values = list(values)
    if target is None:
        return next((idx for idx, value in enumerate(values) if value is None), None)

    finite_values = [
        (idx, float(value))
        for idx, value in enumerate(values)
        if value is not None and np.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return min(finite_values, key=lambda item: abs(item[1] - float(target)))[0]


def best_from_grid(grid: np.ndarray, infl_values: list[float], loc_values: list[float | None]) -> dict:
    finite_mask = np.isfinite(grid)
    if not np.any(finite_mask):
        return {
            "best_score": np.nan,
            "best_infl": None,
            "best_loc_radius": None,
            "best_i": None,
            "best_j": None,
        }

    flat_idx = int(np.nanargmin(np.where(finite_mask, grid, np.nan)))
    best_i, best_j = np.unravel_index(flat_idx, grid.shape)
    return {
        "best_score": float(grid[best_i, best_j]),
        "best_infl": infl_values[best_i],
        "best_loc_radius": loc_values[best_j],
        "best_i": int(best_i),
        "best_j": int(best_j),
    }


def beautify_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(out_path)
    return saved_paths


def plot_l63_grid(output_dir: Path, formats: list[str], dpi: int) -> tuple[list[Path], list[dict]]:
    fig, axes = plt.subplots(
        1,
        len(OBS_FNS),
        figsize=(12.2, 3.7),
        sharey=True,
        constrained_layout=True,
    )

    spec_by_obs = {spec.obs_fn: spec for spec in L63_SPECS}
    summary_rows = []
    for col_idx, obs_fn in enumerate(OBS_FNS):
        ax = axes[col_idx]
        spec = spec_by_obs[obs_fn]
        result = load_grid_search(spec.pt_path)
        grid = as_numpy(result["metric_grids"]["mean_pf_sed"])
        infl_values = [float(value) for value in result["infl_values"]]
        loc_values = result["loc_radius_values"]
        curve = grid[:, 0]
        finite = np.isfinite(curve)
        best = best_from_grid(grid, infl_values, loc_values)

        ax.plot(
            np.asarray(infl_values)[finite],
            curve[finite],
            color=LINE_COLOR,
            marker="o",
            markersize=5.5,
            linewidth=2.0,
        )
        if best["best_i"] is not None:
            ax.scatter(
                [best["best_infl"]],
                [best["best_score"]],
                marker="*",
                s=190,
                color=STAR_COLOR,
                edgecolor=STAR_COLOR,
                zorder=4,
            )

        ax.set_title(OBS_FN_TITLES[obs_fn], fontsize=COLUMN_LABEL_FONTSIZE, pad=8)
        ax.set_xlabel("inflation rate", fontsize=AXIS_LABEL_FONTSIZE)
        if col_idx == 0:
            ax.set_ylabel("SED", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(alpha=0.3)
        beautify_axis(ax)

        summary_rows.append(
            {
                "dataset": spec.dataset,
                "method": spec.method,
                "obs_fn": obs_fn,
                "metric": "SED",
                "best_infl": best["best_infl"],
                "best_loc_radius": best["best_loc_radius"],
                "best_score": best["best_score"],
                "finite_cells": int(np.isfinite(grid).sum()),
                "total_cells": int(grid.size),
                "path": str(spec.pt_path),
            }
        )

    saved_paths = save_figure(
        fig=fig,
        output_dir=output_dir,
        stem="lorenz63_enkf_N10_sed_grid",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return saved_paths, summary_rows


def plot_l96_grid(output_dir: Path, formats: list[str], dpi: int) -> tuple[list[Path], list[dict]]:
    methods = ("EnKF", "LETKF")
    fig, axes = plt.subplots(
        len(methods),
        len(OBS_FNS),
        figsize=(12.5, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    spec_by_key = {(spec.method, spec.obs_fn): spec for spec in L96_SPECS}
    loaded = {
        key: load_grid_search(spec.pt_path)
        for key, spec in spec_by_key.items()
    }
    all_es_values = []
    for result in loaded.values():
        grid = as_numpy(result["metric_grids"]["mean_es1"])
        all_es_values.append(grid[np.isfinite(grid)])
    finite_es_values = np.concatenate([values for values in all_es_values if values.size > 0])
    vmin = float(np.nanmin(finite_es_values))
    vmax = float(np.nanmax(finite_es_values))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    images = []
    summary_rows = []
    for row_idx, method in enumerate(methods):
        for col_idx, obs_fn in enumerate(OBS_FNS):
            ax = axes[row_idx, col_idx]
            spec = spec_by_key[(method, obs_fn)]
            result = loaded[(method, obs_fn)]
            grid = as_numpy(result["metric_grids"]["mean_es1"])
            infl_values = [float(value) for value in result["infl_values"]]
            loc_values = [None if value is None else float(value) for value in result["loc_radius_values"]]
            best = best_from_grid(grid, infl_values, loc_values)

            # Transpose so x is inflation rate and y is localization radius.
            image_grid = np.ma.masked_invalid(grid.T)
            im = ax.imshow(
                image_grid,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            images.append(im)

            if best["best_i"] is not None and best["best_j"] is not None:
                ax.scatter(
                    [best["best_i"]],
                    [best["best_j"]],
                    marker="*",
                    s=190,
                    color=STAR_COLOR,
                    edgecolor=STAR_COLOR,
                    linewidth=0.8,
                    zorder=4,
                )

            if row_idx == 0:
                ax.set_title(OBS_FN_TITLES[obs_fn], fontsize=COLUMN_LABEL_FONTSIZE, pad=8)
            if row_idx == len(methods) - 1:
                ax.set_xlabel("inflation rate", fontsize=AXIS_LABEL_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel("localization radius", fontsize=AXIS_LABEL_FONTSIZE)
                ax.text(
                    -0.38,
                    0.5,
                    method,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=METHOD_LABEL_FONTSIZE,
                    fontweight="bold",
                )

            ax.set_xticks(np.arange(len(infl_values)))
            ax.set_xticklabels([fmt_float(value) for value in infl_values])
            ax.set_yticks(np.arange(len(loc_values)))
            ax.set_yticklabels([fmt_float(value) for value in loc_values])
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

            summary_rows.append(
                {
                    "dataset": spec.dataset,
                    "method": spec.method,
                    "obs_fn": obs_fn,
                    "metric": "ES",
                    "best_infl": best["best_infl"],
                    "best_loc_radius": best["best_loc_radius"],
                    "best_score": best["best_score"],
                    "finite_cells": int(np.isfinite(grid).sum()),
                    "total_cells": int(grid.size),
                    "path": str(spec.pt_path),
                }
            )

    cbar = fig.colorbar(images[-1], ax=axes, location="right", shrink=0.92, pad=0.02)
    cbar.set_label("ES", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    saved_paths = save_figure(
        fig=fig,
        output_dir=output_dir,
        stem="lorenz96_enkf_letkf_N10_es_grid",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return saved_paths, summary_rows


def print_summary(title: str, rows: list[dict]) -> None:
    headers = [
        "dataset",
        "method",
        "obs_fn",
        "metric",
        "best_infl",
        "best_loc_radius",
        "best_score",
        "finite_cells",
    ]
    print(f"\n{title}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if header in {"best_infl", "best_loc_radius", "best_score"}:
                values.append(fmt_float(value))
            elif header == "finite_cells":
                values.append(f"{row['finite_cells']}/{row['total_cells']}")
            else:
                values.append(str(value))
        print("| " + " | ".join(values) + " |")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Lorenz63/Lorenz96 grid-search visualizations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to save, e.g. png pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for raster output formats.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    saved_paths = []

    l63_paths, l63_rows = plot_l63_grid(
        output_dir=args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
    )
    saved_paths.extend(l63_paths)
    print_summary("Lorenz63 EnKF grid-search SED summary", l63_rows)

    l96_paths, l96_rows = plot_l96_grid(
        output_dir=args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
    )
    saved_paths.extend(l96_paths)
    print_summary("Lorenz96 EnKF/LETKF grid-search ES summary", l96_rows)

    print("\nSaved figure(s):")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
