#!/usr/bin/env /home/bhchen/miniconda3/bin/python
"""
Print default model/training settings for selected datasets and NN methods.

The script intentionally reuses config.cli.get_parameters() and
train_test_utils.set_models() so the parameter counts match the training code.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

warnings.filterwarnings("ignore", message="Keyboard interaction.*")

from config.cli import get_parameters
from networks import NaiveNetwork, SetTransformer
from train_test_utils import set_models


DEFAULT_DATASETS = ("linear", "doubling1d", "lorenz63", "lorenz96")
DEFAULT_METHODS = ("CorrTerms", "EtE-LRes")


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    method: str
    total_params: int
    set_transformer_params: int
    st_type: str
    st_latent_dim: int
    st_output_dim_arg: int
    st_output_feature_dim: int
    st_num_seeds: int
    st_encoder_sab_blocks: int
    st_decoder_sab_blocks: int
    st_attention_heads: int
    train_traj_num: int
    train_length: int
    test_traj_num: int
    test_length: int
    learning_rate: float
    batch_size: int
    clamp: float


def count_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def contains_set_transformer(module: nn.Module) -> bool:
    return any(isinstance(child, SetTransformer) for child in module.modules())


def first_set_transformer(modules: Iterable[nn.Module]) -> SetTransformer:
    for module in modules:
        for child in module.modules():
            if isinstance(child, SetTransformer):
                return child
    raise RuntimeError("No SetTransformer module was constructed.")


def build_args(dataset: str, method: str):
    argv = [
        "print_default_model_summary.py",
        "--dataset",
        dataset,
        "--v",
        method,
        "--device",
        "cpu",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            return get_parameters()
    finally:
        sys.argv = old_argv


def summarize_one(dataset: str, method: str) -> SummaryRow:
    args = build_args(dataset, method)
    with contextlib.redirect_stdout(io.StringIO()):
        model_list = set_models(args)

    total_params = sum(count_params(module) for module in model_list)
    st_modules = [
        module
        for module in model_list
        if not isinstance(module, NaiveNetwork) and contains_set_transformer(module)
    ]
    st_params = sum(count_params(module) for module in st_modules)
    st_model = first_set_transformer(st_modules)

    return SummaryRow(
        dataset=args.dataset,
        method=args.v,
        total_params=total_params,
        set_transformer_params=st_params,
        st_type=args.st_type,
        st_latent_dim=args.hidden_dim,
        st_output_dim_arg=args.st_output_dim,
        st_output_feature_dim=st_model.fc_out.out_features,
        st_num_seeds=args.st_num_seeds,
        st_encoder_sab_blocks=len(st_model.enc),
        st_decoder_sab_blocks=len(st_model.dec),
        st_attention_heads=st_model.enc[0].mab.num_heads,
        train_traj_num=args.train_traj_num,
        train_length=args.train_steps,
        test_traj_num=args.test_traj_num,
        test_length=args.test_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        clamp=args.clamp,
    )


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def print_markdown(rows: list[SummaryRow]) -> None:
    headers = [
        "dataset",
        "method",
        "total_params",
        "set_transformer_params",
        "st_type",
        "st_latent_dim",
        "st_output_feature_dim",
        "st_encoder_sab_blocks",
        "st_decoder_sab_blocks",
        "st_attention_heads",
        "train_traj_num",
        "train_length",
        "test_traj_num",
        "test_length",
        "learning_rate",
        "batch_size",
        "clamp",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [format_value(getattr(row, header)) for header in headers]
        print("| " + " | ".join(values) + " |")


def print_csv(rows: list[SummaryRow]) -> None:
    headers = list(SummaryRow.__dataclass_fields__.keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=headers)
    writer.writeheader()
    for row in rows:
        writer.writerow({header: getattr(row, header) for header in headers})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print default NN parameter counts and training settings."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to summarize.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="NN methods/versions to summarize.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        summarize_one(dataset=dataset, method=method)
        for dataset in args.datasets
        for method in args.methods
    ]

    if args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
