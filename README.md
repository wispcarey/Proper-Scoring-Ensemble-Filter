# Proper Scoring Ensemble Filter

Code for the submitted paper **"Learning Probabilistic Filters with Strictly Proper Scoring Rules"**.

This repository contains the research implementation for Proper Scoring Ensemble Filter (PSEF) experiments, including energy-score training, ensemble forecast-analysis training, neural analysis maps such as EtE-LRes and CorrTerms, and classical data-assimilation baselines.

## Repository Scope

The repository is intended to support the paper, not to preserve every internal experiment artifact. Generated outputs, cluster launch files, and notebooks are not part of the public code release.

Core paper-relevant components are organized under `psef/`:

- `psef/losses/`: ES, NL2, L2, and diagnostic ensemble metrics.
- `psef/models/`: Set Transformer blocks, permutation-invariant components, EtE-LRes helpers, and CorrTerms helpers.
- `psef/filters/`: EnKF, ESRF, LETKF, IEnKF/iEnKS entry points, BPF, and localization utilities.
- `psef/dynamics/`: Lorenz '63, Lorenz '96, doubling-angle dynamics, and observation maps.
- `psef/data/`: linear-Gaussian datasets and dataloader helpers.
- `psef/training/`: training, evaluation, checkpoint, and model setup helpers.
- `psef/plotting/`: paper plotting and particle-filter visualization helpers.

## Installation

Use the setup script to create `.venv` and install `requirements.txt`. It uses
`python3.12` by default; override `PYTHON_BIN` if your interpreter has another
name.

```bash
scripts/setup_venv.sh
PYTHON_BIN=python scripts/setup_venv.sh
```

All scripts below use `python` by default. To run them through the local virtual
environment without activating it, set `PYTHON_BIN=.venv/bin/python`.

## Tests

After installing dependencies, run checks through scripts:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_tests.sh
PYTHON_BIN=.venv/bin/python scripts/import_check.sh
```

These checks do not run expensive experiments, GPU training, grid searches, or data generation.

## Main Entry Points

Inspect the main command-line interfaces:

```bash
ENTRYPOINT=train scripts/cli_help.sh
ENTRYPOINT=evaluate scripts/cli_help.sh
ENTRYPOINT=benchmark scripts/cli_help.sh
ENTRYPOINT=grid_search scripts/cli_help.sh
ENTRYPOINT=pf_generate scripts/cli_help.sh
ENTRYPOINT=pf_process scripts/cli_help.sh
```

Check the public CUDA command templates without launching training or evaluation:

```bash
DRY_RUN=true scripts/train_example.sh
DRY_RUN=true scripts/benchmark_example.sh
DRY_RUN=true scripts/grid_search_example.sh
```

More dry-run training examples:

```bash
DATASET=lorenz96 MODEL=CorrTerms LOSS=nl2 DRY_RUN=true scripts/train_example.sh
DATASET=linear MODEL=EtE-LRes LOSS=es DRY_RUN=true scripts/train_example.sh
DATASET=doubling1d MODEL=EtE-LRes LOSS=l2 DRY_RUN=true scripts/train_example.sh
```

More dry-run benchmark examples:

```bash
DATASET=lorenz96 METHOD=LETKF DRY_RUN=true scripts/benchmark_example.sh
DATASET=linear METHOD=EnKF DRY_RUN=true scripts/benchmark_example.sh
DATASET=doubling1d METHOD=ESRF DRY_RUN=true scripts/benchmark_example.sh
```

Dry-run grid-search examples:

```bash
DATASET=linear METHOD=EnKF DRY_RUN=true scripts/grid_search_example.sh
DATASET=lorenz96 METHOD=LETKF DRY_RUN=true scripts/grid_search_example.sh
DATASET=doubling1d METHOD=ESRF DRY_RUN=true scripts/grid_search_example.sh
```

To run one of these templates for real, remove `DRY_RUN=true`. Training,
evaluation, and grid-search scripts use `--device cuda` by default. To evaluate a
trained neural PSEF checkpoint, pass the checkpoint path through the evaluation
script:

```bash
CP_LOAD_PATH=save/path/to/checkpoint.pth DRY_RUN=true scripts/evaluate_example.sh
```

Checkpoints are produced under `save/` by training runs and are not bundled in
this repository.

## Script Defaults

The public scripts encode the intended defaults:

- `lorenz63` and `doubling1d` automatically use `--no_localization`.
- `LOSS=es` in training/evaluation scripts explicitly uses `--es_p 1`.
- Training scripts use `--no_running_loss`; the CLI default also disables running loss.
- `SAVE_EPOCH` defaults to `25`, matching the CLI default.

Use `LOCALIZATION=false` to force `--no_localization` on other datasets, or leave
`LOCALIZATION=auto` for the dataset-aware default.

## PF Verification

PF verification requires cached particle-filter data before the verifying run.
Generate and process the cache through scripts with matching dataset, observation
function, `sigma_y`, test trajectory count/length, `PF_N`, and seed settings.

Generate per-seed PF caches:

```bash
DATASET=lorenz63 SEED=42 PF_N=1000 DRY_RUN=true scripts/gen_pf_results_example.sh
DATASET=lorenz63 SEED=43 PF_N=1000 DRY_RUN=true scripts/gen_pf_results_example.sh
```

Process the saved PF caches and write the averaged cache:

```bash
DATASET=lorenz63 PF_N=1000 DRY_RUN=true scripts/process_pf_results_example.sh
```

Run benchmark or neural evaluation with PF verification:

```bash
DATASET=lorenz63 METHOD=ESRF PF_VERIFICATION=true PF_VERIFICATION_SEED=none PF_N=1000 DRY_RUN=true scripts/benchmark_example.sh
DATASET=lorenz63 CP_LOAD_PATH=save/path/to/checkpoint.pth PF_VERIFICATION=true PF_VERIFICATION_SEED=none PF_N=1000 DRY_RUN=true scripts/evaluate_example.sh
```

Grid-search with PF verification also uses the script entry point:

```bash
DATASET=lorenz63 METHOD=ESRF PF_VERIFICATION=true PF_VERIFICATION_SEED=none PF_N=1000 DRY_RUN=true scripts/grid_search_example.sh
```

Grid-search and paper plotting helpers remain in `scripts/` and `psef/plotting/`.

## Experiments

The paper-relevant experiment families are:

- Linear-Gaussian model.
- Doubling-angle / doubling1d model.
- Lorenz '63.
- Lorenz '96.
- Observation maps such as identity/default, square, arctan, cosine/cos2pi, and related variants.

See `experiments/` and `scripts/` for command templates. Some commands expect checkpoints or generated results under `save/`; these files are intentionally not included.

## Data And Checkpoints

Datasets, checkpoints, cached particle-filter results, and generated figures are produced locally and are ignored by Git. A fresh run may create directories such as `data/` and `save/`.

No pretrained checkpoints are bundled.

## License

MIT License, copyright Bohan Chen.
