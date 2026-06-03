# Scripts

This directory contains small public launch templates and helper scripts.

- `setup_venv.sh`: creates `.venv` and installs `requirements.txt`.
- `run_tests.sh`: runs the pytest suite.
- `import_check.sh`: runs a minimal package import check.
- `cli_help.sh`: prints help for public entry points via `ENTRYPOINT=...`.
- `train_example.sh`: template for a small PSEF training command.
- `evaluate_example.sh`: template for neural checkpoint evaluation.
- `benchmark_example.sh`: template for one classical-filter benchmark command.
- `grid_search_example.sh`: template for one benchmark grid-search command.
- `gen_pf_results_example.sh`: generates cached particle-filter results.
- `process_pf_results_example.sh`: processes PF caches and writes averaged PF caches.
- `print_default_snr_table.py`: regenerates `save/default_snr/default_snr.csv` for
  `--adaptive_sigma_y`.
- `print_default_model_summary.py`: prints model summaries for default configs.
- `visualize_grid_search_l63_l96.py`: helper for visualizing saved grid-search CSVs.

The shell templates default to CUDA training/evaluation commands and accept
environment-variable overrides for local runs.

Script defaults intentionally match the public experiment setup:

- `lorenz63` and `doubling1d` use `--no_localization`.
- Energy-score training/evaluation uses `--es_p 1`.
- Training/evaluation templates use `--no_running_loss`.
- Training `EPOCHS` defaults to `100`.
- Training `SAVE_EPOCH` defaults to `25`.
