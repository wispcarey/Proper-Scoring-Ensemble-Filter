# Linear-Gaussian Experiments

Runnable command templates:

```bash
DATASET=linear MODEL=EtE-LRes LOSS=es DRY_RUN=true scripts/train_example.sh
DATASET=linear METHOD=EnKF DRY_RUN=true scripts/benchmark_example.sh
DATASET=linear METHOD=linear_uncertainty DRY_RUN=true scripts/benchmark_example.sh
DATASET=linear METHOD=EnKF DRY_RUN=true scripts/grid_search_example.sh
```

Remove `DRY_RUN=true` to launch the corresponding CUDA run from the repository root.
