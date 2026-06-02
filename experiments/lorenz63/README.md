# Lorenz '63 Experiments

Runnable command templates:

```bash
DATASET=lorenz63 MODEL=EtE-LRes LOSS=es DRY_RUN=true scripts/train_example.sh
DATASET=lorenz63 METHOD=ESRF DRY_RUN=true scripts/benchmark_example.sh
DATASET=lorenz63 METHOD=ESRF DRY_RUN=true scripts/grid_search_example.sh
```

Remove `DRY_RUN=true` to launch the corresponding CUDA run from the repository root.
