# Lorenz '96 Experiments

Runnable command templates:

```bash
DATASET=lorenz96 MODEL=CorrTerms LOSS=nl2 DRY_RUN=true scripts/train_example.sh
DATASET=lorenz96 METHOD=LETKF DRY_RUN=true scripts/benchmark_example.sh
DATASET=lorenz96 METHOD=LETKF DRY_RUN=true scripts/grid_search_example.sh
```

Remove `DRY_RUN=true` to launch the corresponding CUDA run from the repository root.
