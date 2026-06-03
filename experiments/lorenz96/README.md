# Lorenz '96 Experiments

Runnable command templates:

```bash
DATASET=lorenz96 MODEL=CorrTerms LOSS=nl2 scripts/train_example.sh
DATASET=lorenz96 METHOD=LETKF scripts/benchmark_example.sh
DATASET=lorenz96 METHOD=LETKF scripts/grid_search_example.sh
```

Run these commands from the repository root. The training template defaults to `EPOCHS=100`.
