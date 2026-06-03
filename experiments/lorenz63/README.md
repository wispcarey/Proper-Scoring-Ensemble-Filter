# Lorenz '63 Experiments

Runnable command templates:

```bash
DATASET=lorenz63 MODEL=EtE-LRes LOSS=es scripts/train_example.sh
DATASET=lorenz63 METHOD=ESRF scripts/benchmark_example.sh
DATASET=lorenz63 METHOD=ESRF scripts/grid_search_example.sh
```

Run these commands from the repository root. The training template defaults to `EPOCHS=100`.
