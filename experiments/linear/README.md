# Linear-Gaussian Experiments

Runnable command templates:

```bash
DATASET=linear MODEL=EtE-LRes LOSS=es scripts/train_example.sh
DATASET=linear METHOD=EnKF scripts/benchmark_example.sh
DATASET=linear METHOD=linear_uncertainty scripts/benchmark_example.sh
DATASET=linear METHOD=EnKF scripts/grid_search_example.sh
```

Run these commands from the repository root. The training template defaults to `EPOCHS=100`.
