# Doubling1d Experiments

Typical entry points:

```bash
python train.py --dataset doubling1d --N 30 --loss_type es --device cuda --no_localization
python gen_pf_results.py --dataset doubling1d --seed 42 --device cuda --pf_verification --pf_N 100000
```

Use `scripts/train_example.sh` as the public training template. Notebooks were removed from the final public tree.
