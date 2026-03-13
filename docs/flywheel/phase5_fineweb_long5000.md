# Phase 5 FineWeb 5000-Step Benchmark

Question:
Does the best short-horizon scheduled variant remain competitive over a longer matched-LR horizon?

Primary artifacts:
- `assets/fineweb_long5000_lr_micro_val_vs_scale.png`
- `outputs/flywheel_long5000_lr_micro_20260308/long5000_lr_micro_results.csv`
- `outputs/flywheel_long5000_lr_micro_20260308/summary.txt`

Shared setup:
- same FineWeb family as `docs/flywheel/root_repro_index.md`
- device for these historical runs: `mps`
- methods compared: `baseline_muon` vs `law_powerg2_osc`
- LR scales compared: `1.6`, `2.0`, `2.4`
- horizon: `5000` steps

Representative baseline command template (`scale=2.0`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=mps --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=5000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=False \
  --out_dir=out_fineweb_long5000_baseline_muon_s2p0_seed1337 --wandb_log=False
```

Representative scheduled command template (`scale=2.0`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=mps --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=5000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_law --power_p_low=0.0 --power_p_high=0.2 \
  --power_entropy_low=0.66 --power_entropy_high=0.74 \
  --power_entropy_law=power --power_entropy_gamma=2.0 \
  --power_entropy_linear_coeff=1.0 \
  --power_entropy_osc_amp=0.45 --power_entropy_osc_period=8 \
  --out_dir=out_fineweb_long5000_law_powerg2_osc_s2p0_seed1337 --wandb_log=False
```

Observed result from the CSV summary:
- baseline is better at every tested scale
- best baseline `best_val`: `6.0932` at scale `2.4`
- best scheduled `best_val`: `6.2311` at scale `2.4`

Conclusion:
The long-horizon matched-LR benchmark rejects the current scheduled candidate.
