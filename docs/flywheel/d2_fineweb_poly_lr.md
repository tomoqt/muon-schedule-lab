# D2 FineWeb Polynomial LR Sweep

Question:
Do polynomial scheduled updates produce a real improvement under matched LR?

Primary artifacts:
- `assets/fineweb_small_lr_sweep_poly_v2_val_vs_lr.png`
- `assets/fineweb_small_lr_sweep_poly_v2_meanstd_val_vs_lr.png`

Shared setup:
- see `docs/flywheel/root_repro_index.md`
- polynomial backend runs use `power_backend=poly`, `power_poly_degree=5`

Representative baseline command (`scale=2.0`, seed `1337`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=mps --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=1000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=False \
  --out_dir=out_fineweb_small_lr_sweep_poly_v2_baseline_muon_s2p0_seed1337 --wandb_log=False
```

Representative scheduled command (`power(g=2)` law, no oscillation, same LR scale and seed):
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=mps --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=1000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_law --power_p_low=0.0 --power_p_high=0.12 \
  --power_entropy_low=0.66 --power_entropy_high=0.74 \
  --power_entropy_law=power --power_entropy_gamma=2.0 \
  --power_entropy_linear_coeff=1.0 \
  --out_dir=out_fineweb_small_lr_sweep_poly_v2_law_powerg2_s2p0_seed1337 --wandb_log=False
```

Observed result:
- mean final-val advantage for the best scheduled variants is only around `0.008-0.016`
- seed standard deviation is around `0.09-0.10`

Conclusion:
The short-horizon signal is real enough to study, but too small to call decisive.
