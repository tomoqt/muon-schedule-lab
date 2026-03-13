# C2 Alternating Schedules On Shakespeare-Char

Question:
Can simple periodic switching between low and high `p` improve exploration/exploitation balance?

Primary artifact:
- `assets/char_alternating_horizons.png`

Shared setup:
- see `docs/flywheel/root_repro_index.md`

Representative fixed alternation (`period=20`, `p_low=0`, `p_high=1`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_shakespeare_char.py \
  --dataset=shakespeare_char --use_muon=True --device=cpu --dtype=float32 \
  --batch_size=8 --block_size=64 --gradient_accumulation_steps=1 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.001 --muon_lr=0.01 --min_lr=1e-5 \
  --warmup_iters=100 --lr_decay_iters=5000 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=fixed_alternating --power_p_low=0.0 --power_p_high=1.0 \
  --power_alternation_period=20 \
  --out_dir=out_long2000_fixed_alt_p01_period20 --wandb_log=False
```

Representative entropy alternation (`p_low=0`, `p_high=1`, thresholds `0.64/0.66`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_shakespeare_char.py \
  --dataset=shakespeare_char --use_muon=True --device=cpu --dtype=float32 \
  --batch_size=8 --block_size=64 --gradient_accumulation_steps=1 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.001 --muon_lr=0.01 --min_lr=1e-5 \
  --warmup_iters=100 --lr_decay_iters=5000 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_alternating --power_p_low=0.0 --power_p_high=1.0 \
  --power_entropy_low=0.64 --power_entropy_high=0.66 --power_entropy_initial_mode=low \
  --out_dir=out_long2000_entropy_alt_p01 --wandb_log=False
```

Observed result at 2000 steps:
- baseline final val: `1.9162`
- fixed alternating period 20: `2.0858`
- entropy alternating: `2.2687`

Conclusion:
Simple alternating schedules are consistently worse than baseline across short and medium horizons.
