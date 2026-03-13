# D1 FineWeb Baseline vs Fixed-Alpha Sanity Check

Question:
Does fixed `alpha=0.5` at least match baseline Muon once LR is tuned on FineWeb?

Primary artifact:
- `assets/fineweb_baseline_fixed_alpha_meanstd.png`

Shared setup:
- see `docs/flywheel/root_repro_index.md`
- corrected v2 sweep uses `seed in {1337,1338,1339}` and LR scales `{0.5,1.0,1.5,2.0}`

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

Representative fixed-alpha command (`alpha=0.5`, so `p=0`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=mps --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=1000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=fixed_alternating --power_p_low=0.0 --power_p_high=0.0 \
  --power_alternation_period=50 \
  --out_dir=out_fineweb_small_lr_sweep_poly_v2_fixed_alpha05_s2p0_seed1337 --wandb_log=False
```

Observed result:
- at LR scale `2.0`, seed `1337`, baseline final val: `6.7832`
- fixed alpha=0.5 final val: `6.7878`
- over 3 seeds, fixed alpha remains slightly worse at every LR scale

Conclusion:
Fixed `p=0` is competitive, but it does not beat baseline Muon in this setup.
