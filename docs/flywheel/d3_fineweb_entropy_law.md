# D3 FineWeb Entropy-Law Coefficient And Oscillation Sweeps

Question:
Does coefficient tuning plus narrow oscillation make the entropy-law branch materially stronger?

Primary artifact:
- `assets/fineweb_entropy_law_meanstd.png`

Shared setup:
- see `docs/flywheel/root_repro_index.md`

Representative oscillatory law command (`power(g=2)`, `p_high=0.2`, oscillation amplitude `0.45`, period `8`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=mps --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=1000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_law --power_p_low=0.0 --power_p_high=0.2 \
  --power_entropy_low=0.66 --power_entropy_high=0.74 \
  --power_entropy_law=power --power_entropy_gamma=2.0 \
  --power_entropy_linear_coeff=1.0 \
  --power_entropy_osc_amp=0.45 --power_entropy_osc_period=8 \
  --out_dir=out_fineweb_small_lr_sweep_poly_v2_law_powerg2_osc_s2p0_seed1337 --wandb_log=False
```

Observed result:
- the oscillatory and non-oscillatory `power(g=2)` laws are the strongest scheduled FineWeb runs
- at LR scale `2.0`, seed `1337`, both land around `6.7776`
- the gap over baseline is tiny relative to seed variance

Conclusion:
This branch identifies the best scheduled family, but not a decisive winner.
