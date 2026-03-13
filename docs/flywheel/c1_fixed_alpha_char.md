# C1 Fixed-Alpha Sweeps On Shakespeare-Char

Question:
Can a single fixed singular-value exponent beat baseline Muon on `shakespeare_char`?

Primary artifact:
- `assets/char_fixed_alpha_full.png`
- `assets/alpha_refine_below_05.png`

Shared setup:
- see `docs/flywheel/root_repro_index.md`
- fixed alpha is encoded as a constant `p = 1 - 2*alpha`
- implementation detail: constant `p` is represented with `power_schedule_type=fixed_alternating` and `power_p_low=power_p_high=p`

Representative baseline:
```bash
uv run --with-requirements requirements.txt python train.py config/train_shakespeare_char.py \
  --dataset=shakespeare_char --use_muon=True --device=cpu --dtype=float32 \
  --batch_size=8 --block_size=64 --gradient_accumulation_steps=1 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.001 --muon_lr=0.01 --min_lr=1e-5 \
  --warmup_iters=100 --lr_decay_iters=5000 \
  --enable_power_schedules=False \
  --out_dir=out_long2000_baseline_muon --wandb_log=False
```

Representative fixed-alpha run (`alpha=0.5`, so `p=0`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_shakespeare_char.py \
  --dataset=shakespeare_char --use_muon=True --device=cpu --dtype=float32 \
  --batch_size=8 --block_size=64 --gradient_accumulation_steps=1 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.001 --muon_lr=0.01 --min_lr=1e-5 \
  --warmup_iters=100 --lr_decay_iters=5000 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=fixed_alternating --power_p_low=0.0 --power_p_high=0.0 \
  --power_alternation_period=20 \
  --out_dir=out_long2000_fixed_alpha_0p5_p_0p0 --wandb_log=False
```

Observed result:
- baseline final val: `1.9162`
- fixed `alpha=0.5` final val: `1.9848`
- the refined sweep below `alpha=0.5` also stayed above baseline

Conclusion:
The fixed-alpha family peaks near Muon-like `p=0`, but none of the tested fixed exponents beat baseline Muon.
