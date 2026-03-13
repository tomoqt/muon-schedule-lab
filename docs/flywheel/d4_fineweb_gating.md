# D4 FineWeb 500/2000-Step Gating Reruns

Question:
If the short-horizon scheduled edge is real, does it survive when the horizon is extended from `500` to `2000` steps?

Primary artifact:
- `assets/fineweb_gating_500_2000.png`

Shared setup:
- same FineWeb v2 family as in `docs/flywheel/root_repro_index.md`
- these runs were executed on Flywheel-managed compute with `device=cuda` on a single A10G
- LR scale fixed at `2.0`

Exact run directories compared in the figure:
- `out_flywheel_fineweb_gate500_baseline_muon_s2p0_seed1337`
- `out_flywheel_fineweb_gate500_law_powerg2_osc_s2p0_seed1337`
- `out_flywheel_fineweb_gate2000_baseline_muon_s2p0_seed1337`
- `out_flywheel_fineweb_gate2000_law_powerg2_osc_s2p0_seed1337`

Baseline 500-step command:
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=cuda --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=500 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=False \
  --out_dir=out_flywheel_fineweb_gate500_baseline_muon_s2p0_seed1337 --wandb_log=False
```

Scheduled 500-step command:
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=cuda --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=500 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_law --power_p_low=0.0 --power_p_high=0.2 \
  --power_entropy_low=0.66 --power_entropy_high=0.74 \
  --power_entropy_law=power --power_entropy_gamma=2.0 \
  --power_entropy_linear_coeff=1.0 \
  --power_entropy_osc_amp=0.45 --power_entropy_osc_period=8 \
  --out_dir=out_flywheel_fineweb_gate500_law_powerg2_osc_s2p0_seed1337 --wandb_log=False
```

Baseline 2000-step command:
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=cuda --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=False \
  --out_dir=out_flywheel_fineweb_gate2000_baseline_muon_s2p0_seed1337 --wandb_log=False
```

Scheduled 2000-step command:
```bash
uv run --with-requirements requirements.txt python train.py config/train_fineweb_small.py \
  --dataset=fineweb --use_muon=True --device=cuda --dtype=float32 \
  --batch_size=4 --block_size=256 --gradient_accumulation_steps=1 \
  --n_layer=4 --n_head=4 --n_embd=256 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.0002 --muon_lr=0.02 --min_lr=0.00002 \
  --warmup_iters=100 --lr_decay_iters=5000 --seed=1337 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_law --power_p_low=0.0 --power_p_high=0.2 \
  --power_entropy_low=0.66 --power_entropy_high=0.74 \
  --power_entropy_law=power --power_entropy_gamma=2.0 \
  --power_entropy_linear_coeff=1.0 \
  --power_entropy_osc_amp=0.45 --power_entropy_osc_period=8 \
  --out_dir=out_flywheel_fineweb_gate2000_law_powerg2_osc_s2p0_seed1337 --wandb_log=False
```

Observed result:
- `500` steps: baseline `7.0907`, scheduled `7.0553`
- `2000` steps: baseline `6.5463`, scheduled `6.5893`

Conclusion:
The early scheduled gain is real but short-lived; by 2000 steps the baseline is better.
