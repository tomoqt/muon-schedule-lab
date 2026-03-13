# C4 Continuous Entropy-Law Schedules On Shakespeare-Char

Question:
Can a smooth entropy-to-`p` mapping do better than hard switching?

Primary artifacts:
- `assets/entropy_law_sweep.png`
- `assets/entropy_law_coarse_wide_sweep.png`
- `assets/repeat_seed_oscillation_comparison.png`

Shared setup:
- see `docs/flywheel/root_repro_index.md`

Representative command (`power` law with `gamma=2`):
```bash
uv run --with-requirements requirements.txt python train.py config/train_shakespeare_char.py \
  --dataset=shakespeare_char --use_muon=True --device=cpu --dtype=float32 \
  --batch_size=8 --block_size=64 --gradient_accumulation_steps=1 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.001 --muon_lr=0.01 --min_lr=1e-5 \
  --warmup_iters=100 --lr_decay_iters=5000 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_law --power_p_low=0.0 --power_p_high=0.12 \
  --power_entropy_low=0.68 --power_entropy_high=0.78 \
  --power_entropy_law=power --power_entropy_gamma=2.0 \
  --power_entropy_linear_coeff=1.0 \
  --out_dir=out_long2000_entropy_law_sweep_power_g2_c1_p0_p012_h68_78 --wandb_log=False
```

Observed result:
- representative final val: `1.9938`
- better than the hard-threshold branch but still worse than baseline `1.9162`
- repeat-seed checks keep the scheduled family below baseline

Conclusion:
Continuous laws are the strongest scheduled family on char, but they still fail the baseline comparison.
