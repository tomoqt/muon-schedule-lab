# C3 Entropy-Threshold Schedules On Shakespeare-Char

Question:
Can switching `p` only when SVD entropy crosses tuned low/high thresholds outperform fixed-alpha or baseline Muon?

Primary artifacts:
- `assets/entropy_hyst_config_sweep.png`
- `assets/entropy_hyst_followup_p_high_sweep.png`

Shared setup:
- see `docs/flywheel/root_repro_index.md`

Representative follow-up command (best threshold configuration found in this branch):
```bash
uv run --with-requirements requirements.txt python train.py config/train_shakespeare_char.py \
  --dataset=shakespeare_char --use_muon=True --device=cpu --dtype=float32 \
  --batch_size=8 --block_size=64 --gradient_accumulation_steps=1 \
  --n_layer=2 --n_head=2 --n_embd=64 \
  --max_iters=2000 --eval_interval=100 --eval_iters=20 \
  --learning_rate=0.001 --muon_lr=0.01 --min_lr=1e-5 \
  --warmup_iters=100 --lr_decay_iters=5000 \
  --enable_power_schedules=True --power_backend=poly --power_poly_degree=5 \
  --power_schedule_type=entropy_alternating --power_p_low=0.0 --power_p_high=0.04 \
  --power_entropy_low=0.695 --power_entropy_high=0.715 --power_entropy_initial_mode=low \
  --out_dir=out_long2000_entropy_hyst_followup_p0_p0p04_h695_715 --wandb_log=False
```

Observed result:
- best follow-up configuration final val: `2.0024`
- still worse than baseline Muon `1.9162`
- larger `p_high` values degrade quickly

Conclusion:
Threshold tuning makes the branch less bad, but it does not produce a competitive winner.
