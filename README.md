# Muon Schedule Lab

A clean experimental repo combining:

- [`Muongpt`](https://github.com/tomoqt/Muongpt) as the GPT training base
- scheduled singular-value power transforms from the `scheduled-muon` experiments

The core idea is to control update geometry with a singular-value exponent `p`.
Lower `p` is treated as more exploratory behavior. Higher `p` is treated as more exploitative behavior.

## Math in one place
Let a matrix update be `G = U S V^T`.

- Standard SGD keeps `p=1`: `G = U S^1 V^T`.
- Muon-like zeroth-power behavior corresponds to `p=0`: `U S^0 V^T`.
- General family: `G_p = U S^p V^T`.

Equivalent alpha form:

`G_alpha = G (G^T G)^(-alpha) = U S^(1-2alpha) V^T`, so `p = 1 - 2alpha`.

## What is implemented
Muon parameter groups can now run either:

- Newton-Schulz Muon update (original behavior), or
- SVD-power update with scheduled exponent `p`.

Supported schedules:

- `anneal`: linear annealing from `p_start` to `p_end`
- `anneal_cosine`: cosine annealing from `p_start` to `p_end`
- `fixed_alternating`: alternate between `p_low` and `p_high` every `power_alternation_period` steps
- `entropy_alternating`: switch between `p_low` and `p_high` using SVD-entropy hysteresis thresholds

Entropy-alternating uses gradient-matrix SVD entropy in `[0,1]`:

- switch to `p_high` when entropy rises above `power_entropy_high`
- switch to `p_low` when entropy falls below `power_entropy_low`

## Key files
- `train.py`: training loop and schedule integration
- `muon.py`: optimizer internals with optional power-SVD update
- `power_schedule.py`: schedule classes + entropy utilities
- `scripts/schedule_smoke.py`: small local smoke test on random data

## Install
```bash
pip install -r requirements.txt
```

## Local smoke test (small)
This verifies all three schedule classes and optimizer integration without dataset prep:

```bash
python scripts/schedule_smoke.py
```

## Dataset prep (for real training)
Example small dataset:

```bash
python data/shakespeare_char/prepare.py
```

## Run training with schedules
Single-process example with Muon + annealing schedule:

```bash
python train.py \
  --dataset=shakespeare_char \
  --use_muon=True \
  --enable_power_schedules=True \
  --power_schedule_type=anneal \
  --power_p_start=1.0 \
  --power_p_end=0.0 \
  --max_iters=200 \
  --batch_size=16 \
  --block_size=128 \
  --compile=False
```

Fixed alternating example:

```bash
python train.py \
  --dataset=shakespeare_char \
  --use_muon=True \
  --enable_power_schedules=True \
  --power_schedule_type=fixed_alternating \
  --power_p_low=0.0 \
  --power_p_high=1.0 \
  --power_alternation_period=50 \
  --max_iters=200 \
  --batch_size=16 \
  --block_size=128 \
  --compile=False
```

Entropy alternating example:

```bash
python train.py \
  --dataset=shakespeare_char \
  --use_muon=True \
  --enable_power_schedules=True \
  --power_schedule_type=entropy_alternating \
  --power_p_low=0.0 \
  --power_p_high=1.0 \
  --power_entropy_low=0.45 \
  --power_entropy_high=0.65 \
  --max_iters=200 \
  --batch_size=16 \
  --block_size=128 \
  --compile=False
```

## Notes
- SVD-based updates are significantly heavier than Newton-Schulz Muon.
- Entropy-based switching is intentionally simple and intended as an experimental baseline.
- This repo is meant to make schedule ablations easy to run and compare.
