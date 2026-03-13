# Scheduled Muon Graph Repro Index

Repo: https://github.com/tomoqt/muon-schedule-lab
Reference experiment-code commit: `3d39205`

Minimal local setup:
```bash
git clone https://github.com/tomoqt/muon-schedule-lab
cd muon-schedule-lab
uv run --with-requirements requirements.txt python -c "import torch; print(torch.__version__)"
```

This graph has two layers of evidence:
1. matrix-level plots in the root and phase-1/phase-2 nodes
2. language-model training branches on `shakespeare_char` and `fineweb`

Matrix-level provenance note:
- the earliest matrix plots came from the original `scheduledmuon.py` notebook export
- that file is not checked into this repo anymore
- those root matrix figures should therefore be read as archived evidence
- the current repo does preserve the algebra, optimizer implementation, schedule code, and all training-side experiments

Rebuild the repo-level summary figures:
```bash
uv run --with-requirements requirements.txt python scripts/build_flywheel_artifacts.py
```

Important bookkeeping note:
Historical checkpoints in this repo were produced before the `train.py` fix that stopped
`best_val_loss` from being overwritten on every save when `always_save_checkpoint=True`.
For those older runs, checkpoint `best_val_loss` should be read as the final validation loss
at the run horizon. The long-5000 CSV in `outputs/flywheel_long5000_lr_micro_20260308/`
contains both `best_val` and `final_val` explicitly.

Shared Shakespeare-char training family:
- config: `config/train_shakespeare_char.py`
- device: `cpu`
- dtype: `float32`
- batch_size: `8`
- block_size: `64`
- grad accumulation: `1`
- model: `n_layer=2`, `n_head=2`, `n_embd=64`
- optimizer: `use_muon=True`, `learning_rate=1e-3`, `muon_lr=1e-2`, `min_lr=1e-5`
- schedule horizon: `max_iters=2000`, `eval_interval=100`, `eval_iters=20`, `warmup_iters=100`, `lr_decay_iters=5000`

Shared FineWeb v2 training family:
- config: `config/train_fineweb_small.py`
- dtype: `float32`
- batch_size: `4`
- block_size: `256`
- grad accumulation: `1`
- model: `n_layer=4`, `n_head=4`, `n_embd=256`
- optimizer: `use_muon=True`, `learning_rate=2e-4`, `muon_lr=2e-2`, `min_lr=2e-5`
- schedule backend: `power_backend=poly`, `power_poly_degree=5`
- eval: `eval_interval=100`, `eval_iters=20`, `warmup_iters=100`, `lr_decay_iters=5000`
- corrected LR sweep seeds: `1337`, `1338`, `1339`

Branch mapping:
- `C1`: fixed-alpha sweeps on char
- `C2`: explicit alternating schedules on char
- `C3`: entropy-threshold switching on char
- `C4`: continuous entropy-law schedules on char
- `D1`: FineWeb baseline vs fixed-alpha sanity check
- `D2`: FineWeb LR sweeps with polynomial backend
- `D3`: FineWeb entropy-law coefficient and oscillation sweeps
- `D4`: FineWeb 500/2000-step gating reruns on Flywheel A10G
- `Phase 5`: FineWeb 5000-step matched-LR benchmark

Read each branch note together with its attached figure(s):
- the markdown note gives the question, the exact run family, and representative rerun commands
- the image/CSV artifacts show the observed values used in the branch conclusion
