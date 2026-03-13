# Decision 3 Keep Optimizer Invariants Fixed

Claim:
Schedule comparisons are only meaningful if the non-schedule parts of the optimizer contract stay fixed.

What was held fixed across the main branches:
- Muon on matrix parameters
- AdamW on non-matrix parameters
- schedule backend choice when a family is being compared
- model size, dataset family, and horizon inside each sweep
- matched LR whenever FineWeb results are compared directly

Where this is documented:
- `docs/flywheel/root_repro_index.md`
- branch reproduction notes
- `config/train_shakespeare_char.py`
- `config/train_fineweb_small.py`

Useful local checks:
```bash
sed -n '1,220p' config/train_shakespeare_char.py
sed -n '1,220p' config/train_fineweb_small.py
sed -n '1,120p' docs/flywheel/root_repro_index.md
```

Conclusion:
This decision is what makes the negative results credible rather than a moving-target comparison.
