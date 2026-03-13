# Phase 4 FineWeb Small Transfer

Role:
This phase tests whether the char-side schedule ideas transfer to a more realistic language-model setting once learning rates are matched.

Child branches:
- `D1`: fixed alpha sanity check
- `D2`: corrected polynomial LR sweep
- `D3`: entropy-law family sweep
- `D4`: 500/2000-step gate on Flywheel compute

How to reproduce this phase:
- shared setup is in `docs/flywheel/root_repro_index.md`
- each branch note carries the representative `uv run` commands
- the plotted summaries are rebuilt by `scripts/build_flywheel_artifacts.py`

Phase-level conclusion:
The FineWeb signal is weaker than the optimistic early char intuition. There is a small short-horizon edge for some scheduled variants, but it is not yet stable enough to count as a decisive improvement.
