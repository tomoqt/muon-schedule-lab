# Phase 3 Shakespeare-Char Discovery

Role:
This phase asks whether the scheduled-power idea survives first contact with real language-model training on a small, cheap benchmark.

Child branches:
- `C1`: fixed alpha
- `C2`: explicit alternation
- `C3`: entropy-threshold switching
- `C4`: continuous entropy laws

How to reproduce this phase:
- use `docs/flywheel/root_repro_index.md` for the shared char config
- use each branch note for the exact representative commands
- rebuild the summary figures with:
```bash
uv run --with-requirements requirements.txt python scripts/build_flywheel_artifacts.py
```

Phase-level conclusion:
All scheduled families lose to baseline Muon on the char benchmark. `C4` is the best scheduled family, but it still does not overturn the baseline comparison.
