# Phase 2 Optimizer Implementation

Role:
This phase turns the algebra into code that can be used inside training runs.

What this phase establishes:
- scheduled power updates live inside `muon.py`
- schedule selection and entropy control live in `power_schedule.py`
- `train.py` wires the schedule state into Muon parameter groups
- polynomial backend is the default, exact backend is explicit opt-in

What to read in the graph:
- attached artifact `Phase-2 polynomial vs Muon comparison`
- child nodes `Decision 3`, `Branch B1`, and `Argument B`

Useful local checks:
```bash
uv run --with-requirements requirements.txt python scripts/schedule_smoke.py
sed -n '70,110p' train.py
sed -n '1,170p' muon.py
sed -n '1,220p' power_schedule.py
```

Conclusion:
This is the implementation boundary between the matrix-level argument and the training experiments.
