# Phase 0 Notebook Audit And Baseline Reconstruction

Role:
This node anchors the project history. It records where the root matrix-level artifacts came from and what is still reproducible in the current repo.

What to read in the graph:
- root node artifacts: `Compare at alpha=0.5`, `Exact alpha sweep`, `Polynomial degree-3 alpha sweep`, `Polynomial degree-5 alpha sweep`, `Tempered Muon tau sweep`
- child nodes: `Decision 1` and `Phase 1`

Current reproducibility status:
- The original notebook export `scheduledmuon.py` is not checked into this repo.
- Because of that, the exact script that generated the earliest matrix-level plots is archival rather than fully replayable from `main`.
- What is reproducible today is the algebra, the optimizer implementation, and the downstream language-model experiments that those plots motivated.

Code paths to inspect:
- `README.md`
- `muon.py`
- `power_schedule.py`

Useful local checks:
```bash
sed -n '1,140p' README.md
sed -n '1,170p' muon.py
sed -n '1,220p' power_schedule.py
```

Conclusion:
Treat the root matrix figures as archived evidence and the training code plus branch notes as the current executable implementation.
