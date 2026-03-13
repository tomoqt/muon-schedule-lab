# A1 Exact SVD Alpha Sweeps

Role:
This branch uses the exact SVD transformation as the clean control for alpha and `p` sweeps.

What is attached:
- `Exact alpha sweep`

Current reproducibility status:
- The exact-sweep figure is archived in the graph from the earlier notebook workflow.
- The current repo still contains the exact update path used for training experiments in `muon.py` via `power_backend=exact`.
- The current repo does not contain the original matrix-plotting script that produced this exact figure.

Code paths to inspect:
- `muon.py`, exact branch inside `power_svd_update`
- `README.md`, exact-backend discussion

Useful local checks:
```bash
sed -n '95,140p' muon.py
sed -n '220,255p' README.md
```

Conclusion:
Use this branch as the exact-control reference for the matrix-level story, with the caveat that the figure itself is archival.
