# B1 Quintic Polynomial Backend Checks

Role:
This branch tests whether moving from the historical cubic approximation to a quintic polynomial gives a better practical default.

What is attached:
- `B1 quintic sweep`

Code paths to inspect:
- `muon.py`, polynomial coefficient fit and matrix polynomial evaluation
- `train.py`, default `power_poly_degree=5`

Useful local checks:
```bash
sed -n '40,120p' muon.py
sed -n '90,100p' train.py
```

Current reproducibility status:
- The attached quintic sweep is archived from the earlier matrix workflow.
- The current repo does implement quintic polynomial updates directly and uses them as the training default.

Conclusion:
This branch justifies the repo default: polynomial scheduled runs should be quintic unless you explicitly choose otherwise.
