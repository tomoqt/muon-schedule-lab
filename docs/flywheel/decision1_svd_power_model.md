# Decision 1 SVD Power Model

Claim:
Model Muon and its generalization in singular-value coordinates. If `G = U S V^T`, define the scheduled family as `G_p = U S^p V^T`.

Equivalent identity:
`G (G^T G)^(-alpha) = U S^(1-2alpha) V^T`, so `p = 1 - 2alpha`.

Why this matters:
- `p = 1` gives the unmodified update
- `p = 0` gives Muon-like zeroth-power behavior
- `p < 0` amplifies smaller singular directions
- `p > 1` amplifies larger singular directions

Code paths to inspect:
- `README.md` math section
- `muon.py`, function `power_svd_update`

Useful local checks:
```bash
sed -n '1,90p' README.md
sed -n '60,150p' muon.py
```

Evidence in the graph:
- root matrix artifacts
- `Branch A1`, `Branch A2`, and `Branch B1`

Conclusion:
This is the mathematical definition that the rest of the graph tests.
