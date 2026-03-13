# A2 Polynomial Alpha Sweeps

Role:
This branch checks whether a polynomial surrogate can follow the exact alpha sweep closely enough to be useful.

What is attached:
- `Polynomial degree-3 alpha sweep`

Current reproducibility status:
- The current repo implements polynomial scheduled updates in `muon.py`.
- The attached degree-3 matrix sweep is archival from the earlier notebook workflow.
- The current default training backend has moved to quintic (`power_poly_degree=5`), so this branch is best read as the historical cubic control.

Code paths to inspect:
- `muon.py`, `_build_poly_coeffs`, `_poly_matrix`, `_power_poly_update`
- `README.md`, poly-backend discussion

Useful local checks:
```bash
sed -n '40,120p' muon.py
sed -n '235,255p' README.md
```

Conclusion:
This branch motivates the later switch from cubic to quintic polynomial settings.
