# Argument B Poly Default With Exact Opt-In

Claim:
The practical default should be the polynomial backend, with exact SVD reserved for explicit controls and correctness checks.

Why:
- exact SVD is the clean reference but more expensive
- polynomial updates are the path used in the larger training sweeps
- the repo now defaults to `power_backend=poly` and `power_poly_degree=5`

Code paths to inspect:
- `train.py`
- `muon.py`
- `README.md`

Useful local checks:
```bash
sed -n '90,100p' train.py
sed -n '95,140p' muon.py
sed -n '25,40p' README.md
```

Conclusion:
This argument is a practical policy choice, not a claim that poly is always more accurate than exact.
