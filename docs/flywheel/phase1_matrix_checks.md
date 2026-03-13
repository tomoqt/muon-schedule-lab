# Phase 1 Matrix Checks And Alpha-Range Extension

Role:
This phase stress-tests the scheduled-power idea before full language-model training.

What this phase covers:
- `Branch A1`: exact SVD alpha sweeps
- `Branch A2`: polynomial alpha sweeps
- `Argument A`: negative-power regime for `p < 0`
- `Phase 2`: optimizer implementation that follows from these checks

Current reproducibility status:
- The archived matrix figures are present in Flywheel.
- The original notebook plotting script is not present in this repo.
- The exact and polynomial update paths used later in training are present in `muon.py` and can be inspected directly.

Code paths to inspect:
- `muon.py`
- `README.md`

Useful local checks:
```bash
sed -n '1,170p' muon.py
sed -n '150,260p' README.md
```

Conclusion:
This phase establishes the algebraic control range and the numerical issues that later shaped the optimizer defaults.
