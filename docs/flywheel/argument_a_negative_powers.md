# Argument A Negative Powers For Alpha Greater Than 0.5

Claim:
When `alpha > 0.5`, the equivalent exponent is `p = 1 - 2alpha < 0`, so the update applies a negative singular-value power.

Implication:
- smaller singular values are amplified rather than flattened
- exact SVD path needs an `svd_eps` floor to avoid division-like blowups on tiny singular values

Code paths to inspect:
- `muon.py`, exact branch inside `power_svd_update`
- root and A1 matrix artifacts

Useful local checks:
```bash
sed -n '95,140p' muon.py
```

Conclusion:
Negative powers are part of the intended search space, but they are also where numerical fragility is most obvious.
