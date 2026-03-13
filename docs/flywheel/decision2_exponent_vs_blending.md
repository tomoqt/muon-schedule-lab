# Decision 2 Interpolate In Exponent, Not In Raw Updates

Claim:
A convex blend between Muon and the raw update is not the same object as exponent interpolation.

Key algebra:
If `G = U S V^T`, then a tempered blend is
`tau U V^T + (1 - tau) G = U (tau I + (1 - tau) S) V^T`.
That is affine in `S`, not `S^p`.

Why this matters:
- exponent interpolation preserves the intended spectral family `U S^p V^T`
- raw blending changes the singular values along a different path

Evidence in the graph:
- root artifact `Tempered Muon tau sweep`
- root reproduction index
- downstream scheduled branches, which all use exponent control rather than blended updates

Useful local checks:
```bash
sed -n '1,90p' README.md
sed -n '1,170p' muon.py
```

Conclusion:
This decision defines the project properly. Tempered Muon is a comparison device, not the main interpolation rule.
