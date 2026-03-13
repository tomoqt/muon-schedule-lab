# Argument E Long-Run Matched LR Overrules Short-Run Wins

Claim:
The correct final ranking should be based on the longest matched-LR comparison that the repo currently contains, not on the most optimistic short-horizon slice.

Evidence in the graph:
- `D4`: scheduled variant slightly ahead at `500` steps, behind at `2000`
- `Phase 5`: baseline ahead at every tested LR scale over `5000` steps

Artifacts to inspect:
- `D4 FineWeb gating rerun`
- `Phase 5 FineWeb 5000-step benchmark`
- `Phase 5 benchmark CSV`
- `Phase 5 text summary`

Conclusion:
Short-run gains are worth studying, but they do not presently overturn the long-run baseline result.
