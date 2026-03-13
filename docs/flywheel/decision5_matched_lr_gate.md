# Decision 5 Require Matched LR Adjudication

Claim:
Any scheduled-vs-baseline claim has to survive matched-learning-rate comparison. Otherwise the result may just be optimizer retuning disguised as a new method.

Evidence in the graph:
- `D1` and `D2` keep baseline and scheduled runs on the same LR scales
- `D4` holds LR scale fixed at `2.0` while extending horizon
- `Phase 5` compares the best scheduled candidate against baseline across the same long-run LR grid

Useful local checks:
```bash
sed -n '1,220p' docs/flywheel/d1_fineweb_baseline_fixed.md
sed -n '1,220p' docs/flywheel/d2_fineweb_poly_lr.md
sed -n '1,220p' docs/flywheel/phase5_fineweb_long5000.md
```

Conclusion:
This node is the methodological gate that prevents overclaiming from short-run or unmatched-LR wins.
