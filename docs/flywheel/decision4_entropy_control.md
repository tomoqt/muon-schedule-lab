# Decision 4 Entropy Control Is Plausible But Fragile

Role:
This node records the design choice to keep exploring entropy-driven schedules even after the first threshold-style versions looked unstable.

What is attached:
- `Decision-4 tau-annealed behavior`

Evidence chain:
- `C2` shows that naive alternation is weak
- `C3` shows threshold tuning helps but remains below baseline
- `C4` shows smooth entropy laws are the strongest scheduled family on char

Useful local checks:
```bash
sed -n '1,220p' power_schedule.py
sed -n '190,225p' README.md
```

Conclusion:
Entropy is a sensible control signal, but the evidence says it needs careful tuning and still has not produced a baseline win here.
