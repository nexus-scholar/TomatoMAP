# Risks and Decisions

## Main risks

### Risk 1: Missing labels never arrive
Mitigation:
- keep Paper 1 centered on semi-supervised learning.

### Risk 2: Too many experiments for available compute
Mitigation:
- only one main model family,
- narrow ablations,
- no large sweeps. [web:77][web:79]

### Risk 3: Topic drift
Mitigation:
- freeze paper scope by Week 4.

### Risk 4: Weak novelty claim
Mitigation:
- emphasize label scarcity, practical learning, and TomatoMAP-specific setting rather than architecture novelty. [web:39][web:40][web:43]

### Risk 5: Writing starts too late
Mitigation:
- write from Week 1.

## Decision rules

### If the 2k+ labels arrive early
Use them for:
- one upper-bound supervised comparison,
- stronger discussion,
- maybe a follow-up paper plan.

Do NOT redesign Paper 1 completely.

### If the 2k+ labels do not arrive
Continue exactly as planned.

### If compute becomes tighter than expected
Drop:
- second model,
- minor ablations,
- optional robustness runs.

Keep:
- supervised baseline,
- semi-supervised main run,
- one label-fraction ablation.

## Non-negotiables

- First paper must be submission-ready in 12 weeks.
- The first paper must not depend on external uncertainty.
- The project must remain thesis-coherent.

