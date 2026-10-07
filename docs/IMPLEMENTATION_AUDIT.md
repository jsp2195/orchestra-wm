# Initial continuation audit

This records the state inspected before completing the MVP, not the final status.

| Original milestones | Initial status | Evidence / remaining work |
|---|---|---|
| 1–6 simulator and road families | COMPLETE (simplified) | Four topologies, lane-constrained motion, yielding and advisories |
| 7 simulator tests | PARTIAL | 8 simulator cases passed; identity, outage, jerk and oracle checks to add |
| 8 rendering | MISSING | No renderer or demo |
| 9–10 sensing and joint actions | COMPLETE | Probe telemetry, camera masks and lane loops present |
| 11 action audit | MISSING | No measured audit |
| 12–13 collection and dataset | PARTIAL | Collector present; no generated corpus or diagnostics |
| 14–18 baselines/model/memory/API | PARTIAL | Code and 4 tests present; masks need padded-scene coverage |
| 19–24 training and evaluations | MISSING | Trainer lacked validation logging; no executed training/evaluation |
| 25–27 oracle and learned planning | PARTIAL | CEM and separated scorer code present; no task validation |
| 28–32 rank, coordination, held-out | MISSING | No experiments |
| 33–38 figures/demo/docs/ledger | MISSING | Outputs and scripts mostly empty |
| 39–42 full tests, clean smoke, small, commit | MISSING | 12 initial tests passed; CPU-only machine, small unavailable |

The existing simulator and model were retained. Later integration fixes and final evidence are recorded in DECISIONS.md and outputs/EXPERIMENT_SUMMARY.md.
