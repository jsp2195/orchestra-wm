# ORCHESTRA-I24 implementation checklist

Phase-3 base: a378434438fd0abbec934be4c93ff2d0e8504af7, verified against origin. Phase-1/2 scientific findings and source/results are protected by outputs/i24/preservation.json. New work is separately namespaced.

- [COMPLETE] G0: inspect repository/resources; branch phase3-i24-world-model. CPU only; no competing research process observed; 29 GB available. Limit training to two CPU threads, zero loader workers, 2 GB working-set target, 2 GB scratch cap, 15-minute smoke / 60-minute pilot wall budget. No automatic GPU use.
- [BLOCKED] G1 real data: no mounted authentic trajectories; official data website returns network 403 and tutorial requires an account. Inspect official versioned documentation and implement only its explicit schema.
- [COMPLETE ENGINEERING / BLOCKED REAL QC] G2 engineering: typed scenes, authentic adapter, causal resampling, Parquet partitions, QC, emulated sensing, buffered splits and target isolation. Real clip QC remains BLOCKED.
- [COMPLETE] G3: compact stochastic joint model, cheap baselines, contract tests and fixture-only complete pipeline.
- [BLOCKED] G4 real pilot: requires authorized I-24 files, acquisition metadata and map. No fixture result can satisfy this gate.
- [COMPLETE ENGINEERING / BLOCKED REAL EVIDENCE] G5 engineering: matched evaluation, proper scores, calibration, relational/memory/sensing controls and cluster intervals; real evidence BLOCKED.
- [COMPLETE FIXTURE / BLOCKED REAL SHOWCASE] G6 engineering: provenance-linked fixture demo, model card, Cavnue schema interface and report. Real-data showpiece and scientific claims BLOCKED.
- [GATED] G7: research configuration, no automatic execution.

The primary real hypothesis and acceptance rules are recorded in I24_EVALUATION_PROTOCOL.md before any held-out model comparison. The Phase-2 next experiment is not being executed or reinterpreted: Phase 3 asks a separate passive real-trajectory prediction question. There is no causal action/control claim.

## Fixture integration findings (not real test outcomes)

The first full engineering run passed 46 tests and completed in 99 s. Browser payload compression was added without changing numerical arrays. A real-loader review found that region bounds had to be applied before the agent-count cap; this was corrected before any real source execution. The first fixture blind zone missed the displayed vehicles, so its declared static footprint now covers 15–35% of the corridor to exercise missingness. This changes fixture engineering coverage, not a scientific acceptance criterion. Final smoke is rerun after these fixes; original Phase-1/2 artifacts remain untouched.

## Final engineering status

- G0 COMPLETE: current branch is isolated from verified Phase-2 base; 268 protected tracked files/prefixes are checked.
- G1 documentation/schema COMPLETE; authentic source access BLOCKED (0 files, 0 bytes).
- G2 fixture/canonical/causal-mask/split engineering COMPLETE; real-source geometry, sampling, regime and wave QC BLOCKED.
- G3 COMPLETE: six from-scratch fixture models, seeded autonomous rollouts, baselines, artifact validation and browser demo. A full final checkpoint rebuild takes about 100 s on this CPU; restart validation reuses matching checkpoints. Exact timing is recorded in outputs/i24/execution_history.json.
- G4 BLOCKED: no real pilot checkpoint or real autonomous generation.
- G5 engineering COMPLETE: proper scores, cluster intervals, kinematics, optional training-only calibrated IDM, graph/memory/sensing controls and matched-history diagnostics. Real information/architecture claims BLOCKED.
- G6 fixture showcase / model card / acquisition and Cavnue interface documentation COMPLETE. Real-data showpiece and real-Cavnue validation BLOCKED.
- G7 configured and GATED; not run.

No scientific threshold or primary metric was changed. The fixture forecasts have appreciable drift, poor interval calibration and road/overlap violations. They establish functioning learned stochastic generation and measurable diagnostics, not realistic real-world skill. Source-schema tests use explicitly fabricated unit examples; they do not count as authentic dataset access.
