# ORCHESTRA-I24 implementation checklist

Phase-3 base: a378434438fd0abbec934be4c93ff2d0e8504af7, verified against origin. Phase-1/2 scientific findings and source/results are protected by outputs/i24/preservation.json. New work is separately namespaced.

- [COMPLETE] G0: inspect repository/resources; branch phase3-i24-world-model. CPU only; no competing research process observed; 29 GB available. Limit training to two CPU threads, zero loader workers, 2 GB working-set target, 2 GB scratch cap, 15-minute smoke / 60-minute pilot wall budget. No automatic GPU use.
- [BLOCKED] G1 real data: no mounted authentic trajectories; official data website returns network 403 and tutorial requires an account. Inspect official versioned documentation and implement only its explicit schema.
- [IN PROGRESS] G2 engineering: typed scenes, authentic adapter, causal resampling, Parquet partitions, QC, emulated sensing, buffered splits and target isolation. Real clip QC remains BLOCKED.
- [PENDING] G3: compact stochastic joint model, cheap baselines, contract tests and fixture-only complete pipeline.
- [BLOCKED] G4 real pilot: requires authorized I-24 files, acquisition metadata and map. No fixture result can satisfy this gate.
- [PENDING] G5 engineering: matched evaluation, proper scores, calibration, relational/memory/sensing controls and cluster intervals; real evidence BLOCKED.
- [PENDING] G6 engineering: provenance-linked fixture demo, model card, Cavnue schema interface and report. Real-data showpiece and scientific claims BLOCKED.
- [GATED] G7: research configuration, no automatic execution.

The primary real hypothesis and acceptance rules are recorded in I24_EVALUATION_PROTOCOL.md before any held-out model comparison. The Phase-2 next experiment is not being executed or reinterpreted: Phase 3 asks a separate passive real-trajectory prediction question. There is no causal action/control claim.
