# ORCHESTRA-I24 model card

## Intended scope and current evidence

Passive multi-agent highway forecasting from reconstructed roadside tracks. **No authentic I-24 training has occurred.** The only executed checkpoint campaign is `i24_smoke.yaml`, `SYNTHETIC_FIXTURE`, seed 101. The model is not validated for real vehicles, infrastructure, causal action responses or Cavnue deployment. Phase-1/2 weights are not reused. All real scientific claims are BLOCKED pending authorized data and pilot QC.

## Architecture

Compact graph variational recurrent state-space model (`orchestra_wm/i24/model.py`): per-agent kinematic tokens and GRU memory, geometry-relative neighbor messages, lane/bin field tokens and GRU memory, scene-conditioned Gaussian innovation prior and training-only posterior, continuous acceleration integration and probabilistic macro decoder. Neighbor edges encode relative s/d/velocity, lane proximity and leader direction. Fields are speed (m/s), density (veh/km/lane), flow (veh/hour/lane). The present field approximation is density×mean longitudinal speed, not an independently measured detector count. Temporal-bin definitions and empty-bin support are recorded.

The global latent innovation is shared across agents in the full model; decoder residual variance is learned. Both affect training likelihoods and sampled autonomous rollout losses. This is not a deterministic model with noise added only at inference. A diagonal-Gaussian conditional variational objective with KL regularization is the first implementation, chosen for bounded CPU cost and transparent posterior/prior separation. No diffusion/video model is claimed.

Autonomous prior transitions refresh agent and field features from their own generated outputs. Training adds posterior reconstruction terms using targets, but the recurrent state for the next training step is the prior-generated state. ELBO components plus scaled multistep micro/macro and consistency losses are recorded. Fixed physical normalizers are preregistered; no validation/test normalization fitting. No teacher-forced future state initializes imagination.

API:

```python
state = model.initial_state({'road': road_map, 'track_ids': past_track_ids})
state = model.observe(state, observation_dict, visible_mask, dt)
next_state, heads = model.imagine_step(state, known_future_context=None)
samples = model.imagine(state, horizon_steps=100, num_samples=8, rng_seed=101)
```

`known_future_context` accepts only the same typed fixed map. Arbitrary future arrays are rejected. `condition_on_actions` always raises for passive data. Targets do not appear in `Belief`, `Observation` or `imagine`. IDs in the fixed-cohort roster come exclusively from pre-cutoff detections; unseen future entrants cannot reveal future counts. Padding masks are explicit in the collator; sequential scene batches avoid padding during smoke training. Temporary hidden entities retain memory; predicted exits are based on generated boundaries. Births can enter the separate open-road generator under a declared Poisson past-appearance-rate assumption. That assumption can confuse reappearance with true inflow and is not a learned or validated arrival process.

## Controls

- Independent: no cross-agent messages, pooled scene noise or field feedback in agent evolution; own-agent recurrent dynamics plus independent stochastic innovations.
- Deterministic relational: same interaction pathway with stochastic draws disabled.
- Macro-only: only historical field representation drives field forecasts; placeholder micro motion is not scored as a learned agent forecast.
- Micro-only: removes field-to-agent feedback; reports macro fields derived from generated agents.
- No-cross-scale: removes micro/field feedback and consistency loss; field innovation is independent of agent innovation.
- Full-model inference controls: graph messages removed, history memory reset, residual histories permuted between spatial slots, and emulated sensing masks. No retraining for these controls.

The architecture comparison is approximate matched parameter/update capacity; unused heads make equal total parameter counts an imperfect effective-capacity control. Report counts from training_metadata.json. One scene per optimizer update conserves memory; `sampled_scene_passes` is not an epoch count. Smoke trains 100 updates with 2-second autonomous training unroll, evaluates 5/10/20-second horizons, and does not demonstrate convergence. Pilot uses a 10-second training unroll and 1000 updates, subject to data/resource gates.

## Evaluation and limitations

Read `I24_EVALUATION_PROTOCOL.md` for the frozen primary proper-score comparison and cluster intervals. All current numeric outputs, graph/memory/sensing effects and calibration estimates are **fixture-only integration results**. Longer 20-second forecasts extrapolate beyond smoke's training horizon; report drift and physics violations. There is no output clipping that hides negative macro values, overlaps or lateral departures. Smooth field aggregation differs from hard-bin target extraction; consistency error is measured rather than claimed exact.

v1.0 source lanes are approximate and offline reconstruction can use future video. Masks are emulated, not measured camera outages. No verified wave-onset events, real IDM calibration, cross-day generalization or partner transfer have been executed. A synthetic oscillating-speed fixture does not establish real traffic-wave skill. The model has no causal control semantics.

## Related work and novelty boundary

- Gloudemans et al. (2023), *I-24 MOTION: An instrument for freeway traffic science*, TR-C 155, 104311: instrument and anonymized reconstructed trajectory data, not this forecast model.
- Wang et al., *Automatic vehicle trajectory data reconstruction at scale*, arXiv:2212.07907; official postprocessing-lite: offline merging/stitching/rectification, not autonomous future generation.
- Ji et al. (2024), *Virtual Trajectories for I-24 MOTION: Data and Tools*, FISTS: aggregate speed fields and virtual trajectories. This proposal instead learns a joint conditional future distribution; no demonstrated advantage is claimed.
- Ji et al. (2026), *Scalable analysis of stop-and-go waves: Representation, measurements and insights*, TR-C 182, 105385 (citation from official wave-analysis repository): wave identification/measurement is established prior work, not a new ORCHESTRA finding.
- Social-LSTM (Alahi et al., CVPR 2016), Trajectron++ (Salzmann et al., ECCV 2020), TrafficSim (Suo et al., CVPR 2021): interaction-aware stochastic trajectory prediction/learned multi-agent simulation predates this code. Graph trajectory forecasting and shared latent innovations are not claimed novel components.

The untested contribution is the controlled information-value experiment: whether micro headway/relative motion improves calibrated future macro distributions given similar macro histories, separately from architectural advantages. No paper-level novelty or breakthrough has been established. Traffic breakdown precursors and wave propagation are known phenomena; this campaign must quantify additional out-of-sample predictive information rather than rename them.

## Practical portability limits

The authentic adapter and calibrated-IDM code are covered by documented-schema unit fixtures, not executed real-source validation. Causal resampling and buffered split logic must be inspected on real clips before a scientific pilot is accepted. Canonical partitions and optional input metadata cannot establish actual track identity quality or correct geographic calibration by themselves. The browser bundle embeds gzip-compressed numerical evaluation arrays and requires a modern browser with DecompressionStream; serve it locally if your browser policy blocks `file://` navigation. A compact GIF is an alternate export, not an independent simulation.
