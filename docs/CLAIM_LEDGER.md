# Claim ledger

Generated from executed results. Scientific PASS at smoke scale maps to PARTIALLY SUPPORTED, not conclusive support. All learned models have training seed 17; listed seeds are held-out simulator seeds.

| Claim | Metric / actual result | Config | Evaluation seeds | Status |
| --- | --- | --- | --- | --- |
| model predicts multi-agent evolution | {"orchestra_position_error_m_h10": 7.168714125951131, "constant_velocity_m": 8.947917381922403, "independent_m": 7.354815244674683} | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| autonomous imagination works | {"max_horizon": 40, "parameters": 1012878} | smoke | [17, 29, 43] | SUPPORTED |
| correct actions beat shuffled actions | {"shuffled_minus_correct_m_h10": 1.2443589369455976, "relative_degradation": 0.1735818886180649} | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| joint interactions are learned | {"orchestra_position_error_m": 10.886176546414694, "independent_position_error_m": 9.06163239479065, "orchestra_joint_contrast_error": 1.5663599594434103, "independent_joint_contrast_error": 1.518763653933999} | smoke | [17, 29, 43] | NOT SUPPORTED |
| recurrent memory helps under occlusion | {"intact_reappearance_m": 8.428774197896322, "reset_reappearance_m": 13.135957400004068, "memoryless_reappearance_m": 11.263243993123373} | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| counterfactual outcome structure is meaningful | {"pairwise_outcome_pearson": 0.6154813266236977, "pairwise_outcome_spearman": 0.6172241664687907} | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| plan rankings correlate with simulator truth | {"plan_spearman": 0.5960446560020847, "plan_pearson": 0.5348565070622769, "top_5_overlap": 0.5, "top_10_overlap": 0.5833333333333334, "regret": 0.9113166485975187, "selected_true_rank": 6.166666666666667} | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| centralized orchestration beats independent control | {"orchestra_objective": 4.875889221404989, "local_reactive_objective": 6.238364614049594, "independent_learned_mpc_objective": 4.875889221404989} | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| held-out generalization works | {"ID_position_m": 4.420005877812703, "density_position_m": 4.997404257456462, "behavior_position_m": 4.999167998631795, "topology_position_m": 3.419279098510742, "controlled_regimes_with_positive_rank_and_planning_gain": 1} | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| learned planning beats random | reference minus learned objective = 5.4547 | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| learned planning beats noop | reference minus learned objective = 6.0683 | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| learned planning beats heuristic | reference minus learned objective = 8.2088 | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| learned planning approaches oracle | reference minus learned objective = -0.4326 | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |
| model generalizes to unseen layouts | topology OOD rank=0.7422 | smoke | [17, 29, 43] | PARTIALLY SUPPORTED |

<!-- PHASE2_RESULTS -->
# Phase 2 — measured claims

Phase-1 entries above are preserved. C is primary. Counterfactual correlation and pairing degradation can occur without non-additive reasoning; passing those directional gates supports narrower claims only.

| Claim | Metric/gate | Actual result | Config/seeds | Status |
|---|---|---|---|---|
| joint interactions learned | INT-C | {"ci_high": 14.069780035204575, "ci_low": -6.801566742689738, "mean": 3.6341066462574187, "n_seeds": 3, "pass": false, "sd": 4.200923861662304, "se": 2.4254045223758536} | phase2.yaml / 101,202,303 | NOT SUPPORTED |
| factorial counterfactuals learned | INT-D | {"objective_pearson": 0.573818161385601, "outcome_distance_pearson": 0.9894158588023917, "pass": true} | phase2.yaml / 101,202,303 | PARTIALLY SUPPORTED |
| joint pairing matters | INT-E | {"ci_high": 1.2835438915224346, "ci_low": 1.087524938967835, "mean": 1.1855344152451348, "n_seeds": 3, "pass": true, "sd": 0.03945412358328637, "se": 0.02277884887145115} | phase2.yaml / 101,202,303 | PARTIALLY SUPPORTED |
| plan ranking works in interaction-heavy states | INT-F | {"pass": false, "regret": 1.4804249196965251, "selected_percentile": 18.01470588235294, "spearman": 0.4821551427588579, "top10_overlap": 0.625} | phase2.yaml / 101,202,303 | NOT SUPPORTED |
| planner avoids policy collapse | INT-G | {"command_frequencies": {"-1": 0.0, "0": 0.006944444444444444, "1": 0.9930555555555556}, "joint_strategies": 4, "pass": false} | phase2.yaml / 101,202,303 | NOT SUPPORTED |
| centralized coordination beats independent learned MPC | INT-H | {"bootstrap_high": 0.0, "bootstrap_low": -0.5276421015057714, "bootstrap_replicates": 5000, "ci_high": 0.4363647136906334, "ci_low": -0.7006157563509592, "mean": -0.13212552133016292, "model": "C_difference", "n_seeds": 3, "pass": false, "sd": 0.22884811592036763, "se": 0.13212552133016295, "task": "all"} | phase2.yaml / 101,202,303 | NOT SUPPORTED |

ORCHESTRA remains a functioning multi-agent world model, but the experiment did not establish a measurable centralized coordination advantage over an independent learned-controller baseline.

Generalization: UNTESTED unless INT-H passes. See [Phase-2 experiment report](../outputs/phase2/EXPERIMENT_SUMMARY.md) for all measurements.
