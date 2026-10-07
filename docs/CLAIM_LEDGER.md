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
