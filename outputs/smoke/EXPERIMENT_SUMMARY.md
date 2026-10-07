# ORCHESTRA-WM experiment summary

Synthetic research only. Smoke results are integration-scale evidence.

## IMPLEMENTED

Four road families; structured sensing; 1M-parameter recurrent transformer; five trained variants; autonomous prediction, action shuffle, interactions, memory, counterfactuals, CEM/MPC, OOD and scaling; plots; HTML and GIF.

## ACTUALLY EXECUTED

Config `smoke`; train seed 17; evaluation seeds [17, 29, 43]; 64 episodes/3072 transitions; 192 updates per model; device cpu.

Python 3.12.14; PyTorch 2.14.1+cpu; hardware Linux-6.18.44-x86_64-with-glibc2.41; complete pipeline time 294.3 s.

| Area | Result | Measured metrics | Interpretation |
| --- | --- | --- | --- |
| SIMULATOR | PASS | tests=23 passed in 6.45s; families=4.0000 | Deterministic synthetic dynamics, sensing, action masks and collision checks pass. |
| DATA / ACTION-EXCITATION | PASS | position_divergence_m_at_10=5.6780; transitions=3072.0000; maintain_fraction=0.1304 | Paired simulator branches respond materially to joint actions. |
| WORLD MODEL | PASS | orchestra_position_error_m_h10=7.1687; constant_velocity_m=8.9479; independent_m=7.3548 | Ten-step errors are compared against the same held-out simulator trajectories; training is short and convergence incomplete. |
| AUTONOMOUS IMAGINATION | PASS | max_horizon=40.0000; parameters=1012878.0000 | Autonomous rollouts and tests excluding future observations completed. |
| ACTION SENSITIVITY | PASS | shuffled_minus_correct_m_h10=1.2444; relative_degradation=0.1736 | Correct actions improve forecasts at the primary horizon. |
| MULTI-AGENT INTERACTION | FAIL | orchestra_position_error_m=10.8862; independent_position_error_m=9.0616; orchestra_joint_contrast_error=1.5664; independent_joint_contrast_error=1.5188 | Both trajectory error and the nonlinear joint-cost contrast are measured; a contrast improvement alone does not establish joint-model superiority. |
| MEMORY | PASS | intact_reappearance_m=8.4288; reset_reappearance_m=13.1360; memoryless_reappearance_m=11.2632 | Forecasts are scored before a forcibly hidden background vehicle reappears. |
| COUNTERFACTUAL | PASS | pairwise_outcome_pearson=0.6155; pairwise_outcome_spearman=0.6172 | Pairwise geometry of predicted joint outcomes is compared with simulator branches. |
| DECISION-FIDELITY | PASS | plan_spearman=0.5960; plan_pearson=0.5349; top_5_overlap=0.5000; top_10_overlap=0.5833; regret=0.9113; selected_true_rank=6.1667 | Cost ranking is distinct from trajectory accuracy; positive correlation does not ensure low regret. |
| ORACLE PLANNING | PASS | oracle_objective=4.4432; random_objective=10.3305; noop_objective=10.9442; independent_objective=6.2384 | The finite-budget oracle tests task feasibility with the same CEM budget; it is not a globally optimal ceiling. |
| LEARNED ORCHESTRATION | PASS | orchestra_objective=4.8759; random_objective=10.3305; noop_objective=10.9442; proceed_fraction=1.0000 | All reported outcomes come from executing first actions and replanning in the true simulator. |
| CENTRALIZED VS INDEPENDENT | PARTIAL | orchestra_objective=4.8759; local_reactive_objective=6.2384; independent_learned_mpc_objective=4.8759 | Local-reactive gains do not isolate the benefit of shared interaction modeling; the independently learned MPC baseline must also be considered. |
| GENERALIZATION | PASS | ID_position_m=4.4200; density_position_m=4.9974; behavior_position_m=4.9992; topology_position_m=3.4193; controlled_regimes_with_positive_rank_and_planning_gain=1.0000 | ID, density, behavior and topology shifts are reported separately; the single training seed limits robustness claims. |

## PASSED

SIMULATOR, DATA / ACTION-EXCITATION, WORLD MODEL, AUTONOMOUS IMAGINATION, ACTION SENSITIVITY, MEMORY, COUNTERFACTUAL, DECISION-FIDELITY, ORACLE PLANNING, LEARNED ORCHESTRATION, GENERALIZATION.

## FAILED / PARTIAL

MULTI-AGENT INTERACTION: FAIL, CENTRALIZED VS INDEPENDENT: PARTIAL.

## UNTESTED

Small (no GPU available), full (intentionally unrun), multiple training seeds, uncertainty/calibration, active sensing, anonymous track association, real-world operation.

## Primary comparison

| Model | Rollout error h10 (m) | Action gap (m) | Counterfactual Pearson | Plan-rank Spearman | MPC true objective |
| --- | --- | --- | --- | --- | --- |
| persistence | 17.6138 | 0.0000 | undefined | undefined | 10.9442 |
| constant_velocity | 8.9479 | 0.0000 | undefined | undefined | 10.9442 |
| independent | 7.3548 | -0.0009 | 0.4157 | 0.5938 | 4.8759 |
| no_actions | 7.3781 | 0.0000 | undefined | undefined | 10.9442 |
| orchestra | 7.1687 | 1.2444 | 0.6155 | 0.5960 | 4.8759 |
| privileged | 8.5781 | -0.4223 | 0.7373 | 0.5860 | 4.8759 |
| oracle | 0 (simulator reference) | N/A | 1.0000 | 1.0000 | 4.4432 |

Trajectory fidelity, counterfactual geometry and decision fidelity are separate measurements. A good plan ranking cannot repair an action-shuffle failure, and choosing the same aggressive plan as an independent learned controller is not evidence for interaction-aware coordination.

## Next experiment

Train three independent initializations on a larger, balanced factorial intervention corpus and test whether ten-step correct-action gains and joint-interaction gains persist over the independent learned MPC baseline.
