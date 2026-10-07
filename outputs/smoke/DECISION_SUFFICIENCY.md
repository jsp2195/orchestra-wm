# Decision sufficiency

| Model | Rollout error h10 (m) | Action gap (m) | Counterfactual Pearson | Plan-rank Spearman | MPC true objective |
| --- | --- | --- | --- | --- | --- |
| persistence | 17.6138 | 0.0000 | undefined | undefined | 10.9442 |
| constant_velocity | 8.9479 | 0.0000 | undefined | undefined | 10.9442 |
| independent | 7.3548 | -0.0009 | 0.4157 | 0.5938 | 4.8759 |
| no_actions | 7.3781 | 0.0000 | undefined | undefined | 10.9442 |
| orchestra | 7.1687 | 1.2444 | 0.6155 | 0.5960 | 4.8759 |
| privileged | 8.5781 | -0.4223 | 0.7373 | 0.5860 | 4.8759 |
| oracle | 0 (simulator reference) | N/A | 1.0000 | 1.0000 | 4.4432 |

Lower error/objective is better; positive action gap is better. Undefined correlations mean constant predictions, not missing execution. All learned models share one training seed; evaluation uses three paired simulator seeds. Persistence/CV use observable kinematic surrogates and select no-op when all candidates tie. Oracle geometry/rank correlations are the diagnostic self-reference.
