# Released results

## Controlled speaker-overlap test set

| Model | WA (%) | UA (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Pitch-Fusion, probability ensemble | 72.63 | 70.84 | 71.26 |
| Residual acoustic adaptation, probability ensemble | 73.16 | 72.48 | 72.43 |

The full residual model improves WA by 0.53, UA by 1.63, and Macro-F1 by 1.17 percentage points
relative to the reproduced Pitch-Fusion ensemble on this controlled-overlap test set. These
numbers do not measure unseen-speaker generalization.

Machine-readable per-seed, classwise, confusion-matrix, and validation-ablation tables are stored
in `results/`.
