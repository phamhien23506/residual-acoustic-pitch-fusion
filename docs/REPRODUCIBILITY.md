# Reproducibility protocol

## Evaluation protocol

This experiment uses a speaker-overlapping controlled stratified split. It is not a
speaker-independent evaluation. Speaker identities may occur in more than one partition, while
each utterance and each audio hash occur in exactly one partition.

The released split contains 4,992 eligible utterances: 3,494 for training, 749 for validation,
and 749 for testing. The allocation seed is 2025 and the target ratio is 70/15/15. Speaker 0 has
2,217 eligible utterances allocated as 1,552/333/332.

## Model selection

Pitch-Fusion checkpoints and residual-adapter checkpoints are selected using validation UA.
The reported controlled-overlap test table uses probability ensembles across seeds 42, 2025,
and 3407. Test metrics are not inputs to the split algorithm.

## Residual adapter

The paper configuration uses Prosody and Log-Mel encoders, a learned two-branch fusion module,
and nine disagreement statistics computed from three Pitch-Fusion anchors. It applies direct
residual correction without a correction gate:

```text
final_logits = anchor_logits + residual_logits
```

The implementation has 73,083 total parameters and 70,522 trainable parameters. The automated
test in `tests/test_residual_adapter.py` verifies both counts and the direct-addition equation.

## Data availability

ViSEC audio is not redistributed here. The split CSV files provide dataset row locators, labels,
speaker identifiers, and audio hashes required to reconstruct the exact partitions from an
authorized copy of ViSEC. Cached Wav2Vec2, pitch, and Log-Mel features are also excluded.
