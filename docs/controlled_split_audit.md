# Controlled Speaker-Overlapping Split Audit

> This experiment uses a speaker-overlapping controlled stratified split. It is not a speaker-independent evaluation.

## Configuration

- Protocol: `controlled_speaker_overlap_v1`
- Source manifest: `data/clean_manifest.csv`
- Source manifest SHA-256: `f65352be07c6d770851a9f082590e60f7b76c60ab3058139757e9d719f9017c6`
- Seed: `2025`
- Controlled-speaker rule: more than `20` eligible utterances
- Special speaker ID: `0`
- Requested ratios: Train 70%, Validation 15%, Test 15%
- Allocation inputs: speaker ID, emotion, accent, stable sample ID, and audio SHA-256 only
- Model predictions, losses, F1 scores, and test performance were not loaded or used.

The frozen source split is unchanged. The new allocation is written only to separate controlled-split files.

## Source eligibility and exclusions

- Source rows: 5,280
- Eligible rows allocated by this pipeline: 4,992
- Rows already marked ineligible by the frozen cleaning manifest: 288
- Newly removed by this controlled splitter: 0

| Frozen exclusion reason | Rows | Meaning |
| --- | --- | --- |
| identical_audio_has_conflicting_metadata | 20 | exact audio has conflicting metadata; excluded conservatively |
| identical_audio_represented_by_lower_source_row | 268 | redundant exact-audio copy; the lower source-row representative is retained |

The 268 rows in the second category are duplicate copies, not 268 additional duplicate groups. This splitter does not delete them again; it consumes only the 4,992 rows already marked eligible.

## Global split sizes

| Split | Utterances | Percentage | Speakers |
| --- | --- | --- | --- |
| Train | 3494 | 69.992% | 142 |
| Validation | 749 | 15.004% | 105 |
| Test | 749 | 15.004% | 102 |

## Speaker control and overlap

- Eligible speakers: 147
- Controlled speakers (> 20 utterances): 37
- Controlled speakers with both Train and Test samples: 37/37

| Identity intersection | Speakers | % of 147 eligible speakers |
| --- | --- | --- |
| Train / Validation | 100 | 68.027% |
| Train / Test | 99 | 67.347% |
| Validation / Test | 87 | 59.184% |
| All three | 84 | 57.143% |

Speaker overlap is intentional. Utterance overlap is prohibited.

## Emotion distribution

| Split | angry | happy | neutral | sad |
| --- | --- | --- | --- | --- |
| Train | 1029 | 684 | 1055 | 726 |
| Validation | 217 | 146 | 223 | 163 |
| Test | 219 | 143 | 227 | 160 |

## Accent distribution

| Split | central | north | south |
| --- | --- | --- | --- |
| Train | 193 | 828 | 2473 |
| Validation | 41 | 176 | 532 |
| Test | 41 | 179 | 529 |

## Accent x emotion coverage

- Missing cells across all splits: 0
- No missing accent x emotion cells.

## Speaker ID 0

Speaker ID 0 is treated as one physical speaker. In the controlled speaker-overlapping split, this speaker is intentionally represented in Train, Validation, and Test; in the speaker-independent protocol, all utterances from this speaker are assigned to Train only.

| Split | Count | Percentage of Speaker 0 |
| --- | --- | --- |
| Train | 1552 | 70.004% |
| Validation | 333 | 15.020% |
| Test | 332 | 14.975% |

Speaker 0 emotion distribution:

| Split | angry | happy | neutral | sad |
| --- | --- | --- | --- | --- |
| Train | 599 | 293 | 359 | 301 |
| Validation | 128 | 63 | 77 | 65 |
| Test | 128 | 62 | 77 | 65 |

- Accent: `{'south': 2217}`
- Released source distribution: `{'huggingface': 2217}`; release IDs represented: 1
- Storage-shard distribution: `{'data/train-00000-of-00001.parquet': 2217}`
- Source-group inspection: no explicit reliable source-video/session column is present; audio_locator only identifies parquet rows and audio_path_raw is a unique WAV filename.
- The released manifest does not expose a reliable YouTube video or recording-session mapping for Speaker 0.
- Numeric WAV filenames and parquet row positions were not treated as sessions because that would be an unsupported inference.

## Integrity checks

| Check | Result |
| --- | --- |
| Every eligible utterance appears exactly once | PASS (4,992/4,992) |
| Identical audio hashes crossing partitions | PASS (0) |
| Controlled speakers represented in Train and Test | PASS (37/37) |
| Global sizes match deterministic quotas | PASS |
| Frozen split modified | PASS (No) |
| Model performance used for allocation | PASS (No) |

## Method summary

Each controlled speaker is allocated separately with deterministic emotion stratification and approximately 70/15/15 quotas. The remaining speakers are allocated at utterance level using deterministic accent x emotion stratification to meet the exact global quotas. A final metadata-only swap can repair an empty Test accent x emotion cell without changing split sizes. Exact audio hashes are required to be unique before allocation and are checked again after allocation.

Coverage repairs applied: `[]`
