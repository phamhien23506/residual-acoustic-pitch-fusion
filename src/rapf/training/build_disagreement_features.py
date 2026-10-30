"""Add three-seed disagreement statistics to the selected ensemble anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    ROOT / "artifacts/b6/residual_pitch_ensemble_v3/anchor/seed42.parquet",
    ROOT / "artifacts/b6/residual_pitch_v2_seed2025/anchor/robust_pitch_train_val_anchor.parquet",
    ROOT / "artifacts/b6/residual_pitch_ensemble_v3/anchor/seed3407.parquet",
)
ENSEMBLE_PATH = (
    ROOT / "artifacts/b6/residual_pitch_ensemble_v3/anchor/ensemble_train_val_anchor.parquet"
)
OUTPUT_DIR = ROOT / "artifacts/b6/residual_pitch_ensemble_v4/anchor"
LOG_COLUMNS = [f"anchor_log_probability_{index}" for index in range(4)]


def main() -> None:
    seeds = [
        pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)
        for path in SOURCE_PATHS
    ]
    ensemble = pd.read_parquet(ENSEMBLE_PATH).sort_values("sample_id").reset_index(
        drop=True
    )
    if not all(ensemble["sample_id"].equals(seed["sample_id"]) for seed in seeds):
        raise ValueError("Seed anchors and ensemble anchor are not aligned")
    probabilities = np.stack(
        [np.exp(seed[LOG_COLUMNS].to_numpy(dtype=np.float64)) for seed in seeds],
        axis=1,
    )
    variance = probabilities.var(axis=1)
    votes = np.stack([probabilities[:, index].argmax(axis=1) for index in range(3)])
    vote_fraction = np.stack(
        [(votes == label).mean(axis=0) for label in range(4)], axis=1
    )
    agreement = vote_fraction.max(axis=1, keepdims=True)
    auxiliary = np.concatenate((variance, vote_fraction, agreement), axis=1)
    output = ensemble.copy()
    for index in range(auxiliary.shape[1]):
        output[f"anchor_aux_{index}"] = auxiliary[:, index].astype(np.float32)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "disagreement_aware_train_val_anchor.parquet"
    output.to_parquet(output_path, index=False)
    report = {
        "experiment": "three_seed_disagreement_aware_anchor",
        "test_loaded_or_evaluated": False,
        "rows": len(output),
        "anchor_aux_size": int(auxiliary.shape[1]),
        "auxiliary_features": [
            "class_probability_variance_happy",
            "class_probability_variance_neutral",
            "class_probability_variance_sad",
            "class_probability_variance_angry",
            "vote_fraction_happy",
            "vote_fraction_neutral",
            "vote_fraction_sad",
            "vote_fraction_angry",
            "maximum_vote_agreement",
        ],
        "mean_maximum_vote_agreement": float(agreement.mean()),
        "unanimous_vote_ratio": float(np.mean(agreement[:, 0] == 1.0)),
    }
    (OUTPUT_DIR / "disagreement_anchor_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
