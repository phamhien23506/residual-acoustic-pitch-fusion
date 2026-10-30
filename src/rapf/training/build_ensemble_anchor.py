"""Build a fixed-candidate three-seed Train/Validation ensemble anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    "seed42": ROOT / "artifacts/b6/residual_pitch_ensemble_v3/anchor/seed42.parquet",
    "seed2025": (
        ROOT / "artifacts/b6/residual_pitch_v2_seed2025/anchor"
        / "robust_pitch_train_val_anchor.parquet"
    ),
    "seed3407": ROOT / "artifacts/b6/residual_pitch_ensemble_v3/anchor/seed3407.parquet",
}
OUTPUT_DIR = ROOT / "artifacts/b6/residual_pitch_ensemble_v3/anchor"
LOG_COLUMNS = [f"anchor_log_probability_{index}" for index in range(4)]


def metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    truth = frame["emotion_id"].to_numpy()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "ua": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def main() -> None:
    frames = {
        name: pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)
        for name, path in SOURCES.items()
    }
    reference = frames["seed2025"]
    if not all(reference["sample_id"].equals(frame["sample_id"]) for frame in frames.values()):
        raise ValueError("Seed anchors do not contain identical ordered samples")
    probabilities = {
        name: np.exp(frame[LOG_COLUMNS].to_numpy(dtype=np.float64))
        for name, frame in frames.items()
    }
    fixed_candidates = {
        "equal_mean": (1 / 3, 1 / 3, 1 / 3),
        "favor_2025_60": (0.20, 0.60, 0.20),
        "favor_2025_50": (0.25, 0.50, 0.25),
        "seed42_seed2025": (0.50, 0.50, 0.00),
        "seed2025_seed3407": (0.00, 0.50, 0.50),
    }
    candidates: dict[str, dict[str, object]] = {}
    validation_mask = reference["split"].eq("validation").to_numpy()
    for name, weights in fixed_candidates.items():
        combined = (
            weights[0] * probabilities["seed42"]
            + weights[1] * probabilities["seed2025"]
            + weights[2] * probabilities["seed3407"]
        )
        combined /= combined.sum(axis=1, keepdims=True)
        candidates[name] = {
            "weights_seed42_seed2025_seed3407": list(weights),
            "validation": metrics(reference.loc[validation_mask], combined[validation_mask]),
            "probabilities": combined,
        }
    geometric = np.exp(
        sum(np.log(np.clip(value, 1e-8, 1.0)) for value in probabilities.values()) / 3
    )
    geometric /= geometric.sum(axis=1, keepdims=True)
    candidates["geometric_mean"] = {
        "weights_seed42_seed2025_seed3407": [1 / 3, 1 / 3, 1 / 3],
        "validation": metrics(reference.loc[validation_mask], geometric[validation_mask]),
        "probabilities": geometric,
    }
    selected_name = max(
        candidates,
        key=lambda name: (
            candidates[name]["validation"]["ua"],
            candidates[name]["validation"]["macro_f1"],
        ),
    )
    selected = candidates[selected_name]
    selected_probabilities = selected.pop("probabilities")
    for candidate in candidates.values():
        candidate.pop("probabilities", None)
    output = reference[["sample_id", "split", "emotion", "emotion_id"]].copy()
    for index, column in enumerate(LOG_COLUMNS):
        output[column] = np.log(
            np.clip(selected_probabilities[:, index], 1e-8, 1.0)
        ).astype(np.float32)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_DIR / "ensemble_train_val_anchor.parquet", index=False)
    report = {
        "experiment": "fixed_candidate_three_seed_anchor_selection",
        "selection_data": "Validation only",
        "test_loaded_or_evaluated": False,
        "candidates": candidates,
        "selected": selected_name,
        "selected_validation": selected["validation"],
        "single_seed2025_validation_ua": 0.7045810803124503,
        "validation_ua_delta": selected["validation"]["ua"] - 0.7045810803124503,
    }
    (OUTPUT_DIR / "ensemble_anchor_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
