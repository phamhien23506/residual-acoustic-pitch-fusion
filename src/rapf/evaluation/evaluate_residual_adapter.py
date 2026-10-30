"""Evaluate a selected residual acoustic adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rapf.experiment_utils import classification_metrics
from rapf.training.train_residual_adapter import (
    AnchoredResidualPitchFusion,
    ResidualConfig,
    ResidualDataset,
    anchor_metrics,
    evaluate,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[3]
LABELS = ["happy", "neutral", "sad", "angry"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir", default="artifacts/b6/residual_pitch_v2_seed2025/pilot"
    )
    parser.add_argument(
        "--baseline-predictions",
        default=(
            "artifacts/b6/pitch_fusion_official_controlled_seed2025/evaluation/"
            "pitch_fusion_official_controlled_seed2025_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", default="artifacts/b6/residual_pitch_v2_seed2025/evaluation"
    )
    parser.add_argument("--model-name", default="residual_pitch_v2_seed2025")
    parser.add_argument("--expected-baseline-ua", type=float, default=0.699448446187761)
    parser.add_argument(
        "--test-manifest", default="data/manifests/controlled_test.csv"
    )
    parser.add_argument(
        "--train-manifest", default="data/manifests/controlled_train.csv"
    )
    parser.add_argument("--expected-test-size", type=int, default=749)
    parser.add_argument(
        "--protocol-name", default="controlled speaker-overlapping 70/15/15"
    )
    parser.add_argument(
        "--prelocked-architecture",
        action="store_true",
        help=(
            "Evaluate an architecture locked before this protocol's Test was opened, "
            "even when its validation stage-gate did not indicate a gain."
        ),
    )
    parser.add_argument(
        "--seed-predictions",
        nargs=3,
        help="Three seed Test prediction CSVs used to construct anchor disagreement features.",
    )
    parser.add_argument(
        "--post-hoc-test-sweep",
        action="store_true",
        help=(
            "Mark this evaluation as part of an exploratory post-hoc Test sweep. "
            "Such results must not be presented as confirmatory model selection."
        ),
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    pilot_dir = resolve(args.pilot_dir)
    output_dir = resolve(args.output_dir)
    report_path = output_dir / f"{args.model_name}_report.json"
    if report_path.exists():
        raise RuntimeError(
            "Residual v2 Test report already exists; refusing a second evaluation"
        )
    pilot = json.loads((pilot_dir / "pilot_report.json").read_text(encoding="utf-8"))
    if pilot["decision"] != "GO_FULL_RESIDUAL" and not args.prelocked_architecture:
        raise RuntimeError("Validation stage-gate did not authorize Test evaluation")
    config = ResidualConfig(**pilot["config"])
    checkpoint = torch.load(
        pilot_dir / "residual_v2_best.pt", map_location="cpu", weights_only=False
    )
    if int(checkpoint["best_epoch"]) != int(pilot["residual"]["best_epoch"]):
        raise RuntimeError("Pilot report and checkpoint epoch disagree")

    baseline_path = resolve(args.baseline_predictions)
    baseline = pd.read_csv(baseline_path)
    manifest = pd.read_csv(
        resolve(args.test_manifest),
        usecols=[
            "sample_id",
            "emotion",
            "emotion_id",
            "speaker_id",
            "accent_clean",
            "gender_clean",
        ],
    )
    train_metadata = pd.read_csv(
        resolve(args.train_manifest),
        usecols=["speaker_id", "accent_clean", "gender_clean", "emotion_id"],
    )
    features = pd.read_parquet(
        ROOT / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet"
    )
    wav_columns = [f"wav2vec2_{index:03d}" for index in range(768)]
    feature_columns = [
        "sample_id", "log_mel_artifact_path", "pitch_artifact_path", *wav_columns
    ]
    probability_columns = [f"probability_{label}" for label in LABELS]
    anchor = baseline[["sample_id", *probability_columns]].copy()
    for index, probability_column in enumerate(probability_columns):
        anchor[f"anchor_log_probability_{index}"] = np.log(
            np.clip(anchor[probability_column].to_numpy(), 1e-8, 1.0)
        )
    frame = manifest.merge(anchor, on="sample_id", validate="one_to_one").merge(
        features[feature_columns], on="sample_id", validate="one_to_one"
    )
    for source, target in (
        ("speaker_id", "speaker_index"),
        ("accent_clean", "accent_index"),
        ("gender_clean", "gender_index"),
    ):
        vocabulary = {
            value: index + 1
            for index, value in enumerate(
                sorted(train_metadata[source].astype(str).unique())
            )
        }
        frame[target] = (
            frame[source].astype(str).map(vocabulary).fillna(0).astype(int)
        )
    if config.speaker_conditioning:
        expected_vocab_sizes = {
            "speaker_vocab_size": int(frame["speaker_index"].max()) + 1,
            "accent_vocab_size": int(frame["accent_index"].max()) + 1,
            "gender_vocab_size": int(frame["gender_index"].max()) + 1,
        }
        for field, observed_minimum in expected_vocab_sizes.items():
            if getattr(config, field) < observed_minimum:
                raise ValueError(
                    f"Checkpoint {field}={getattr(config, field)} cannot encode "
                    f"Test index requiring size {observed_minimum}"
                )
    global_counts = np.bincount(
        train_metadata["emotion_id"].to_numpy(dtype=int), minlength=len(LABELS)
    ).astype(np.float64)
    global_prior = global_counts / global_counts.sum()
    speaker_counts = {
        str(speaker): np.bincount(
            group["emotion_id"].to_numpy(dtype=int), minlength=len(LABELS)
        ).astype(np.float64)
        for speaker, group in train_metadata.groupby("speaker_id")
    }
    smoothing = 10.0
    for class_index in range(len(LABELS)):
        frame[f"speaker_prior_{class_index}"] = [
            (
                speaker_counts.get(
                    str(speaker), np.zeros(len(LABELS), dtype=np.float64)
                )[class_index]
                + smoothing * global_prior[class_index]
            )
            / (
                speaker_counts.get(
                    str(speaker), np.zeros(len(LABELS), dtype=np.float64)
                ).sum()
                + smoothing
            )
            for speaker in frame["speaker_id"]
        ]
    if config.anchor_aux_size:
        if not args.seed_predictions:
            raise ValueError("Disagreement-aware checkpoint requires --seed-predictions")
        seed_probabilities = []
        for value in args.seed_predictions:
            seed_frame = pd.read_csv(resolve(value))
            aligned = manifest[["sample_id"]].merge(
                seed_frame[["sample_id", *probability_columns]],
                on="sample_id",
                validate="one_to_one",
            )
            seed_probabilities.append(
                aligned[probability_columns].to_numpy(dtype=np.float64)
            )
        stacked = np.stack(seed_probabilities, axis=1)
        variance = stacked.var(axis=1)
        votes = np.stack([stacked[:, index].argmax(axis=1) for index in range(3)])
        vote_fraction = np.stack(
            [(votes == label).mean(axis=0) for label in range(len(LABELS))], axis=1
        )
        agreement = vote_fraction.max(axis=1, keepdims=True)
        if config.anchor_aux_size == 9:
            auxiliary = np.concatenate((variance, vote_fraction, agreement), axis=1)
        elif config.anchor_aux_size == 15:
            mean_probability = np.exp(
                frame[
                    [f"anchor_log_probability_{index}" for index in range(4)]
                ].to_numpy(dtype=np.float64)
            )
            entropy = -(
                mean_probability
                * np.log(np.clip(mean_probability, 1e-12, 1.0))
            ).sum(axis=1, keepdims=True) / np.log(4.0)
            sorted_probability = np.sort(mean_probability, axis=1)
            margin = (
                sorted_probability[:, -1] - sorted_probability[:, -2]
            )[:, None]
            auxiliary = np.concatenate(
                (
                    mean_probability,
                    variance,
                    vote_fraction,
                    agreement,
                    entropy,
                    margin,
                ),
                axis=1,
            )
        else:
            raise ValueError(
                f"Unsupported disagreement feature size: {config.anchor_aux_size}"
            )
        if auxiliary.shape[1] != config.anchor_aux_size:
            raise ValueError("Checkpoint and constructed anchor auxiliary size disagree")
        auxiliary_frame = manifest[["sample_id"]].copy()
        for index in range(auxiliary.shape[1]):
            auxiliary_frame[f"anchor_aux_{index}"] = auxiliary[:, index]
        frame = frame.merge(auxiliary_frame, on="sample_id", validate="one_to_one")
    frame["split"] = "test"
    if len(frame) != int(args.expected_test_size) or frame["sample_id"].duplicated().any():
        raise ValueError(
            f"Expected {int(args.expected_test_size)} unique Test utterances"
        )
    baseline_test = anchor_metrics(frame)
    expected_baseline_ua = float(args.expected_baseline_ua)
    if abs(baseline_test["ua"] - expected_baseline_ua) > 1e-9:
        raise RuntimeError(
            f"Baseline Test anchor mismatch: {baseline_test['ua']:.12f}"
        )

    seed_everything(config.seed)
    model = AnchoredResidualPitchFusion(config)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda")
    model.to(device).eval()
    loader = DataLoader(
        ResidualDataset(frame, config.max_frames, config.anchor_aux_size),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_metrics, predictions = evaluate(
        model,
        loader,
        device,
        config.residual_penalty,
        None,
        config.anchor_consistency_weight,
        config.correct_anchor_consistency_weight,
        config.auxiliary_classification_weight,
        config.quality_auxiliary_weight,
        config.happy_neutral_specialist_weight,
        config.focal_gamma,
        config.label_smoothing,
        config.supervised_contrastive_weight,
        config.supervised_contrastive_temperature,
        config.anchor_uncertainty_weight,
        config.cross_modal_alignment_weight,
        config.cross_modal_alignment_temperature,
        config.anchor_error_auxiliary_weight,
        config.branch_auxiliary_weight,
        config.gate_direction_weight,
    )
    detailed = classification_metrics(
        predictions["true_emotion"].to_numpy(),
        predictions["predicted_emotion"].to_numpy(),
        frame["accent_clean"].to_numpy(),
        LABELS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output_dir / f"{args.model_name}_test_predictions.csv", index=False
    )
    report = {
        "schema_version": "b6.residual_pitch.v2.exploratory",
        "model": args.model_name,
        "protocol": str(args.protocol_name),
        "warning": (
            "Architecture and hyperparameters were locked before this protocol's Test "
            "evaluation; Test was not used for checkpoint or architecture selection."
            if args.prelocked_architecture
            else
            "Exploratory result. The shared Test set had already been observed in prior "
            "experiments; this result is not a fresh confirmatory estimate."
        ),
        "architecture": (
            "Anchored residual Pitch-Fusion configuration loaded exactly from the "
            "referenced pilot report. See source_pilot_dir and config for the "
            "candidate-specific architecture."
        ),
        "source_pilot_dir": str(pilot_dir),
        "config": pilot["config"],
        "selection": {
            "criterion": (
                "Post-hoc Test sweep; exploratory only"
                if args.post_hoc_test_sweep
                else "Validation balanced accuracy only"
            ),
            "best_epoch": int(checkpoint["best_epoch"]),
            "pilot_validation": pilot["residual"]["validation"],
            "pilot_validation_ua_delta": pilot["residual"]["validation_ua_delta"],
            "prelocked_architecture": bool(args.prelocked_architecture),
        },
        "baseline_test": baseline_test,
        "residual_test": {
            **test_metrics,
            "detailed": detailed,
            "ua_delta": test_metrics["ua"] - baseline_test["ua"],
            "macro_f1_delta": test_metrics["macro_f1"] - baseline_test["macro_f1"],
            "accuracy_delta": test_metrics["accuracy"] - baseline_test["accuracy"],
            "mean_correction_strength": float(
                predictions["correction_strength"].mean()
            ),
            "mean_residual_l2": float(predictions["residual_l2"].mean()),
            "mean_prosody_gate": float(predictions["gate_prosody"].mean()),
            "mean_spectral_gate": float(predictions["gate_spectral"].mean()),
        },
        "test_evaluation_attempts_for_this_candidate": 1,
        "test_used_for_candidate_selection": bool(args.post_hoc_test_sweep),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "baseline_test": baseline_test,
                "residual_test": test_metrics,
                "ua_delta": report["residual_test"]["ua_delta"],
                "macro_f1_delta": report["residual_test"]["macro_f1_delta"],
                "test_evaluation_attempts_for_this_candidate": 1,
                "test_used_for_candidate_selection": bool(args.post_hoc_test_sweep),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
