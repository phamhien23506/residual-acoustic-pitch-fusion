from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = ("happy", "neutral", "sad", "angry")
PROBABILITY_COLUMNS = tuple(f"probability_{label}" for label in LABELS)
D2_SEEDS = (42, 2025, 3407)
BOOTSTRAP_SEED = 2025
N_RESAMPLES = 2_000


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_experiment_root = repository_root.parent / "DAP391m_Group3_FA26"
    parser = argparse.ArgumentParser(
        description="Create the paired-bootstrap Test improvement forest plot."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=default_experiment_root,
        help="Path to the full experiment repository containing Test predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "figures",
        help="Directory for PDF and PNG figure outputs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repository_root / "results",
        help="Directory for the numerical source table.",
    )
    return parser.parse_args()


def metric_triplet(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    wa = np.mean(y_true == y_pred)
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in LABELS:
        true_label = y_true == label
        pred_label = y_pred == label
        true_positive = np.sum(true_label & pred_label)
        recall = true_positive / np.sum(true_label)
        predicted_count = np.sum(pred_label)
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(float(recall))
        f1_scores.append(float(f1))
    return np.asarray((wa, np.mean(recalls), np.mean(f1_scores)), dtype=np.float64)


def read_and_align(path: Path, sample_ids: pd.Series, truth: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "true_emotion", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) != 749 or frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} must contain 749 unique Test utterances")
    aligned = frame.set_index("sample_id").loc[sample_ids].reset_index()
    if not np.array_equal(aligned["true_emotion"].to_numpy(), truth.to_numpy()):
        raise ValueError(f"Ground-truth mismatch in {path}")
    return aligned


def load_predictions(experiment_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    manifest_path = experiment_root / "data" / "manifests" / "controlled_test.csv"
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 749 or manifest["sample_id"].duplicated().any():
        raise ValueError("controlled_test.csv must contain 749 unique utterances")
    if set(manifest["emotion"]) != set(LABELS):
        raise ValueError("Unexpected emotion labels in controlled_test.csv")

    sample_ids = manifest["sample_id"]
    truth = manifest["emotion"]
    raw_root = experiment_root / "DAGR_PF_FINAL_TEST_RESULTS" / "raw_predictions"
    baseline_path = (
        raw_root
        / "A0_ensemble"
        / "pitch_fusion_official_controlled_ensemble_seed42_2025_3407_test_predictions.csv"
    )
    baseline = read_and_align(baseline_path, sample_ids, truth)

    d2_paths = [
        raw_root / f"D2_seed{seed}" / f"D2_seed{seed}_test_predictions.csv"
        for seed in D2_SEEDS
    ]
    d2_frames = [read_and_align(path, sample_ids, truth) for path in d2_paths]
    d2_probabilities = np.mean(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64) for frame in d2_frames],
        axis=0,
    )
    d2_prediction = np.asarray(LABELS)[np.argmax(d2_probabilities, axis=1)]
    baseline_prediction = baseline["predicted_emotion"].to_numpy()
    return truth.to_numpy(), baseline_prediction, d2_prediction, [manifest_path, baseline_path, *d2_paths]


def stratified_paired_bootstrap(
    truth: np.ndarray,
    d2_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = 100.0 * (
        metric_triplet(truth, d2_prediction) - metric_triplet(truth, baseline_prediction)
    )
    strata = [np.flatnonzero(truth == label) for label in LABELS]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_deltas = np.empty((N_RESAMPLES, 3), dtype=np.float64)
    for iteration in range(N_RESAMPLES):
        indices = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        bootstrap_deltas[iteration] = 100.0 * (
            metric_triplet(truth[indices], d2_prediction[indices])
            - metric_triplet(truth[indices], baseline_prediction[indices])
        )
    lower, upper = np.percentile(bootstrap_deltas, (2.5, 97.5), axis=0)
    return observed, lower, upper


def write_results_table(
    path: Path,
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    d2_prediction: np.ndarray,
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    baseline_metrics = 100.0 * metric_triplet(truth, baseline_prediction)
    d2_metrics = 100.0 * metric_triplet(truth, d2_prediction)
    table = pd.DataFrame(
        {
            "metric": ("WA", "UA", "Macro-F1"),
            "baseline_pct": baseline_metrics,
            "d2_pct": d2_metrics,
            "delta_percentage_points": observed,
            "ci95_low_percentage_points": lower,
            "ci95_high_percentage_points": upper,
            "bootstrap_type": "utterance-paired-class-stratified-percentile",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_resamples": N_RESAMPLES,
            "n_test": len(truth),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def create_forest_plot(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ("Times New Roman", "Times", "DejaVu Serif"),
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    positions = np.asarray((2.0, 1.0, 0.0))
    colors = ("#4b5563", "#2f6f9f", "#4b5563")
    figure, axis = plt.subplots(figsize=(4.8, 2.2), facecolor="white")
    figure.subplots_adjust(left=0.19, right=0.985, bottom=0.25, top=0.96)
    axis.set_facecolor("white")
    axis.grid(axis="x", color="0.86", linewidth=0.5, alpha=0.75)
    axis.set_axisbelow(True)
    axis.axvline(0.0, color="0.38", linestyle=(0, (4, 3)), linewidth=0.85, zorder=2)
    for index, color in enumerate(colors):
        axis.errorbar(
            observed[index],
            positions[index],
            xerr=np.asarray(
                [[observed[index] - lower[index]], [upper[index] - observed[index]]]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            elinewidth=1.1,
            capsize=2.4,
            capthick=0.85,
            zorder=3,
        )
    axis.set_yticks(positions, ("WA", "UA", "Macro-F1"))
    axis.set_xlabel("Improvement over Pitch-Fusion (percentage points)")
    span_low = min(float(np.min(lower)), 0.0)
    span_high = max(float(np.max(upper)), 0.0)
    margin = max(0.18, 0.045 * (span_high - span_low))
    axis.set_xlim(span_low - margin, span_high + margin)
    axis.set_ylim(-0.28, 2.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("0.25")
    axis.spines["bottom"].set_color("0.25")
    axis.spines["left"].set_linewidth(0.75)
    axis.spines["bottom"].set_linewidth(0.75)
    axis.tick_params(axis="x", direction="out", length=3.0, width=0.65, pad=3)
    axis.tick_params(axis="y", length=0, pad=5)

    figure.savefig(output_dir / "fig_test_improvement_ci.pdf", facecolor="white")
    figure.savefig(output_dir / "fig_test_improvement_ci.svg", facecolor="white")
    figure.savefig(output_dir / "fig_test_improvement_ci.png", dpi=300, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    truth, baseline_prediction, d2_prediction, sources = load_predictions(experiment_root)
    observed, lower, upper = stratified_paired_bootstrap(
        truth, d2_prediction, baseline_prediction
    )
    results_path = args.results_dir / "test_improvement_ci.csv"
    table = write_results_table(
        results_path,
        truth,
        baseline_prediction,
        d2_prediction,
        observed,
        lower,
        upper,
    )
    create_forest_plot(observed, lower, upper, args.output_dir)

    print(table.to_string(index=False))
    print("\nSources:")
    for source in sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()
