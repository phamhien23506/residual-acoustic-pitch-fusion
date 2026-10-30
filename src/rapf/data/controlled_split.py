"""Create a reproducible speaker-overlapping controlled ViSEC split.

The frozen speaker-disjoint split is read only as a source manifest and is
never modified. Allocation uses metadata only; model predictions and metrics
are not accepted as inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SPEAKER_THRESHOLD = 20
SPECIAL_SPEAKER_ID = "0"
SEED = 2025
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLITS = tuple(RATIOS)
ACCENTS = ("central", "north", "south")
EMOTIONS = ("angry", "happy", "neutral", "sad")
PROTOCOL_VERSION = "controlled_speaker_overlap_v1"
SOURCE_GROUP_CANDIDATES = (
    "source_group_id",
    "source_video_id",
    "video_id",
    "youtube_video_id",
    "session_id",
    "source_session_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, scope: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{scope}|{value}".encode("utf-8")).hexdigest()


def numeric_text_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def largest_remainder_counts(
    total: int,
    ratios: dict[str, float] = RATIOS,
    *,
    seed: int = SEED,
    scope: str = "global",
) -> dict[str, int]:
    """Convert ratios to deterministic integer quotas that sum to total."""
    exact = {split: total * ratios[split] for split in SPLITS}
    counts = {split: math.floor(exact[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    order = sorted(
        SPLITS,
        key=lambda split: (
            -(exact[split] - counts[split]),
            stable_key(seed, scope, split),
        ),
    )
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def _stratum_key(values: Any) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        values = (values,)
    return tuple("<missing>" if pd.isna(value) else str(value) for value in values)


def allocate_stratified(
    frame: pd.DataFrame,
    target_counts: dict[str, int],
    strata_columns: list[str],
    *,
    seed: int,
    scope: str,
) -> dict[int, str]:
    """Allocate rows to exact quotas while approximating each stratum ratio."""
    if sum(target_counts.values()) != len(frame):
        raise ValueError(f"Quota mismatch in {scope}: {target_counts} for {len(frame)} rows")
    if frame.empty:
        return {}

    remaining = dict(target_counts)
    target_ratios = {
        split: target_counts[split] / len(frame) if len(frame) else 0.0
        for split in SPLITS
    }
    assignments: dict[int, str] = {}
    groupby_key: str | list[str] = strata_columns[0] if len(strata_columns) == 1 else strata_columns
    grouped = frame.groupby(groupby_key, sort=True, dropna=False).groups
    groups = sorted(grouped.items(), key=lambda item: (_stratum_key(item[0]), len(item[1])))

    for values, indices in groups:
        stratum = "/".join(_stratum_key(values))
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: stable_key(seed, f"{scope}|{stratum}", str(frame.at[index, "sample_id"])),
        )
        ideal = {split: len(ordered) * target_ratios[split] for split in SPLITS}
        local = {split: 0 for split in SPLITS}
        for index in ordered:
            candidates = [split for split in SPLITS if remaining[split] > 0]
            if not candidates:
                raise RuntimeError(f"No quota remains while allocating {scope}")
            selected = max(
                candidates,
                key=lambda split: (
                    ideal[split] - local[split],
                    remaining[split] / max(target_counts[split], 1),
                    stable_key(seed, f"{scope}|tie|{frame.at[index, 'sample_id']}", split),
                ),
            )
            assignments[index] = selected
            local[selected] += 1
            remaining[selected] -= 1

    if any(remaining.values()):
        raise RuntimeError(f"Unfilled quota in {scope}: {remaining}")
    return assignments


def detect_reliable_source_group(frame: pd.DataFrame) -> tuple[str | None, str]:
    """Use only explicit group metadata; never infer sessions from row order."""
    for column in SOURCE_GROUP_CANDIDATES:
        if column not in frame:
            continue
        values = frame[column].astype("string").str.strip()
        present = values.notna() & values.ne("")
        if not present.all():
            return None, f"{column} exists but is incomplete"
        unique = int(values.nunique())
        if 1 < unique < len(frame):
            return column, f"using explicit {column} ({unique} groups)"
    return None, (
        "no explicit reliable source-video/session column is present; audio_locator "
        "only identifies parquet rows and audio_path_raw is a unique WAV filename"
    )


def _repair_missing_test_cells(
    frame: pd.DataFrame,
    assignments: dict[int, str],
    controlled_speakers: set[str],
    *,
    seed: int,
) -> list[dict[str, str]]:
    """Swap rows without changing split sizes to fill feasible test cross-cells."""
    repairs: list[dict[str, str]] = []
    while True:
        test_counts = {
            (accent, emotion): sum(
                assignments[int(index)] == "test"
                for index in frame.index[
                    frame["accent_clean"].eq(accent) & frame["emotion"].eq(emotion)
                ]
            )
            for accent in ACCENTS
            for emotion in EMOTIONS
        }
        missing = [cell for cell, count in test_counts.items() if count == 0]
        if not missing:
            break
        changed = False
        for accent, emotion in missing:
            candidates = [
                int(index)
                for index in frame.index[
                    frame["accent_clean"].eq(accent) & frame["emotion"].eq(emotion)
                ]
                if assignments[int(index)] != "test"
            ]
            candidates.sort(
                key=lambda index: stable_key(seed, "coverage-candidate", str(frame.at[index, "sample_id"]))
            )
            for candidate in candidates:
                old_split = assignments[candidate]
                candidate_speaker = str(frame.at[candidate, "speaker_id"])
                if candidate_speaker in controlled_speakers:
                    old_count = sum(
                        assignments[int(index)] == old_split
                        for index in frame.index[frame["speaker_id"].astype(str).eq(candidate_speaker)]
                    )
                    if old_count <= 1:
                        continue
                donors = [
                    int(index)
                    for index in frame.index
                    if assignments[int(index)] == "test"
                    and test_counts[(str(frame.at[index, "accent_clean"]), str(frame.at[index, "emotion"]))] > 1
                ]
                donors.sort(
                    key=lambda index: (
                        str(frame.at[index, "accent_clean"]) != accent,
                        str(frame.at[index, "emotion"]) != emotion,
                        stable_key(seed, "coverage-donor", str(frame.at[index, "sample_id"])),
                    )
                )
                donor = next(
                    (
                        index
                        for index in donors
                        if not (
                            str(frame.at[index, "speaker_id"]) in controlled_speakers
                            and sum(
                                assignments[int(member)] == "test"
                                for member in frame.index[
                                    frame["speaker_id"].astype(str).eq(str(frame.at[index, "speaker_id"]))
                                ]
                            )
                            <= 1
                        )
                    ),
                    None,
                )
                if donor is None:
                    continue
                assignments[candidate], assignments[donor] = "test", old_split
                repairs.append(
                    {
                        "filled_cell": f"{accent}/{emotion}",
                        "added_sample": str(frame.at[candidate, "sample_id"]),
                        "swapped_sample": str(frame.at[donor, "sample_id"]),
                    }
                )
                changed = True
                break
        if not changed:
            break
    return repairs


def build_controlled_split(
    source: pd.DataFrame,
    *,
    threshold: int = SPEAKER_THRESHOLD,
    special_speaker_id: str = SPECIAL_SPEAKER_ID,
    seed: int = SEED,
    ratios: dict[str, float] = RATIOS,
    protocol_version: str = PROTOCOL_VERSION,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return eligible rows with a new controlled split and its audit facts."""
    frame = source.copy()
    source_rows = len(frame)
    if "is_eligible_for_split" in frame:
        eligible = frame["is_eligible_for_split"].astype(str).str.lower().eq("true")
    else:
        eligible = frame["split"].astype(str).ne("excluded")
    excluded = frame.loc[~eligible]
    if "exclusion_reason" in excluded:
        exclusion_reasons = {
            str(reason): int(count)
            for reason, count in excluded["exclusion_reason"]
            .fillna("<missing reason>")
            .value_counts()
            .sort_index()
            .items()
        }
    else:
        exclusion_reasons = {"marked ineligible by source manifest": int(len(excluded))}
    frame = frame.loc[eligible].copy().reset_index(drop=True)
    frame["speaker_id"] = frame["speaker_id"].astype(str)

    required = {"sample_id", "speaker_id", "emotion", "accent_clean", "audio_sha256"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Eligible sample_id values must be unique")
    if frame["audio_sha256"].isna().any() or frame["audio_sha256"].duplicated().any():
        raise ValueError(
            "Eligible audio hashes must be complete and unique before controlled splitting"
        )

    global_targets = largest_remainder_counts(len(frame), ratios, seed=seed, scope="global")
    speaker_counts = frame.groupby("speaker_id").size()
    controlled = {
        str(speaker)
        for speaker, count in speaker_counts.items()
        if int(count) > threshold
    }
    if special_speaker_id in speaker_counts:
        controlled.add(str(special_speaker_id))

    assignments: dict[int, str] = {}
    controlled_totals = {split: 0 for split in SPLITS}
    for speaker in sorted(controlled, key=numeric_text_key):
        subset = frame[frame["speaker_id"].eq(speaker)]
        quotas = largest_remainder_counts(
            len(subset), ratios, seed=seed, scope=f"speaker-{speaker}"
        )
        if len(subset) >= 2:
            if quotas["train"] == 0:
                quotas["train"] = 1
                quotas[max(("validation", "test"), key=quotas.get)] -= 1
            if quotas["test"] == 0:
                quotas["test"] = 1
                quotas[max(("train", "validation"), key=quotas.get)] -= 1
        allocated = allocate_stratified(
            subset,
            quotas,
            ["emotion"],
            seed=seed,
            scope=f"controlled-speaker-{speaker}",
        )
        assignments.update(allocated)
        for split in SPLITS:
            controlled_totals[split] += quotas[split]

    remaining_targets = {
        split: global_targets[split] - controlled_totals[split] for split in SPLITS
    }
    if min(remaining_targets.values()) < 0:
        raise RuntimeError(
            f"Controlled per-speaker quotas exceed global quotas: {remaining_targets}"
        )
    uncontrolled = frame[~frame["speaker_id"].isin(controlled)]
    assignments.update(
        allocate_stratified(
            uncontrolled,
            remaining_targets,
            ["accent_clean", "emotion"],
            seed=seed,
            scope="uncontrolled-global",
        )
    )

    repairs = _repair_missing_test_cells(frame, assignments, controlled, seed=seed)
    frame["original_frozen_split"] = frame.get("split", "")
    frame["original_split_version"] = frame.get("split_version", "")
    frame["split"] = [assignments[index] for index in frame.index]
    frame["split_version"] = protocol_version
    frame["split_seed"] = seed
    frame["split_protocol"] = "speaker-overlapping controlled stratified"
    frame["controlled_speaker"] = frame["speaker_id"].isin(controlled)

    source_group_column, source_group_note = detect_reliable_source_group(
        frame[frame["speaker_id"].eq(str(special_speaker_id))]
    )
    source_group_overlap = None
    if source_group_column:
        source_group_overlap = int(
            (
                frame[frame["speaker_id"].eq(str(special_speaker_id))]
                .groupby(source_group_column)["split"]
                .nunique()
                > 1
            ).sum()
        )
        if source_group_overlap:
            raise ValueError(
                f"Reliable source groups cross splits for speaker {special_speaker_id}: "
                f"{source_group_overlap}"
            )

    facts = {
        "seed": seed,
        "threshold": threshold,
        "special_speaker_id": str(special_speaker_id),
        "ratios": ratios,
        "protocol_version": protocol_version,
        "source_rows": source_rows,
        "eligible_rows": len(frame),
        "excluded_rows": int(len(excluded)),
        "exclusion_reasons": exclusion_reasons,
        "global_targets": global_targets,
        "controlled_speakers": sorted(controlled, key=numeric_text_key),
        "coverage_repairs": repairs,
        "source_group_column": source_group_column,
        "source_group_note": source_group_note,
        "source_group_overlap": source_group_overlap,
    }
    validate_controlled_split(frame, facts)
    return frame, facts


def validate_controlled_split(frame: pd.DataFrame, facts: dict[str, Any]) -> None:
    if len(frame) != frame["sample_id"].nunique():
        raise ValueError("Every utterance must appear exactly once")
    if set(frame["split"]) != set(SPLITS):
        raise ValueError(f"Unexpected split labels: {sorted(set(frame['split']))}")
    observed = frame["split"].value_counts().to_dict()
    if observed != facts["global_targets"]:
        raise ValueError(f"Global sizes differ from targets: {observed}")
    cross_hashes = int((frame.groupby("audio_sha256")["split"].nunique() > 1).sum())
    if cross_hashes:
        raise ValueError(f"Identical audio crosses partitions: {cross_hashes} hashes")
    controlled_counts = (
        frame[frame["speaker_id"].isin(facts["controlled_speakers"])]
        .groupby(["speaker_id", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SPLITS, fill_value=0)
    )
    failures = controlled_counts[(controlled_counts["train"] == 0) | (controlled_counts["test"] == 0)]
    if not failures.empty:
        raise ValueError(f"Controlled speakers missing Train or Test: {list(failures.index)}")


def speaker_report(frame: pd.DataFrame, facts: dict[str, Any]) -> pd.DataFrame:
    counts = frame.groupby(["speaker_id", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=SPLITS, fill_value=0)
    rows = []
    controlled = set(facts["controlled_speakers"])
    for speaker, values in counts.iterrows():
        total = int(values.sum())
        rows.append(
            {
                "speaker_id": str(speaker),
                "utterances": total,
                "controlled": str(speaker) in controlled,
                "special_speaker": str(speaker) == facts["special_speaker_id"],
                "train_count": int(values["train"]),
                "validation_count": int(values["validation"]),
                "test_count": int(values["test"]),
                "train_pct": round(100 * values["train"] / total, 4),
                "validation_pct": round(100 * values["validation"] / total, 4),
                "test_pct": round(100 * values["test"] / total, 4),
                "partitions_present": int((values > 0).sum()),
                "train_test_overlap_identity": bool(values["train"] and values["test"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "speaker_id", key=lambda series: series.map(numeric_text_key)
    )


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = [str(value) for value in headers]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_audit(
    frame: pd.DataFrame,
    facts: dict[str, Any],
    source_path: Path,
    source_hash: str,
) -> str:
    sizes = frame["split"].value_counts().reindex(SPLITS)
    speakers = {split: set(frame.loc[frame["split"].eq(split), "speaker_id"]) for split in SPLITS}
    all_speakers = set(frame["speaker_id"])
    overlap_counts = [
        ("Train / Validation", len(speakers["train"] & speakers["validation"])),
        ("Train / Test", len(speakers["train"] & speakers["test"])),
        ("Validation / Test", len(speakers["validation"] & speakers["test"])),
        ("All three", len(speakers["train"] & speakers["validation"] & speakers["test"])),
    ]
    overlap_rows = [
        (label, count, f"{100 * count / len(all_speakers):.3f}%")
        for label, count in overlap_counts
    ]
    emotion = frame.groupby(["split", "emotion"]).size().unstack(fill_value=0).reindex(index=SPLITS, columns=EMOTIONS, fill_value=0)
    accent = frame.groupby(["split", "accent_clean"]).size().unstack(fill_value=0).reindex(index=SPLITS, columns=ACCENTS, fill_value=0)
    cross = frame.groupby(["split", "accent_clean", "emotion"]).size()
    missing = [
        (split, accent_name, emotion_name)
        for split in SPLITS
        for accent_name in ACCENTS
        for emotion_name in EMOTIONS
        if int(cross.get((split, accent_name, emotion_name), 0)) == 0
    ]
    duplicate_cross = int((frame.groupby("audio_sha256")["split"].nunique() > 1).sum())
    report = speaker_report(frame, facts)
    controlled_report = report[report["controlled"]]
    special = report[report["speaker_id"].eq(facts["special_speaker_id"])].iloc[0]
    special_frame = frame[frame["speaker_id"].eq(facts["special_speaker_id"])]
    special_emotion = special_frame.groupby(["split", "emotion"]).size().unstack(fill_value=0).reindex(index=SPLITS, columns=EMOTIONS, fill_value=0)
    special_sources = special_frame["source_name"].value_counts(dropna=False).to_dict() if "source_name" in special_frame else {}
    special_releases = special_frame["source_release_id"].nunique(dropna=False) if "source_release_id" in special_frame else 0
    special_shards = (
        special_frame["audio_locator"].astype(str).str.split("#row=", n=1).str[0].value_counts().to_dict()
        if "audio_locator" in special_frame
        else {}
    )

    lines = [
        "# Controlled Speaker-Overlapping Split Audit",
        "",
        "> This experiment uses a speaker-overlapping controlled stratified split. It is not a speaker-independent evaluation.",
        "",
        "## Configuration",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        f"- Source manifest: `{source_path.as_posix()}`",
        f"- Source manifest SHA-256: `{source_hash}`",
        f"- Seed: `{facts['seed']}`",
        f"- Controlled-speaker rule: more than `{facts['threshold']}` eligible utterances",
        f"- Special speaker ID: `{facts['special_speaker_id']}`",
        "- Requested ratios: Train 70%, Validation 15%, Test 15%",
        "- Allocation inputs: speaker ID, emotion, accent, stable sample ID, and audio SHA-256 only",
        "- Model predictions, losses, F1 scores, and test performance were not loaded or used.",
        "",
        "The frozen source split is unchanged. The new allocation is written only to separate controlled-split files.",
        "",
        "## Source eligibility and exclusions",
        "",
        f"- Source rows: {facts['source_rows']:,}",
        f"- Eligible rows allocated by this pipeline: {facts['eligible_rows']:,}",
        f"- Rows already marked ineligible by the frozen cleaning manifest: {facts['excluded_rows']:,}",
        "- Newly removed by this controlled splitter: 0",
        "",
        markdown_table(
            ("Frozen exclusion reason", "Rows", "Meaning"),
            (
                (
                    reason,
                    count,
                    "redundant exact-audio copy; the lower source-row representative is retained"
                    if reason == "identical_audio_represented_by_lower_source_row"
                    else "exact audio has conflicting metadata; excluded conservatively"
                    if reason == "identical_audio_has_conflicting_metadata"
                    else "reason recorded by the frozen cleaning manifest",
                )
                for reason, count in facts["exclusion_reasons"].items()
            ),
        ),
        "",
        "The 268 rows in the first category are duplicate copies, not 268 additional duplicate groups. This splitter does not delete them again; it consumes only the 4,992 rows already marked eligible.",
        "",
        "## Global split sizes",
        "",
        markdown_table(
            ("Split", "Utterances", "Percentage", "Speakers"),
            (
                (split.title(), int(sizes[split]), f"{100 * sizes[split] / len(frame):.3f}%", len(speakers[split]))
                for split in SPLITS
            ),
        ),
        "",
        "## Speaker control and overlap",
        "",
        f"- Eligible speakers: {len(all_speakers)}",
        f"- Controlled speakers (> {facts['threshold']} utterances): {len(facts['controlled_speakers'])}",
        f"- Controlled speakers with both Train and Test samples: {int(controlled_report['train_test_overlap_identity'].sum())}/{len(controlled_report)}",
        "",
        markdown_table(("Identity intersection", "Speakers", f"% of {len(all_speakers)} eligible speakers"), overlap_rows),
        "",
        "Speaker overlap is intentional. Utterance overlap is prohibited.",
        "",
        "## Emotion distribution",
        "",
        markdown_table(("Split", *EMOTIONS), ((split.title(), *map(int, emotion.loc[split])) for split in SPLITS)),
        "",
        "## Accent distribution",
        "",
        markdown_table(("Split", *ACCENTS), ((split.title(), *map(int, accent.loc[split])) for split in SPLITS)),
        "",
        "## Accent x emotion coverage",
        "",
        f"- Missing cells across all splits: {len(missing)}",
    ]
    if missing:
        lines.extend(f"- `{split} / {accent_name} / {emotion_name}`" for split, accent_name, emotion_name in missing)
    else:
        lines.append("- No missing accent x emotion cells.")
    lines.extend(
        [
            "",
            "## Speaker ID 0",
            "",
            "Speaker ID 0 is treated as a released metadata identifier, not as a verified physical identity.",
            "",
            markdown_table(
                ("Split", "Count", "Percentage of Speaker 0"),
                (
                    (
                        split.title(),
                        int(special[f"{split}_count"]),
                        f"{special[f'{split}_pct']:.3f}%",
                    )
                    for split in SPLITS
                ),
            ),
            "",
            "Speaker 0 emotion distribution:",
            "",
            markdown_table(("Split", *EMOTIONS), ((split.title(), *map(int, special_emotion.loc[split])) for split in SPLITS)),
            "",
            f"- Accent: `{special_frame['accent_clean'].value_counts().to_dict()}`",
            f"- Released source distribution: `{special_sources}`; release IDs represented: {special_releases}",
            f"- Storage-shard distribution: `{special_shards}`",
            f"- Source-group inspection: {facts['source_group_note']}.",
            "- The released manifest does not expose a reliable YouTube video or recording-session mapping for Speaker 0.",
            "- Numeric WAV filenames and parquet row positions were not treated as sessions because that would be an unsupported inference.",
            "",
            "## Integrity checks",
            "",
            markdown_table(
                ("Check", "Result"),
                (
                    ("Every eligible utterance appears exactly once", f"PASS ({len(frame):,}/{len(frame):,})"),
                    ("Identical audio hashes crossing partitions", f"PASS ({duplicate_cross})"),
                    ("Controlled speakers represented in Train and Test", f"PASS ({len(controlled_report)}/{len(controlled_report)})"),
                    ("Global sizes match deterministic quotas", "PASS"),
                    ("Frozen split modified", "PASS (No)"),
                    ("Model performance used for allocation", "PASS (No)"),
                ),
            ),
            "",
            "## Method summary",
            "",
            "Each controlled speaker is allocated separately with deterministic emotion stratification and approximately 70/15/15 quotas. The remaining speakers are allocated at utterance level using deterministic accent x emotion stratification to meet the exact global quotas. A final metadata-only swap can repair an empty Test accent x emotion cell without changing split sizes. Exact audio hashes are required to be unique before allocation and are checked again after allocation.",
            "",
            f"Coverage repairs applied: `{json.dumps(facts['coverage_repairs'], ensure_ascii=False)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/clean_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "splits")
    parser.add_argument("--speaker-report", type=Path, default=root / "splits/controlled_speaker_report.csv")
    parser.add_argument("--audit", type=Path, default=root / "docs/controlled_split_audit.md")
    parser.add_argument("--threshold", type=int, default=SPEAKER_THRESHOLD)
    parser.add_argument("--special-speaker", default=SPECIAL_SPEAKER_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    source_hash_before = sha256_file(source_path)
    source = pd.read_csv(source_path, dtype={"speaker_id": "string"})
    controlled, facts = build_controlled_split(
        source,
        threshold=args.threshold,
        special_speaker_id=str(args.special_speaker),
        seed=args.seed,
    )
    output_dir = args.output_dir.resolve()
    for split, filename in (
        ("train", "controlled_train.csv"),
        ("validation", "controlled_validation.csv"),
        ("test", "controlled_test.csv"),
    ):
        write_csv_atomic(controlled[controlled["split"].eq(split)], output_dir / filename)
    write_csv_atomic(speaker_report(controlled, facts), args.speaker_report.resolve())
    audit = build_audit(controlled, facts, source_path, source_hash_before)
    write_text_atomic(audit, args.audit.resolve())
    source_hash_after = sha256_file(source_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Frozen source manifest changed during controlled split generation")
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": PROTOCOL_VERSION,
                "rows": len(controlled),
                "sizes": controlled["split"].value_counts().reindex(SPLITS).to_dict(),
                "controlled_speakers": len(facts["controlled_speakers"]),
                "source_manifest_unchanged": True,
                "model_performance_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
