from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_published_splits_are_disjoint_and_complete() -> None:
    paths = {
        "train": ROOT / "splits/controlled_train.csv",
        "validation": ROOT / "splits/controlled_validation.csv",
        "test": ROOT / "splits/controlled_test.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}

    assert {name: len(frame) for name, frame in frames.items()} == {
        "train": 3494,
        "validation": 749,
        "test": 749,
    }
    sample_sets = {name: set(frame["sample_id"]) for name, frame in frames.items()}
    hash_sets = {name: set(frame["audio_sha256"]) for name, frame in frames.items()}
    assert sample_sets["train"].isdisjoint(sample_sets["validation"])
    assert sample_sets["train"].isdisjoint(sample_sets["test"])
    assert sample_sets["validation"].isdisjoint(sample_sets["test"])
    assert hash_sets["train"].isdisjoint(hash_sets["validation"])
    assert hash_sets["train"].isdisjoint(hash_sets["test"])
    assert hash_sets["validation"].isdisjoint(hash_sets["test"])


def test_speaker_zero_allocation_matches_audit() -> None:
    expected = {"train": 1552, "validation": 333, "test": 332}
    filenames = {
        "train": "controlled_train.csv",
        "validation": "controlled_validation.csv",
        "test": "controlled_test.csv",
    }
    observed = {}
    for split, filename in filenames.items():
        frame = pd.read_csv(ROOT / "splits" / filename, dtype={"speaker_id": "string"})
        observed[split] = int(frame["speaker_id"].eq("0").sum())
    assert observed == expected
