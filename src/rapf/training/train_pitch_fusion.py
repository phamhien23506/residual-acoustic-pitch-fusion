"""Train the official Pitch-fusion architecture on controlled ViSEC splits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from rapf.experiment_utils import (
    classification_metrics,
    load_b6_table,
    load_config,
    prediction_rows,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from rapf.models.pitch_fusion import OfficialWav2Vec2CrossAttentionPitchForSER

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_NAME = "pitch_fusion_paper_aligned"
OFFICIAL_LABELS = ["happy", "neutral", "sad", "angry"]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_wav(value: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(value), "rb") as stream:
        if stream.getframerate() != 16000 or stream.getnchannels() != 1:
            raise ValueError("Official Pitch-fusion requires mono 16 kHz PCM audio")
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if width == 3:
        source = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        packed = (
            source[:, 0].astype(np.int32)
            | source[:, 1].astype(np.int32) << 8
            | source[:, 2].astype(np.int32) << 16
        )
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        return packed.astype(np.float32) / 8388608.0
    raise ValueError(f"Unsupported PCM sample width: {width}")


def interpolate_official_pitch(path: Path, waveform_length: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        pitch = np.asarray(stored["f0_hz"], dtype=np.float32).reshape(-1)
    pitch = np.nan_to_num(pitch)
    if pitch.size == 0:
        return np.zeros(waveform_length, dtype=np.float32)
    return np.interp(
        np.linspace(0, waveform_length, waveform_length),
        np.linspace(0, pitch.size, pitch.size),
        pitch,
    ).astype(np.float32)


class OfficialPitchFusionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        audio_bytes: list[bytes],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.audio_bytes = audio_bytes
        # WAV decoding and full-length pitch interpolation are deterministic but
        # comparatively expensive.  With num_workers=0 (required by the Windows
        # in-memory Parquet loader), doing both again on every epoch wastes CPU
        # time and keeps the laptop hot while the GPU waits.  Cache the exact
        # float32 arrays after their first use.  This is a runtime-only
        # optimization: it does not alter examples, ordering, padding, seeds, or
        # any model/optimizer operation.
        self._decoded_cache: list[tuple[np.ndarray, np.ndarray, int] | None] = [
            None
        ] * len(self.frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        cached = self._decoded_cache[index]
        if cached is not None:
            return cached
        row = self.frame.iloc[index]
        audio = decode_wav(self.audio_bytes[int(row["source_row_index"])])
        pitch = interpolate_official_pitch(
            resolve(str(row["pitch_artifact_path"])), len(audio)
        )
        cached = (audio, pitch, int(row["emotion_id"]))
        self._decoded_cache[index] = cached
        return cached


def collate(batch):
    maximum = max(len(row[0]) for row in batch)
    audio = torch.zeros(len(batch), maximum, dtype=torch.float32)
    pitch = torch.zeros(len(batch), maximum, dtype=torch.float32)
    labels = torch.empty(len(batch), dtype=torch.long)
    for index, (audio_value, pitch_value, label) in enumerate(batch):
        length = len(audio_value)
        audio[index, :length] = torch.from_numpy(audio_value.copy())
        pitch[index, :length] = torch.from_numpy(pitch_value.copy())
        labels[index] = label
    return audio, pitch, labels


def build_model(
    config: dict[str, Any], local_files_only: bool
) -> OfficialWav2Vec2CrossAttentionPitchForSER:
    snapshot = resolve(config["backbone"]["local_snapshot"])
    source = str(snapshot) if snapshot.is_dir() else config["backbone"]["model_id"]
    model = OfficialWav2Vec2CrossAttentionPitchForSER.from_pretrained(
        source,
        revision=None if snapshot.is_dir() else config["backbone"]["revision"],
        local_files_only=local_files_only,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        final_dropout=0.1,
        mask_time_prob=0.05,
        layerdrop=0.1,
        num_labels=4,
        classifier_proj_size=256,
    )
    # Transformers 4.57 materializes direct MultiheadAttention parameters that
    # are absent from the backbone checkpoint as zeros. The upstream code was
    # written against 4.30 and relies on PyTorch's Xavier initialization.
    for block in (
        model.attention_audio_pitch,
        model.attention_pitch_audio,
        model.self_attention,
    ):
        weight = block.self_attention.in_proj_weight
        if torch.count_nonzero(weight).item() == 0:
            nn.init.xavier_uniform_(weight)
        if block.self_attention.in_proj_bias is not None:
            nn.init.zeros_(block.self_attention.in_proj_bias)
    if config["backbone"]["freeze_feature_encoder"]:
        model.freeze_feature_extractor()
    return model


def precision_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "fp16" if value else "none"
    mode = str(value).lower()
    if mode not in {"none", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed precision mode: {value!r}")
    return mode


def autocast_options(device: torch.device, precision: str) -> dict[str, Any]:
    return {
        "device_type": device.type,
        "enabled": device.type == "cuda" and precision != "none",
        "dtype": torch.bfloat16 if precision == "bf16" else torch.float16,
    }


def evaluate(model, loader, device, precision: str):
    model.eval()
    predictions, probabilities = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for audio, pitch, labels in loader:
            audio, pitch, labels = (
                value.to(device, non_blocking=True)
                for value in (audio, pitch, labels)
            )
            with torch.autocast(**autocast_options(device, precision)):
                logits = model(audio, pitch).logits
                loss = criterion(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(probabilities),
        total_loss / len(loader.dataset),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/pitch_fusion.json"
    )
    parser.add_argument(
        "--features",
        default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet",
    )
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=[
            "splits/controlled_train.csv",
            "splits/controlled_validation.csv",
            "splits/controlled_test.csv",
        ],
    )
    parser.add_argument(
        "--raw-parquet", default="data/raw/visec_hf/train-00000-of-00001.parquet"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train/select/export with Validation only; do not construct or evaluate a Test loader.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--run-name-override")
    parser.add_argument(
        "--recover-export-after-write-failure",
        action="store_true",
        help="Reuse a completed checkpoint and transparently record the extra test evaluation.",
    )
    return parser.parse_args()


def lineage_signature(
    config_path: Path,
    features_path: Path,
    manifest_paths: list[Path],
    raw_path: Path,
    runtime_overrides: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    lineage = {
        "feature_sha256": sha256_file(features_path),
        "manifest_sha256": {str(path): sha256_file(path) for path in manifest_paths},
        "raw_parquet_sha256": sha256_file(raw_path),
        "config_sha256": sha256_file(config_path),
        "upstream_commit": "2e8e05f940fce265f2d2090556b1b6d7e84e7368",
    }
    if runtime_overrides:
        lineage["runtime_overrides"] = runtime_overrides
    encoded = json.dumps(lineage, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), lineage


def main() -> None:
    args = parse_args()
    config_path, features_path, raw_path = map(
        resolve, (args.config, args.features, args.raw_parquet)
    )
    manifest_paths = [resolve(path) for path in args.manifest]
    config = load_config(config_path)
    if (args.seed_override is None) != (args.run_name_override is None):
        raise ValueError("Seed and run-name overrides must be provided together")
    runtime_overrides = None
    if args.seed_override is not None:
        runtime_overrides = {
            "seed": int(args.seed_override),
            "run_name": str(args.run_name_override),
        }
        config["seed"] = runtime_overrides["seed"]
        config["run_name"] = runtime_overrides["run_name"]
        config["models"] = {runtime_overrides["run_name"]: {}}
        config["report_dir"] = str(
            ROOT / "artifacts/b6" / runtime_overrides["run_name"] / "evaluation"
        )
    run_name = str(config.get("run_name", DEFAULT_MODEL_NAME))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError(f"Unsafe run_name: {run_name!r}")
    frame, _ = load_b6_table(
        features_path,
        manifest_paths,
        config,
        require_grouped_cv_folds=bool(config.get("require_grouped_cv_folds", False)),
    )
    observed_sizes = frame["split"].value_counts().to_dict()
    if observed_sizes != config["expected_split_sizes"]:
        raise ValueError(
            f"Unexpected split sizes: observed={observed_sizes}, "
            f"expected={config['expected_split_sizes']}"
        )
    observed_labels = {
        (str(emotion), int(emotion_id))
        for emotion, emotion_id in frame[["emotion", "emotion_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    expected_labels = {(emotion, index) for index, emotion in enumerate(OFFICIAL_LABELS)}
    if observed_labels != expected_labels:
        raise ValueError(
            f"Official emotion ID mapping mismatch: {sorted(observed_labels)}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    audio_column = pq.read_table(raw_path, columns=["path"])["path"].to_pylist()
    audio_bytes = [row["bytes"] for row in audio_column]
    if int(frame["source_row_index"].max()) >= len(audio_bytes):
        raise ValueError("Manifest source_row_index exceeds raw parquet")

    if args.preflight_only:
        longest = sorted(
            frame.index,
            key=lambda index: len(audio_bytes[int(frame.at[index, "source_row_index"])]),
            reverse=True,
        )[:1]
        cache_frame = frame.loc[longest]
    else:
        cache_frame = frame

    seed_everything(int(config["seed"]))
    model = build_model(config, local_files_only=not args.allow_download)
    parameters = sum(value.numel() for value in model.parameters())
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    precision = precision_mode(config["training"]["mixed_precision"])
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "run_name": run_name,
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
                "test_loaded_for_training": False,
                "parameters": parameters,
                "upstream_commit": config["upstream"]["commit"],
            }
        ),
        flush=True,
    )
    if args.preflight_only:
        probe = OfficialPitchFusionDataset(cache_frame, audio_bytes)
        audio, pitch, labels = collate([probe[index] for index in range(len(probe))])
        model.to(device).train()
        with torch.autocast(**autocast_options(device, precision)):
            output = model(audio.to(device), pitch.to(device), labels.to(device))
        output.loss.backward()
        if tuple(output.logits.shape) != (len(probe), 4):
            raise RuntimeError(f"Invalid preflight logits: {tuple(output.logits.shape)}")
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise RuntimeError("Preflight produced a non-finite gradient")
        print(
            json.dumps(
                {
                    "forward_backward_smoke_test": "PASS",
                    "logits_shape": list(output.logits.shape),
                    "longest_batch_samples": int(audio.shape[1]),
                    "gradients_finite": True,
                    "precision": precision,
                }
            ),
            flush=True,
        )
        return

    training = config["training"]
    options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate,
    }
    train_dataset = OfficialPitchFusionDataset(train, audio_bytes)
    validation_loader = DataLoader(
        OfficialPitchFusionDataset(validation, audio_bytes),
        shuffle=False,
        **options,
    )
    test_loader = None
    if not args.skip_test:
        test_loader = DataLoader(
            OfficialPitchFusionDataset(test, audio_bytes),
            shuffle=False,
            **options,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    batches_per_epoch = math.ceil(len(train_dataset) / options["batch_size"])
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = steps_per_epoch * int(training["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training["warmup_steps"]),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
        init_scale=float(training.get("grad_scaler_init_scale", 65536.0)),
    )
    artifact_dir = ROOT / "artifacts/b6" / run_name
    report_dir = resolve(
        config.get("report_dir", str(ROOT / "reports/tables/b6"))
    )
    checkpoint_path = artifact_dir / "checkpoint.pt"
    best_model_path = artifact_dir / "best_model_state.pt"
    model_path = artifact_dir / f"{run_name}.pt"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    signature, lineage = lineage_signature(
        config_path, features_path, manifest_paths, raw_path, runtime_overrides
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    elapsed_before = 0.0
    skipped_optimizer_steps = 0
    early_stopping = config["early_stopping"]
    if early_stopping["metric"] != "validation_balanced_accuracy":
        raise ValueError("Only validation_balanced_accuracy early stopping is supported")
    if early_stopping["mode"] != "max":
        raise ValueError("Validation balanced accuracy must be maximized")
    patience = int(early_stopping["patience"])
    min_delta = float(early_stopping["min_delta"])
    if patience < 1 or min_delta < 0:
        raise ValueError("Early-stopping patience must be positive and min_delta nonnegative")
    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    if args.recover_export_after_write_failure and args.force:
        raise ValueError("Export recovery cannot be combined with --force")
    if args.recover_export_after_write_failure and not checkpoint_path.exists():
        raise RuntimeError("Export recovery requires an existing completed checkpoint")
    if checkpoint_path.exists() and not args.force:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("lineage_signature") != signature:
            raise RuntimeError("Checkpoint lineage does not match this controlled run")
        if not args.recover_export_after_write_failure:
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            scheduler.load_state_dict(saved["scheduler_state"])
            scaler.load_state_dict(saved["scaler_state"])
        history = saved["history"]
        start_epoch = int(saved["epoch"]) + 1
        elapsed_before = float(saved.get("runtime_seconds", 0.0))
        skipped_optimizer_steps = int(saved.get("skipped_optimizer_steps", 0))
        best_score = float(saved["best_validation_balanced_accuracy"])
        best_epoch = int(saved["best_epoch"])
        epochs_without_improvement = int(saved["epochs_without_improvement"])
        if not best_model_path.exists():
            raise RuntimeError("Best-model checkpoint is missing for this resumed run")
        print(f"resume checkpoint: epoch {saved['epoch']}", flush=True)
        del saved
        gc.collect()

    started = time.perf_counter()
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        if epochs_without_improvement >= patience:
            break
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **options
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (audio, pitch, labels) in enumerate(train_loader, 1):
            audio, pitch, labels = (
                value.to(device, non_blocking=True)
                for value in (audio, pitch, labels)
            )
            with torch.autocast(**autocast_options(device, precision)):
                loss = model(audio, pitch, labels).loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                else:
                    skipped_optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * accumulation * len(labels)
        predicted, _, validation_loss = evaluate(
            model, validation_loader, device, precision
        )
        validation_balanced_accuracy = float(
            balanced_accuracy_score(validation["emotion_id"].to_numpy(), predicted)
        )
        predicted_text = np.asarray(OFFICIAL_LABELS)[predicted]
        validation_macro_f1 = classification_metrics(
            validation["emotion"].to_numpy(),
            predicted_text,
            validation["accent_clean"].to_numpy(),
            config["emotion_order"],
        )["overall"]["macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train),
                "validation_loss": validation_loss,
                "validation_balanced_accuracy": validation_balanced_accuracy,
                "validation_macro_f1": validation_macro_f1,
            }
        )
        improved = validation_balanced_accuracy > best_score + min_delta
        if improved:
            best_score = validation_balanced_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            best_temporary = best_model_path.with_suffix(".tmp")
            torch.save(
                {
                    "epoch": best_epoch,
                    "validation_balanced_accuracy": best_score,
                    "model_state": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "lineage_signature": signature,
                },
                best_temporary,
            )
            os.replace(best_temporary, best_model_path)
        else:
            epochs_without_improvement += 1
        runtime_seconds = elapsed_before + time.perf_counter() - started
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "runtime_seconds": runtime_seconds,
                "skipped_optimizer_steps": skipped_optimizer_steps,
                "best_validation_balanced_accuracy": best_score,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "lineage_signature": signature,
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        print(
            f"{run_name} epoch {epoch}/{training['epochs']} "
            f"train_loss={total_loss / len(train):.4f} "
            f"val_bal_acc={validation_balanced_accuracy:.4f} "
            f"val_macro_f1={validation_macro_f1:.4f} "
            f"best_epoch={best_epoch} "
            f"early_stop={epochs_without_improvement}/{patience} "
            f"skipped_steps={skipped_optimizer_steps}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            print(
                f"early stopping at epoch {epoch}; restoring epoch {best_epoch} "
                f"(validation balanced accuracy={best_score:.4f})",
                flush=True,
            )
            break

    if not history or best_epoch < 1 or not best_model_path.exists():
        raise RuntimeError("Training produced no restorable best checkpoint")
    best_saved = torch.load(best_model_path, map_location="cpu", weights_only=False)
    if best_saved.get("lineage_signature") != signature:
        raise RuntimeError("Best-model lineage does not match this controlled run")
    model.load_state_dict(best_saved["model_state"])
    del best_saved
    gc.collect()
    model.to(device)
    model.eval()
    validation_predicted, validation_probabilities, _ = evaluate(
        model, validation_loader, device, precision
    )
    validation_text = np.asarray(OFFICIAL_LABELS)[validation_predicted]
    validation_metrics = classification_metrics(
        validation["emotion"].to_numpy(),
        validation_text,
        validation["accent_clean"].to_numpy(),
        config["emotion_order"],
    )
    write_csv_atomic(
        prediction_rows(
            validation, validation_text, validation_probabilities, OFFICIAL_LABELS
        ),
        report_dir / f"{run_name}_validation_predictions.csv",
    )
    cpu_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state": cpu_state,
            "official_labels_by_id": OFFICIAL_LABELS,
            "config": config,
            "epoch": best_epoch,
            "stopped_epoch": int(history[-1]["epoch"]),
            "validation_balanced_accuracy": best_score,
        },
        model_path,
    )
    if args.skip_test:
        report = {
            "schema_version": config["schema_version"],
            "model": run_name,
            "display_name": config["display_name"],
            "protocol": config.get("evaluation_protocol"),
            "warning": config.get("evaluation_warning"),
            "upstream": config["upstream"],
            "reproduction_boundary": config["reproduction_boundary"],
            "model_selection": "best validation balanced accuracy with configured early stopping",
            "early_stopping": early_stopping,
            "configured_max_epochs": int(training["epochs"]),
            "stopped_epoch": int(history[-1]["epoch"]),
            "exported_epoch": best_epoch,
            "best_validation_epoch_observed": best_epoch,
            "best_validation_balanced_accuracy": best_score,
            "validation": validation_metrics,
            "test": None,
            "history": history,
            "test_loaded_or_evaluated": False,
            "test_used_for_model_selection": False,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "lineage": lineage,
            "runtime_seconds": elapsed_before + time.perf_counter() - started,
            "device": str(device),
            "runtime_overrides": runtime_overrides,
        }
        write_json_atomic(report, report_dir / f"{run_name}_report.json")
        print(
            json.dumps(
                {
                    "status": "PASS_VALIDATION_ONLY",
                    "validation_macro_f1": validation_metrics["overall"]["macro_f1"],
                    "validation_uar": validation_metrics["overall"]["uar"],
                    "exported_epoch": best_epoch,
                    "stopped_epoch": int(history[-1]["epoch"]),
                    "test_loaded_or_evaluated": False,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    print("validation inference and export: PASS; starting test", flush=True)
    assert test_loader is not None
    test_predicted, test_probabilities, _ = evaluate(
        model, test_loader, device, precision
    )
    test_text = np.asarray(OFFICIAL_LABELS)[test_predicted]
    test_metrics = classification_metrics(
        test["emotion"].to_numpy(),
        test_text,
        test["accent_clean"].to_numpy(),
        config["emotion_order"],
    )
    write_csv_atomic(
        prediction_rows(test, test_text, test_probabilities, OFFICIAL_LABELS),
        report_dir / f"{run_name}_test_predictions.csv",
    )
    report = {
        "schema_version": config["schema_version"],
        "model": run_name,
        "display_name": config["display_name"],
        "protocol": config.get(
            "evaluation_protocol", "controlled speaker-overlapping 70/15/15"
        ),
        "warning": config.get(
            "evaluation_warning",
            "This is a speaker-overlapping in-distribution evaluation, not a "
            "speaker-independent evaluation.",
        ),
        "upstream": config["upstream"],
        "reproduction_boundary": config["reproduction_boundary"],
        "model_selection": (
            "best validation balanced accuracy with configured early stopping; "
            "test was not used for model selection"
        ),
        "early_stopping": early_stopping,
        "configured_max_epochs": int(training["epochs"]),
        "stopped_epoch": int(history[-1]["epoch"]),
        "exported_epoch": best_epoch,
        "best_validation_epoch_observed": best_epoch,
        "best_validation_balanced_accuracy": best_score,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "test_evaluated_once_after_training": (
            not args.recover_export_after_write_failure
        ),
        "test_evaluation_attempts": (
            4 if args.recover_export_after_write_failure else 1
        ),
        "test_used_for_model_selection": False,
        "export_recovery": (
            "The first post-training evaluation completed, but CSV export failed "
            "before results were persisted. A first recovery attempt was stopped because "
            "retained optimizer state caused severe paging. A second recovery was aborted "
            "with its PTY before validation export. The final background export loaded the "
            "unchanged best checkpoint without optimizer state. No training, selection, "
            "or tuning occurred during any recovery."
            if args.recover_export_after_write_failure
            else None
        ),
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "lineage": lineage,
        "runtime_seconds": elapsed_before + time.perf_counter() - started,
        "device": str(device),
        "runtime_overrides": runtime_overrides,
    }
    write_json_atomic(report, report_dir / f"{run_name}_report.json")
    print(
        json.dumps(
            {
                "status": "PASS",
                "validation_macro_f1": validation_metrics["overall"]["macro_f1"],
                "test_macro_f1": test_metrics["overall"]["macro_f1"],
                "exported_epoch": best_epoch,
                "stopped_epoch": int(history[-1]["epoch"]),
                "test_evaluated_once_after_training": (
                    not args.recover_export_after_write_failure
                ),
                "test_evaluation_attempts": (
                    4 if args.recover_export_after_write_failure else 1
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
