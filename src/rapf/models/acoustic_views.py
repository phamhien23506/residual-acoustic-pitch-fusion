"""Cached acoustic views, encoders, and reusable training utilities."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
EMOTIONS = ("happy", "neutral", "sad", "angry")
EMOTION_TO_ID = {emotion: index for index, emotion in enumerate(EMOTIONS)}


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 2025
    train_samples: int = 1200
    validation_samples: int = 400
    max_frames: int = 300
    batch_size: int = 24
    max_epochs: int = 12
    patience: int = 3
    min_delta: float = 0.002
    overfit_gap: float = 0.15
    overfit_patience: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    hidden_size: int = 128
    branch_dropout: float = 0.15
    dropout: float = 0.25
    auxiliary_loss_weight: float = 0.15


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sample_stratified(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size >= len(frame):
        return frame.sort_values("sample_id").reset_index(drop=True)
    selected, _ = train_test_split(
        frame,
        train_size=size,
        random_state=seed,
        shuffle=True,
        stratify=frame["emotion"],
    )
    return selected.sort_values("sample_id").reset_index(drop=True)


def load_pilot_frames(config: PilotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = ROOT / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet"
    features = pd.read_parquet(feature_path)
    wav_columns = [f"wav2vec2_{index:03d}" for index in range(768)]
    required = {
        "sample_id",
        "emotion",
        "log_mel_artifact_path",
        "pitch_artifact_path",
        *wav_columns,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"B4 feature table is missing columns: {missing[:10]}")
    features = features[
        ["sample_id", "emotion", "log_mel_artifact_path", "pitch_artifact_path", *wav_columns]
    ].copy()

    def merge_manifest(filename: str) -> pd.DataFrame:
        manifest = pd.read_csv(ROOT / "data/manifests" / filename, usecols=["sample_id", "emotion"])
        merged = manifest[["sample_id", "emotion"]].merge(
            features.drop(columns="emotion"), on="sample_id", how="left", validate="one_to_one"
        )
        if merged[["log_mel_artifact_path", "pitch_artifact_path"]].isna().any().any():
            raise ValueError(f"Missing cached features after joining {filename}")
        return merged

    # Intentionally no controlled_test.csv read in this pilot.
    train = sample_stratified(
        merge_manifest("controlled_train.csv"), config.train_samples, config.seed
    )
    validation = sample_stratified(
        merge_manifest("controlled_val.csv"), config.validation_samples, config.seed + 1
    )
    return train, validation


def delta(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if values.shape[-1] > 1:
        result[..., 1:-1] = (values[..., 2:] - values[..., :-2]) * 0.5
        result[..., 0] = values[..., 1] - values[..., 0]
        result[..., -1] = values[..., -1] - values[..., -2]
    return result


def crop_or_pad(values: np.ndarray, frames: int) -> tuple[np.ndarray, int]:
    length = min(values.shape[-1], frames)
    if values.shape[-1] > frames:
        start = (values.shape[-1] - frames) // 2
        values = values[..., start : start + frames]
    if values.shape[-1] < frames:
        padding = [(0, 0)] * values.ndim
        padding[-1] = (0, frames - values.shape[-1])
        values = np.pad(values, padding, mode="constant")
    return np.ascontiguousarray(values, dtype=np.float32), length


class CachedMultiViewDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_frames: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.max_frames = max_frames
        self.wav_columns = [f"wav2vec2_{index:03d}" for index in range(768)]
        self.wav = self.frame[self.wav_columns].to_numpy(dtype=np.float32, copy=True)
        self.labels = self.frame["emotion"].map(EMOTION_TO_ID).to_numpy(dtype=np.int64)
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _load_auxiliary(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        cached = self.cache.get(index)
        if cached is not None:
            return cached
        row = self.frame.iloc[index]
        mel = np.load(resolve(row["log_mel_artifact_path"]), allow_pickle=False).astype(
            np.float32, copy=False
        )
        pitch_file = np.load(resolve(row["pitch_artifact_path"]), allow_pickle=False)
        try:
            f0 = np.asarray(pitch_file["f0_hz"], dtype=np.float32)
            voiced = np.asarray(pitch_file["voiced_flag"], dtype=np.float32)
            probability = np.asarray(pitch_file["voiced_probability"], dtype=np.float32)
        finally:
            pitch_file.close()
        width = min(mel.shape[-1], f0.shape[-1], voiced.shape[-1], probability.shape[-1])
        mel = mel[:, :width]
        f0, voiced, probability = f0[:width], voiced[:width], probability[:width]

        log_f0 = np.log(np.clip(f0, 1.0, None))
        voiced_mask = voiced > 0.5
        if voiced_mask.any():
            center = float(log_f0[voiced_mask].mean())
            scale = float(log_f0[voiced_mask].std())
            log_f0 = (log_f0 - center) / max(scale, 1e-4)
        else:
            log_f0 = np.zeros_like(log_f0)
        log_f0 = np.where(voiced_mask, log_f0, 0.0).astype(np.float32)

        frame_energy = mel.mean(axis=0).astype(np.float32)
        frame_energy = (frame_energy - frame_energy.mean()) / max(float(frame_energy.std()), 1e-4)
        prosody = np.stack(
            [
                log_f0,
                delta(log_f0),
                delta(delta(log_f0)),
                frame_energy,
                delta(frame_energy),
                probability,
            ],
            axis=0,
        )

        mel_mean = mel.mean(axis=1, keepdims=True)
        mel_std = mel.std(axis=1, keepdims=True)
        mel_norm = (mel - mel_mean) / np.maximum(mel_std, 1e-4)
        spectral = np.stack([mel_norm, delta(mel_norm), delta(delta(mel_norm))], axis=0)
        prosody, length = crop_or_pad(prosody, self.max_frames)
        spectral, spectral_length = crop_or_pad(spectral, self.max_frames)
        length = min(length, spectral_length)
        quality = np.asarray(
            [
                float(voiced_mask.mean()),
                float(probability[voiced_mask].mean()) if voiced_mask.any() else 0.0,
                min(width / self.max_frames, 1.0),
            ],
            dtype=np.float32,
        )
        result = prosody, spectral, quality, length
        self.cache[index] = result
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        prosody, spectral, quality, length = self._load_auxiliary(index)
        return {
            "sample_id": str(self.frame.iloc[index]["sample_id"]),
            "wav": torch.from_numpy(self.wav[index]),
            "prosody": torch.from_numpy(prosody),
            "spectral": torch.from_numpy(spectral),
            "quality": torch.from_numpy(quality),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def masked_statistics(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    # sequence: batch x channels x time
    width = sequence.shape[-1]
    mask = torch.arange(width, device=sequence.device)[None, :] < lengths[:, None]
    weights = mask[:, None, :].to(sequence.dtype)
    denominator = weights.sum(dim=-1).clamp_min(1.0)
    mean = (sequence * weights).sum(dim=-1) / denominator
    variance = ((sequence - mean[..., None]).square() * weights).sum(dim=-1) / denominator
    return torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)


class MaskedAttentiveStatisticsPooling(nn.Module):
    """Learn frame importance, then return weighted mean and deviation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        bottleneck = max(channels // 2, 16)
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, 1, kernel_size=1),
        )

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        width = sequence.shape[-1]
        mask = torch.arange(width, device=sequence.device)[None, :] < lengths[:, None]
        scores = self.attention(sequence).squeeze(1).masked_fill(~mask, -1e4)
        weights = scores.softmax(dim=-1)[:, None, :]
        mean = (sequence * weights).sum(dim=-1)
        variance = ((sequence - mean[..., None]).square() * weights).sum(dim=-1)
        return torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)


class MultiScaleProsodyEncoder(nn.Module):
    def __init__(
        self, hidden: int, dropout: float, attentive_pooling: bool = False,
        dual_statistics_pooling: bool = False,
    ) -> None:
        super().__init__()
        width = 48
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(6, width, kernel_size=kernel, padding=kernel // 2),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel in (3, 7, 15)
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(width * 3, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.dual_statistics_pooling = dual_statistics_pooling
        self.pool = (
            MaskedAttentiveStatisticsPooling(hidden)
            if attentive_pooling or dual_statistics_pooling
            else None
        )
        pooled_size = hidden * (4 if dual_statistics_pooling else 2)
        self.output = nn.Sequential(nn.Linear(pooled_size, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sequence = self.mix(torch.cat([branch(values) for branch in self.branches], dim=1))
        global_statistics = masked_statistics(sequence, lengths)
        if self.dual_statistics_pooling:
            statistics = torch.cat(
                (global_statistics, self.pool(sequence, lengths)), dim=-1
            )
        elif self.pool is not None:
            statistics = self.pool(sequence, lengths)
        else:
            statistics = global_statistics
        return self.output(statistics)


class SpectralEncoder(nn.Module):
    def __init__(
        self, hidden: int, dropout: float, attentive_pooling: bool = False,
        dual_statistics_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.dual_statistics_pooling = dual_statistics_pooling
        self.pool = (
            MaskedAttentiveStatisticsPooling(64)
            if attentive_pooling or dual_statistics_pooling
            else None
        )
        pooled_size = 64 * (4 if dual_statistics_pooling else 2)
        self.output = nn.Sequential(nn.Linear(pooled_size, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.network(values).mean(dim=2)
        downsampled = torch.div(lengths + 1, 2, rounding_mode="floor")
        downsampled = torch.div(downsampled + 1, 2, rounding_mode="floor").clamp_min(1)
        global_statistics = masked_statistics(encoded, downsampled)
        if self.dual_statistics_pooling:
            statistics = torch.cat(
                (global_statistics, self.pool(encoded, downsampled)), dim=-1
            )
        elif self.pool is not None:
            statistics = self.pool(encoded, downsampled)
        else:
            statistics = global_statistics
        return self.output(statistics)


class SslOnlyPilot(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden, len(EMOTIONS))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        vector = self.encoder(batch["wav"])
        return {"logits": self.classifier(vector)}


class QualityAwareMultiViewPilot(nn.Module):
    def __init__(self, config: PilotConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.branch_dropout = config.branch_dropout
        self.ssl = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.prosody = MultiScaleProsodyEncoder(hidden, config.dropout)
        self.spectral = SpectralEncoder(hidden, config.dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3 + 3, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 3),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(hidden, len(EMOTIONS))
        self.auxiliary = nn.ModuleList([nn.Linear(hidden, len(EMOTIONS)) for _ in range(3)])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        branches = torch.stack(
            [
                self.ssl(batch["wav"]),
                self.prosody(batch["prosody"], batch["length"]),
                self.spectral(batch["spectral"], batch["length"]),
            ],
            dim=1,
        )
        gate_logits = self.gate(torch.cat([branches.flatten(1), batch["quality"]], dim=-1))
        if self.training and self.branch_dropout > 0:
            dropped = torch.rand(branches.shape[0], 2, device=branches.device) < self.branch_dropout
            branch_mask = torch.cat(
                [torch.zeros(branches.shape[0], 1, dtype=torch.bool, device=branches.device), dropped],
                dim=1,
            )
            gate_logits = gate_logits.masked_fill(branch_mask, -1e4)
        gates = gate_logits.softmax(dim=-1)
        fused = self.fusion((branches * gates[..., None]).sum(dim=1))
        auxiliary_logits = torch.stack(
            [head(branches[:, index]) for index, head in enumerate(self.auxiliary)], dim=1
        )
        return {
            "logits": self.classifier(fused),
            "gates": gates,
            "auxiliary_logits": auxiliary_logits,
        }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def calculate_loss(
    output: dict[str, torch.Tensor], labels: torch.Tensor, auxiliary_weight: float
) -> torch.Tensor:
    loss = F.cross_entropy(output["logits"], labels)
    auxiliary = output.get("auxiliary_logits")
    if auxiliary is not None and auxiliary_weight > 0:
        losses = [F.cross_entropy(auxiliary[:, index], labels) for index in range(auxiliary.shape[1])]
        loss = loss + auxiliary_weight * torch.stack(losses).mean()
    return loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    auxiliary_weight: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    gate_rows: list[np.ndarray] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch)
        loss = calculate_loss(output, batch["label"], auxiliary_weight)
        losses.extend([float(loss.item())] * len(batch["label"]))
        predicted = output["logits"].argmax(dim=-1)
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        sample_ids.extend(raw_batch["sample_id"])
        if "gates" in output:
            gate_rows.extend(output["gates"].cpu().numpy())
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "ua": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
    prediction_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_emotion": [EMOTIONS[index] for index in labels],
            "predicted_emotion": [EMOTIONS[index] for index in predictions],
        }
    )
    if gate_rows:
        gates = np.asarray(gate_rows)
        prediction_frame[["gate_ssl", "gate_prosody", "gate_spectral"]] = gates
    return metrics, prediction_frame


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    train_evaluation_loader: DataLoader,
    validation_loader: DataLoader,
    config: PilotConfig,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_ua = -math.inf
    bad_epochs = 0
    overfit_streak = 0
    stop_reason = "max_epochs"
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = calculate_loss(output, batch["label"], config.auxiliary_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_metrics, _ = evaluate(
            model, train_evaluation_loader, device, config.auxiliary_loss_weight
        )
        validation_metrics, _ = evaluate(
            model, validation_loader, device, config.auxiliary_loss_weight
        )
        gap = train_metrics["ua"] - validation_metrics["ua"]
        improved = validation_metrics["ua"] > best_validation_ua + config.min_delta
        if improved:
            best_validation_ua = validation_metrics["ua"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch >= 4 and gap > config.overfit_gap and not improved:
            overfit_streak += 1
        else:
            overfit_streak = 0
        history.append(
            {
                "model": name,
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "ua_gap": gap,
                "improved": improved,
            }
        )
        print(
            f"{name} epoch={epoch:02d} train_ua={train_metrics['ua']:.4f} "
            f"val_ua={validation_metrics['ua']:.4f} val_f1={validation_metrics['macro_f1']:.4f} "
            f"gap={gap:.4f} bad={bad_epochs}",
            flush=True,
        )
        if overfit_streak >= config.overfit_patience:
            stop_reason = "overfit_guard"
            break
        if bad_epochs >= config.patience:
            stop_reason = "early_stopping"
            break

    model.load_state_dict(best_state)
    best_metrics, predictions = evaluate(
        model, validation_loader, device, config.auxiliary_loss_weight
    )
    summary = {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "stopped_epoch": int(history[-1]["epoch"]),
        "stop_reason": stop_reason,
        "best_validation": best_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if {"gate_ssl", "gate_prosody", "gate_spectral"}.issubset(predictions.columns):
        summary["mean_validation_gates"] = {
            column.removeprefix("gate_"): float(predictions[column].mean())
            for column in ("gate_ssl", "gate_prosody", "gate_spectral")
        }
        summary["mean_validation_gates_by_emotion"] = (
            predictions.groupby("true_emotion")
            [["gate_ssl", "gate_prosody", "gate_spectral"]]
            .mean()
            .to_dict(orient="index")
        )
    return model, pd.DataFrame(history), summary, predictions


def run(config: PilotConfig, output_dir: Path) -> dict[str, Any]:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_frame, validation_frame = load_pilot_frames(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame[["sample_id", "emotion"]].assign(split="train").to_csv(
        output_dir / "pilot_train_samples.csv", index=False
    )
    validation_frame[["sample_id", "emotion"]].assign(split="validation").to_csv(
        output_dir / "pilot_validation_samples.csv", index=False
    )

    train_dataset = CachedMultiViewDataset(train_frame, config.max_frames)
    validation_dataset = CachedMultiViewDataset(validation_frame, config.max_frames)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    evaluation_train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    results: dict[str, Any] = {}
    histories: list[pd.DataFrame] = []
    for name, model in (
        ("ssl_only_proxy", SslOnlyPilot(config.hidden_size, config.dropout)),
        ("qampf_proxy", QualityAwareMultiViewPilot(config)),
    ):
        trained, history, summary, predictions = train_model(
            name,
            model,
            train_loader,
            evaluation_train_loader,
            validation_loader,
            config,
            device,
        )
        # Re-evaluate train deterministically after training. The training loader remains shuffled
        # only for gradient updates; the report uses the restored best checkpoint.
        final_train, _ = evaluate(
            trained, evaluation_train_loader, device, config.auxiliary_loss_weight
        )
        summary["best_checkpoint_train"] = final_train
        results[name] = summary
        histories.append(history)
        predictions.to_csv(output_dir / f"{name}_validation_predictions.csv", index=False)
        torch.save(trained.state_dict(), output_dir / f"{name}_best.pt")

    baseline = results["ssl_only_proxy"]["best_validation"]
    qampf = results["qampf_proxy"]["best_validation"]
    qampf_train = results["qampf_proxy"]["best_checkpoint_train"]
    delta_ua = qampf["ua"] - baseline["ua"]
    final_gap = qampf_train["ua"] - qampf["ua"]
    if delta_ua >= 0.005 and final_gap <= config.overfit_gap:
        decision = "GO_END_TO_END_ABLATION"
    elif delta_ua > 0:
        decision = "HOLD_SMALL_GAIN"
    else:
        decision = "STOP_NO_PROXY_GAIN"

    report = {
        "experiment": "qampf_validation_only_proxy_pilot",
        "protocol": "controlled_speaker_overlap_v1",
        "warning": (
            "This is a validation-only proxy using cached utterance-level Wav2Vec2 embeddings. "
            "It is not an end-to-end Pitch-Fusion result and Test was not loaded."
        ),
        "test_manifest_loaded": False,
        "device": str(device),
        "config": asdict(config),
        "sample_counts": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "train_by_emotion": train_frame["emotion"].value_counts().sort_index().to_dict(),
            "validation_by_emotion": validation_frame["emotion"].value_counts().sort_index().to_dict(),
        },
        "models": results,
        "comparison": {
            "validation_ua_delta_qampf_minus_ssl": delta_ua,
            "validation_macro_f1_delta_qampf_minus_ssl": qampf["macro_f1"] - baseline["macro_f1"],
            "qampf_train_validation_ua_gap_at_best_checkpoint": final_gap,
            "decision": decision,
        },
    }
    pd.concat(histories, ignore_index=True).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "pilot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/b6/qampf_proxy_pilot_seed2025"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--regularized",
        action="store_true",
        help="Use the compact high-regularization preset after an overfit pilot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PilotConfig()
    if args.smoke:
        config = PilotConfig(
            train_samples=64,
            validation_samples=32,
            max_frames=160,
            batch_size=8,
            max_epochs=1,
            patience=1,
            overfit_patience=1,
        )
    elif args.regularized:
        config = PilotConfig(
            train_samples=1200,
            validation_samples=400,
            max_frames=300,
            batch_size=32,
            max_epochs=10,
            patience=3,
            min_delta=0.002,
            overfit_gap=0.12,
            overfit_patience=2,
            learning_rate=2e-4,
            weight_decay=2e-2,
            hidden_size=64,
            branch_dropout=0.30,
            dropout=0.45,
            auxiliary_loss_weight=0.10,
        )
    report = run(config, resolve(args.output_dir))
    print(json.dumps(report["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
