"""Train the residual acoustic adapter on frozen Pitch-Fusion anchor logits."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rapf.models.acoustic_views import (
    CachedMultiViewDataset,
    MultiScaleProsodyEncoder,
    SpectralEncoder,
    masked_statistics,
    move_batch,
    resolve,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[3]
LABELS = ("happy", "neutral", "sad", "angry")
ANCHOR = (
    ROOT / "artifacts/anchors/disagreement_aware_train_val_anchor.parquet"
)
OUTPUT = ROOT / "artifacts/residual_adapter/seed2025"


@dataclass(frozen=True)
class ResidualConfig:
    seed: int = 2025
    max_frames: int = 300
    hidden_size: int = 64
    batch_size: int = 32
    max_epochs: int = 12
    patience: int = 3
    min_delta: float = 0.002
    learning_rate: float = 2e-4
    weight_decay: float = 0.02
    dropout: float = 0.40
    branch_dropout: float = 0.25
    residual_penalty: float = 0.02
    ema_decay: float = 0.0
    maximum_train_validation_gap: float = 0.20
    overfit_patience: int = 2
    minimum_required_ua_gain: float = 0.005
    anchor_aux_size: int = 0
    class_weight_power: float = 0.0
    anchor_consistency_weight: float = 0.0
    consensus_aux_index: int = -1
    auxiliary_classification_weight: float = 0.0
    branch_auxiliary_weight: float = 0.0
    interaction_fusion: bool = False
    dual_residual_experts: bool = False
    class_conditional_experts: bool = False
    cosine_auxiliary_head: bool = False
    cosine_temperature: float = 0.2
    hybrid_auxiliary_head: bool = False
    attentive_pooling: bool = False
    dual_statistics_pooling: bool = False
    uncertainty_film: bool = False
    anchor_conditioned_branch_gate: bool = False
    anchor_residual_conditioning: bool = False
    feature_augmentation: bool = False
    augmentation_noise_std: float = 0.02
    augmentation_time_mask_ratio: float = 0.08
    augmentation_frequency_mask_ratio: float = 0.10
    ssl_residual_branch: bool = False
    segment_context_branch: bool = False
    segment_count: int = 5
    temporal_delta_branch: bool = False
    temporal_delta_lags: tuple[int, ...] = (1,)
    cross_modal_attention: bool = False
    cross_attention_stride: int = 4
    quality_auxiliary_weight: float = 0.0
    anchor_error_auxiliary_weight: float = 0.0
    classwise_correction_gate: bool = False
    logit_aware_correction_gate: bool = False
    identity_centered_logit_gate: bool = False
    gate_direction_weight: float = 0.0
    speaker_conditioning: bool = False
    metadata_film: bool = False
    metadata_residual_concat: bool = False
    speaker_vocab_size: int = 0
    accent_vocab_size: int = 0
    gender_vocab_size: int = 0
    speaker_prior_branch: bool = False
    speaker_logit_adapter: bool = False
    anchor_augmentation: bool = False
    anchor_noise_std: float = 0.20
    anchor_temperature_max: float = 1.50
    happy_neutral_specialist_weight: float = 0.0
    focal_gamma: float = 0.0
    label_smoothing: float = 0.0
    supervised_contrastive_weight: float = 0.0
    supervised_contrastive_temperature: float = 0.10
    cross_modal_alignment_weight: float = 0.0
    cross_modal_alignment_temperature: float = 0.10
    branch_token_transformer: bool = False
    deep_prosody_backbone: bool = False
    deep_spectral_backbone: bool = False
    anchor_uncertainty_weight: float = 0.0
    correction_gate_initial_bias: float = -2.2
    mask_disagreement: bool = False
    branch_mode: str = "both"
    branch_fusion: str = "learned"
    neutralize_branch_quality: bool = False
    neutral_branch_quality_values: tuple[float, ...] = ()
    initialize_from_checkpoint: str = ""
    freeze_non_gate: bool = False
    shuffle_branch_quality: bool = False
    quality_feature_modulation: bool = False
    disagreement_mode: str = "original"
    correct_anchor_consistency_weight: float = 0.0
    force_correction_alpha_one: bool = False
    static_anchor_residual: bool = False
    prosody_feature_mask: str = "none"
    spectral_feature_mask: str = "none"
    preload_workers: int = 8


def apply_acoustic_feature_masks(
    prosody: torch.Tensor,
    spectral: torch.Tensor,
    prosody_feature_mask: str,
    spectral_feature_mask: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply train/validation-matched destructive feature-group ablations."""
    prosody = prosody.clone()
    spectral = spectral.clone()
    if prosody_feature_mask == "pitch_only":
        prosody[3:6] = 0
    elif prosody_feature_mask == "energy_voicing_only":
        prosody[0:3] = 0
    elif prosody_feature_mask != "none":
        raise ValueError(f"Unknown prosody feature mask: {prosody_feature_mask}")
    if spectral_feature_mask == "base_only":
        spectral[1:3] = 0
    elif spectral_feature_mask != "none":
        raise ValueError(f"Unknown spectral feature mask: {spectral_feature_mask}")
    return prosody, spectral


class ResidualDataset(CachedMultiViewDataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        max_frames: int,
        anchor_aux_size: int = 0,
        *,
        augment: bool = False,
        seed: int = 2025,
        noise_std: float = 0.02,
        time_mask_ratio: float = 0.08,
        frequency_mask_ratio: float = 0.10,
        anchor_augment: bool = False,
        anchor_noise_std: float = 0.20,
        anchor_temperature_max: float = 1.50,
        mask_disagreement: bool = False,
        prosody_feature_mask: str = "none",
        spectral_feature_mask: str = "none",
        static_anchor_only: bool = False,
    ) -> None:
        super().__init__(frame, max_frames)
        columns = [f"anchor_log_probability_{index}" for index in range(4)]
        self.anchor = frame[columns].to_numpy(dtype=np.float32, copy=True)
        auxiliary_columns = [
            f"anchor_aux_{index}" for index in range(anchor_aux_size)
        ]
        missing = [column for column in auxiliary_columns if column not in frame]
        if missing:
            raise ValueError(f"Missing anchor auxiliary columns: {missing}")
        self.anchor_aux = (
            frame[auxiliary_columns].to_numpy(dtype=np.float32, copy=True)
            if auxiliary_columns
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        if mask_disagreement:
            self.anchor_aux.fill(0.0)
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.noise_std = noise_std
        self.time_mask_ratio = time_mask_ratio
        self.frequency_mask_ratio = frequency_mask_ratio
        self.anchor_augment = anchor_augment
        self.anchor_noise_std = anchor_noise_std
        self.anchor_temperature_max = anchor_temperature_max
        self.prosody_feature_mask = prosody_feature_mask
        self.spectral_feature_mask = spectral_feature_mask
        self.static_anchor_only = static_anchor_only
        self.branch_quality_neutral: torch.Tensor | None = None
        self.branch_quality_override: torch.Tensor | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.static_anchor_only:
            result = {
                "sample_id": str(self.frame.iloc[index]["sample_id"]),
                "wav": torch.zeros(768, dtype=torch.float32),
                "prosody": torch.zeros(6, 1, dtype=torch.float32),
                "spectral": torch.zeros(3, 80, 1, dtype=torch.float32),
                "quality": torch.zeros(3, dtype=torch.float32),
                "length": torch.tensor(1, dtype=torch.long),
                "label": torch.tensor(self.labels[index], dtype=torch.long),
            }
        else:
            result = super().__getitem__(index)
            result["prosody"], result["spectral"] = apply_acoustic_feature_masks(
                result["prosody"], result["spectral"],
                self.prosody_feature_mask, self.spectral_feature_mask,
            )
        generator = None
        if self.augment or self.anchor_augment:
            generator = torch.Generator().manual_seed(
                self.seed + self.epoch * 1_000_003 + index
            )
        if self.augment:
            prosody = result["prosody"].clone()
            spectral = result["spectral"].clone()
            if self.noise_std > 0:
                prosody += torch.randn(
                    prosody.shape, generator=generator, dtype=prosody.dtype
                ) * self.noise_std
                spectral += torch.randn(
                    spectral.shape, generator=generator, dtype=spectral.dtype
                ) * self.noise_std
            valid_length = int(result["length"].item())
            maximum_time_mask = int(valid_length * self.time_mask_ratio)
            if maximum_time_mask > 0:
                width = int(
                    torch.randint(1, maximum_time_mask + 1, (1,), generator=generator)
                )
                start = int(
                    torch.randint(
                        0, max(1, valid_length - width + 1), (1,), generator=generator
                    )
                )
                prosody[:, start : start + width] = 0
                spectral[..., start : start + width] = 0
            frequency_bins = spectral.shape[-2]
            maximum_frequency_mask = int(
                frequency_bins * self.frequency_mask_ratio
            )
            if maximum_frequency_mask > 0:
                width = int(
                    torch.randint(
                        1, maximum_frequency_mask + 1, (1,), generator=generator
                    )
                )
                start = int(
                    torch.randint(
                        0,
                        max(1, frequency_bins - width + 1),
                        (1,),
                        generator=generator,
                    )
                )
                spectral[:, start : start + width, :] = 0
            result["prosody"] = prosody
            result["spectral"] = spectral
        anchor = torch.from_numpy(self.anchor[index]).clone()
        if self.anchor_augment:
            temperature = 1.0 + float(
                torch.rand((), generator=generator)
            ) * (self.anchor_temperature_max - 1.0)
            anchor = anchor / temperature
            if self.anchor_noise_std > 0:
                anchor = anchor + torch.randn(
                    anchor.shape, generator=generator, dtype=anchor.dtype
                ) * self.anchor_noise_std
        result["anchor"] = anchor
        result["anchor_aux"] = torch.from_numpy(self.anchor_aux[index])
        result["branch_quality"] = (
            self.branch_quality_override[index].clone()
            if self.branch_quality_override is not None
            else self.branch_quality_neutral.clone()
            if self.branch_quality_neutral is not None
            else result["quality"].clone()
        )
        result["speaker_index"] = torch.tensor(
            int(self.frame.iloc[index]["speaker_index"]), dtype=torch.long
        )
        result["accent_index"] = torch.tensor(
            int(self.frame.iloc[index]["accent_index"]), dtype=torch.long
        )
        result["gender_index"] = torch.tensor(
            int(self.frame.iloc[index]["gender_index"]), dtype=torch.long
        )
        result["speaker_prior"] = torch.tensor(
            [
                float(self.frame.iloc[index][f"speaker_prior_{class_index}"])
                for class_index in range(len(LABELS))
            ],
            dtype=torch.float32,
        )
        return result


class DeepResidualProsodyEncoder(nn.Module):
    """Multi-scale pitch encoder with lightweight dilated residual blocks."""

    def __init__(self, hidden: int, dropout: float) -> None:
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
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden,
                        hidden,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden,
                    ),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dilation in (2, 4)
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sequence = self.mix(
            torch.cat([branch(values) for branch in self.branches], dim=1)
        )
        for block in self.blocks:
            sequence = sequence + 0.5 * block(sequence)
        return self.output(masked_statistics(sequence, lengths))


class DeepResidualSpectralEncoder(nn.Module):
    """Frequency-aware CNN with lightweight temporal residual refinement."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(48, hidden, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden, hidden, kernel_size=3, padding=dilation,
                        dilation=dilation, groups=hidden,
                    ),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sequence = self.frontend(values).mean(dim=2)
        for block in self.temporal_blocks:
            sequence = sequence + 0.5 * block(sequence)
        downsampled = torch.div(lengths + 1, 2, rounding_mode="floor")
        downsampled = torch.div(
            downsampled + 1, 2, rounding_mode="floor"
        ).clamp_min(1)
        return self.output(masked_statistics(sequence, downsampled))


class AnchoredResidualPitchFusion(nn.Module):
    def __init__(self, config: ResidualConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.branch_dropout = config.branch_dropout
        if config.branch_mode not in {"both", "prosody", "spectral"}:
            raise ValueError(f"Unknown branch mode: {config.branch_mode}")
        if config.branch_fusion not in {"learned", "fixed_equal"}:
            raise ValueError(f"Unknown branch fusion: {config.branch_fusion}")
        if config.branch_fusion == "fixed_equal" and config.branch_mode != "both":
            raise ValueError("Fixed-equal fusion requires both acoustic branches")
        self.branch_mode = config.branch_mode
        self.branch_fusion = config.branch_fusion
        if config.disagreement_mode not in {"original", "entropy_margin", "permuted"}:
            raise ValueError(f"Unknown disagreement mode: {config.disagreement_mode}")
        self.disagreement_mode = config.disagreement_mode
        self.quality_feature_modulation = config.quality_feature_modulation
        self.force_correction_alpha_one = config.force_correction_alpha_one
        self.static_anchor_residual = config.static_anchor_residual
        self.consensus_aux_index = config.consensus_aux_index
        self.interaction_fusion = config.interaction_fusion
        self.dual_residual_experts = config.dual_residual_experts
        self.class_conditional_experts = config.class_conditional_experts
        self.classwise_correction_gate = config.classwise_correction_gate
        self.identity_centered_logit_gate = config.identity_centered_logit_gate
        if config.identity_centered_logit_gate and not config.logit_aware_correction_gate:
            raise ValueError(
                "Identity-centered modulation requires --logit-aware-correction-gate"
            )
        self.speaker_conditioning = config.speaker_conditioning
        self.metadata_film_enabled = config.metadata_film
        self.metadata_residual_concat = config.metadata_residual_concat
        self.speaker_prior_branch = config.speaker_prior_branch
        self.speaker_logit_adapter_enabled = config.speaker_logit_adapter
        self.branch_token_transformer_enabled = config.branch_token_transformer
        if config.metadata_film and not config.speaker_conditioning:
            raise ValueError("Metadata FiLM requires speaker conditioning")
        if config.metadata_residual_concat and not config.speaker_conditioning:
            raise ValueError("Metadata residual conditioning requires speaker conditioning")
        if config.metadata_film and config.metadata_residual_concat:
            raise ValueError("Choose one direct metadata-conditioning method")
        if config.dual_residual_experts and config.class_conditional_experts:
            raise ValueError("Choose either dual or class-conditional residual experts")
        self.uncertainty_film = config.uncertainty_film
        self.anchor_conditioned_branch_gate = config.anchor_conditioned_branch_gate
        self.anchor_residual_conditioning = config.anchor_residual_conditioning
        self.ssl_residual_branch = config.ssl_residual_branch
        self.segment_context_branch = config.segment_context_branch
        self.temporal_delta_branch = config.temporal_delta_branch
        self.temporal_delta_lags = config.temporal_delta_lags
        self.cross_modal_attention = config.cross_modal_attention
        if config.temporal_delta_branch and (
            not config.temporal_delta_lags
            or any(lag <= 0 or lag >= config.max_frames for lag in config.temporal_delta_lags)
        ):
            raise ValueError("Temporal-delta lags must be between 1 and max_frames - 1")
        if config.cross_modal_attention and config.cross_attention_stride <= 0:
            raise ValueError("Cross-attention stride must be positive")
        self.segment_count = config.segment_count
        self.branch_count = (
            2
            + int(config.ssl_residual_branch)
            + int(config.segment_context_branch)
            + int(config.temporal_delta_branch)
            + int(config.cross_modal_attention)
            + int(
                config.speaker_conditioning
                and not config.metadata_film
                and not config.metadata_residual_concat
            )
            + int(config.speaker_prior_branch)
        )
        self.prosody = (
            DeepResidualProsodyEncoder(hidden, config.dropout)
            if config.deep_prosody_backbone
            else MultiScaleProsodyEncoder(
                hidden,
                config.dropout,
                config.attentive_pooling,
                config.dual_statistics_pooling,
            )
        )
        self.spectral = (
            DeepResidualSpectralEncoder(hidden, config.dropout)
            if config.deep_spectral_backbone
            else SpectralEncoder(
                hidden,
                config.dropout,
                config.attentive_pooling,
                config.dual_statistics_pooling,
            )
        )
        self.static_residual_head = (
            nn.Linear(len(LABELS), len(LABELS))
            if config.static_anchor_residual
            else None
        )
        if self.static_residual_head is not None:
            nn.init.zeros_(self.static_residual_head.weight)
            nn.init.zeros_(self.static_residual_head.bias)
        self.ssl = (
            nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.ssl_residual_branch
            else None
        )
        if config.segment_context_branch:
            self.segment_fusion = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            segment_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.segment_transformer = nn.TransformerEncoder(
                segment_layer, num_layers=1
            )
            self.segment_attention = nn.Linear(hidden, 1)
        else:
            self.segment_fusion = None
            self.segment_transformer = None
            self.segment_attention = None
        self.delta_fusion = (
            nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.temporal_delta_branch
            else None
        )
        self.delta_scale_attention = (
            nn.Linear(hidden, 1)
            if config.temporal_delta_branch and len(config.temporal_delta_lags) > 1
            else None
        )
        if config.cross_modal_attention:
            self.cross_attention_stride = config.cross_attention_stride
            maximum_tokens = math.ceil(config.max_frames / config.cross_attention_stride)
            self.cross_position = nn.Parameter(torch.zeros(1, maximum_tokens, hidden))
            nn.init.normal_(self.cross_position, mean=0.0, std=0.02)
            self.cross_prosody_projection = nn.Sequential(
                nn.LayerNorm(6), nn.Linear(6, hidden), nn.GELU()
            )
            self.cross_spectral_projection = nn.Sequential(
                nn.LayerNorm(240), nn.Linear(240, hidden), nn.GELU()
            )
            self.pitch_queries_spectral = nn.MultiheadAttention(
                hidden, num_heads=4, dropout=config.dropout, batch_first=True
            )
            self.spectral_queries_pitch = nn.MultiheadAttention(
                hidden, num_heads=4, dropout=config.dropout, batch_first=True
            )
            self.cross_fusion = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.cross_attention_stride = 1
            self.register_parameter("cross_position", None)
            self.cross_prosody_projection = None
            self.cross_spectral_projection = None
            self.pitch_queries_spectral = None
            self.spectral_queries_pitch = None
            self.cross_fusion = None
        if config.speaker_conditioning:
            if min(
                config.speaker_vocab_size,
                config.accent_vocab_size,
                config.gender_vocab_size,
            ) <= 0:
                raise ValueError("Speaker-conditioning vocabulary sizes are required")
            self.speaker_embedding = nn.Embedding(
                config.speaker_vocab_size, 32, padding_idx=0
            )
            self.accent_embedding = nn.Embedding(
                config.accent_vocab_size, 8, padding_idx=0
            )
            self.gender_embedding = nn.Embedding(
                config.gender_vocab_size, 4, padding_idx=0
            )
            self.metadata_fusion = nn.Sequential(
                nn.LayerNorm(44),
                nn.Linear(44, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.metadata_film_layer = (
                nn.Linear(hidden, hidden * 2) if config.metadata_film else None
            )
            if self.metadata_film_layer is not None:
                nn.init.zeros_(self.metadata_film_layer.weight)
                nn.init.zeros_(self.metadata_film_layer.bias)
        else:
            self.speaker_embedding = None
            self.accent_embedding = None
            self.gender_embedding = None
            self.metadata_fusion = None
            self.metadata_film_layer = None
        self.speaker_prior_projection = (
            nn.Sequential(
                nn.LayerNorm(len(LABELS)),
                nn.Linear(len(LABELS), hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.speaker_prior_branch
            else None
        )
        if config.speaker_logit_adapter:
            if not config.speaker_conditioning or config.speaker_vocab_size <= 0:
                raise ValueError("Speaker logit adapter requires speaker conditioning")
            self.speaker_logit_embedding = nn.Embedding(
                config.speaker_vocab_size, len(LABELS), padding_idx=0
            )
            nn.init.zeros_(self.speaker_logit_embedding.weight)
            self.speaker_logit_strength = nn.Parameter(torch.tensor(-1.5))
        else:
            self.speaker_logit_embedding = None
            self.register_parameter("speaker_logit_strength", None)
        if config.interaction_fusion and self.branch_count != 2:
            raise ValueError("Interaction fusion supports two branches only")
        self.branch_gate = nn.Sequential(
            nn.Linear(
                hidden * self.branch_count
                + 3
                + (6 if config.anchor_conditioned_branch_gate else 0),
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, self.branch_count),
        )
        self.quality_modulation = (
            nn.Sequential(
                nn.LayerNorm(3),
                nn.Linear(3, hidden),
                nn.GELU(),
                nn.Linear(hidden, 4 * hidden),
            )
            if config.quality_feature_modulation
            else None
        )
        if self.quality_modulation is not None:
            nn.init.zeros_(self.quality_modulation[-1].weight)
            nn.init.zeros_(self.quality_modulation[-1].bias)
        self.branch_auxiliary_heads = (
            nn.ModuleList(
                [nn.Linear(hidden, len(LABELS)) for _ in range(self.branch_count)]
            )
            if config.branch_auxiliary_weight > 0
            else None
        )
        if config.branch_token_transformer:
            branch_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.branch_token_transformer = nn.TransformerEncoder(
                branch_layer, num_layers=1
            )
            self.branch_positions = nn.Parameter(
                torch.zeros(1, self.branch_count, hidden)
            )
            nn.init.normal_(self.branch_positions, mean=0.0, std=0.02)
        else:
            self.branch_token_transformer = None
            self.register_parameter("branch_positions", None)
        fusion_input_size = hidden * 3 if config.interaction_fusion else hidden
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_size),
            nn.Linear(fusion_input_size, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        if config.uncertainty_film:
            if config.anchor_aux_size <= 0:
                raise ValueError("Uncertainty FiLM requires anchor auxiliary features")
            self.film = nn.Sequential(
                nn.LayerNorm(config.anchor_aux_size),
                nn.Linear(config.anchor_aux_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden * 2),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        else:
            self.film = None
        residual_input_size = (
            hidden
            + config.anchor_aux_size
            + (hidden if config.metadata_residual_concat else 0)
            + (len(LABELS) if config.anchor_residual_conditioning else 0)
        )
        if config.dual_residual_experts or config.class_conditional_experts:
            expert_count = len(LABELS) if config.class_conditional_experts else 2
            self.residual_experts = nn.ModuleList(
                [
                    nn.Linear(residual_input_size, len(LABELS))
                    for _ in range(expert_count)
                ]
            )
            self.expert_gate = nn.Sequential(
                nn.Linear(residual_input_size, hidden // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden // 2, expert_count),
            )
            if config.class_conditional_experts:
                nn.init.zeros_(self.expert_gate[-1].weight)
                nn.init.zeros_(self.expert_gate[-1].bias)
            self.residual_head = None
        else:
            self.residual_head = nn.Linear(residual_input_size, len(LABELS))
            self.residual_experts = None
            self.expert_gate = None
        if config.happy_neutral_specialist_weight > 0:
            self.happy_neutral_specialist = nn.Linear(residual_input_size, 2)
            nn.init.normal_(self.happy_neutral_specialist.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.happy_neutral_specialist.bias)
            self.happy_neutral_strength = nn.Parameter(torch.tensor(-1.5))
        else:
            self.happy_neutral_specialist = None
            self.register_parameter("happy_neutral_strength", None)
        self.cosine_temperature = config.cosine_temperature
        if config.auxiliary_classification_weight > 0 and (
            config.cosine_auxiliary_head or config.hybrid_auxiliary_head
        ):
            self.auxiliary_head = (
                nn.Linear(hidden, len(LABELS))
                if config.hybrid_auxiliary_head
                else None
            )
            self.auxiliary_prototypes = nn.Parameter(
                torch.empty(len(LABELS), hidden)
            )
            nn.init.normal_(self.auxiliary_prototypes, mean=0.0, std=0.02)
        else:
            self.auxiliary_head = (
                nn.Linear(hidden, len(LABELS))
                if config.auxiliary_classification_weight > 0
                else None
            )
            self.register_parameter("auxiliary_prototypes", None)
        correction_context_size = (
            hidden
            + 5
            + config.anchor_aux_size
            + (hidden if config.metadata_residual_concat else 0)
        )
        if config.logit_aware_correction_gate:
            # A shared per-class gate.  The same weights are applied to every
            # emotion, so the mechanism does not hard-code a favourable class.
            # Six local features expose the actual proposed logit edit before
            # the gate decides how much of that edit should be applied.
            self.correction_gate = None
            self.logit_correction_gate = nn.Sequential(
                nn.LayerNorm(correction_context_size + 6),
                nn.Linear(correction_context_size + 6, hidden // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.correction_gate = nn.Sequential(
                nn.Linear(correction_context_size, hidden // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(
                    hidden // 2,
                    len(LABELS) if config.classwise_correction_gate else 1,
                ),
            )
            self.logit_correction_gate = None
        self.quality_auxiliary_head = (
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden // 2, 3),
                nn.Sigmoid(),
            )
            if config.quality_auxiliary_weight > 0
            else None
        )
        self.anchor_error_head = (
            nn.Sequential(
                nn.LayerNorm(hidden + config.anchor_aux_size + 2),
                nn.Linear(hidden + config.anchor_aux_size + 2, hidden // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden // 2, 1),
            )
            if config.anchor_error_auxiliary_weight > 0
            else None
        )
        if self.anchor_error_head is not None:
            nn.init.zeros_(self.anchor_error_head[-1].weight)
            nn.init.zeros_(self.anchor_error_head[-1].bias)
        if self.residual_head is not None:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
        else:
            for expert in self.residual_experts:
                nn.init.normal_(expert.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(expert.bias)
        active_correction_gate = (
            self.logit_correction_gate
            if self.logit_correction_gate is not None
            else self.correction_gate
        )
        assert active_correction_gate is not None
        nn.init.zeros_(active_correction_gate[-1].weight)
        nn.init.constant_(
            active_correction_gate[-1].bias,
            0.0
            if config.identity_centered_logit_gate
            else config.correction_gate_initial_bias,
        )
        if config.branch_mode == "prosody":
            for parameter in self.spectral.parameters():
                parameter.requires_grad_(False)
        elif config.branch_mode == "spectral":
            for parameter in self.prosody.parameters():
                parameter.requires_grad_(False)
        if config.branch_mode != "both" or config.branch_fusion == "fixed_equal":
            for parameter in self.branch_gate.parameters():
                parameter.requires_grad_(False)
        if config.force_correction_alpha_one:
            for parameter in active_correction_gate.parameters():
                parameter.requires_grad_(False)
        if config.static_anchor_residual:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.static_residual_head.parameters():
                parameter.requires_grad_(True)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.static_anchor_residual:
            raw_residual = self.static_residual_head(batch["anchor"])
            batch_size = batch["anchor"].shape[0]
            correction_strength = torch.ones(
                batch_size, 1, device=batch["anchor"].device,
                dtype=batch["anchor"].dtype,
            )
            zeros = torch.zeros(
                batch_size, self.hidden_size,
                device=batch["anchor"].device, dtype=batch["anchor"].dtype,
            )
            return {
                "logits": batch["anchor"] + raw_residual,
                "embedding": zeros,
                "prosody_embedding": zeros,
                "spectral_embedding": zeros,
                "anchor_logits": batch["anchor"],
                "raw_residual": raw_residual,
                "residual": raw_residual,
                "correction_strength": correction_strength,
                "branch_gates": torch.full(
                    (batch_size, 2), 0.5, device=batch["anchor"].device,
                    dtype=batch["anchor"].dtype,
                ),
                "consensus": torch.zeros(
                    batch_size, device=batch["anchor"].device,
                    dtype=batch["anchor"].dtype,
                ),
            }
        prosody_embedding = (
            self.prosody(batch["prosody"], batch["length"])
            if self.branch_mode != "spectral" else None
        )
        spectral_embedding = (
            self.spectral(batch["spectral"], batch["length"])
            if self.branch_mode != "prosody" else None
        )
        reference = prosody_embedding if prosody_embedding is not None else spectral_embedding
        assert reference is not None
        branch_values = [
            prosody_embedding if prosody_embedding is not None else torch.zeros_like(reference),
            spectral_embedding if spectral_embedding is not None else torch.zeros_like(reference),
        ]
        if self.ssl is not None:
            branch_values.append(self.ssl(batch["wav"]))
        if self.segment_transformer is not None:
            width = batch["prosody"].shape[-1]
            boundaries = torch.linspace(
                0, width, self.segment_count + 1, device=batch["length"].device
            ).round().long()
            segment_values = []
            segment_validity = []
            for segment_index in range(self.segment_count):
                start = int(boundaries[segment_index].item())
                stop = int(boundaries[segment_index + 1].item())
                segment_width = stop - start
                segment_lengths = (batch["length"] - start).clamp(
                    min=0, max=segment_width
                )
                valid = segment_lengths > 0
                safe_lengths = segment_lengths.clamp_min(1)
                prosody_segment = self.prosody(
                    batch["prosody"][..., start:stop], safe_lengths
                )
                spectral_segment = self.spectral(
                    batch["spectral"][..., start:stop], safe_lengths
                )
                segment_values.append(
                    self.segment_fusion(
                        torch.cat((prosody_segment, spectral_segment), dim=-1)
                    )
                )
                segment_validity.append(valid)
            segment_sequence = torch.stack(segment_values, dim=1)
            segment_valid = torch.stack(segment_validity, dim=1)
            segment_sequence = self.segment_transformer(
                segment_sequence, src_key_padding_mask=~segment_valid
            )
            segment_scores = self.segment_attention(segment_sequence).squeeze(-1)
            segment_scores = segment_scores.masked_fill(~segment_valid, -1e4)
            segment_weights = segment_scores.softmax(dim=-1)
            segment_context = (
                segment_sequence * segment_weights[..., None]
            ).sum(dim=1)
            branch_values.append(segment_context)
        if self.delta_fusion is not None:
            delta_scales = []
            delta_validity = []
            for lag in self.temporal_delta_lags:
                delta_lengths = (batch["length"] - lag).clamp_min(1)
                prosody_delta = (
                    batch["prosody"][..., lag:] - batch["prosody"][..., :-lag]
                )
                spectral_delta = (
                    batch["spectral"][..., lag:] - batch["spectral"][..., :-lag]
                )
                delta_scales.append(self.delta_fusion(
                    torch.cat(
                        (
                            self.prosody(prosody_delta, delta_lengths),
                            self.spectral(spectral_delta, delta_lengths),
                        ),
                        dim=-1,
                    )
                ))
                delta_validity.append(batch["length"] > lag)
            if self.delta_scale_attention is None:
                delta_context = delta_scales[0]
            else:
                stacked_deltas = torch.stack(delta_scales, dim=1)
                valid_scales = torch.stack(delta_validity, dim=1)
                scale_scores = self.delta_scale_attention(stacked_deltas).squeeze(-1)
                scale_scores = scale_scores.masked_fill(~valid_scales, -1e4)
                scale_weights = scale_scores.softmax(dim=-1)
                delta_context = (stacked_deltas * scale_weights[..., None]).sum(dim=1)
            branch_values.append(delta_context)
        if self.cross_fusion is not None:
            stride = self.cross_attention_stride
            prosody_tokens = batch["prosody"].transpose(1, 2)[:, ::stride]
            spectral_tokens = batch["spectral"].permute(0, 3, 1, 2)
            spectral_tokens = spectral_tokens.flatten(2)[:, ::stride]
            prosody_tokens = self.cross_prosody_projection(prosody_tokens)
            spectral_tokens = self.cross_spectral_projection(spectral_tokens)
            token_count = prosody_tokens.shape[1]
            position = self.cross_position[:, :token_count]
            prosody_tokens = prosody_tokens + position
            spectral_tokens = spectral_tokens + position
            token_lengths = torch.div(
                batch["length"] + stride - 1, stride, rounding_mode="floor"
            )
            padding_mask = (
                torch.arange(token_count, device=batch["length"].device)[None]
                >= token_lengths[:, None]
            )
            prosody_cross, _ = self.pitch_queries_spectral(
                prosody_tokens, spectral_tokens, spectral_tokens,
                key_padding_mask=padding_mask, need_weights=False,
            )
            spectral_cross, _ = self.spectral_queries_pitch(
                spectral_tokens, prosody_tokens, prosody_tokens,
                key_padding_mask=padding_mask, need_weights=False,
            )
            valid_tokens = (~padding_mask).unsqueeze(-1)
            denominator = valid_tokens.sum(dim=1).clamp_min(1)
            prosody_context = (
                (prosody_tokens + prosody_cross) * valid_tokens
            ).sum(dim=1) / denominator
            spectral_context = (
                (spectral_tokens + spectral_cross) * valid_tokens
            ).sum(dim=1) / denominator
            branch_values.append(
                self.cross_fusion(torch.cat((prosody_context, spectral_context), dim=-1))
            )
        metadata_context = None
        if self.metadata_fusion is not None:
            metadata_context = self.metadata_fusion(
                torch.cat(
                    (
                        self.speaker_embedding(batch["speaker_index"]),
                        self.accent_embedding(batch["accent_index"]),
                        self.gender_embedding(batch["gender_index"]),
                    ),
                    dim=-1,
                )
            )
            if self.metadata_film_layer is None and not self.metadata_residual_concat:
                branch_values.append(metadata_context)
        if self.speaker_prior_projection is not None:
            branch_values.append(
                self.speaker_prior_projection(batch["speaker_prior"])
            )
        branches = torch.stack(branch_values, dim=1)
        if self.quality_modulation is not None:
            modulation = self.quality_modulation(batch["branch_quality"]).view(
                branches.shape[0], 2, 2, self.hidden_size
            )
            scale, shift = modulation[:, :, 0], modulation[:, :, 1]
            acoustic = (
                branches[:, :2] * (1.0 + 0.25 * torch.tanh(scale))
                + 0.25 * shift
            )
            branches = torch.cat((acoustic, branches[:, 2:]), dim=1)
        if self.branch_token_transformer is not None:
            branches = branches + 0.25 * self.branch_token_transformer(
                branches + self.branch_positions
            )
        anchor_probability = batch["anchor"].softmax(dim=-1)
        entropy = -(
            anchor_probability * anchor_probability.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(len(LABELS))
        top_two = anchor_probability.topk(2, dim=-1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        effective_anchor_aux = batch["anchor_aux"]
        if self.disagreement_mode == "entropy_margin":
            effective_anchor_aux = torch.zeros_like(batch["anchor_aux"])
            if effective_anchor_aux.shape[1] < 2:
                raise ValueError("Entropy-margin disagreement requires at least two auxiliary slots")
            effective_anchor_aux[:, 0:1] = entropy
            effective_anchor_aux[:, 1:2] = margin
        gate_parts = [branches.flatten(1), batch["branch_quality"]]
        if self.anchor_conditioned_branch_gate:
            gate_parts.extend((anchor_probability, entropy, margin))
        gate_logits = self.branch_gate(torch.cat(gate_parts, dim=-1))
        if self.branch_mode == "prosody":
            gate_logits = torch.tensor(
                [0.0, -1e4], device=branches.device, dtype=branches.dtype
            ).expand(branches.shape[0], -1)
        elif self.branch_mode == "spectral":
            gate_logits = torch.tensor(
                [-1e4, 0.0], device=branches.device, dtype=branches.dtype
            ).expand(branches.shape[0], -1)
        elif self.branch_fusion == "fixed_equal":
            gate_logits = torch.zeros_like(gate_logits)
        if (
            self.training and self.branch_dropout > 0
            and self.branch_mode == "both" and self.branch_fusion == "learned"
        ):
            dropped = (
                torch.rand(
                    branches.shape[0], self.branch_count, device=branches.device
                )
                < self.branch_dropout
            )
            both = dropped.all(dim=1)
            dropped[
                both,
                torch.randint(
                    0,
                    self.branch_count,
                    (int(both.sum()),),
                    device=branches.device,
                ),
            ] = False
            gate_logits = gate_logits.masked_fill(dropped, -1e4)
        branch_gates = gate_logits.softmax(dim=-1)
        weighted_sum = (branches * branch_gates[..., None]).sum(dim=1)
        if self.interaction_fusion:
            fusion_input = torch.cat(
                (
                    weighted_sum,
                    (branches[:, 0] - branches[:, 1]).abs(),
                    branches[:, 0] * branches[:, 1],
                ),
                dim=-1,
            )
        else:
            fusion_input = weighted_sum
        fused = self.fusion(fusion_input)
        if self.film is not None:
            film_scale, film_shift = self.film(effective_anchor_aux).chunk(2, dim=-1)
            fused = fused * (1.0 + 0.25 * torch.tanh(film_scale)) + 0.25 * film_shift
        if self.metadata_film_layer is not None:
            metadata_scale, metadata_shift = self.metadata_film_layer(
                metadata_context
            ).chunk(2, dim=-1)
            fused = (
                fused * (1.0 + 0.25 * torch.tanh(metadata_scale))
                + 0.25 * metadata_shift
            )
        residual_parts = [fused, effective_anchor_aux]
        if self.anchor_residual_conditioning:
            residual_parts.append(anchor_probability)
        if self.metadata_residual_concat:
            residual_parts.append(metadata_context)
        residual_features = torch.cat(residual_parts, dim=-1)
        correction_parts = [
            fused,
            batch["quality"],
            entropy,
            margin,
            effective_anchor_aux,
        ]
        if self.metadata_residual_concat:
            correction_parts.append(metadata_context)
        correction_context = torch.cat(correction_parts, dim=-1)
        anchor_error_logit = None
        if self.anchor_error_head is not None:
            anchor_error_logit = self.anchor_error_head(
                torch.cat((fused, effective_anchor_aux, entropy, margin), dim=-1)
            )
        expert_gates = None
        if self.dual_residual_experts or self.class_conditional_experts:
            expert_gate_logits = self.expert_gate(residual_features)
            if self.class_conditional_experts:
                expert_gate_logits = expert_gate_logits + batch["anchor"].detach()
            expert_gates = expert_gate_logits.softmax(dim=-1)
            expert_residuals = torch.stack(
                [expert(residual_features) for expert in self.residual_experts],
                dim=1,
            )
            raw_residual = (
                expert_residuals * expert_gates[..., None]
            ).sum(dim=1)
        else:
            raw_residual = self.residual_head(residual_features)
        if self.logit_correction_gate is not None:
            candidate_logits = batch["anchor"] + raw_residual
            candidate_probability = candidate_logits.softmax(dim=-1)
            centered_anchor = batch["anchor"] - batch["anchor"].mean(
                dim=-1, keepdim=True
            )
            class_count = candidate_logits.shape[-1]
            expanded_candidate = candidate_logits[:, None, :].expand(
                -1, class_count, -1
            )
            diagonal = torch.eye(
                class_count, dtype=torch.bool, device=candidate_logits.device
            )[None, :, :]
            strongest_other = expanded_candidate.masked_fill(
                diagonal, -torch.inf
            ).amax(dim=-1)
            candidate_class_margin = candidate_logits - strongest_other
            local_logit_features = torch.stack(
                (
                    centered_anchor,
                    anchor_probability,
                    raw_residual,
                    raw_residual.abs(),
                    candidate_probability,
                    candidate_class_margin,
                ),
                dim=-1,
            )
            expanded_context = correction_context[:, None, :].expand(
                -1, class_count, -1
            )
            correction_logit = self.logit_correction_gate(
                torch.cat((expanded_context, local_logit_features), dim=-1)
            ).squeeze(-1)
        else:
            assert self.correction_gate is not None
            correction_logit = self.correction_gate(correction_context)
        if anchor_error_logit is not None:
            correction_logit = correction_logit + 0.25 * anchor_error_logit
        if self.force_correction_alpha_one:
            correction_strength = torch.ones_like(correction_logit)
        elif self.identity_centered_logit_gate:
            correction_strength = 1.0 + 0.5 * correction_logit.tanh()
        else:
            correction_strength = correction_logit.sigmoid()
        residual = correction_strength * raw_residual
        speaker_logit_residual = None
        if self.speaker_logit_embedding is not None:
            speaker_logit_residual = (
                self.speaker_logit_strength.sigmoid()
                * self.speaker_logit_embedding(batch["speaker_index"])
            )
            residual = residual + speaker_logit_residual
        logits = batch["anchor"] + residual
        pair_logits = None
        if self.happy_neutral_specialist is not None:
            pair_logits = self.happy_neutral_specialist(residual_features)
            centered_pair = pair_logits - pair_logits.mean(dim=-1, keepdim=True)
            pair_mass = anchor_probability[:, :2].sum(dim=-1, keepdim=True).detach()
            pair_correction = torch.zeros_like(logits)
            pair_correction[:, :2] = (
                self.happy_neutral_strength.sigmoid()
                * pair_mass
                * centered_pair
            )
            logits = logits + pair_correction
        result = {
            "logits": logits,
            "embedding": fused,
            "prosody_embedding": branch_values[0],
            "spectral_embedding": branch_values[1],
            "anchor_logits": batch["anchor"],
            "raw_residual": raw_residual,
            "residual": residual,
            "correction_strength": correction_strength,
            "correction_gate_logit": correction_logit,
            "branch_gates": branch_gates,
            "effective_anchor_aux": effective_anchor_aux,
            "consensus": (
                effective_anchor_aux[:, self.consensus_aux_index]
                if self.consensus_aux_index >= 0
                else torch.zeros_like(correction_strength[:, 0])
            ),
        }
        if self.branch_auxiliary_heads is not None:
            result["branch_auxiliary_logits"] = torch.stack(
                [
                    head(branches[:, index])
                    for index, head in enumerate(self.branch_auxiliary_heads)
                ],
                dim=1,
            )
        if self.auxiliary_head is not None:
            result["auxiliary_logits"] = self.auxiliary_head(fused)
        if self.auxiliary_prototypes is not None:
            cosine_logits = (
                F.normalize(fused, dim=-1)
                @ F.normalize(self.auxiliary_prototypes, dim=-1).t()
            ) / self.cosine_temperature
            if "auxiliary_logits" in result:
                result["auxiliary_cosine_logits"] = cosine_logits
            else:
                result["auxiliary_logits"] = cosine_logits
        if expert_gates is not None:
            result["expert_gates"] = expert_gates
        if pair_logits is not None:
            result["happy_neutral_logits"] = pair_logits
        if self.quality_auxiliary_head is not None:
            result["quality_prediction"] = self.quality_auxiliary_head(fused)
            result["quality_target"] = batch["quality"]
        if anchor_error_logit is not None:
            result["anchor_error_logit"] = anchor_error_logit.squeeze(-1)
        if speaker_logit_residual is not None:
            result["speaker_logit_residual"] = speaker_logit_residual
        return result


def load_frames(
    anchor_path: Path,
    train_manifest: Path,
    validation_manifest: Path,
    expected_train_size: int,
    expected_validation_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = pd.read_parquet(anchor_path)
    if set(anchor["split"]) != {"train", "validation"}:
        raise ValueError("Anchor cache must contain only Train and Validation")
    features = pd.read_parquet(
        ROOT / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet"
    )
    wav_columns = [f"wav2vec2_{index:03d}" for index in range(768)]
    feature_columns = [
        "sample_id", "log_mel_artifact_path", "pitch_artifact_path", *wav_columns
    ]
    frame = anchor.merge(
        features[feature_columns], on="sample_id", how="left", validate="one_to_one"
    )
    metadata = pd.concat(
        [
            pd.read_csv(
                path,
                usecols=["sample_id", "speaker_id", "accent_clean", "gender_clean"],
            )
            for path in (train_manifest, validation_manifest)
        ],
        ignore_index=True,
    )
    frame = frame.merge(metadata, on="sample_id", how="left", validate="one_to_one")
    if frame[["log_mel_artifact_path", "pitch_artifact_path"]].isna().any().any():
        raise ValueError("Missing cached acoustic features for residual training")
    train = frame[frame["split"] == "train"].sort_values("sample_id").reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].sort_values("sample_id").reset_index(drop=True)
    if len(train) != expected_train_size or len(validation) != expected_validation_size:
        raise ValueError(
            "Unexpected Train/Validation sizes: "
            f"{len(train)}/{len(validation)} instead of "
            f"{expected_train_size}/{expected_validation_size}"
        )
    for source, target in (
        ("speaker_id", "speaker_index"),
        ("accent_clean", "accent_index"),
        ("gender_clean", "gender_index"),
    ):
        vocabulary = {
            value: index + 1
            for index, value in enumerate(sorted(train[source].astype(str).unique()))
        }
        train[target] = train[source].astype(str).map(vocabulary).fillna(0).astype(int)
        validation[target] = (
            validation[source].astype(str).map(vocabulary).fillna(0).astype(int)
        )
    global_counts = np.bincount(
        train["emotion_id"].to_numpy(dtype=int), minlength=len(LABELS)
    ).astype(np.float64)
    global_prior = global_counts / global_counts.sum()
    speaker_counts = {
        str(speaker): np.bincount(
            group["emotion_id"].to_numpy(dtype=int), minlength=len(LABELS)
        ).astype(np.float64)
        for speaker, group in train.groupby("speaker_id")
    }
    smoothing = 10.0
    train_priors = np.empty((len(train), len(LABELS)), dtype=np.float32)
    for row_index, row in train.iterrows():
        counts = speaker_counts[str(row["speaker_id"])].copy()
        counts[int(row["emotion_id"])] -= 1.0
        train_priors[row_index] = (
            counts + smoothing * global_prior
        ) / (counts.sum() + smoothing)
    validation_priors = np.empty((len(validation), len(LABELS)), dtype=np.float32)
    for row_index, row in validation.iterrows():
        counts = speaker_counts.get(
            str(row["speaker_id"]), np.zeros(len(LABELS), dtype=np.float64)
        )
        validation_priors[row_index] = (
            counts + smoothing * global_prior
        ) / (counts.sum() + smoothing)
    for class_index in range(len(LABELS)):
        train[f"speaker_prior_{class_index}"] = train_priors[:, class_index]
        validation[f"speaker_prior_{class_index}"] = validation_priors[:, class_index]
    return train, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default=str(ANCHOR))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--train-manifest", default="splits/controlled_train.csv"
    )
    parser.add_argument(
        "--validation-manifest", default="splits/controlled_validation.csv"
    )
    parser.add_argument("--expected-train-size", type=int, default=3494)
    parser.add_argument("--expected-validation-size", type=int, default=749)
    parser.add_argument("--protocol-name", default="controlled_speaker_overlap_v1")
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--branch-dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--residual-penalty", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--correction-gate-initial-bias", type=float, default=-2.2)
    parser.add_argument("--minimum-required-ua-gain", type=float, default=0.005)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--anchor-aux-size", type=int, default=0)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--anchor-consistency-weight", type=float, default=0.0)
    parser.add_argument("--consensus-aux-index", type=int, default=-1)
    parser.add_argument("--auxiliary-classification-weight", type=float, default=0.0)
    parser.add_argument("--branch-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--interaction-fusion", action="store_true")
    parser.add_argument("--dual-residual-experts", action="store_true")
    parser.add_argument("--class-conditional-experts", action="store_true")
    parser.add_argument("--cosine-auxiliary-head", action="store_true")
    parser.add_argument("--cosine-temperature", type=float, default=0.2)
    parser.add_argument("--hybrid-auxiliary-head", action="store_true")
    parser.add_argument("--attentive-pooling", action="store_true")
    parser.add_argument("--dual-statistics-pooling", action="store_true")
    parser.add_argument("--uncertainty-film", action="store_true")
    parser.add_argument("--anchor-conditioned-branch-gate", action="store_true")
    parser.add_argument("--anchor-residual-conditioning", action="store_true")
    parser.add_argument("--feature-augmentation", action="store_true")
    parser.add_argument("--augmentation-noise-std", type=float, default=0.02)
    parser.add_argument("--augmentation-time-mask-ratio", type=float, default=0.08)
    parser.add_argument("--augmentation-frequency-mask-ratio", type=float, default=0.10)
    parser.add_argument("--ssl-residual-branch", action="store_true")
    parser.add_argument("--segment-context-branch", action="store_true")
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument("--temporal-delta-branch", action="store_true")
    parser.add_argument("--temporal-delta-lags", default="1")
    parser.add_argument("--cross-modal-attention", action="store_true")
    parser.add_argument("--cross-attention-stride", type=int, default=4)
    parser.add_argument("--quality-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--anchor-error-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--classwise-correction-gate", action="store_true")
    parser.add_argument(
        "--logit-aware-correction-gate",
        action="store_true",
        help=(
            "Use a permutation-shared per-class gate that observes anchor logits, "
            "the proposed residual edit, and the resulting candidate logits."
        ),
    )
    parser.add_argument(
        "--gate-direction-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary weight for learning whether each proposed logit edit moves "
            "toward or away from the labelled class; Train/Validation only."
        ),
    )
    parser.add_argument(
        "--identity-centered-logit-gate",
        action="store_true",
        help=(
            "Parameterize per-class correction strength as 1 + 0.5*tanh(h), "
            "giving an exact identity initialization and symmetric logit modulation."
        ),
    )
    parser.add_argument("--speaker-conditioning", action="store_true")
    parser.add_argument("--metadata-film", action="store_true")
    parser.add_argument("--metadata-residual-concat", action="store_true")
    parser.add_argument("--speaker-prior-branch", action="store_true")
    parser.add_argument("--speaker-logit-adapter", action="store_true")
    parser.add_argument("--anchor-augmentation", action="store_true")
    parser.add_argument("--anchor-noise-std", type=float, default=0.20)
    parser.add_argument("--anchor-temperature-max", type=float, default=1.50)
    parser.add_argument("--happy-neutral-specialist-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--supervised-contrastive-weight", type=float, default=0.0)
    parser.add_argument(
        "--supervised-contrastive-temperature", type=float, default=0.10
    )
    parser.add_argument("--cross-modal-alignment-weight", type=float, default=0.0)
    parser.add_argument(
        "--cross-modal-alignment-temperature", type=float, default=0.10
    )
    parser.add_argument("--branch-token-transformer", action="store_true")
    parser.add_argument("--deep-prosody-backbone", action="store_true")
    parser.add_argument("--deep-spectral-backbone", action="store_true")
    parser.add_argument("--anchor-uncertainty-weight", type=float, default=0.0)
    parser.add_argument(
        "--mask-disagreement",
        action="store_true",
        help=(
            "Zero all anchor auxiliary features in both Train and Validation. "
            "This preserves input shape and parameter count for controlled ablation."
        ),
    )
    parser.add_argument(
        "--branch-mode", choices=("both", "prosody", "spectral"), default="both"
    )
    parser.add_argument(
        "--branch-fusion", choices=("learned", "fixed_equal"), default="learned"
    )
    parser.add_argument("--neutralize-branch-quality", action="store_true")
    parser.add_argument(
        "--neutral-branch-quality-values",
        default="",
        help="Optional comma-separated Train-derived neutral q vector for reproducible reuse.",
    )
    parser.add_argument("--shuffle-branch-quality", action="store_true")
    parser.add_argument("--quality-feature-modulation", action="store_true")
    parser.add_argument(
        "--disagreement-mode",
        choices=("original", "entropy_margin", "permuted"),
        default="original",
    )
    parser.add_argument("--correct-anchor-consistency-weight", type=float, default=0.0)
    parser.add_argument("--force-correction-alpha-one", action="store_true")
    parser.add_argument("--static-anchor-residual", action="store_true")
    parser.add_argument(
        "--prosody-feature-mask",
        choices=("none", "pitch_only", "energy_voicing_only"), default="none",
    )
    parser.add_argument(
        "--spectral-feature-mask", choices=("none", "base_only"), default="none"
    )
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument(
        "--initialize-from-checkpoint",
        default="",
        help="Optional residual checkpoint used to initialize a two-stage gate run.",
    )
    parser.add_argument(
        "--freeze-non-gate",
        action="store_true",
        help="Freeze the initialized residual/acoustic model and train only the logit gate.",
    )
    return parser.parse_args()


def calculate_loss(
    output: dict[str, torch.Tensor], labels: torch.Tensor, penalty: float,
    class_weights: torch.Tensor | None = None,
    anchor_consistency_weight: float = 0.0,
    correct_anchor_consistency_weight: float = 0.0,
    auxiliary_classification_weight: float = 0.0,
    quality_auxiliary_weight: float = 0.0,
    happy_neutral_specialist_weight: float = 0.0,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    supervised_contrastive_weight: float = 0.0,
    supervised_contrastive_temperature: float = 0.10,
    anchor_uncertainty_weight: float = 0.0,
    cross_modal_alignment_weight: float = 0.0,
    cross_modal_alignment_temperature: float = 0.10,
    anchor_error_auxiliary_weight: float = 0.0,
    branch_auxiliary_weight: float = 0.0,
    gate_direction_weight: float = 0.0,
) -> torch.Tensor:
    if focal_gamma > 0:
        per_sample_ce = F.cross_entropy(
            output["logits"], labels, weight=class_weights, reduction="none",
            label_smoothing=label_smoothing,
        )
        true_probability = output["logits"].softmax(dim=-1).gather(
            1, labels[:, None]
        ).squeeze(1)
        classification_loss = (
            (1.0 - true_probability).pow(focal_gamma) * per_sample_ce
        ).mean()
    else:
        if anchor_uncertainty_weight > 0:
            per_sample_ce = F.cross_entropy(
                output["logits"],
                labels,
                weight=class_weights,
                reduction="none",
                label_smoothing=label_smoothing,
            )
            anchor_probability = output["anchor_logits"].softmax(dim=-1).detach()
            anchor_entropy = -(
                anchor_probability
                * anchor_probability.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(len(LABELS))
            sample_weight = 1.0 + anchor_uncertainty_weight * anchor_entropy
            classification_loss = (
                per_sample_ce * sample_weight
            ).sum() / sample_weight.sum()
        else:
            classification_loss = F.cross_entropy(
                output["logits"], labels, weight=class_weights,
                label_smoothing=label_smoothing,
            )
    loss = classification_loss + penalty * output["residual"].square().mean()
    if anchor_consistency_weight > 0:
        anchor_probability = output["anchor_logits"].softmax(dim=-1).detach()
        per_sample_kl = (
            anchor_probability
            * (
                anchor_probability.clamp_min(1e-8).log()
                - output["logits"].log_softmax(dim=-1)
            )
        ).sum(dim=-1)
        # Three voters yield agreement values 1/3, 2/3, or 1.  Map these
        # to 0, 0.5, and 1 so only consensus receives a preservation cost.
        consensus_weight = ((output["consensus"] - 1 / 3) * 1.5).clamp(0, 1)
        loss = loss + anchor_consistency_weight * (
            consensus_weight * per_sample_kl
        ).mean()
    if correct_anchor_consistency_weight > 0:
        anchor_probability = output["anchor_logits"].softmax(dim=-1).detach()
        per_sample_kl = (
            anchor_probability
            * (
                anchor_probability.clamp_min(1e-8).log()
                - output["logits"].log_softmax(dim=-1)
            )
        ).sum(dim=-1)
        anchor_correct = output["anchor_logits"].argmax(dim=-1).eq(labels).float()
        loss = loss + correct_anchor_consistency_weight * (
            anchor_correct * per_sample_kl
        ).sum() / anchor_correct.sum().clamp_min(1.0)
    if auxiliary_classification_weight > 0:
        auxiliary_loss = F.cross_entropy(
            output["auxiliary_logits"], labels, weight=class_weights
        )
        if "auxiliary_cosine_logits" in output:
            auxiliary_loss = 0.5 * (
                auxiliary_loss
                + F.cross_entropy(
                    output["auxiliary_cosine_logits"],
                    labels,
                    weight=class_weights,
                )
            )
        loss = loss + auxiliary_classification_weight * auxiliary_loss
    if branch_auxiliary_weight > 0:
        branch_logits = output["branch_auxiliary_logits"]
        repeated_labels = labels[:, None].expand(-1, branch_logits.shape[1])
        branch_loss = F.cross_entropy(
            branch_logits.reshape(-1, branch_logits.shape[-1]),
            repeated_labels.reshape(-1),
            weight=class_weights,
        )
        loss = loss + branch_auxiliary_weight * branch_loss
    if quality_auxiliary_weight > 0:
        loss = loss + quality_auxiliary_weight * F.smooth_l1_loss(
            output["quality_prediction"], output["quality_target"]
        )
    if happy_neutral_specialist_weight > 0:
        pair_mask = labels < 2
        if pair_mask.any():
            loss = loss + happy_neutral_specialist_weight * F.cross_entropy(
                output["happy_neutral_logits"][pair_mask], labels[pair_mask]
            )
    if supervised_contrastive_weight > 0:
        embedding = F.normalize(output["embedding"], dim=-1)
        similarity = embedding @ embedding.t()
        similarity = similarity / supervised_contrastive_temperature
        identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
        valid = positive_mask.any(dim=1)
        if valid.any():
            logits_masked = similarity.masked_fill(identity, -torch.inf)
            log_probability = logits_masked - torch.logsumexp(
                logits_masked, dim=1, keepdim=True
            )
            positive_log_probability = (
                log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
                / positive_mask.sum(dim=1).clamp_min(1)
            )
            loss = loss - supervised_contrastive_weight * (
                positive_log_probability[valid].mean()
            )
    if cross_modal_alignment_weight > 0:
        prosody = F.normalize(output["prosody_embedding"], dim=-1)
        spectral = F.normalize(output["spectral_embedding"], dim=-1)
        similarity = (prosody @ spectral.t()) / cross_modal_alignment_temperature
        targets = torch.arange(len(labels), device=labels.device)
        alignment_loss = 0.5 * (
            F.cross_entropy(similarity, targets)
            + F.cross_entropy(similarity.t(), targets)
        )
        loss = loss + cross_modal_alignment_weight * alignment_loss
    if anchor_error_auxiliary_weight > 0:
        anchor_wrong = output["anchor_logits"].argmax(dim=-1).ne(labels).float()
        loss = loss + anchor_error_auxiliary_weight * F.binary_cross_entropy_with_logits(
            output["anchor_error_logit"], anchor_wrong
        )
    if gate_direction_weight > 0:
        if "correction_gate_logit" not in output:
            raise ValueError("Gate-direction loss requires correction gate logits")
        gate_logits = output["correction_gate_logit"]
        if gate_logits.shape[-1] != len(LABELS):
            raise ValueError("Gate-direction loss requires a per-class correction gate")
        direction = output["raw_residual"].detach()
        beneficial_sign = F.one_hot(labels, num_classes=len(LABELS)).mul(2).sub(1)
        direction_target = (beneficial_sign * direction > 0).to(direction.dtype)
        direction_weight = direction.abs()
        direction_loss = F.binary_cross_entropy_with_logits(
            gate_logits,
            direction_target,
            reduction="none",
        )
        loss = loss + gate_direction_weight * (
            direction_loss * direction_weight
        ).sum() / direction_weight.sum().clamp_min(1e-6)
    return loss


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, penalty: float,
    class_weights: torch.Tensor | None = None,
    anchor_consistency_weight: float = 0.0,
    correct_anchor_consistency_weight: float = 0.0,
    auxiliary_classification_weight: float = 0.0,
    quality_auxiliary_weight: float = 0.0,
    happy_neutral_specialist_weight: float = 0.0,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    supervised_contrastive_weight: float = 0.0,
    supervised_contrastive_temperature: float = 0.10,
    anchor_uncertainty_weight: float = 0.0,
    cross_modal_alignment_weight: float = 0.0,
    cross_modal_alignment_temperature: float = 0.10,
    anchor_error_auxiliary_weight: float = 0.0,
    branch_auxiliary_weight: float = 0.0,
    gate_direction_weight: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    losses, labels, predictions = [], [], []
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch)
        loss = calculate_loss(
            output,
            batch["label"],
            penalty,
            class_weights,
            anchor_consistency_weight,
            correct_anchor_consistency_weight,
            auxiliary_classification_weight,
            quality_auxiliary_weight,
            happy_neutral_specialist_weight,
            focal_gamma,
            label_smoothing,
            supervised_contrastive_weight,
            supervised_contrastive_temperature,
            anchor_uncertainty_weight,
            cross_modal_alignment_weight,
            cross_modal_alignment_temperature,
            anchor_error_auxiliary_weight,
            branch_auxiliary_weight,
            gate_direction_weight,
        )
        predicted = output["logits"].argmax(dim=-1)
        losses.extend([float(loss.item())] * len(predicted))
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        probabilities = output["logits"].softmax(dim=-1).cpu().numpy()
        anchor_probabilities = output["anchor_logits"].softmax(dim=-1).cpu().numpy()
        anchor_predictions = anchor_probabilities.argmax(axis=1)
        anchor_entropies = -(
            anchor_probabilities
            * np.log(np.clip(anchor_probabilities, 1e-8, None))
        ).sum(axis=1) / math.log(len(LABELS))
        sorted_anchor_probabilities = np.sort(anchor_probabilities, axis=1)
        anchor_margins = (
            sorted_anchor_probabilities[:, -1]
            - sorted_anchor_probabilities[:, -2]
        )
        anchor_auxiliary = output.get(
            "effective_anchor_aux", batch["anchor_aux"]
        ).cpu().numpy()
        residuals = output["residual"].cpu().numpy()
        raw_residuals = output["raw_residual"].cpu().numpy()
        correction_values = output["correction_strength"].cpu().numpy()
        corrections = correction_values.mean(axis=1)
        gates = output["branch_gates"].cpu().numpy()
        expert_gates = (
            output["expert_gates"].cpu().numpy()
            if "expert_gates" in output
            else None
        )
        for index, sample_id in enumerate(raw_batch["sample_id"]):
            row = {
                "sample_id": sample_id,
                "true_emotion": LABELS[int(batch["label"][index].item())],
                "predicted_emotion": LABELS[int(predicted[index].item())],
                "anchor_predicted_emotion": LABELS[int(anchor_predictions[index])],
                "anchor_entropy": float(anchor_entropies[index]),
                "anchor_margin": float(anchor_margins[index]),
                "correction_strength": float(corrections[index]),
                "gate_prosody": float(gates[index, 0]),
                "gate_spectral": float(gates[index, 1]),
                "residual_l2": float(np.linalg.norm(residuals[index])),
                "applied_correction_l2": float(np.linalg.norm(residuals[index])),
                "length_frames": int(batch["length"][index].item()),
                "quality_voiced_ratio": float(batch["quality"][index, 0].item()),
                "quality_mean_voiced_probability": float(
                    batch["quality"][index, 1].item()
                ),
                "quality_normalized_length": float(batch["quality"][index, 2].item()),
            }
            for class_index, label in enumerate(LABELS):
                row[f"delta_z_{label}"] = float(raw_residuals[index, class_index])
                row[f"applied_delta_z_{label}"] = float(residuals[index, class_index])
            if correction_values.shape[1] > 1:
                for class_index, label in enumerate(LABELS):
                    row[f"correction_strength_{label}"] = float(
                        correction_values[index, class_index]
                    )
            gate_index = 2
            if getattr(model, "ssl", None) is not None:
                row["gate_ssl"] = float(gates[index, gate_index])
                gate_index += 1
            if getattr(model, "segment_transformer", None) is not None:
                row["gate_segment_context"] = float(gates[index, gate_index])
                gate_index += 1
            if getattr(model, "delta_fusion", None) is not None:
                row["gate_temporal_delta"] = float(gates[index, gate_index])
                gate_index += 1
            if getattr(model, "cross_fusion", None) is not None:
                row["gate_cross_modal"] = float(gates[index, gate_index])
                gate_index += 1
            if (
                getattr(model, "metadata_fusion", None) is not None
                and getattr(model, "metadata_film_layer", None) is None
                and not getattr(model, "metadata_residual_concat", False)
            ):
                row["gate_metadata"] = float(gates[index, gate_index])
                gate_index += 1
            if getattr(model, "speaker_prior_projection", None) is not None:
                row["gate_speaker_prior"] = float(gates[index, gate_index])
            if expert_gates is not None:
                for expert_index in range(expert_gates.shape[1]):
                    row[f"expert_gate_{expert_index}"] = float(
                        expert_gates[index, expert_index]
                    )
            row.update(
                {f"probability_{label}": float(probabilities[index, offset]) for offset, label in enumerate(LABELS)}
            )
            row.update(
                {
                    f"anchor_probability_{label}": float(
                        anchor_probabilities[index, offset]
                    )
                    for offset, label in enumerate(LABELS)
                }
            )
            row.update(
                {
                    f"anchor_aux_{offset}": float(anchor_auxiliary[index, offset])
                    for offset in range(anchor_auxiliary.shape[1])
                }
            )
            rows.append(row)
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "ua": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }
    return metrics, pd.DataFrame(rows)


def anchor_metrics(frame: pd.DataFrame) -> dict[str, float]:
    columns = [f"anchor_log_probability_{index}" for index in range(4)]
    predicted = frame[columns].to_numpy().argmax(axis=1)
    truth = frame["emotion_id"].to_numpy()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "ua": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    anchor_path = resolve(args.anchor)
    output_dir = resolve(args.output_dir)
    config = ResidualConfig(
        seed=int(args.seed),
        dropout=float(args.dropout),
        branch_dropout=float(args.branch_dropout),
        learning_rate=float(args.learning_rate),
        hidden_size=int(args.hidden_size),
        batch_size=int(args.batch_size),
        residual_penalty=float(args.residual_penalty),
        ema_decay=float(args.ema_decay),
        correction_gate_initial_bias=float(args.correction_gate_initial_bias),
        minimum_required_ua_gain=float(args.minimum_required_ua_gain),
        min_delta=float(args.min_delta),
        patience=int(args.patience),
        max_epochs=int(args.max_epochs),
        anchor_aux_size=int(args.anchor_aux_size),
        class_weight_power=float(args.class_weight_power),
        anchor_consistency_weight=float(args.anchor_consistency_weight),
        consensus_aux_index=int(args.consensus_aux_index),
        auxiliary_classification_weight=float(
            args.auxiliary_classification_weight
        ),
        branch_auxiliary_weight=float(args.branch_auxiliary_weight),
        interaction_fusion=bool(args.interaction_fusion),
        dual_residual_experts=bool(args.dual_residual_experts),
        class_conditional_experts=bool(args.class_conditional_experts),
        cosine_auxiliary_head=bool(args.cosine_auxiliary_head),
        cosine_temperature=float(args.cosine_temperature),
        hybrid_auxiliary_head=bool(args.hybrid_auxiliary_head),
        attentive_pooling=bool(args.attentive_pooling),
        dual_statistics_pooling=bool(args.dual_statistics_pooling),
        uncertainty_film=bool(args.uncertainty_film),
        anchor_conditioned_branch_gate=bool(args.anchor_conditioned_branch_gate),
        anchor_residual_conditioning=bool(args.anchor_residual_conditioning),
        feature_augmentation=bool(args.feature_augmentation),
        augmentation_noise_std=float(args.augmentation_noise_std),
        augmentation_time_mask_ratio=float(args.augmentation_time_mask_ratio),
        augmentation_frequency_mask_ratio=float(
            args.augmentation_frequency_mask_ratio
        ),
        ssl_residual_branch=bool(args.ssl_residual_branch),
        segment_context_branch=bool(args.segment_context_branch),
        segment_count=int(args.segment_count),
        temporal_delta_branch=bool(args.temporal_delta_branch),
        temporal_delta_lags=tuple(
            int(value) for value in args.temporal_delta_lags.split(",")
        ),
        cross_modal_attention=bool(args.cross_modal_attention),
        cross_attention_stride=int(args.cross_attention_stride),
        quality_auxiliary_weight=float(args.quality_auxiliary_weight),
        anchor_error_auxiliary_weight=float(args.anchor_error_auxiliary_weight),
        classwise_correction_gate=bool(args.classwise_correction_gate),
        logit_aware_correction_gate=bool(args.logit_aware_correction_gate),
        identity_centered_logit_gate=bool(args.identity_centered_logit_gate),
        gate_direction_weight=float(args.gate_direction_weight),
        speaker_conditioning=bool(args.speaker_conditioning),
        metadata_film=bool(args.metadata_film),
        metadata_residual_concat=bool(args.metadata_residual_concat),
        speaker_prior_branch=bool(args.speaker_prior_branch),
        speaker_logit_adapter=bool(args.speaker_logit_adapter),
        anchor_augmentation=bool(args.anchor_augmentation),
        anchor_noise_std=float(args.anchor_noise_std),
        anchor_temperature_max=float(args.anchor_temperature_max),
        happy_neutral_specialist_weight=float(args.happy_neutral_specialist_weight),
        focal_gamma=float(args.focal_gamma),
        label_smoothing=float(args.label_smoothing),
        supervised_contrastive_weight=float(args.supervised_contrastive_weight),
        supervised_contrastive_temperature=float(
            args.supervised_contrastive_temperature
        ),
        cross_modal_alignment_weight=float(args.cross_modal_alignment_weight),
        cross_modal_alignment_temperature=float(
            args.cross_modal_alignment_temperature
        ),
        branch_token_transformer=bool(args.branch_token_transformer),
        deep_prosody_backbone=bool(args.deep_prosody_backbone),
        deep_spectral_backbone=bool(args.deep_spectral_backbone),
        anchor_uncertainty_weight=float(args.anchor_uncertainty_weight),
        mask_disagreement=bool(args.mask_disagreement),
        branch_mode=str(args.branch_mode),
        branch_fusion=str(args.branch_fusion),
        neutralize_branch_quality=bool(args.neutralize_branch_quality),
        neutral_branch_quality_values=tuple(
            float(value)
            for value in str(args.neutral_branch_quality_values).split(",")
            if value.strip()
        ),
        shuffle_branch_quality=bool(args.shuffle_branch_quality),
        quality_feature_modulation=bool(args.quality_feature_modulation),
        disagreement_mode=str(args.disagreement_mode),
        correct_anchor_consistency_weight=float(args.correct_anchor_consistency_weight),
        force_correction_alpha_one=bool(args.force_correction_alpha_one),
        static_anchor_residual=bool(args.static_anchor_residual),
        prosody_feature_mask=str(args.prosody_feature_mask),
        spectral_feature_mask=str(args.spectral_feature_mask),
        preload_workers=int(args.preload_workers),
        initialize_from_checkpoint=str(args.initialize_from_checkpoint),
        freeze_non_gate=bool(args.freeze_non_gate),
    )
    seed_everything(config.seed)
    train, validation = load_frames(
        anchor_path,
        resolve(args.train_manifest),
        resolve(args.validation_manifest),
        int(args.expected_train_size),
        int(args.expected_validation_size),
    )
    if config.speaker_conditioning:
        config = replace(
            config,
            speaker_vocab_size=int(train["speaker_index"].max()) + 1,
            accent_vocab_size=int(train["accent_index"].max()) + 1,
            gender_vocab_size=int(train["gender_index"].max()) + 1,
        )
    train_dataset = ResidualDataset(
        train,
        config.max_frames,
        config.anchor_aux_size,
        augment=config.feature_augmentation,
        seed=config.seed,
        noise_std=config.augmentation_noise_std,
        time_mask_ratio=config.augmentation_time_mask_ratio,
        frequency_mask_ratio=config.augmentation_frequency_mask_ratio,
        anchor_augment=config.anchor_augmentation,
        anchor_noise_std=config.anchor_noise_std,
        anchor_temperature_max=config.anchor_temperature_max,
        mask_disagreement=config.mask_disagreement,
        prosody_feature_mask=config.prosody_feature_mask,
        spectral_feature_mask=config.spectral_feature_mask,
        static_anchor_only=config.static_anchor_residual,
    )
    validation_dataset = ResidualDataset(
        validation,
        config.max_frames,
        config.anchor_aux_size,
        mask_disagreement=config.mask_disagreement,
        prosody_feature_mask=config.prosody_feature_mask,
        spectral_feature_mask=config.spectral_feature_mask,
        static_anchor_only=config.static_anchor_residual,
    )
    if config.neutralize_branch_quality:
        if config.shuffle_branch_quality:
            raise ValueError("Choose neutralized or shuffled branch quality, not both")
        if config.neutral_branch_quality_values:
            if len(config.neutral_branch_quality_values) != 3:
                raise ValueError("Neutral branch-quality vector must have exactly 3 values")
            neutral_quality = torch.tensor(
                config.neutral_branch_quality_values, dtype=torch.float32
            )
        else:
            neutral_quality = torch.from_numpy(
                np.stack([
                    train_dataset._load_auxiliary(index)[2]
                    for index in range(len(train_dataset))
                ]).mean(axis=0).astype(np.float32)
            )
        train_dataset.branch_quality_neutral = neutral_quality
        validation_dataset.branch_quality_neutral = neutral_quality
    if config.shuffle_branch_quality:
        for split_index, dataset in enumerate((train_dataset, validation_dataset)):
            quality = torch.from_numpy(
                np.stack([
                    dataset._load_auxiliary(index)[2]
                    for index in range(len(dataset))
                ]).astype(np.float32)
            )
            generator = torch.Generator().manual_seed(2025 + split_index)
            dataset.branch_quality_override = quality[
                torch.randperm(len(dataset), generator=generator)
            ]
    if config.disagreement_mode == "permuted":
        for split_index, dataset in enumerate((train_dataset, validation_dataset)):
            generator = np.random.default_rng(2025 + split_index)
            dataset.anchor_aux = dataset.anchor_aux[
                generator.permutation(len(dataset))
            ].copy()
    if not config.static_anchor_residual and config.preload_workers > 0:
        print(
            f"preloading acoustic cache with {config.preload_workers} workers",
            flush=True,
        )
        for dataset in (train_dataset, validation_dataset):
            with ThreadPoolExecutor(max_workers=config.preload_workers) as pool:
                list(pool.map(dataset._load_auxiliary, range(len(dataset))))
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=True,
    )
    train_evaluation_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    baseline_train = anchor_metrics(train)
    baseline_validation = anchor_metrics(validation)
    model = AnchoredResidualPitchFusion(config).to("cuda")
    if config.initialize_from_checkpoint:
        checkpoint_path = resolve(config.initialize_from_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_state = checkpoint["model_state"]
        target_state = model.state_dict()
        compatible_state = {
            name: value
            for name, value in source_state.items()
            if name in target_state
            and target_state[name].shape == value.shape
            and not name.startswith("correction_gate.")
            and not name.startswith("logit_correction_gate.")
        }
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        print(
            f"initialized {len(compatible_state)} tensors from {checkpoint_path}; "
            f"new_or_unloaded={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    if config.freeze_non_gate:
        if model.logit_correction_gate is None:
            raise ValueError("--freeze-non-gate requires --logit-aware-correction-gate")
        if not config.initialize_from_checkpoint:
            raise ValueError("--freeze-non-gate requires --initialize-from-checkpoint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.logit_correction_gate.parameters():
            parameter.requires_grad_(True)
    device = torch.device("cuda")
    class_weights = None
    if config.class_weight_power > 0:
        counts = np.bincount(train["emotion_id"].to_numpy(), minlength=len(LABELS))
        weights = np.power(counts.astype(np.float64), -config.class_weight_power)
        weights /= weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    ema_state = copy.deepcopy(model.state_dict()) if config.ema_decay > 0 else None
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = baseline_validation["ua"]
    bad_epochs = 0
    overfit_streak = 0
    stop_reason = "max_epochs"
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = calculate_loss(
                output,
                batch["label"],
                config.residual_penalty,
                class_weights,
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
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_state is not None:
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        if value.is_floating_point():
                            ema_state[name].mul_(config.ema_decay).add_(
                                value, alpha=1.0 - config.ema_decay
                            )
                        else:
                            ema_state[name].copy_(value)
        live_state = None
        if ema_state is not None:
            live_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_state)
        train_metrics, _ = evaluate(
            model, train_evaluation_loader, device, config.residual_penalty,
            class_weights, config.anchor_consistency_weight,
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
        validation_metrics, _ = evaluate(
            model, validation_loader, device, config.residual_penalty,
            class_weights, config.anchor_consistency_weight,
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
        evaluated_state = copy.deepcopy(model.state_dict())
        if live_state is not None:
            model.load_state_dict(live_state)
        gap = train_metrics["ua"] - validation_metrics["ua"]
        improved = validation_metrics["ua"] > best_score + config.min_delta
        if improved:
            best_score = validation_metrics["ua"]
            best_epoch = epoch
            best_state = evaluated_state
            bad_epochs = 0
        else:
            bad_epochs += 1
        if gap > config.maximum_train_validation_gap and not improved:
            overfit_streak += 1
        else:
            overfit_streak = 0
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "train_validation_ua_gap": gap,
                "improved_over_anchor": improved,
            }
        )
        print(
            f"residual_v2 epoch={epoch:02d} train_ua={train_metrics['ua']:.4f} "
            f"val_ua={validation_metrics['ua']:.4f} val_f1={validation_metrics['macro_f1']:.4f} "
            f"gap={gap:.4f} best={best_score:.4f} bad={bad_epochs}/{config.patience} "
            f"overfit={overfit_streak}/{config.overfit_patience}",
            flush=True,
        )
        if overfit_streak >= config.overfit_patience:
            stop_reason = "overfit_guard"
            break
        if bad_epochs >= config.patience:
            stop_reason = "early_stopping"
            break
    model.load_state_dict(best_state)
    final_validation, predictions = evaluate(
        model, validation_loader, device, config.residual_penalty, class_weights,
        config.anchor_consistency_weight, config.correct_anchor_consistency_weight,
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
    delta = final_validation["ua"] - baseline_validation["ua"]
    final_gap = (
        history[best_epoch - 1]["train_validation_ua_gap"] if best_epoch > 0
        else baseline_train["ua"] - baseline_validation["ua"]
    )
    decision = (
        "GO_FULL_RESIDUAL"
        if delta >= config.minimum_required_ua_gain
        and final_gap <= config.maximum_train_validation_gap
        else "STOP_NO_RELIABLE_VALIDATION_GAIN"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": best_state, "config": asdict(config), "best_epoch": best_epoch},
        output_dir / "residual_v2_best.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    report = {
        "experiment": "anchored_residual_pitch_fusion_v2_validation_pilot",
        "anchor_path": str(anchor_path),
        "protocol": str(args.protocol_name),
        "test_loaded_or_evaluated": False,
        "config": asdict(config),
        "parameter_count": {
            "total": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable": int(sum(parameter.numel() for parameter in trainable_parameters)),
        },
        "baseline_anchor": {
            "train": baseline_train,
            "validation": baseline_validation,
        },
        "residual": {
            "best_epoch": best_epoch,
            "stopped_epoch": int(history[-1]["epoch"]),
            "stop_reason": stop_reason,
            "validation": final_validation,
            "validation_ua_delta": delta,
            "train_validation_ua_gap_at_selected_checkpoint": final_gap,
            "mean_correction_strength": float(predictions["correction_strength"].mean()),
            "mean_residual_l2": float(predictions["residual_l2"].mean()),
            "mean_prosody_gate": float(predictions["gate_prosody"].mean()),
            "mean_spectral_gate": float(predictions["gate_spectral"].mean()),
        },
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    for gate_name in (
        "gate_ssl", "gate_segment_context", "gate_temporal_delta",
        "gate_cross_modal",
        "gate_metadata",
        "gate_speaker_prior",
    ):
        if gate_name in predictions:
            report["residual"][f"mean_{gate_name}"] = float(
                predictions[gate_name].mean()
            )
    expert_gate_columns = sorted(
        column for column in predictions if column.startswith("expert_gate_")
    )
    if expert_gate_columns:
        report["residual"]["mean_expert_gates"] = {
            column: float(predictions[column].mean())
            for column in expert_gate_columns
        }
    (output_dir / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "baseline_validation_ua": baseline_validation["ua"],
                "residual_validation_ua": final_validation["ua"],
                "validation_ua_delta": delta,
                "decision": decision,
                "test_loaded_or_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
