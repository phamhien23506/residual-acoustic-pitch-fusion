"""Pitch-Fusion and acoustic encoder building blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
)


class OfficialPitchFusionAttentionLayer(nn.Module):
    """Attention block ported from Lamnu1902/Pitch-fusion at commit 2e8e05f."""

    def __init__(self, dimension: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout)
        self.linear1 = nn.Linear(dimension, dimension * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dimension * 4, dimension)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_time_first = source.transpose(0, 1)
        target_time_first = target.transpose(0, 1)
        attended = self.self_attention(
            target_time_first,
            source_time_first,
            source_time_first,
            attn_mask=None,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )[0]
        output = self.norm1(source_time_first + self.dropout1(attended))
        transformed = self.linear2(self.dropout(F.relu(self.linear1(output))))
        return self.norm2(output + self.dropout2(transformed)).transpose(0, 1)


class OfficialWav2Vec2CrossAttentionPitchForSER(Wav2Vec2PreTrainedModel):
    """Faithful model architecture from the official Pitch-fusion repository."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.pitch_encoder = Wav2Vec2FeatureEncoder(config)
        self.pitch_projection = Wav2Vec2FeatureProjection(config)
        self.attention_audio_pitch = OfficialPitchFusionAttentionLayer(
            config.hidden_size, 16
        )
        self.attention_pitch_audio = OfficialPitchFusionAttentionLayer(
            config.hidden_size, 16
        )
        self.self_attention = OfficialPitchFusionAttentionLayer(
            config.hidden_size * 2, 16
        )
        self.projector = nn.Linear(
            config.hidden_size * 2, config.classifier_proj_size
        )
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def freeze_feature_extractor(self) -> None:
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        audio_input: torch.Tensor,
        pitch_input: torch.Tensor,
        labels: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        input_attention_mask = None
        if audio_lengths is not None:
            input_attention_mask = (
                torch.arange(audio_input.shape[1], device=audio_input.device)[None, :]
                < audio_lengths[:, None]
            ).long()
        acoustic = self.wav2vec2(
            audio_input, attention_mask=input_attention_mask
        ).last_hidden_state
        pitch = self.pitch_encoder(pitch_input).transpose(1, 2)
        pitch, _ = self.pitch_projection(pitch)
        width = min(acoustic.shape[1], pitch.shape[1])
        acoustic, pitch = acoustic[:, :width], pitch[:, :width]
        frame_padding_mask = None
        if audio_lengths is not None:
            frame_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                audio_lengths
            ).clamp(max=width)
            frame_padding_mask = (
                torch.arange(width, device=audio_input.device)[None, :]
                >= frame_lengths[:, None]
            )
        # The official attention graph is retained, but it runs in FP32 because
        # end-to-end BF16/FP16 updates are unstable for the raw-Hz pitch stream.
        with torch.autocast(device_type=acoustic.device.type, enabled=False):
            acoustic = acoustic.float()
            pitch = pitch.float()
            audio_pitch = self.attention_audio_pitch(
                acoustic, pitch, frame_padding_mask
            )
            pitch_audio = self.attention_pitch_audio(
                pitch, acoustic, frame_padding_mask
            )
            fused = torch.cat((audio_pitch, pitch_audio), dim=-1)
            fused = self.self_attention(fused, fused, frame_padding_mask)
            projected = self.projector(fused)
            if frame_padding_mask is None:
                pooled = projected.mean(dim=1)
            else:
                valid = (~frame_padding_mask).unsqueeze(-1).to(projected.dtype)
                pooled = (projected * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            logits = self.classifier(pooled)
            loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


class CrossAttentionBlock(nn.Module):
    """Paper-aligned cross/self-attention block with residual feed-forward layers."""

    def __init__(self, dimension: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, target: torch.Tensor, source: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            target, source, source, key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        output = self.norm1(target + self.dropout(attended))
        return self.norm2(output + self.dropout(self.feed_forward(output)))


class PitchSequenceEncoder(nn.Module):
    """Wav2Vec2-style Conv1D encoder for a waveform-length pitch sequence."""

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        kernels = (10, 3, 3, 3, 3, 2, 2)
        strides = (5, 2, 2, 2, 2, 2, 2)
        layers: list[nn.Module] = []
        source = 1
        for kernel, stride in zip(kernels, strides):
            layers.extend((nn.Conv1d(source, 512, kernel, stride=stride), nn.GELU()))
            source = 512
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden_size))

    def forward(self, pitch: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(pitch[:, None, :]).transpose(1, 2))


class PaperAlignedPitchFusionHead(nn.Module):
    """Bidirectional cross-attention head described by the official ViSEC code."""

    def __init__(
        self, hidden_size: int = 768, projection_size: int = 256,
        classes: int = 4, heads: int = 16, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pitch_encoder = PitchSequenceEncoder(hidden_size)
        self.audio_to_pitch = CrossAttentionBlock(hidden_size, heads, dropout)
        self.pitch_to_audio = CrossAttentionBlock(hidden_size, heads, dropout)
        self.self_attention = CrossAttentionBlock(hidden_size * 2, heads, dropout)
        self.projector = nn.Linear(hidden_size * 2, projection_size)
        self.classifier = nn.Linear(projection_size, classes)

    def forward(
        self, acoustic: torch.Tensor, pitch_waveform: torch.Tensor,
        frame_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pitch = self.pitch_encoder(pitch_waveform)
        width = min(acoustic.shape[1], pitch.shape[1])
        acoustic, pitch = acoustic[:, :width], pitch[:, :width]
        mask = frame_padding_mask[:, :width] if frame_padding_mask is not None else None
        acoustic_attended = self.audio_to_pitch(acoustic, pitch, mask)
        pitch_attended = self.pitch_to_audio(pitch, acoustic, mask)
        fused = self.self_attention(
            torch.cat((acoustic_attended, pitch_attended), dim=-1),
            torch.cat((acoustic_attended, pitch_attended), dim=-1), mask,
        )
        projected = self.projector(fused)
        if mask is None:
            pooled = projected.mean(dim=1)
        else:
            valid = (~mask).unsqueeze(-1).to(projected.dtype)
            pooled = (projected * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


def masked_temporal_statistics(
    sequence: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Return masked mean and standard deviation for B x C x T features."""

    width = sequence.shape[-1]
    mask = torch.arange(width, device=sequence.device)[None, :] < lengths[:, None]
    weights = mask[:, None, :].to(sequence.dtype)
    denominator = weights.sum(dim=-1).clamp_min(1.0)
    mean = (sequence * weights).sum(dim=-1) / denominator
    variance = (
        (sequence - mean[..., None]).square() * weights
    ).sum(dim=-1) / denominator
    return torch.cat((mean, variance.clamp_min(1e-6).sqrt()), dim=-1)


def temporal_delta(values: torch.Tensor) -> torch.Tensor:
    """Compute a deterministic first-order delta along the final dimension."""

    result = torch.zeros_like(values)
    if values.shape[-1] > 1:
        result[..., 1:-1] = (values[..., 2:] - values[..., :-2]) * 0.5
        result[..., 0] = values[..., 1] - values[..., 0]
        result[..., -1] = values[..., -1] - values[..., -2]
    return result


class MultiScaleProsodyCnn(nn.Module):
    """Encode F0, energy and voicing trajectories at three time scales."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
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
            nn.Conv1d(width * 3, hidden_size, kernel_size=1),
            nn.GroupNorm(8, hidden_size),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.mix(
            torch.cat([branch(values) for branch in self.branches], dim=1)
        )
        return self.output(masked_temporal_statistics(encoded, lengths))


class MultiViewSpectralCnn(nn.Module):
    """Encode log-Mel, delta and delta-delta maps with a compact 2-D CNN."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
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
        self.output = nn.Sequential(
            nn.Linear(64 * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, log_mel: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        delta = temporal_delta(log_mel)
        views = torch.stack((log_mel, delta, temporal_delta(delta)), dim=1)
        encoded = self.network(views).mean(dim=2)
        downsampled = torch.div(lengths + 1, 2, rounding_mode="floor")
        downsampled = torch.div(
            downsampled + 1, 2, rounding_mode="floor"
        ).clamp_min(1)
        return self.output(masked_temporal_statistics(encoded, downsampled))


class QualityAwareMultiViewPitchFusionForSER(nn.Module):
    """QAM-PF: Wav2Vec2, multi-scale prosody and spectral CNN fusion."""

    def __init__(
        self,
        wav2vec2: Wav2Vec2Model,
        hidden_size: int = 128,
        num_labels: int = 4,
        dropout: float = 0.45,
        branch_dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if hidden_size % 8:
            raise ValueError("hidden_size must be divisible by 8")
        self.wav2vec2 = wav2vec2
        self.branch_dropout = branch_dropout
        acoustic_width = int(wav2vec2.config.hidden_size)
        self.acoustic = nn.Sequential(
            nn.LayerNorm(acoustic_width * 2),
            nn.Linear(acoustic_width * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prosody = MultiScaleProsodyCnn(hidden_size, dropout)
        self.spectral = MultiViewSpectralCnn(hidden_size, dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3 + 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.auxiliary_classifiers = nn.ModuleList(
            [nn.Linear(hidden_size, num_labels) for _ in range(3)]
        )

    def freeze_feature_extractor(self) -> None:
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        audio_input: torch.Tensor,
        audio_lengths: torch.Tensor,
        prosody_input: torch.Tensor,
        log_mel_input: torch.Tensor,
        auxiliary_lengths: torch.Tensor,
        quality_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_mask = (
            torch.arange(audio_input.shape[1], device=audio_input.device)[None, :]
            < audio_lengths[:, None]
        )
        acoustic_sequence = self.wav2vec2(
            audio_input, attention_mask=input_mask.long()
        ).last_hidden_state
        acoustic_lengths = self.wav2vec2._get_feat_extract_output_lengths(
            audio_lengths
        ).clamp(max=acoustic_sequence.shape[1])
        acoustic_vector = self.acoustic(
            masked_temporal_statistics(
                acoustic_sequence.transpose(1, 2), acoustic_lengths
            )
        )
        branches = torch.stack(
            (
                acoustic_vector,
                self.prosody(prosody_input, auxiliary_lengths),
                self.spectral(log_mel_input, auxiliary_lengths),
            ),
            dim=1,
        )
        gate_logits = self.gate(
            torch.cat((branches.flatten(1), quality_input), dim=-1)
        )
        if self.training and self.branch_dropout > 0:
            optional_dropped = (
                torch.rand(branches.shape[0], 2, device=branches.device)
                < self.branch_dropout
            )
            branch_mask = torch.cat(
                (
                    torch.zeros(
                        branches.shape[0], 1, dtype=torch.bool,
                        device=branches.device,
                    ),
                    optional_dropped,
                ),
                dim=1,
            )
            gate_logits = gate_logits.masked_fill(branch_mask, -1e4)
        gates = gate_logits.softmax(dim=-1)
        fused = self.fusion((branches * gates[..., None]).sum(dim=1))
        auxiliary_logits = torch.stack(
            [
                classifier(branches[:, index])
                for index, classifier in enumerate(self.auxiliary_classifiers)
            ],
            dim=1,
        )
        return {
            "logits": self.classifier(fused),
            "gates": gates,
            "auxiliary_logits": auxiliary_logits,
        }
