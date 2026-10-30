from __future__ import annotations

import torch

from rapf.training.train_residual_adapter import AnchoredResidualPitchFusion, ResidualConfig


def test_paper_configuration_has_expected_parameter_count() -> None:
    config = ResidualConfig(anchor_aux_size=9, force_correction_alpha_one=True)
    model = AnchoredResidualPitchFusion(config)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert total == 73_083
    assert trainable == 70_522


def test_final_logits_are_direct_residual_addition() -> None:
    torch.manual_seed(2025)
    config = ResidualConfig(anchor_aux_size=9, force_correction_alpha_one=True)
    model = AnchoredResidualPitchFusion(config).eval()
    batch_size, frames = 2, 32
    batch = {
        "prosody": torch.randn(batch_size, 6, frames),
        "spectral": torch.randn(batch_size, 3, 80, frames),
        "length": torch.full((batch_size,), frames, dtype=torch.long),
        "anchor": torch.randn(batch_size, 4),
        "anchor_aux": torch.randn(batch_size, 9),
        "branch_quality": torch.rand(batch_size, 3),
        "quality": torch.rand(batch_size, 3),
    }

    with torch.no_grad():
        output = model(batch)

    assert torch.all(output["correction_strength"] == 1)
    assert torch.allclose(
        output["logits"], output["anchor_logits"] + output["raw_residual"]
    )
