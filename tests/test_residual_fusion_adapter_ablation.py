from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from models import (  # noqa: E402
    ADAPTER_ABLATION_VARIANTS,
    Encoder,
    FrozenEncoderResidualFusionAblationExit,
    ResidualFusionAblationExitHead,
)
from phase5_residual_fusion_ablation_exits import optimizer_for, parameter_counts  # noqa: E402
import residual_fusion_adapter_ablation_jobs as jobs  # noqa: E402
import residual_fusion_adapter_ablation_report as report  # noqa: E402


def representations(dim=8):
    return {2: torch.randn(3, 4, dim), 4: torch.randn(3, 3, dim), 6: torch.randn(3, 2, dim), 8: torch.randn(3, 1, dim)}


@pytest.mark.parametrize("variant", ADAPTER_ABLATION_VARIANTS)
def test_zero_residual_identity_gate_shape_and_source_configuration(variant):
    torch.manual_seed(4)
    head = ResidualFusionAblationExitHead(8, 3, 6, (2, 4), variant, bottleneck_dim=5, gate_dim=4)
    reps = representations()
    fused, diagnostics = head._fuse_features(reps, return_diagnostics=True)
    torch.testing.assert_close(fused, reps[6].mean(dim=1), rtol=0, atol=0)
    assert set(diagnostics) == {2, 4}
    assert all(bool(((entry["gate"] > 0) & (entry["gate"] < 1)).all()) for entry in diagnostics.values())
    assert not hasattr(head, "fusion_logits")
    if variant.startswith("bottleneck_"):
        assert all(module.in_features == 5 for module in head.source_gate_projections.values())
        assert len(head.single_projections) == 0
    else:
        assert all(module.in_features == 8 for module in head.source_gate_projections.values())
        assert len(head.down_projections) == len(head.up_projections) == 0


def test_l2_variants_have_no_source_modules_and_are_seeded_identically():
    tokens = torch.randn(2, 3, 8)
    outputs = []
    for variant in ADAPTER_ABLATION_VARIANTS:
        torch.manual_seed(91)
        head = ResidualFusionAblationExitHead(8, 2, 2, (), variant)
        assert not head.source_norms and not head.source_gate_projections and head.anchor_gate_projection is None
        outputs.append(head({2: tokens}))
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


def test_centered_sigmoid_and_first_step_adapter_gradients():
    for variant in ("single_sigmoid", "single_linear", "bottleneck_linear"):
        torch.manual_seed(12)
        head = ResidualFusionAblationExitHead(8, 2, 4, (2,), variant, bottleneck_dim=4, gate_dim=4)
        source = torch.randn(3, 8)
        correction, _ = head._source_correction_and_gate_feature(2, source)
        torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0, atol=0)
        head(representations()).sum().backward()
        if variant.startswith("bottleneck"):
            assert torch.count_nonzero(head.up_projections["2"].weight.grad) > 0
        else:
            assert torch.count_nonzero(head.single_projections["2"].weight.grad) > 0


def test_validation_and_frozen_prefix_and_optimizer_coverage():
    with pytest.raises(ValueError, match="adapter_variant"):
        ResidualFusionAblationExitHead(8, 2, 4, (2,), "wrong")
    with pytest.raises(ValueError, match="requires source_layers"):
        ResidualFusionAblationExitHead(8, 2, 6, (2,), "single_linear")
    encoder = Encoder(img_size=(16, 16), patch_size=8, embed_dim=8, depth=8, num_heads=2)
    calls = [0] * 8
    hooks = [block.register_forward_hook(lambda _m, _i, _o, index=index: calls.__setitem__(index, calls[index] + 1)) for index, block in enumerate(encoder.blocks)]
    model = FrozenEncoderResidualFusionAblationExit(encoder, 6, 2, "single_gelu")
    model(torch.randn(2, 1, 16, 16)).sum().backward()
    assert calls[:6] == [1] * 6 and calls[6:] == [0, 0]
    assert all(parameter.grad is None for parameter in encoder.parameters())
    args = type("Args", (), {"learning_rate": 1e-3, "weight_decay": .05})()
    optimizer = optimizer_for(model, args)
    groups = [{id(parameter) for parameter in group["params"]} for group in optimizer.param_groups]
    assert not groups[0] & groups[1]
    assert groups[0] | groups[1] == {id(parameter) for parameter in model.head.parameters()}
    assert parameter_counts(model.head)["total"] == sum(parameter.numel() for parameter in model.head.parameters())
    for hook in hooks:
        hook.remove()


def test_report_fails_closed_and_job_layout_is_ten_jobs(tmp_path):
    with pytest.raises(FileNotFoundError, match="required current packet missing"):
        report.load_packet("p0", "current", current_root=tmp_path, gelu_root=tmp_path, ablation_root=tmp_path)
    scripts = [jobs.job_script(method, variant, "gpu-l40", tmp_path) for variant in jobs.VARIANTS for method in jobs.METHODS]
    assert len(scripts) == 10
    assert all("#SBATCH --gres=gpu:l40:1" in script and "--head-seed 0" in script for script in scripts)
    assert "--adapter-variant single_sigmoid" in jobs.job_script("p2", "single_sigmoid", "gpu-l40", tmp_path)
