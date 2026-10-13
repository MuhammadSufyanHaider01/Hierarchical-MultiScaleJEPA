from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from models import Encoder, FrozenEncoderResidualFusionExit, ResidualProjectedFusionExitHead  # noqa: E402
from phase5_residual_fusion_exits import optimizer_for  # noqa: E402
import residual_fusion_v2_report as v2_report  # noqa: E402


def test_l2_is_exact_anchor_path_and_zero_up_projections():
    torch.manual_seed(11)
    head = ResidualProjectedFusionExitHead(8, 3, 2, [])
    tokens = torch.randn(4, 5, 8)
    baseline_norm = torch.nn.LayerNorm(8)
    baseline_linear = torch.nn.Linear(8, 3)
    baseline_norm.load_state_dict(head.output_norm.state_dict())
    baseline_linear.load_state_dict(head.output_projection.state_dict())
    torch.testing.assert_close(head({2: tokens}), baseline_linear(baseline_norm(tokens.mean(1))), rtol=0, atol=0)
    assert not head.source_gate_projections
    assert head.anchor_gate_projection is None
    for up in head.up_projections.values():
        assert torch.count_nonzero(up.weight) == 0
        assert torch.count_nonzero(up.bias) == 0


def test_dynamic_gates_are_independent_bounded_and_shapes_are_checked():
    head = ResidualProjectedFusionExitHead(8, 2, 6, [2, 4], bottleneck_dim=5, gate_dim=3)
    reps = {2: torch.randn(3, 4, 8), 4: torch.randn(3, 2, 8), 6: torch.randn(3, 1, 8)}
    _, diagnostics = head(reps, return_diagnostics=True)
    assert set(diagnostics) == {2, 4}
    assert all(bool(((d["gate"] > 0) & (d["gate"] < 1)).all()) for d in diagnostics.values())
    assert not hasattr(head, "fusion_logits")
    with pytest.raises(ValueError, match="source_layers"):
        ResidualProjectedFusionExitHead(8, 2, 6, [2])
    with pytest.raises(ValueError, match="missing"):
        head({2: reps[2], 6: reps[6]})
    with pytest.raises(ValueError, match="shape"):
        head({2: torch.randn(3, 8), 4: reps[4], 6: reps[6]})
    with pytest.raises(ValueError, match="matching batch"):
        head({2: reps[2], 4: torch.randn(2, 2, 8), 6: reps[6]})


def test_frozen_encoder_and_l6_stops_after_prefix():
    encoder = Encoder(img_size=(16, 16), patch_size=8, embed_dim=8, depth=8, num_heads=2).eval()
    calls = [0] * len(encoder.blocks)
    hooks = [block.register_forward_hook(lambda _m, _i, _o, index=i: calls.__setitem__(index, calls[index] + 1))
             for i, block in enumerate(encoder.blocks)]
    model = FrozenEncoderResidualFusionExit(encoder, 6, 3)
    x = torch.randn(2, 1, 16, 16)
    model(x).sum().backward()
    assert calls[:6] == [1] * 6
    assert calls[6:] == [0, 0]
    assert all(p.grad is None for p in encoder.parameters())
    assert all(not p.requires_grad for p in encoder.parameters())
    assert not model.encoder.training
    for hook in hooks:
        hook.remove()


def test_optimizer_decay_groups_match_v2_protocol():
    model = FrozenEncoderResidualFusionExit(
        Encoder(img_size=(16, 16), patch_size=8, embed_dim=8, depth=8, num_heads=2), 4, 2)
    args = type("Args", (), {"learning_rate": 1e-3, "weight_decay": 0.05})()
    optimizer = optimizer_for(model, args)
    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    assert optimizer.param_groups[1]["weight_decay"] == 0
    assert id(model.head.down_projections["2"].weight) in decay_ids
    assert id(model.head.up_projections["2"].weight) in decay_ids
    assert id(model.head.output_projection.weight) in decay_ids
    for p in model.head.parameters():
        if p.ndim == 1 or p is model.head.anchor_gate_projection.weight or p is model.head.source_gate_projections["2"].weight:
            assert id(p) in no_decay_ids
    assert not (decay_ids & no_decay_ids)


def test_report_fails_closed_when_seed_zero_current_packet_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(v2_report, "BASE", tmp_path)
    with pytest.raises(FileNotFoundError, match="required current packet missing"):
        v2_report.load_packet("p0", "current")
