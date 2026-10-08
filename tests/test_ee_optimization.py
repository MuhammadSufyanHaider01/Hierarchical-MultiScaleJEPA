from __future__ import annotations

import json
import sys

import pytest
import torch

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from models import (  # noqa: E402
    CausalFusionExitHead, Encoder, FrozenEncoderFusionExit,
    FrozenEncoderFusionTeacherStudent,
)
from phase5_fusion_distill import validate_teacher  # noqa: E402
from phase5_fusion_distill import distillation_loss  # noqa: E402
from ee_optimization_jobs import SEEDS as JOB_SEEDS, job_script  # noqa: E402
import ee_optimization_report as ee_report  # noqa: E402


def test_fusion_weights_are_normalized_and_l2_is_baseline_equivalent():
    torch.manual_seed(4)
    tokens = torch.randn(2, 7, 12)
    single = CausalFusionExitHead(12, 3, [2])
    baseline = torch.nn.Sequential(torch.nn.LayerNorm(12), torch.nn.Linear(12, 3))
    baseline[0].load_state_dict(single.norm.state_dict())
    baseline[1].load_state_dict(single.projection.state_dict())
    torch.testing.assert_close(single({2: tokens}), baseline(tokens.mean(dim=1)), rtol=0, atol=0)
    fusion = CausalFusionExitHead(12, 3, [2, 4, 6])
    assert fusion.fusion_weights == {2: pytest.approx(1 / 3), 4: pytest.approx(1 / 3), 6: pytest.approx(1 / 3)}
    assert all(value >= 0 for value in fusion.fusion_weights.values())
    assert sum(fusion.fusion_weights.values()) == pytest.approx(1.0)
    with torch.no_grad():
        fusion.fusion_logits.copy_(torch.tensor([-100.0, -100.0, 100.0]))
    pooled = torch.stack([tokens.mean(1)] * 3, dim=1)
    expected = fusion.projection(fusion.norm(pooled[:, 2]))
    torch.testing.assert_close(fusion({2: tokens, 4: tokens, 6: tokens}), expected, rtol=1e-6, atol=1e-6)


def test_fusion_head_rejects_bad_mapping_and_shapes():
    head = CausalFusionExitHead(8, 2, [2, 4])
    with pytest.raises(ValueError, match="missing"):
        head({2: torch.randn(1, 3, 8)})
    with pytest.raises(ValueError, match="shape"):
        head({2: torch.randn(1, 3, 8), 4: torch.randn(1, 8)})


def test_fusion_and_teacher_student_freeze_backbone_and_teacher():
    torch.manual_seed(3)
    encoder = Encoder(img_size=(16, 16), patch_size=8, embed_dim=8, depth=8, num_heads=2).eval()
    inputs = torch.randn(2, 1, 16, 16)
    fusion = FrozenEncoderFusionExit(encoder, 4, 3)
    fusion(inputs).sum().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    teacher = CausalFusionExitHead(8, 3, [2, 4])
    model = FrozenEncoderFusionTeacherStudent(encoder, 4, 3, teacher.state_dict())
    before = {name: value.clone() for name, value in model.teacher.state_dict().items()}
    student, soft = model(inputs)
    (student.square().mean() + soft.detach().square().mean()).backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.teacher.state_dict().items())
    assert all(parameter.grad is not None for parameter in model.student.parameters())


def test_teacher_validation_rejects_identity_mismatch(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    metrics = tmp_path / "exit_metrics.json"
    model = CausalFusionExitHead(4, 2, [2])
    torch.save({"head_algorithm": "causal_fusion", "exit_layer": 2,
                "source_layers": [2], "head": model.state_dict(), "epoch": 3}, checkpoint)
    metrics.write_text(json.dumps({"head_algorithm": "causal_fusion", "method": "p0",
        "pretrain_seed": 0, "head_seed": 1, "checkpoint_sha256": "ckpt", "split_manifest_sha256": "split",
        "tasks": {"rfs": {"exits": {"2": {"head_path": str(checkpoint), "best_epoch": 3}}}}}))
    with pytest.raises(ValueError, match="head_seed mismatch"):
        validate_teacher(checkpoint, metrics, method="p0", pretrain_seed=0, head_seed=2,
                         task="rfs", layer=2, checkpoint_sha="ckpt", split_sha="split")


def test_zero_kd_weight_is_task_loss_for_classification_and_position():
    from types import SimpleNamespace
    labels = torch.tensor([0, 1])
    student = torch.randn(2, 3, requires_grad=True)
    teacher = torch.randn(2, 3)
    ce = torch.nn.CrossEntropyLoss()
    classification = SimpleNamespace(target_type="classification")
    actual = distillation_loss(student, teacher, labels, classification, ce, temperature=2.0, weight=0.0)
    torch.testing.assert_close(actual, ce(student, labels))
    target = torch.randn(2, 2)
    regression = SimpleNamespace(target_type="position")
    student_pos, teacher_pos = torch.randn(2, 2), torch.randn(2, 2)
    actual_pos = distillation_loss(student_pos, teacher_pos, target, regression, ce, temperature=2.0, weight=0.0)
    torch.testing.assert_close(actual_pos, torch.nn.functional.mse_loss(student_pos, target))


def test_stage_job_commands_and_missing_report_input_fail_closed(tmp_path, monkeypatch):
    assert JOB_SEEDS == tuple(range(7))
    assert ee_report.SEEDS == tuple(range(7))
    fusion = job_script("fusion", "p2", 4, "gpu-a100")
    kd = job_script("fusion_kd", "p2", 4, "gpu-a100")
    benchmark = job_script("benchmark", "p2", 0, "gpu-a100")
    assert "phase5_fusion_exits.py" in fusion
    assert "phase5_fusion_distill.py" in kd
    assert "--fusion-teacher-root" in kd
    assert "--head-seed 4" in kd
    assert "phase5_fusion_benchmark.py" in benchmark
    assert "--measure-iterations 200" in benchmark
    monkeypatch.setattr(ee_report, "RUNS", tmp_path)
    with pytest.raises(FileNotFoundError, match="required causal_fusion packet missing"):
        ee_report.read_packet("p0", "causal_fusion", 0)
