from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets import InputAvailabilityDataset
from models import Encoder, FrozenEncoderExit
from phase5 import TaskSpec


class _Samples(Dataset):
    def __init__(self):
        self.sample_shape = (4, 32, 64)
        self.samples = torch.arange(2 * 4 * 32 * 64, dtype=torch.float32).reshape(2, 4, 32, 64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], torch.tensor(index), {"source": index}


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "input_information_exits.py"
    spec = importlib.util.spec_from_file_location("input_information_exits", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "input_information_report.py"
    spec = importlib.util.spec_from_file_location("input_information_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InputAvailabilityDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _Samples()

    def test_antenna_prefix_preserves_labels_metadata_and_order(self):
        view = InputAvailabilityDataset(self.dataset, mode="antenna_prefix", level=2)
        sample, label, metadata = view[1]
        self.assertEqual(view.sample_shape, (2, 32, 64))
        torch.testing.assert_close(sample, self.dataset.samples[1, :2])
        self.assertEqual(label.item(), 1)
        self.assertEqual(metadata, {"source": 1})

    def test_leading_temporal_window_is_patch_aligned_and_nested(self):
        short = InputAvailabilityDataset(self.dataset, mode="leading_temporal_patches", level=2)
        full = InputAvailabilityDataset(self.dataset, mode="leading_temporal_patches", level=4)
        short_sample = short[0][0]
        full_sample = full[0][0]
        self.assertEqual(short.sample_shape, (4, 32, 32))
        torch.testing.assert_close(short_sample, self.dataset.samples[0, ..., :32])
        torch.testing.assert_close(short_sample, full_sample[..., :32])
        torch.testing.assert_close(full_sample, self.dataset.samples[0])

    def test_invalid_availability_levels_fail_without_mutating_base(self):
        with self.assertRaises(ValueError):
            InputAvailabilityDataset(self.dataset, mode="antenna_prefix", level=5)
        with self.assertRaises(ValueError):
            InputAvailabilityDataset(self.dataset, mode="leading_temporal_patches", level=5)
        self.assertEqual(self.dataset.sample_shape, (4, 32, 64))


class InputInformationRunnerValidationTest(unittest.TestCase):
    def test_har_antenna_study_uses_only_available_channels(self):
        runner = _runner_module()
        study = runner.STUDIES["antenna-human-activity"]
        self.assertEqual(study["task"], "human-activity")
        self.assertEqual(study["mode"], "antenna_prefix")
        self.assertEqual(study["levels"], (1, 2, 3))
        self.assertEqual(runner._condition_name("antenna-human-activity", 3), "C3")

    def test_saved_head_loads_and_backbone_remains_frozen(self):
        runner = _runner_module()
        encoder = Encoder(img_size=(32, 32), patch_size=8, embed_dim=16, depth=8, num_heads=2)
        reference = FrozenEncoderExit(encoder, exit_layer=2, num_outputs=3)
        spec = TaskSpec(
            name="toy",
            path="toy.h5",
            target_type="classification",
            num_outputs=3,
            higher_is_better=True,
            metric_name="macro_accuracy",
            sample_shape=(1, 32, 32),
        )
        with tempfile.TemporaryDirectory() as directory:
            head_path = Path(directory) / "L2.pt"
            torch.save({"head": reference.head.state_dict()}, head_path)
            loaded = runner._load_head(encoder, exit_layer=2, spec=spec, path=head_path, device=torch.device("cpu"))
            logits = loaded(torch.randn(2, 1, 32, 32))
            logits.square().mean().backward()
            self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
            self.assertTrue(all(parameter.grad is not None for parameter in loaded.head.parameters()))

    def test_missing_required_heads_fail_before_inference(self):
        runner = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            manifest = root / "splits.json"
            source = root / "exit_metrics.json"
            checkpoint.write_bytes(b"checkpoint")
            manifest.write_text(json.dumps({"tasks": {}}))
            source.write_text(json.dumps({
                "method": "p0",
                "checkpoint": str(checkpoint),
                "split_manifest": str(manifest),
                "protocol": {"tolerance": 0.05},
                "tasks": {},
            }))
            with self.assertRaisesRegex(RuntimeError, "nrpos-outdoor.*human-activity"):
                runner._validate_source_metrics(source)

    def test_test_stage_rejects_non_selection_artifact_before_accessing_data(self):
        runner = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.json"
            artifact.write_text(json.dumps({"schema_version": 1, "stage": "incomplete"}))
            with self.assertRaisesRegex(ValueError, "completed selection artifact"):
                runner._run_test(artifact, object(), torch.device("cpu"))

    def test_relative_head_paths_are_resolved_from_the_source_run_root(self):
        runner = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source-repository"
            run = root / "runs" / "formal" / "p0" / "downstream"
            run.mkdir(parents=True)
            (root / "checkpoint.ckpt").write_bytes(b"checkpoint")
            (root / "splits.json").write_text(json.dumps({"tasks": {}}))
            tasks = {}
            for task in ("nrpos-outdoor", "human-activity"):
                exits = {}
                for layer in (2, 4, 6, 8):
                    head = root / "heads" / task / f"L{layer}.pt"
                    head.parent.mkdir(parents=True, exist_ok=True)
                    head.write_bytes(b"head")
                    exits[str(layer)] = {"head_path": str(head.relative_to(root))}
                tasks[task] = {"exits": exits}
            source = run / "exit_metrics.json"
            source.write_text(json.dumps({
                "method": "p0",
                "checkpoint": "checkpoint.ckpt",
                "split_manifest": "splits.json",
                "protocol": {"tolerance": 0.05},
                "tasks": tasks,
            }))
            _, checkpoint, manifest, source_root, _ = runner._validate_source_metrics(source)
            self.assertEqual(checkpoint, root / "checkpoint.ckpt")
            self.assertEqual(manifest, root / "splits.json")
            self.assertEqual(source_root, root)

    def test_report_rows_include_validation_gaps_and_selected_compute(self):
        report = _report_module()
        study = {
            "task": "nrpos-outdoor",
            "task_spec": {"metric_name": "mean_distance_error"},
            "conditions": [{
                "condition": "C4",
                "available_channels": 4,
                "input_shape": [4, 224, 224],
                "token_count": 784,
                "validation": {str(layer): {"mean_distance_error": float(layer)} for layer in (2, 4, 6, 8)},
                "validation_l8_relative_gap": {str(layer): float(layer - 8) for layer in (2, 4, 6, 8)},
                "selected_exit_earliest_within_tolerance": 6,
                "selected_exit_compute": {"flops": 123, "latency_ms": {"median": 1.5}},
            }],
        }
        row = report._rows("antenna", study)[0]
        self.assertEqual(row["selected_exit"], 6)
        self.assertEqual(row["selected_flops"], 123)
        self.assertEqual(row["validation_L2_mean_distance_error"], 2.0)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            report._write_antenna_exit_plot(output_dir, study, stem="antenna_exit_vs_channels")
            self.assertTrue((output_dir / "antenna_exit_vs_channels.png").is_file())
            self.assertTrue((output_dir / "antenna_exit_vs_channels.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
