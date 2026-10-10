import argparse
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import phase5_primary_report as report  # noqa: E402


class CNNOnlyReportTest(unittest.TestCase):
    def _split_manifest(self, path: Path, *, marker: int = 0) -> None:
        path.write_text(json.dumps({
            "seed": 2025,
            "marker": marker,
            "tasks": {task: {"spec": {"path": f"host-specific/{task}.h5"}} for task in report.TASK_ORDER[:6]},
        }))

    def _profile_metadata(self, name: str, *, resolved_marker: str = "shared") -> dict:
        profile, path, checksum = report.load_mask_profile(name)
        return {
            "name": name,
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": checksum,
            "overrides": profile["masks"],
            "resolved": {"profile": name, "marker": resolved_marker},
        }

    def _packet(
        self,
        root: Path,
        method: str,
        manifest: Path,
        *,
        mask_profile: dict | None = None,
        pretrain_split_marker: int = 0,
    ) -> None:
        seed_root = root / method / "seed_000"
        checkpoint = seed_root / "checkpoints" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{method}-checkpoint".encode())
        (seed_root / "experiment.json").write_text(json.dumps({
            "metadata": {
                "phase5": {"method": method},
                "seed": 0,
                "config": {"encoder": {"backbone": "convnext_v2_hier"}},
                "mask_profile": mask_profile,
            },
            "runtime": {
                "method": method,
                "seed": 0,
                "mask_profile": mask_profile,
                "microbatch_size": 16,
                "accumulate_grad_batches": 4,
                "effective_batch_size": 64,
                "max_epochs": 100,
                "schedule_horizon_epochs": 100,
                "precision": "bf16-mixed",
                "attention_backend": "fused",
                "epoch_metrics_only": True,
                "performance_safe": True,
                "data_loader_workers": 3,
                "hdf5_worker_budget": 12,
                "data_loader_workers_per_cache": [3, 3, 3, 3],
                "data_loader_prefetch_factor": 4,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "validation_seed": 2025,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1.0e-4,
                "resume_schedule_horizon_extended": True,
            },
        }))
        self._split_manifest(seed_root / "pretrain_split.json", marker=pretrain_split_marker)
        tasks = {}
        entries = []
        for task in report.TASK_ORDER[:6]:
            metric, _, _ = report.TASK_METRICS[task]
            exits = {}
            for layer in report.EXIT_LAYERS:
                value = 0.70 + layer / 1000 if task != "nrpos-outdoor" else 2.0 - layer / 100
                exits[str(layer)] = {
                    "test": {metric: value},
                    "validation": {metric: value},
                    "best_epoch": 3,
                    "epochs_completed": 4,
                    "stopped_early": False,
                }
                entries.append({
                    "task": task,
                    "exit_layer": layer,
                    "flops": layer * 1000,
                    "sample_shape": [1, 1, 224, 224],
                    "latency": {"mean_ms": float(layer), "median_ms": float(layer), "p95_ms": float(layer + 0.1), "standard_deviation_ms": 0.01},
                })
            chosen_layer = 8
            selected = exits[str(chosen_layer)]["test"]
            tasks[task] = {
                "spec": {"metric_name": metric},
                "exits": exits,
                "selected_exit_best_validation": chosen_layer,
                "selected_test_best_validation": selected,
                "selected_exit_earliest_within_tolerance": 2,
                "selected_test_earliest_within_tolerance": exits["2"]["test"],
            }
        downstream = seed_root / "downstream"
        downstream.mkdir(exist_ok=True)
        (downstream / "exit_metrics.json").write_text(json.dumps({
            "method": method,
            "pretrain_seed": 0,
            "dataset_suite": "primary",
            "checkpoint": str(checkpoint),
            "split_manifest": str(manifest),
            "protocol": {"test_evaluation": True},
            "tasks": tasks,
        }))
        (seed_root / "benchmarks.json").write_text(json.dumps({"entries": entries}))

    def _vit_p0_root(self, root: Path, manifest: Path, profile: str = "multi_block_n6") -> Path:
        vit_root = root / profile
        self._packet(vit_root / "head_seed_000", "p0", manifest)
        benchmark_source = vit_root / "head_seed_000/p0/seed_000/benchmarks.json"
        benchmark_target = vit_root / "benchmarks/p0/seed_000/benchmarks.json"
        benchmark_target.parent.mkdir(parents=True)
        shutil.copyfile(benchmark_source, benchmark_target)
        return vit_root

    def _vit_p2_root(self, root: Path, manifest: Path, profile: str = "multi_block_n6") -> Path:
        vit_root = root / profile
        self._packet(vit_root, "p2", manifest)
        return vit_root

    def test_cnn_packet_includes_vit_p0_p2_and_derived_b0_baselines(self):
        original_tasks, original_methods = report.TASKS, report.METHODS
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "runs"
                manifest = Path(directory) / "split.json"
                self._split_manifest(manifest)
                vit_root = self._vit_p0_root(Path(directory), manifest)
                vit_p2_root = self._vit_p2_root(Path(directory), manifest)
                self._packet(root, "cnn_p0", manifest)
                self._packet(root, "cnn_p2", manifest)
                stale = root / "cnn_p1" / "seed_000" / "downstream"
                stale.mkdir(parents=True)
                (stale / "exit_metrics.json").write_text('{"intentionally": "ignored"}')
                output = Path(directory) / "report"
                output.mkdir()
                args = argparse.Namespace(
                    cnn_methods=["cnn_p0", "cnn_p2"],
                    vit_p0_run_root=vit_root,
                    vit_p2_run_root=vit_p2_root,
                    masking_label="M-block / span-time, N_m=6",
                    report_title="Synthetic CNN-only packet",
                    provenance_note="test fixture",
                )
                report._run_cnn_only_report(args, root, output)
                self.assertTrue((output / "README.md").is_file())
                self.assertTrue((output / "table_all_methods_downstream.csv").is_file())
                with (output / "table_all_methods_downstream.csv").open(newline="") as handle:
                    downstream_rows = list(csv.DictReader(handle))
                with (output / "table_all_methods_inference.csv").open(newline="") as handle:
                    inference_rows = list(csv.DictReader(handle))
                self.assertEqual(len(inference_rows), len(downstream_rows))
                self.assertIn("flops", inference_rows[0])
                self.assertIn("latency_mean_ms", inference_rows[0])
                self.assertNotIn("metric", inference_rows[0])
                self.assertNotIn("display_value", inference_rows[0])
                self.assertNotEqual(
                    list(inference_rows[0]), list(downstream_rows[0])
                )
                self.assertTrue((output / "early_exit_performance.png").is_file())
                readme = (output / "README.md").read_text()
                self.assertIn("P0 — LatentWave", readme)
                self.assertIn("P2 — Proposed Hierarchical", readme)
                self.assertIn("B0 — Original LatentWave", readme)
                self.assertIn("C0 — CNN", readme)
                self.assertIn("C2 — CNN", readme)
                manifest_payload = json.loads((output / "manifest.json").read_text())
                self.assertEqual(
                    [entry.split(" — ")[0] for entry in manifest_payload["methods"]],
                    ["B0", "P0", "P2", "C0", "C2"],
                )
                self.assertEqual(manifest_payload["comparison_mode"], "vit_b0_p0_p2_with_cnn_c0_c2")
                self.assertIn("b0_reference", manifest_payload)
                self.assertIn("vit_p2_baseline", manifest_payload)
        finally:
            report.TASKS, report.METHODS = original_tasks, original_methods

    def test_cnn_packet_accepts_existing_masking_study_vit_p0_layout(self):
        original_tasks, original_methods = report.TASKS, report.METHODS
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                manifest = directory / "split.json"
                self._split_manifest(manifest)
                vit_root = directory / "multi_block_n6"
                self._packet(vit_root, "p0", manifest)
                self._packet(vit_root, "p2", manifest)
                cnn_root = directory / "runs"
                self._packet(cnn_root, "cnn_p0", manifest)
                self._packet(cnn_root, "cnn_p2", manifest)
                output = directory / "report"
                output.mkdir()
                args = argparse.Namespace(
                    cnn_methods=["cnn_p0", "cnn_p2"],
                    vit_p0_run_root=vit_root,
                    vit_p2_run_root=vit_root,
                    masking_label="M-block / span-time, N_m=6",
                    report_title="Existing M-block packet layout",
                    provenance_note=None,
                )
                report._run_cnn_only_report(args, cnn_root, output)
                manifest_payload = json.loads((output / "manifest.json").read_text())
                self.assertTrue(
                    manifest_payload["vit_p0_baseline"]["metrics_path"].endswith(
                        "multi_block_n6/p0/seed_000/downstream/exit_metrics.json"
                    )
                )
        finally:
            report.TASKS, report.METHODS = original_tasks, original_methods

    def test_incompatible_split_packets_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first, second = directory / "first.json", directory / "second.json"
            self._split_manifest(first, marker=0)
            self._split_manifest(second, marker=1)
            packets = {
                "cnn_p0": {"tasks": {task: {} for task in report.TASK_ORDER[:6]}, "split_manifest": str(first)},
                "cnn_p2": {"tasks": {task: {} for task in report.TASK_ORDER[:6]}, "split_manifest": str(second)},
            }
            with self.assertRaisesRegex(ValueError, "different persisted split assignments"):
                report._discover_tasks_from_packets(packets)

    def test_profiled_packet_records_region_identity_and_validates_provenance(self):
        original_tasks, original_methods = report.TASKS, report.METHODS
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "runs"
                manifest = Path(directory) / "downstream-split.json"
                self._split_manifest(manifest)
                vit_root = self._vit_p0_root(Path(directory), manifest, "region_n6")
                vit_p2_root = self._vit_p2_root(Path(directory), manifest, "region_n6")
                profile = self._profile_metadata("region_n6")
                self._packet(root, "cnn_p0", manifest, mask_profile=profile)
                self._packet(root, "cnn_p2", manifest, mask_profile=profile)
                output = Path(directory) / "report"
                output.mkdir()
                args = argparse.Namespace(
                    cnn_methods=["cnn_p0", "cnn_p2"],
                    vit_p0_run_root=vit_root,
                    vit_p2_run_root=vit_p2_root,
                    mask_profile="region_n6",
                    masking_label="Region masking / random 2D blocks, N_m=6",
                    report_title="Synthetic region CNN-only packet",
                    provenance_note=None,
                )
                report._run_cnn_only_report(args, root, output)
                generated = json.loads((output / "manifest.json").read_text())
                self.assertEqual(generated["report"], "vit_b0_p0_p2_cnn_c0_c2_region_n6_single_seed")
                self.assertEqual(generated["canonical_mask_profile"]["name"], "region_n6")
                self.assertIn("random contiguous two-dimensional", (output / "README.md").read_text())
        finally:
            report.TASKS, report.METHODS = original_tasks, original_methods

    def test_profiled_packet_rejects_mismatched_profile_or_pretraining_split(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, manifest = directory / "runs", directory / "downstream-split.json"
            self._split_manifest(manifest)
            region = self._profile_metadata("region_n6")
            time = self._profile_metadata("time_n6")
            self._packet(root, "cnn_p0", manifest, mask_profile=region)
            self._packet(root, "cnn_p2", manifest, mask_profile=time)
            packets = {
                method: report._load_exit_results(root, method)[0]
                for method in ("cnn_p0", "cnn_p2")
            }
            with self.assertRaisesRegex(ValueError, "mask profile 'name'"):
                report._validate_cnn_profile_provenance(packets, "region_n6")

            self._packet(root, "cnn_p2", manifest, mask_profile=region, pretrain_split_marker=1)
            packets["cnn_p2"] = report._load_exit_results(root, "cnn_p2")[0]
            with self.assertRaisesRegex(ValueError, "validation splits differ"):
                report._validate_cnn_profile_provenance(packets, "region_n6")

    def test_report_rejects_vit_baseline_with_different_downstream_split(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, manifest = directory / "runs", directory / "downstream-split.json"
            self._split_manifest(manifest)
            vit_root = self._vit_p0_root(directory, manifest)
            vit_p2_root = self._vit_p2_root(directory, manifest)
            other_manifest = directory / "other-split.json"
            self._split_manifest(other_manifest, marker=1)
            self._packet(root, "cnn_p0", other_manifest)
            self._packet(root, "cnn_p2", other_manifest)
            args = argparse.Namespace(
                cnn_methods=["cnn_p0", "cnn_p2"],
                vit_p0_run_root=vit_root,
                vit_p2_run_root=vit_p2_root,
                mask_profile=None,
                masking_label="frequency masking",
                report_title="Split mismatch fixture",
                provenance_note=None,
            )
            with self.assertRaisesRegex(ValueError, "different persisted split assignments"):
                report._run_cnn_only_report(args, root, directory / "report")


if __name__ == "__main__":
    unittest.main()
