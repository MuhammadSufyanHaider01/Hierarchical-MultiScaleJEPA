"""Checks for the paper-results data contracts and generated values."""
import csv
import importlib.util
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "paper_results"))
import pipeline


class PaperResultsTests(unittest.TestCase):
    def test_duplicate_and_incomplete_keys_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pipeline.index_unique([{"id": "1"}, {"id": "1"}], ("id",))
        with self.assertRaisesRegex(ValueError, "missing"):
            pipeline.require_keys({("1",): {}}, {("1",), ("2",)}, "input")

    def test_seed_summary_rejects_wrong_deviation_and_seed(self):
        values = {str(i): i / 10 for i in range(7)}
        row = {"per_seed": json.dumps(values), "mean": statistics.mean(values.values()), "sample_sd": 0}
        with self.assertRaisesRegex(ValueError, "sample SD"):
            pipeline.seed_values(row)
        row["sample_sd"] = statistics.stdev(values.values())
        self.assertEqual(pipeline.seed_values(row), values)
        del values["6"]
        row["per_seed"] = json.dumps(values)
        with self.assertRaisesRegex(ValueError, "seeds 0-6"):
            pipeline.seed_values(row)

    def test_fusion_reports_have_matched_provenance_and_all_fixed_exits(self):
        rows = pipeline.load_fusion()
        self.assertEqual(len(rows), 96)
        value = next(r for r in rows if (r["method"], r["algorithm"], r["task"], r["exit"]) == ("p0", "current", "rfs", 2))
        self.assertAlmostEqual(value["mean"], 59.12995764187404)
        self.assertAlmostEqual(value["sample_sd"], 3.989683015799484)
        position = next(r for r in rows if r["task"] == "nrpos-outdoor")
        self.assertEqual(position["unit"], "m")

    def test_selected_and_ablation_tables_have_expected_rows(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            pipeline.make_selected()
            pipeline.make_ablation()
            for filename, expected in (("pretraining_comparison.csv", 96), ("adapter_ablation.csv", 84)):
                with (Path(tmp) / "tables" / filename).open(encoding="utf-8", newline="") as f:
                    self.assertEqual(len(list(csv.DictReader(f))), expected)
            provenance = json.loads((Path(tmp) / "data/checkpoint_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["multi_block_n6"]["p0"]["actual_epoch"], 64)
            self.assertEqual(provenance["frequency_n6"]["p1"]["checkpoint_sha256"], provenance["multi_block_n6"]["p1"]["checkpoint_sha256"])

    def test_cost_never_substitutes_fusion_measurements(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            pipeline.make_cost()
            with (Path(tmp) / "tables/ee_performance_cost.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 96)
            for row in rows:
                if row["algorithm"] == "single_gelu":
                    self.assertEqual(row["flops"], "")
                    self.assertEqual(row["latency_mean_ms"], "")
                else:
                    self.assertGreater(int(row["flops"]), 0)
                    self.assertIn("unavailable", row["cost_status"])

    def test_paper_legends_use_descriptive_method_names(self):
        fig = pipeline.plot_fusion(pipeline.load_fusion(), uncertainty=False)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(labels, ["LatentWave + linear EE", "LatentWave + fusion EE", "Fine-to-coarse + linear EE", "Fine-to-coarse + fusion EE"])
        pipeline.plt.close(fig)

    def test_representation_titles_match_task_and_axis(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            captured = []
            def capture(fig, stem):
                captured.extend(t.get_text() for t in fig.texts)
                pipeline.plt.close(fig)
            with mock.patch.object(pipeline, "save_fig", side_effect=capture):
                pipeline.make_representations("rfs")
            self.assertIn("RF classification: 4 of 14 input-patch columns", captured)

    def test_trajectory_covers_layers_methods_and_input_lengths(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            pipeline.make_interference_trajectory()
            with (Path(tmp) / "data/interference_separation_vs_input.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 64)
            self.assertEqual({r["temporal_patch_columns"] for r in rows}, {"2", "4", "8", "14"})
            self.assertEqual({r["exit"] for r in rows}, {"2", "4", "6", "8"})
            reference = next(r for r in rows if r["method"] == "p0" and r["exit"] == "2" and r["temporal_patch_columns"] == "4")
            self.assertAlmostEqual(float(reference["class_separation_ratio"]), 0.12755348403248387)

    def test_pca_layout_uses_validated_coordinates_and_trajectories(self):
        import hashlib
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs/representation_trajectories/multi_block_n6/interference/seed_000"
            test, validation = base / "test", base / "validation"
            test.mkdir(parents=True)
            validation.mkdir()
            layers, columns = [2, 4, 6, 8], [2, 4, 8, 14]
            ids = np.arange(6, dtype=np.int64)
            digest = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
            labels = np.array([0, 0, 0, 1, 1, 1])
            data = np.arange(6 * 4 * 4 * 2, dtype=float).reshape(6, 4, 4, 2) / 20
            np.savez_compressed(test / "coordinates.npz", labels=labels, indices=ids, **{m: data for m in pipeline.METHOD_ORDER})
            checks = {m: {"sha256": "checkpoint-" + m} for m in pipeline.METHOD_ORDER}
            definition = {"layers": layers, "temporal_columns": columns}
            common = {"task": "interference", "checkpoints": checks, "split_manifest_sha256": "manifest", "feature_definition": definition, "diagnostic_limit_per_class": None}
            (test / "provenance.json").write_text(json.dumps({**common, "split": "test", "sample_indices_sha256": digest, "split_indices_sha256": digest}))
            (validation / "provenance.json").write_text(json.dumps({**common, "split": "validation", "split_indices_sha256": "validation-digest"}))
            (validation / "pca_metadata.json").write_text(json.dumps({"task": "interference", "fit_split": "validation", "checkpoint_sha256": {m: checks[m]["sha256"] for m in checks}, "split_indices_sha256": "validation-digest", "feature_definition": definition, "explained_variance_ratio": {m: [.3, .2] for m in checks}}))
            with mock.patch.object(pipeline, "ROOT", root), mock.patch.object(pipeline, "OUT", root / "out"):
                status, _ = pipeline.optional_pca()
                self.assertEqual(status, "complete")
                with (root / "out/data/interference_pca_centroids.csv").open(encoding="utf-8", newline="") as f:
                    centroids = list(csv.DictReader(f))
                self.assertEqual(len(centroids), 32)
                self.assertAlmostEqual(float(centroids[0]["pc1"]), float(data[:3, 0, 0, 0].mean()))
                self.assertTrue((root / "out/figures/main/interference_pca_trajectories.pdf").is_file())
                metadata = json.loads((validation / "pca_metadata.json").read_text())
                metadata["checkpoint_sha256"]["p0"] = "wrong"
                (validation / "pca_metadata.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "checkpoints differ"):
                    pipeline.optional_pca()

    def test_paired_representation_matches_source_differences(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            status, _ = pipeline.make_paired_representation()
            self.assertEqual(status, "complete")
            with (Path(tmp) / "data/interference_paired_separation.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 28)
            row = next(r for r in rows if r["panel"] == "difference" and r["method"] == "p2" and r["layer"] == "2")
            self.assertEqual(row["reference_method"], "p0")
            self.assertEqual(row["temporal_patch_columns"], "14")
            self.assertAlmostEqual(float(row["estimate"]), 0.0425056748440152)
            self.assertAlmostEqual(float(row["ci_low"]), 0.008361039269575436)
            self.assertAlmostEqual(float(row["ci_high"]), 0.06848942793222187)
            self.assertTrue((Path(tmp) / "figures/main/interference_paired_separation.pdf").is_file())

    def test_full_input_representation_uses_fourteen_columns(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            pipeline.make_representations("interference", True)
            with (Path(tmp) / "data/layerwise_representation_analysis.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 16)
            self.assertEqual({r["temporal_patch_columns"] for r in rows}, {"14"})
            row = next(r for r in rows if r["method"] == "p0" and r["exit"] == "2")
            self.assertAlmostEqual(float(row["class_separation_ratio"]), 0.07350271772603799)

    def test_temporal_accuracy_uses_five_full_input_trained_heads(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            status, _ = pipeline.make_temporal_performance()
            self.assertEqual(status, "complete")
            with (Path(tmp) / "data/interference_accuracy_vs_time_patches.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 64)
            self.assertEqual({r["temporal_patch_columns"] for r in rows}, {"2", "4", "8", "14"})
            self.assertEqual({r["head_seed_count"] for r in rows}, {"5"})
            self.assertEqual({r["head_training_columns"] for r in rows}, {"14"})
            row = next(r for r in rows if r["method"] == "p0" and r["exit"] == "2" and r["temporal_patch_columns"] == "4")
            self.assertAlmostEqual(float(row["mean"]), 43.22857142857143)
            self.assertAlmostEqual(float(row["sample_sd"]), 0.5258617805515994)
            self.assertTrue((Path(tmp) / "figures/main/interference_accuracy_vs_time_patches.pdf").is_file())

    def test_human_activity_full_input_representation(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            status, _ = pipeline.make_paired_representation("human-activity")
            self.assertEqual(status, "complete")
            with (Path(tmp) / "data/human_activity_paired_separation.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 28)
            self.assertEqual({r["temporal_patch_columns"] for r in rows}, {"14"})
            row = next(r for r in rows if r["panel"] == "difference" and r["method"] == "p2" and r["layer"] == "2")
            self.assertAlmostEqual(float(row["estimate"]), 0.18924282687571048)
            self.assertAlmostEqual(float(row["ci_low"]), 0.07727343750846938)
            self.assertAlmostEqual(float(row["ci_high"]), 0.28586183101957624)

    def test_human_activity_temporal_accuracy_uses_six_classes(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            status, _ = pipeline.make_temporal_performance("human-activity")
            self.assertEqual(status, "complete")
            with (Path(tmp) / "data/human_activity_accuracy_vs_time_patches.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 64)
            self.assertEqual({r["head_seed_count"] for r in rows}, {"5"})
            self.assertEqual({r["head_training_columns"] for r in rows}, {"14"})
            row = next(r for r in rows if r["method"] == "p0" and r["exit"] == "2" and r["temporal_patch_columns"] == "4")
            self.assertAlmostEqual(float(row["mean"]), 64.0)
            self.assertAlmostEqual(float(row["sample_sd"]), 1.6029486718059448)
            self.assertEqual(row["head_seeds"], "not recorded in checked-in summary")

    def test_rf_and_jamming_figures_use_input_columns(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            for task in ("rfs", "jamming"):
                self.assertEqual(pipeline.make_paired_representation(task)[0], "complete")
                self.assertEqual(pipeline.make_temporal_performance(task)[0], "complete")
                with (Path(tmp) / f"data/{task}_paired_separation.csv").open(encoding="utf-8", newline="") as f:
                    paired = list(csv.DictReader(f))
                with (Path(tmp) / f"data/{task}_accuracy_vs_input_patches.csv").open(encoding="utf-8", newline="") as f:
                    accuracy = list(csv.DictReader(f))
                self.assertEqual(len(paired), 28)
                self.assertEqual(len(accuracy), 64)
                self.assertEqual({r["input_patch_columns"] for r in paired}, {"14"})
                self.assertEqual({r["input_patch_columns"] for r in accuracy}, {"2", "4", "8", "14"})
                self.assertNotIn("temporal_patch_columns", accuracy[0])
                if task == "rfs":
                    row = next(r for r in paired if r["panel"] == "absolute" and r["method"] == "p0" and r["layer"] == "2")
                    self.assertAlmostEqual(float(row["estimate"]), 0.7846268204107151)
                    self.assertGreater(float(row["ci_low"]), float(row["estimate"]))
                    row = next(r for r in accuracy if r["method"] == "p0" and r["exit"] == "2" and r["input_patch_columns"] == "4")
                    self.assertAlmostEqual(float(row["mean"]), 44.195543541788425)

    def test_nr_positioning_figure_protocols_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            self.assertEqual(pipeline.make_nrpos_depth()[0], "partial")
            self.assertEqual(pipeline.make_nrpos_time()[0], "partial")
            with (Path(tmp) / "data/nrpos_full_input_error_vs_exit.csv").open(encoding="utf-8", newline="") as f:
                depth = list(csv.DictReader(f))
            with (Path(tmp) / "data/nrpos_validation_error_vs_time_patches.csv").open(encoding="utf-8", newline="") as f:
                time = list(csv.DictReader(f))
            self.assertEqual(len(depth), 16)
            self.assertEqual({r["split"] for r in depth}, {"test"})
            self.assertEqual({r["head_seed_count"] for r in depth}, {"7"})
            self.assertEqual(len(time), 32)
            self.assertEqual({r["split"] for r in time}, {"validation"})
            self.assertEqual({r["head_seed_count"] for r in time}, {"1"})
            row = next(r for r in time if r["method"] == "p2" and r["exit"] == "2" and r["temporal_patch_columns"] == "2")
            self.assertAlmostEqual(float(row["mean_distance_error_m"]), 14.316057205200195)
            self.assertEqual(row["checkpoint_sha256"], "not recorded")

    def test_bootstrap_interval_may_exclude_point_estimate(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "OUT", Path(tmp)):
            pipeline.make_representations("rfs")
            with (Path(tmp) / "data/representations_rfs.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 16)
            self.assertTrue(any(not (float(r["ci_low"]) <= float(r["class_separation_ratio"]) <= float(r["ci_high"])) for r in rows))
            self.assertTrue((Path(tmp) / "figures/supplementary/representations_rfs.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
