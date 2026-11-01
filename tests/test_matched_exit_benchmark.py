"""Tests for matched benchmark joins and actual early-exit wrappers."""
import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/paper_results'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import pipeline
import matched_exit_benchmark as benchmark


def fixture():
    provenance = pipeline.fusion_provenance()
    params = pipeline.load_parameters()
    entries = []
    for method in ('p0', 'p2'):
        for algorithm in pipeline.ALGORITHMS:
            for task in pipeline.TASK_ORDER:
                for layer in pipeline.EXITS:
                    entry = provenance[(method, algorithm, 0)]
                    entries.append({'method': method, 'algorithm': algorithm, 'task': task, 'exit': layer,
                        'input_shape': [1, 4, 224, 224], 'num_outputs': 2,
                        'head_parameters': int(params[(method, algorithm, task, str(layer))]['total']),
                        'checkpoint_sha256': entry['checkpoint_sha256'], 'packet_sha256': entry['metrics_sha256'],
                        'head_weights': 'initialized', 'head_sha256': None,
                        'execution': {'executed_blocks': list(range(1, layer+1)),
                                      'fusion_layers': [x for x in pipeline.EXITS if x <= layer] if algorithm == 'single_gelu' else []},
                        'flops': layer * 1000000000 + (1000000 if algorithm == 'single_gelu' else 0),
                        'counted_operators': {'synthetic_matmul': layer * 1000000000 + (1000000 if algorithm == 'single_gelu' else 0)},
                        'latency': {'mean_ms': 1., 'median_ms': 1., 'standard_deviation_ms': 0.1, 'p95_ms': 1.2}})
    protocol = {'batch_size': 1, 'precision': 'float32', 'matmul_precision': 'high', 'allow_tf32_matmul': True,
        'allow_tf32_cudnn': True, 'warmup_iterations': 50, 'measure_iterations': 200, 'synchronize_each_inference': True,
        'flop_counter': 'torch.utils.flop_counter.FlopCounterMode', 'flop_convention': 'two operations',
        'flop_exclusions': 'unregistered operations', 'device': 'cuda:0', 'gpu_name': 'test GPU',
        'gpu_memory_bytes': 100, 'torch_version': 'test', 'cuda_version': 'test', 'python_version': 'test',
        'revision': 'abc', 'working_tree_dirty': False, 'input_source': 'authenticated original seed-0 evaluation packets',
        'head_execution': 'fixed architecture'}
    return {'schema_version': 1, 'study': 'multi_block_n6_matched_exit', 'entries': entries, 'protocol': protocol}


class MatchedTests(unittest.TestCase):
    def test_real_wrappers_execute_only_prefix(self):
        import torch
        import importlib.util
        spec = importlib.util.spec_from_file_location('exit_heads_for_benchmark_test', ROOT / 'src/models/exit_heads.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        FrozenEncoderExit = module.FrozenEncoderExit
        FrozenEncoderResidualFusionAblationExit = module.FrozenEncoderResidualFusionAblationExit
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(8)])
                self.pos_embed = torch.nn.Parameter(torch.zeros(1, 3, 4))
            def forward_to_layers(self, x, layers):
                result = {}
                for index, block in enumerate(self.blocks, 1):
                    x = block(x)
                    if index in layers:
                        result[index] = x
                    if index == max(layers):
                        break
                return result
            def forward_to_layer(self, x, layer):
                return self.forward_to_layers(x, (layer,))[layer]
        for layer in (2, 4, 8):
            for head in ('current', 'single_gelu'):
                encoder = Encoder()
                model = FrozenEncoderExit(encoder, layer, 3) if head == 'current' else FrozenEncoderResidualFusionAblationExit(encoder, layer, 3, 'single_gelu')
                result = benchmark.check_execution(model, torch.randn(1, 3, 4), layer, head, 3)
                self.assertEqual(result['executed_blocks'], list(range(1, layer+1)))

    def test_report_joins_and_cost_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'benchmark.json'
            path.write_text(json.dumps(fixture()), encoding='utf-8')
            with mock.patch.object(pipeline, 'BENCHMARK', path), mock.patch.object(pipeline, 'OUT', Path(tmp) / 'out'):
                self.assertEqual(pipeline.make_selected_cost()[0], 'complete')
                with (pipeline.OUT / 'data/selected_exit_cost_by_seed.csv').open() as f:
                    per_seed = list(csv.DictReader(f))
                with (pipeline.OUT / 'tables/selected_exit_performance_cost.csv').open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 24)
                self.assertEqual(len(per_seed), 168)
                row = rows[0]
                group = [r for r in per_seed if all(r[k] == row[k] for k in ('method', 'algorithm', 'task'))]
                expected = sum(int(r['selected_flops']) for r in group) / 7
                self.assertAlmostEqual(float(row['selected_mean_gflops']), expected / 1e9)
                self.assertAlmostEqual(float(row['reduction_vs_same_head_l8_percent']), 100*(1-expected/int(group[0]['same_head_l8_flops'])))
                self.assertEqual(pipeline.make_cost()[0], 'complete')
                with (pipeline.OUT / 'tables/ee_performance_cost.csv').open() as f:
                    self.assertTrue(all(r['cost_status'] == 'matched_benchmark' for r in csv.DictReader(f)))

    def test_historical_costs_preserve_missing_fusion(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, 'OUT', Path(tmp)):
            self.assertEqual(pipeline.make_historical_selected_cost()[0], 'partial')
            with (Path(tmp) / 'tables/historical_selected_performance_cost.csv').open() as f:
                rows = list(csv.DictReader(f))
            with (Path(tmp) / 'data/historical_selected_cost_by_seed.csv').open() as f:
                seeds = list(csv.DictReader(f))
            self.assertEqual(len(rows), 24)
            self.assertEqual(len(seeds), 168)
            for row in rows:
                group = [r for r in seeds if all(r[k] == row[k] for k in ('method', 'algorithm', 'task'))]
                if row['algorithm'] == 'single_gelu':
                    self.assertEqual(row['reported_selected_mean_gflops'], '')
                    self.assertTrue(all(r['reported_flops'] == '' for r in group))
                else:
                    expected = sum(int(r['reported_flops']) for r in group) / 7e9
                    self.assertAlmostEqual(float(row['reported_selected_mean_gflops']), expected)
                    self.assertEqual(row['cost_status'], 'provisional_historical_baseline')

    def test_selected_metric_direction_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'benchmark.json'
            path.write_text(json.dumps(fixture()), encoding='utf-8')
            original_read = pipeline.read_csv
            def bad_direction(source, required):
                rows = original_read(source, required)
                if str(source).endswith('per_seed_validation_selected_results.csv'):
                    rows[0]['higher_is_better'] = 'False'
                return rows
            with mock.patch.object(pipeline, 'BENCHMARK', path), mock.patch.object(pipeline, 'OUT', Path(tmp) / 'out'), mock.patch.object(pipeline, 'read_csv', side_effect=bad_direction):
                with self.assertRaisesRegex(ValueError, 'Metric/direction'):
                    pipeline.make_selected_cost()

    def test_authenticated_specs_must_agree(self):
        packets = {}
        for method in ('p0', 'p2'):
            for head in ('current', 'single_gelu'):
                packets[method, head] = ({'tasks': {task: {'spec': {'sample_shape': [4, 16, 16], 'num_outputs': 3}} for task in benchmark.TASKS}}, {}, None)
        self.assertEqual(len(benchmark.validated_specs(packets)), 6)
        packets['p2', 'single_gelu'][0]['tasks']['rfs']['spec']['sample_shape'] = [4, 8, 16]
        with self.assertRaisesRegex(ValueError, 'specifications differ'):
            benchmark.validated_specs(packets)

    def test_invalid_artifacts_fail_closed(self):
        original = fixture()
        mutations = [lambda d: d['entries'].pop(), lambda d: d['entries'].append(copy.deepcopy(d['entries'][0])),
            lambda d: d['entries'][0].update(checkpoint_sha256='wrong'),
            lambda d: d['entries'][0].update(input_shape=[1, 1]),
            lambda d: d['protocol'].update(precision='float16'),
            lambda d: d['entries'][0]['execution'].update(executed_blocks=[1, 2, 3]),
            lambda d: d['protocol'].update(working_tree_dirty=True)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'benchmark.json'
            for mutate in mutations:
                data = copy.deepcopy(original); mutate(data)
                path.write_text(json.dumps(data), encoding='utf-8')
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    pipeline.load_matched_benchmark(path)
            with mock.patch.object(pipeline, 'BENCHMARK', path.parent / 'missing.json'):
                self.assertEqual(pipeline.make_selected_cost()[0], 'unavailable')


if __name__ == '__main__':
    unittest.main()
