#!/usr/bin/env python3
"""Benchmark the exact multi-block EE architectures; never trains models.

Requires the four original head-seed-0 evaluation packets, named
{p0,p2}_{current,single_gelu}.json, and both checkpoint files. The packets
are authenticated by the checked-in study manifest before task shapes are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('WANDB_KEY', 'offline')
TASKS = ('rfs', 'jamming', 'human-activity', 'interference', 'nrpos-outdoor', 'deepmimo-beam')
EXITS = (2, 4, 6, 8)
METHODS = ('p0', 'p2')
HEADS = ('current', 'single_gelu')
STUDY = ROOT / 'reports/early_exit_optimization/multi_block_n6/residual_dynamic_fusion_multiseed/manifest.json'
PARAMETERS = ROOT / 'reports/early_exit_optimization/multi_block_n6/residual_fusion_adapter_ablation_single_seed/parameter_counts.csv'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_authenticated_packets(packet_dir, checkpoints):
    manifest = json.loads(STUDY.read_text(encoding='utf-8'))
    sources = {(x['method'], x['algorithm']): x for x in manifest['sources'] if x['head_seed'] == 0}
    if set(sources) != {(m, h) for m in METHODS for h in HEADS}:
        raise ValueError('Expected exactly four seed-zero provenance sources')
    packets = {}
    for method in METHODS:
        expected_hash = sources[method, 'current']['checkpoint_sha256']
        if sha(checkpoints[method]) != expected_hash:
            raise ValueError(f'{method}: checkpoint hash mismatch')
        for head in HEADS:
            provenance = sources[method, head]
            path = packet_dir / f'{method}_{head}.json'
            if sha(path) != provenance['metrics_sha256']:
                raise ValueError(f'{path}: evaluation packet hash mismatch')
            packet = json.loads(path.read_text(encoding='utf-8'))
            if packet.get('method') != method or packet.get('head_seed') != 0 or provenance['checkpoint_sha256'] != expected_hash:
                raise ValueError(f'{path}: method, seed or checkpoint mismatch')
            if head == 'single_gelu' and (packet.get('adapter_variant') != head or packet.get('head_algorithm') != 'residual_fusion_adapter_ablation'):
                raise ValueError(f'{path}: wrong fusion architecture')
            if set(packet.get('tasks', {})) != set(TASKS):
                raise ValueError(f'{path}: task coverage mismatch')
            packets[method, head] = (packet, provenance, path)
    return packets


def validated_specs(packets):
    specs = {}
    for task in TASKS:
        values = []
        for packet, _, _ in packets.values():
            spec = packet['tasks'][task]['spec']
            shape = tuple(int(x) for x in spec['sample_shape'])
            outputs = int(spec['num_outputs'])
            if not shape or min(shape) < 1 or outputs < 1:
                raise ValueError(f'{task}: invalid shape or output dimension')
            values.append((shape, outputs))
        if len(set(values)) != 1:
            raise ValueError(f'{task}: evaluated packet specifications differ')
        specs[task] = values[0]
    return specs


def check_execution(model, sample, exit_layer, head_name, outputs):
    """Hooks observe real blocks and fusion head inputs, not inferred FLOPs."""
    import torch
    visited, inputs, handles = [], [], []
    for idx, block in enumerate(model.encoder.blocks, 1):
        handles.append(block.register_forward_hook(lambda _m, _i, _o, idx=idx: visited.append(idx)))
    if head_name == 'single_gelu':
        handles.append(model.head.register_forward_pre_hook(lambda _m, args: inputs.append(tuple(sorted(args[0])))))
    try:
        with torch.inference_mode():
            output = model(sample)
    finally:
        for handle in handles:
            handle.remove()
    if visited != list(range(1, exit_layer + 1)):
        raise ValueError(f'Unexpected executed encoder blocks: {visited} for L{exit_layer}')
    if tuple(output.shape) != (sample.shape[0], outputs):
        raise ValueError(f'Unexpected output shape: {tuple(output.shape)}')
    if head_name == 'single_gelu':
        expected = tuple(x for x in EXITS if x <= exit_layer)
        if inputs != [expected] or tuple(model.source_layers) != expected[:-1]:
            raise ValueError(f'Invalid fusion source layers: {inputs}')
        if exit_layer == 2 and (len(model.head.single_projections) or len(model.head.source_gate_projections)):
            raise ValueError('L2 unexpectedly has source adapters')
    return {'executed_blocks': visited, 'fusion_layers': list(inputs[0]) if inputs else []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0-checkpoint', type=Path, required=True)
    parser.add_argument('--p2-checkpoint', type=Path, required=True)
    parser.add_argument('--packets-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head-root', type=Path, help='Optional fusion heads at <root>/<method>/<task>/L<exit>.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 2:
        parser.error('warmup must be positive and iterations at least 2')
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from configs import configure_method_variant, init_jepa_config
    from models import FrozenEncoderExit, FrozenEncoderResidualFusionAblationExit
    from phase5_benchmark import _new_encoder, _measure_latency
    from checkpointing import environment_metadata
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for paper benchmark measurements')
    checkpoints = {'p0': args.p0_checkpoint, 'p2': args.p2_checkpoint}
    packets = load_authenticated_packets(args.packets_dir, checkpoints)
    specs = validated_specs(packets)
    with PARAMETERS.open(encoding='utf-8-sig', newline='') as f:
        counts = {(r['method'], r['adapter_variant'], r['task'], int(r['exit'])): int(r['total']) for r in csv.DictReader(f)}
    torch.set_float32_matmul_precision('high')
    generator = torch.Generator(device=device).manual_seed(0)
    entries = []
    for method in METHODS:
        config = init_jepa_config(str(ROOT / 'configs' / f'{method}.yml'))
        configure_method_variant(config, method)
        encoder = _new_encoder(config, checkpoints[method], device)
        for task in TASKS:
            shape, outputs = specs[task]
            sample = torch.randn((1, *shape), generator=generator, device=device, dtype=torch.float32)
            for layer in EXITS:
                for head in HEADS:
                    model = (FrozenEncoderExit(encoder, layer, outputs) if head == 'current' else
                             FrozenEncoderResidualFusionAblationExit(encoder, layer, outputs, 'single_gelu')).to(device).eval()
                    count = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
                    if count != counts[method, head, task, layer]:
                        raise ValueError(f'Head parameter count mismatch: {method}/{head}/{task}/L{layer}')
                    weight_source, head_sha = 'initialized', None
                    if args.head_root and head == 'single_gelu':
                        path = args.head_root / method / task / f'L{layer}.pt'
                        entry = packets[method, head][0]['tasks'][task]['exits'][str(layer)]
                        if sha(path) != entry.get('checkpoint_sha256'):
                            raise ValueError(f'Head checkpoint hash mismatch: {path}')
                        payload = torch.load(path, map_location=device, weights_only=False)
                        if (payload.get('adapter_variant') != head or payload.get('exit_layer') != layer or
                            payload.get('source_layers') != [x for x in EXITS if x < layer]):
                            raise ValueError(f'Head identity mismatch: {path}')
                        model.head.load_state_dict(payload['head'], strict=True)
                        weight_source, head_sha = 'trained_seed_0', sha(path)
                    execution = check_execution(model, sample, layer, head, outputs)
                    torch.cuda.synchronize(device)
                    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                        model(sample)
                    torch.cuda.synchronize(device)
                    flops = int(counter.get_total_flops())
                    counted_operators = {str(op): int(value) for op, value in counter.get_flop_counts().get('Global', {}).items()}
                    if flops <= 0:
                        raise ValueError('FLOP counter returned no operations')
                    latency = _measure_latency(model, sample, device, warmup_iterations=args.warmup, measure_iterations=args.iterations)
                    entries.append({'method': method, 'algorithm': head, 'task': task, 'exit': layer,
                        'input_shape': [1, *shape], 'num_outputs': outputs, 'head_parameters': count,
                        'checkpoint_sha256': sha(checkpoints[method]), 'packet_sha256': packets[method, head][1]['metrics_sha256'],
                        'head_weights': weight_source, 'head_sha256': head_sha, 'execution': execution,
                        'flops': flops, 'counted_operators': counted_operators, 'latency': latency})
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        rev, dirty = None, None
    device_props = torch.cuda.get_device_properties(device)
    result = {'schema_version': 1, 'study': 'multi_block_n6_matched_exit', 'entries': entries,
        'protocol': {'batch_size': 1, 'precision': 'float32', 'matmul_precision': torch.get_float32_matmul_precision(),
          'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
          'warmup_iterations': args.warmup, 'measure_iterations': args.iterations, 'synchronize_each_inference': True,
          'flop_counter': 'torch.utils.flop_counter.FlopCounterMode', 'flop_convention': 'PyTorch registered formulas; matrix multiply counts multiply and add as two operations',
          'flop_exclusions': 'Only counted_operators are included; pooling, normalization, GELU, gates, elementwise additions and memory operations may be omitted depending on registered formulas.',
          'device': str(device), 'gpu_name': torch.cuda.get_device_name(device), 'gpu_memory_bytes': device_props.total_memory,
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'slurm_gpu_partition': os.environ.get('SLURM_JOB_PARTITION'),
          'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'python_version': platform.python_version(),
          'environment': environment_metadata(), 'revision': rev, 'working_tree_dirty': dirty,
          'input_source': 'authenticated original seed-0 evaluation packets', 'checkpoint_seed': 0,
          'head_execution': 'fixed architecture; no data-dependent or seed-dependent model branches'} }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Wrote {len(entries)} measurements to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
