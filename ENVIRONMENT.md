# Environment and hardware

## GPU training target

Pretraining, finetuning, and synchronized latency measurements are intended to
run on the machine's NVIDIA RTX 5070 Ti. Before Phase 1 CUDA work, verify all
of the following in the selected environment:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The vendored upstream project specifies Python `>=3.13`, PyTorch `>=2.10.0`,
and a CUDA 12.6 PyTorch index in `pyproject.toml`/`uv.lock`. CUDA 12.6 does not
contain kernels for this GPU's `sm_120` capability, so the local training
environment uses the compatible CUDA 12.8 wheel instead.

The verified local GPU environment is:

- Python 3.14.6 in the repository-local `.venv`
- PyTorch 2.10.0+cu128
- CUDA runtime 12.8
- NVIDIA GeForce RTX 5070 Ti, capability `(12, 0)`

Run the project-side CUDA check with:

```bash
.venv/bin/python scripts/check_cuda.py
```

The selected versions must be captured in checkpoint metadata.

## Current bootstrap runtime

The Phase 0 checks use the existing environment at
`/home/waves-lab/miniconda3/envs/iqfm_jepa/bin/python`:

- Python 3.9.23
- PyTorch 2.3.1+cu121
- h5py 3.10.0
- CUDA runtime reported by PyTorch: 12.1
- `torch.cuda.is_available()`: currently `False`
- `nvidia-smi`: currently reports an NVML mismatch because the loaded kernel
  module is 595.84 while the installed user-space driver is 595.91.07

This older environment remains sufficient for the read-only HDF5 audit but is
not suitable for GPU training: its PyTorch 2.3.1+cu121 build has no `sm_120`
kernels. The repository-local `.venv` is the environment to use for GPU
pretraining and finetuning.

PyTorch CUDA allocation and a 2048x2048 matrix multiplication have been
verified successfully in `.venv`. A reboot is still recommended before long
experiments so `nvidia-smi` and NVML load the installed 595.91.07 module.

## Reproducibility requirements

Record Python, PyTorch, CUDA, driver, GPU name, source commit, configuration,
seed, dataset manifest, and git revision in every training checkpoint. Keep
experiment outputs, credentials, and HDF5 files outside Git.
