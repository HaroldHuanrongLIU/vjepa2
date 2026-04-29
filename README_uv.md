# Reproducible uv Environment

This repository uses `pyproject.toml` plus `uv.lock` for the reproducible
development environment. The checked lock targets Python 3.12 and the CUDA 13.0
PyTorch wheel backend used on the local training machine.

## Create the Environment

```bash
uv python install 3.12
UV_TORCH_BACKEND=cu130 uv sync --locked --group dev
source .venv/bin/activate
```

## Verify

```bash
python -c "import torch, torchvision, decord; print(torch.__version__, torchvision.__version__, decord.__version__)"
python -m pytest tests/test_surgwmbench_dataset.py tests/test_surgwmbench_collate.py
```

## Regenerate the Lock

Use this only when intentionally updating dependencies:

```bash
UV_TORCH_BACKEND=cu130 uv lock --python 3.12
UV_TORCH_BACKEND=cu130 uv sync --locked --group dev
```

For CPU-only machines, regenerate a machine-specific lock with
`UV_TORCH_BACKEND=cpu` and do not mix CPU and CUDA lock updates in the same
commit.
