# SurgWMBench V-JEPA 2 Baseline

This repository is adapted from the official V-JEPA 2 codebase for
SurgWMBench:

```text
SurgWMBench: A Dataset and World-Model Benchmark for Surgical Instrument Motion Planning
```

SurgWMBench work uses `pyproject.toml` plus `uv.lock` for reproducible setup.
The checked lock targets Python 3.12 and the CUDA 13.0 PyTorch wheel backend
used on the local training machine.

The repository intentionally keeps a single README and a single requirements
file, both scoped to the SurgWMBench adaptation.

## Create the Environment

```bash
uv python install 3.12
UV_TORCH_BACKEND=cu130 uv sync --locked --group surgwmbench
source .venv/bin/activate
```

For a requirements-based fallback:

```bash
uv venv --python 3.12 .venv
uv pip install -e . -r requirements.txt
source .venv/bin/activate
```

## Validate SurgWMBench Loading

The local dataset root used for validation is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/all.jsonl \
  --interpolation-method linear \
  --check-files
```

Run the SurgWMBench loader/collator tests:

```bash
python -m pytest tests/test_surgwmbench_dataset.py tests/test_surgwmbench_collate.py
```

Expected result:

```text
9 passed
```

## Implemented Scope

Implemented:

- final-layout `SurgWMBenchClipDataset`
- sparse 20-anchor human-label loading
- dense pseudo-coordinate loading for `linear`, `pchip`, `akima`, and
  `cubic_spline`
- source labels that keep `human` and `interpolated` targets separate
- sparse and dense collators
- read-only validation CLI
- synthetic final-layout tests

Deferred:

- V-JEPA encoder wrapper
- action-conditioned latent predictor
- coordinate decoder / planning head
- training, rollout, and evaluation scripts

## Regenerate the Lock

Use this only when intentionally updating dependencies:

```bash
UV_TORCH_BACKEND=cu130 uv lock --python 3.12
UV_TORCH_BACKEND=cu130 uv sync --locked --group surgwmbench
```

For CPU-only machines, regenerate a machine-specific lock with
`UV_TORCH_BACKEND=cpu` and do not mix CPU and CUDA lock updates in the same
commit.
