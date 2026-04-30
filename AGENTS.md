# AGENTS.md

Instructions for agents working in this repository.

## Repository Scope

This repo is the official V-JEPA 2 codebase adapted for SurgWMBench:

```text
SurgWMBench: A Dataset and World-Model Benchmark for Surgical Instrument Motion Planning
```

Current implemented SurgWMBench scope:

- final-layout SurgWMBench clip loader and collators
- raw-video / extracted-frame SSL smoke pretraining adapter
- sparse 20-anchor action-conditioned baseline
- sparse human-anchor rollout evaluation
- synthetic tests for loader, collators, metrics, model shapes, SSL smoke, train, and eval

Do not treat dense interpolation coordinates as human ground truth. Dense pseudo labels are auxiliary only.

## Environment

Use Python 3.12 with `uv`:

```bash
UV_TORCH_BACKEND=cu130 uv sync --locked --group surgwmbench
source .venv/bin/activate
```

CPU-only machines may need a CPU-specific lock regeneration. Do not mix CPU and CUDA lock changes in the same commit unless explicitly requested.

## Dataset

The local SurgWMBench root used in this workspace is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Use official manifests only:

```text
manifests/train.jsonl
manifests/val.jsonl
manifests/test.jsonl
manifests/all.jsonl
```

Never create random splits. Never infer difficulty from folder names. Never rewrite dataset labels or manifests.

## Key Files

- `src/datasets/surgwmbench.py`: final-layout clip loader
- `src/datasets/surgwmbench_video.py`: raw-video / extracted-frame SSL dataset
- `src/datasets/surgwmbench_collators.py`: sparse, dense, and SSL collators
- `src/models/surgwmbench_vjepa_ac.py`: SurgWMBench V-JEPA 2-AC-style model skeleton
- `src/utils/surgwmbench_metrics.py`: sparse trajectory metrics
- `app/surgwmbench/pretrain_ssl.py`: lightweight SurgWMBench SSL smoke pretraining
- `app/surgwmbench/train_ac.py`: sparse action-conditioned training
- `app/surgwmbench/eval_ac.py`: sparse human-anchor evaluation
- `tools/validate_surgwmbench_loader.py`: read-only loader validation

## Validation Commands

Run the full SurgWMBench targeted synthetic suite:

```bash
.venv/bin/python -m pytest \
  tests/test_surgwmbench_pretrain_ssl.py \
  tests/test_surgwmbench_dataset.py \
  tests/test_surgwmbench_collate.py \
  tests/test_surgwmbench_metrics.py \
  tests/test_surgwmbench_model_shapes.py \
  tests/test_surgwmbench_train_eval.py
```

Validate the real dataset layout:

```bash
.venv/bin/python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/all.jsonl \
  --interpolation-method linear \
  --check-files
```

## Small Real-Data Smoke Workflow

SSL smoke on raw videos:

```bash
.venv/bin/python -m app.surgwmbench.pretrain_ssl \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --output checkpoints/surgwmbench_ssl_smoke.pt \
  --ssl-source raw_videos \
  --encoder-name dummy \
  --image-size 64 \
  --clip-length 4 \
  --stride 4 \
  --latent-dim 16 \
  --hidden-dim 32 \
  --batch-size 2 \
  --num-workers 0 \
  --epochs 1 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --precision fp32 \
  --device cpu \
  --max-videos 1 \
  --max-clips-per-video 2 \
  --max-samples 2
```

Sparse AC smoke using the SSL checkpoint:

```bash
.venv/bin/python -m app.surgwmbench.train_ac \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --output checkpoints/surgwmbench_ac_sparse_smoke.pt \
  --interpolation-method linear \
  --encoder-name dummy \
  --encoder-checkpoint checkpoints/surgwmbench_ssl_smoke.pt \
  --encoder-checkpoint-key encoder \
  --freeze-encoder \
  --image-size 64 \
  --latent-dim 16 \
  --hidden-dim 32 \
  --batch-size 2 \
  --num-workers 0 \
  --epochs 1 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --precision fp32 \
  --device cpu \
  --max-train-samples 4 \
  --max-val-samples 2
```

Sparse rollout evaluation smoke:

```bash
.venv/bin/python -m app.surgwmbench.eval_ac \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/test.jsonl \
  --checkpoint checkpoints/surgwmbench_ac_sparse_smoke.pt \
  --output results/surgwmbench_ac_sparse_smoke_metrics.json \
  --interpolation-method linear \
  --image-size 64 \
  --batch-size 2 \
  --num-workers 0 \
  --device cpu \
  --rollout-mode policy_rollout \
  --context-anchors 1 \
  --horizon 19 \
  --horizons 1 3 5 10 19 \
  --max-samples 2
```

`checkpoints/` and `results/` are ignored. Do not commit generated training artifacts.

## Larger GPU Training

The current larger-run path is:

1. optionally run SurgWMBench SSL domain adaptation with `app.surgwmbench.pretrain_ssl`
2. train sparse AC with `app.surgwmbench.train_ac`
3. evaluate sparse human-anchor rollout with `app.surgwmbench.eval_ac`

When using real V-JEPA checkpoints, load only the encoder and initialize SurgWMBench action/coordinate heads:

```bash
--encoder-name vit_giant
--encoder-checkpoint /path/to/vjepa2_checkpoint.pt
--encoder-checkpoint-key target_encoder
```

If using the SSL adapter checkpoint as initialization for sparse AC:

```bash
--encoder-checkpoint checkpoints/surgwmbench_ssl_vitg.pt
--encoder-checkpoint-key encoder
```

The official V-JEPA 2-AC predictor action dimensions are not the SurgWMBench default `[dx, dy, dt]`. Do not force incompatible predictor weights into the SurgWMBench action-conditioned heads.

## Coding Rules

- Prefer `rg` for searching.
- Keep edits scoped to SurgWMBench unless explicitly asked otherwise.
- Use `apply_patch` for manual edits.
- Use pathlib and typed Python where practical.
- Use PyTorch 2.x APIs: `torch.amp.autocast`, `torch.amp.GradScaler`, and `torch.inference_mode`.
- Avoid TensorFlow.
- Do not import `src.datasets.video_dataset` from SurgWMBench loaders; it imports decord at import time.
- Do not silently clip target coordinates.
- Do not report dense pseudo-coordinate metrics as sparse human-anchor metrics.

## Git Hygiene

Before committing, run targeted tests for the area touched. Keep generated files out of commits:

```text
checkpoints/
results/
.venv/
__pycache__/
.pytest_cache/
```
