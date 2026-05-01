# SurgWMBench V-JEPA 2 Baseline

This repository is adapted from the official V-JEPA 2 codebase for:

```text
SurgWMBench: A Dataset and World-Model Benchmark for Surgical Instrument Motion Planning
```

The SurgWMBench adaptation uses the official dataset manifests and keeps sparse
20-anchor human labels as the primary benchmark target. Dense interpolation
coordinates are auxiliary pseudo labels only and must not be reported as human
ground truth.

The repository intentionally keeps one top-level README and one
`requirements.txt`, both scoped to this SurgWMBench adaptation.

## Environment

Use Python 3.12 with `uv`:

```bash
uv python install 3.12
UV_TORCH_BACKEND=cu130 uv sync --locked --group surgwmbench
source .venv/bin/activate
```

Requirements fallback:

```bash
uv venv --python 3.12 .venv
uv pip install -e . -r requirements.txt
source .venv/bin/activate
```

The checked `uv.lock` targets the CUDA PyTorch wheel backend used on the local
training machine. CPU-only machines may need a CPU-specific lock regeneration;
do not mix CPU and CUDA lock updates in one commit unless intentionally doing
an environment migration.

## Dataset

The local dataset root used in this workspace is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Expected official manifests:

```text
manifests/train.jsonl
manifests/val.jsonl
manifests/test.jsonl
manifests/all.jsonl
```

Do not create random train/val/test splits. All paths are resolved relative to
the dataset root. Difficulty must come from manifests or annotations, not
folder names.

Validate the real dataset layout:

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/all.jsonl \
  --interpolation-method linear \
  --check-files
```

## Implemented Scope

Implemented:

- final-layout `SurgWMBenchClipDataset`
- sparse 20-anchor human-label loading
- dense pseudo-coordinate loading for `linear`, `pchip`, `akima`, and
  `cubic_spline`
- source labels that keep `human` and `interpolated` targets separate
- sparse, dense, and SSL video collators
- read-only dataset validation CLI
- raw-video / extracted-frame SSL smoke pretraining adapter
- V-JEPA-style SurgWMBench action-conditioned model skeleton
- local encoder checkpoint loading and dummy encoder smoke path
- sparse action-conditioned training
- sparse human-anchor rollout evaluation
- trajectory metrics and synthetic tests

Not yet implemented:

- full official V-JEPA mask pretraining on SurgWMBench videos
- dense auxiliary AC training and dense pseudo-coordinate evaluation
- feature caching / extraction CLI
- CEM or goal-conditioned planning

## Key Entry Points

- `src/datasets/surgwmbench.py`: final-layout clip loader
- `src/datasets/surgwmbench_video.py`: raw-video / extracted-frame SSL dataset
- `src/datasets/surgwmbench_collators.py`: sparse, dense, and SSL collators
- `src/models/surgwmbench_vjepa_ac.py`: SurgWMBench V-JEPA 2-AC-style model
- `src/utils/surgwmbench_metrics.py`: sparse trajectory metrics
- `app/surgwmbench/pretrain_ssl.py`: lightweight SurgWMBench SSL smoke pretraining
- `app/surgwmbench/train_ac.py`: sparse action-conditioned training
- `app/surgwmbench/eval_ac.py`: sparse human-anchor rollout evaluation
- `tools/validate_surgwmbench_loader.py`: read-only loader validation

## Tests

Run the targeted SurgWMBench synthetic suite:

```bash
python -m pytest \
  tests/test_surgwmbench_pretrain_ssl.py \
  tests/test_surgwmbench_dataset.py \
  tests/test_surgwmbench_collate.py \
  tests/test_surgwmbench_metrics.py \
  tests/test_surgwmbench_model_shapes.py \
  tests/test_surgwmbench_train_eval.py
```

Expected current result:

```text
19 passed
```

Warnings from upstream `timm` or PyTorch future deprecations may appear during
the repo-native ViT smoke test.

## Small Real-Data Smoke Workflow

First run SSL smoke on raw videos. This is a lightweight latent next-step SSL
adapter, not full official V-JEPA mask pretraining:

```bash
python -m app.surgwmbench.pretrain_ssl \
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

Then run sparse 20-anchor action-conditioned training:

```bash
python -m app.surgwmbench.train_ac \
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

Evaluate sparse human-anchor rollout:

```bash
python -m app.surgwmbench.eval_ac \
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

`checkpoints/` and `results/` are ignored and should not be committed.

## Larger GPU Training

For a larger SurgWMBench SSL adaptation run:

```bash
python -m app.surgwmbench.pretrain_ssl \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --output checkpoints/surgwmbench_ssl_vitg.pt \
  --ssl-source raw_videos \
  --encoder-name vit_giant \
  --encoder-checkpoint /path/to/vjepa2_checkpoint.pt \
  --encoder-checkpoint-key target_encoder \
  --no-freeze-encoder \
  --image-size 384 \
  --clip-length 16 \
  --stride 4 \
  --latent-dim 1024 \
  --hidden-dim 512 \
  --batch-size 4 \
  --num-workers 8 \
  --epochs 10 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --precision amp \
  --device cuda
```

For sparse AC training with the adapted encoder:

```bash
python -m app.surgwmbench.train_ac \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --output checkpoints/vjepa2_ac_surgwmbench_sparse.pt \
  --interpolation-method linear \
  --encoder-name vit_giant \
  --encoder-checkpoint checkpoints/surgwmbench_ssl_vitg.pt \
  --encoder-checkpoint-key encoder \
  --freeze-encoder \
  --image-size 384 \
  --latent-dim 1024 \
  --hidden-dim 512 \
  --batch-size 4 \
  --num-workers 8 \
  --epochs 100 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --precision amp \
  --device cuda
```

If using an official V-JEPA checkpoint directly, load only the encoder:

```bash
--encoder-name vit_giant
--encoder-checkpoint /path/to/vjepa2_checkpoint.pt
--encoder-checkpoint-key target_encoder
```

The official V-JEPA 2-AC predictor action dimensions are not the SurgWMBench
default `[dx, dy, dt]`. Do not force incompatible predictor weights into the
SurgWMBench action-conditioned heads.

## Evaluation Output

`app.surgwmbench.eval_ac` writes JSON with:

- `dataset_name`
- `manifest`
- `checkpoint`
- `interpolation_method`
- `primary_target: sparse_human_anchors`
- `dense_target: null`
- `rollout_mode`
- `metrics_overall`
- `metrics_by_difficulty`
- `num_clips`
- `timestamp`

Dense pseudo-coordinate metrics are not reported by the sparse evaluator.

## Regenerate the Lock

Use this only when intentionally updating dependencies:

```bash
UV_TORCH_BACKEND=cu130 uv lock --python 3.12
UV_TORCH_BACKEND=cu130 uv sync --locked --group surgwmbench
```
