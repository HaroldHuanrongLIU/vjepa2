from __future__ import annotations

import argparse
import json

import torch

from app.surgwmbench.eval_ac import evaluate_ac
from app.surgwmbench.train_ac import train_ac
from tests.surgwmbench_test_utils import make_surgwmbench_root


def _train_args(root, output):
    return argparse.Namespace(
        dataset_root=str(root),
        train_manifest="manifests/train.jsonl",
        val_manifest="manifests/val.jsonl",
        output=str(output),
        interpolation_method="linear",
        encoder_name="dummy",
        encoder_checkpoint=None,
        encoder_checkpoint_key="target_encoder",
        freeze_encoder=True,
        image_size=32,
        latent_dim=16,
        hidden_dim=32,
        batch_size=2,
        num_workers=0,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        precision="fp32",
        grad_clip_norm=1.0,
        seed=7,
        device="cpu",
        max_train_samples=2,
        max_val_samples=2,
        latent_weight=1.0,
        sparse_coord_weight=10.0,
        action_weight=1.0,
        smoothness_weight=0.1,
    )


def test_sparse_train_and_eval_smoke(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    checkpoint_path = tmp_path / "checkpoints" / "surgwmbench_sparse.pt"

    output = train_ac(_train_args(root, checkpoint_path))

    assert output == checkpoint_path
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state" in checkpoint
    assert checkpoint["model_config"]["encoder_name"] == "dummy"
    assert checkpoint["history"]

    metrics_path = tmp_path / "results" / "metrics.json"
    eval_args = argparse.Namespace(
        dataset_root=str(root),
        manifest="manifests/test.jsonl",
        checkpoint=str(checkpoint_path),
        output=str(metrics_path),
        interpolation_method="linear",
        image_size=32,
        batch_size=2,
        num_workers=0,
        device="cpu",
        rollout_mode="policy_rollout",
        context_anchors=1,
        horizon=19,
        horizons=[1, 3, 5, 10, 20],
        max_samples=2,
    )

    metrics_output = evaluate_ac(eval_args)
    payload = json.loads(metrics_output.read_text())

    assert metrics_output == metrics_path
    assert payload["dataset_name"] == "SurgWMBench"
    assert payload["primary_target"] == "sparse_human_anchors"
    assert payload["dense_target"] is None
    assert payload["rollout_mode"] == "policy_rollout"
    assert payload["num_clips"] == 2
    assert "ade" in payload["metrics_overall"]
    assert "horizon_20_error" in payload["metrics_overall"]
    assert set(payload["metrics_by_difficulty"]) == {"low", "null"}
