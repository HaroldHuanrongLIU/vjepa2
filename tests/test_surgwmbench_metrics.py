from __future__ import annotations

import math

import numpy as np
import torch

from src.utils.surgwmbench_metrics import (
    ade,
    discrete_frechet,
    endpoint_error,
    error_by_horizon,
    fde,
    sparse_anchor_metrics,
    symmetric_hausdorff,
    trajectory_length,
    trajectory_length_error,
    trajectory_smoothness,
)


def test_metrics_are_zero_for_identical_straight_line():
    target = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    pred = target.copy()

    assert ade(pred, target) == 0.0
    assert fde(pred, target) == 0.0
    assert endpoint_error(pred, target) == 0.0
    assert discrete_frechet(pred, target) == 0.0
    assert symmetric_hausdorff(pred, target) == 0.0
    assert trajectory_length(pred) == 2.0
    assert trajectory_length_error(pred, target) == 0.0
    assert trajectory_smoothness(pred) == 0.0
    assert error_by_horizon(pred, target, [1, 3]) == {1: 0.0, 3: 0.0}


def test_metrics_on_shifted_trajectory():
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pred = target + torch.tensor([1.0, 0.0])

    assert ade(pred, target) == 1.0
    assert fde(pred, target) == 1.0
    assert endpoint_error(pred, target) == 1.0
    assert discrete_frechet(pred, target) == 1.0
    assert symmetric_hausdorff(pred, target) == 1.0
    assert trajectory_length_error(pred, target) == 0.0


def test_batched_metrics_respect_masks():
    pred = torch.tensor(
        [
            [[0.0, 0.0], [3.0, 0.0], [100.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
        ]
    )
    target = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])

    assert math.isclose(ade(pred, target, mask), 0.8)
    assert math.isclose(fde(pred, target, mask), 1.5)
    assert error_by_horizon(pred, target, [1, 2, 3], mask) == {1: 0.0, 2: 1.5, 3: 1.0}


def test_empty_mask_returns_none():
    pred = torch.zeros(2, 3, 2)
    target = torch.ones(2, 3, 2)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    assert ade(pred, target, mask) is None
    assert fde(pred, target, mask) is None
    assert discrete_frechet(pred, target, mask) is None
    assert symmetric_hausdorff(pred, target, mask) is None
    assert trajectory_length(pred, mask) is None
    assert trajectory_length_error(pred, target, mask) is None
    assert trajectory_smoothness(pred, mask) is None
    assert error_by_horizon(pred, target, [1, 3], mask) == {1: None, 3: None}


def test_sparse_anchor_metrics_keys():
    target = torch.zeros(1, 20, 2)
    pred = target.clone()
    result = sparse_anchor_metrics(pred, target, horizons=[1, 3, 20])

    expected_keys = {
        "ade",
        "fde",
        "discrete_frechet",
        "symmetric_hausdorff",
        "endpoint_error",
        "trajectory_length_error",
        "smoothness",
        "horizon_1_error",
        "horizon_3_error",
        "horizon_20_error",
    }
    assert expected_keys.issubset(result)
    assert all(value == 0.0 for value in result.values())
