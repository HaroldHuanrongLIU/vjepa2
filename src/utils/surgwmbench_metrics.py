"""Trajectory metrics for SurgWMBench sparse-anchor evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray | Sequence[Sequence[float]]


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_batched_coords(value: Any, name: str) -> tuple[np.ndarray, bool]:
    array = _to_numpy(value).astype(np.float64)
    if array.ndim == 2:
        if array.shape[-1] != 2:
            raise ValueError(f"{name} must have shape [T, 2] or [B, T, 2], got {array.shape}")
        return array[None, ...], False
    if array.ndim == 3 and array.shape[-1] == 2:
        return array, True
    raise ValueError(f"{name} must have shape [T, 2] or [B, T, 2], got {array.shape}")


def _as_batched_mask(mask: Any | None, batch_size: int, timesteps: int) -> np.ndarray:
    if mask is None:
        return np.ones((batch_size, timesteps), dtype=bool)
    array = _to_numpy(mask).astype(bool)
    if array.ndim == 1:
        if array.shape[0] != timesteps:
            raise ValueError(f"mask shape {array.shape} does not match timesteps={timesteps}")
        return np.broadcast_to(array[None, :], (batch_size, timesteps)).copy()
    if array.ndim == 2:
        if array.shape != (batch_size, timesteps):
            raise ValueError(f"mask shape {array.shape} must match {(batch_size, timesteps)}")
        return array
    raise ValueError(f"mask must have shape [T] or [B, T], got {array.shape}")


def _prepare_pair(
    pred: ArrayLike, target: ArrayLike, mask: Any | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_np, _ = _as_batched_coords(pred, "pred")
    target_np, _ = _as_batched_coords(target, "target")
    if pred_np.shape != target_np.shape:
        raise ValueError(f"pred and target shapes differ: {pred_np.shape} vs {target_np.shape}")
    mask_np = _as_batched_mask(mask, pred_np.shape[0], pred_np.shape[1])
    return pred_np, target_np, mask_np


def _prepare_coords(coords: ArrayLike, mask: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    coords_np, _ = _as_batched_coords(coords, "coords")
    mask_np = _as_batched_mask(mask, coords_np.shape[0], coords_np.shape[1])
    return coords_np, mask_np


def _valid_pairs(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for pred_i, target_i, mask_i in zip(pred, target, mask):
        valid = mask_i.astype(bool)
        if valid.any():
            pairs.append((pred_i[valid], target_i[valid]))
    return pairs


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _discrete_frechet_single(pred: np.ndarray, target: np.ndarray) -> float | None:
    if len(pred) == 0 or len(target) == 0:
        return None
    ca = np.full((len(pred), len(target)), np.inf, dtype=np.float64)
    for i in range(len(pred)):
        for j in range(len(target)):
            dist = np.linalg.norm(pred[i] - target[j])
            if i == 0 and j == 0:
                ca[i, j] = dist
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], dist)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], dist)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), dist)
    return float(ca[-1, -1])


def _hausdorff_single(pred: np.ndarray, target: np.ndarray) -> float | None:
    if len(pred) == 0 or len(target) == 0:
        return None
    distances = np.linalg.norm(pred[:, None, :] - target[None, :, :], axis=-1)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def ade(pred: ArrayLike, target: ArrayLike, mask: Any | None = None) -> float | None:
    """Average displacement error over valid points."""

    pred_np, target_np, mask_np = _prepare_pair(pred, target, mask)
    distances = np.linalg.norm(pred_np - target_np, axis=-1)
    valid = mask_np.astype(bool)
    if not valid.any():
        return None
    return float(distances[valid].mean())


def fde(pred: ArrayLike, target: ArrayLike, mask: Any | None = None) -> float | None:
    """Final displacement error at each trajectory's final valid point."""

    pred_np, target_np, mask_np = _prepare_pair(pred, target, mask)
    values = [
        float(np.linalg.norm(pred_i[-1] - target_i[-1]))
        for pred_i, target_i in _valid_pairs(pred_np, target_np, mask_np)
    ]
    return _mean_or_none(values)


def endpoint_error(pred: ArrayLike, target: ArrayLike, mask: Any | None = None) -> float | None:
    """Endpoint error, equivalent to FDE for valid masked trajectories."""

    return fde(pred, target, mask)


def discrete_frechet(pred: ArrayLike, target: ArrayLike, mask: Any | None = None) -> float | None:
    """Mean discrete Fréchet distance over valid trajectories."""

    pred_np, target_np, mask_np = _prepare_pair(pred, target, mask)
    values = [
        value
        for pred_i, target_i in _valid_pairs(pred_np, target_np, mask_np)
        if (value := _discrete_frechet_single(pred_i, target_i)) is not None
    ]
    return _mean_or_none(values)


def symmetric_hausdorff(pred: ArrayLike, target: ArrayLike, mask: Any | None = None) -> float | None:
    """Mean symmetric Hausdorff distance over valid trajectories."""

    pred_np, target_np, mask_np = _prepare_pair(pred, target, mask)
    values = [
        value
        for pred_i, target_i in _valid_pairs(pred_np, target_np, mask_np)
        if (value := _hausdorff_single(pred_i, target_i)) is not None
    ]
    return _mean_or_none(values)


def trajectory_length(coords: ArrayLike, mask: Any | None = None) -> float | None:
    """Mean polyline length over valid trajectories."""

    coords_np, mask_np = _prepare_coords(coords, mask)
    values: list[float] = []
    for coords_i, mask_i in zip(coords_np, mask_np):
        valid_coords = coords_i[mask_i.astype(bool)]
        if len(valid_coords) < 2:
            continue
        values.append(float(np.linalg.norm(valid_coords[1:] - valid_coords[:-1], axis=-1).sum()))
    return _mean_or_none(values)


def trajectory_length_error(pred: ArrayLike, target: ArrayLike, mask: Any | None = None) -> float | None:
    """Mean absolute difference between predicted and target polyline lengths."""

    pred_np, target_np, mask_np = _prepare_pair(pred, target, mask)
    values: list[float] = []
    for pred_i, target_i in _valid_pairs(pred_np, target_np, mask_np):
        if len(pred_i) < 2:
            continue
        pred_len = np.linalg.norm(pred_i[1:] - pred_i[:-1], axis=-1).sum()
        target_len = np.linalg.norm(target_i[1:] - target_i[:-1], axis=-1).sum()
        values.append(float(abs(pred_len - target_len)))
    return _mean_or_none(values)


def trajectory_smoothness(coords: ArrayLike, mask: Any | None = None) -> float | None:
    """Mean second-order coordinate difference norm over valid triples."""

    coords_np, mask_np = _prepare_coords(coords, mask)
    values: list[float] = []
    for coords_i, mask_i in zip(coords_np, mask_np):
        valid_coords = coords_i[mask_i.astype(bool)]
        if len(valid_coords) < 3:
            continue
        accel = valid_coords[2:] - 2.0 * valid_coords[1:-1] + valid_coords[:-2]
        values.extend(np.linalg.norm(accel, axis=-1).astype(float).tolist())
    return _mean_or_none(values)


def error_by_horizon(
    pred: ArrayLike, target: ArrayLike, horizons: Sequence[int], mask: Any | None = None
) -> dict[int, float | None]:
    """Point error at 1-indexed valid horizons."""

    pred_np, target_np, mask_np = _prepare_pair(pred, target, mask)
    result: dict[int, float | None] = {}
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"horizons must be positive 1-indexed values, got {horizon}")
        values: list[float] = []
        index = horizon - 1
        for pred_i, target_i in _valid_pairs(pred_np, target_np, mask_np):
            if len(pred_i) > index:
                values.append(float(np.linalg.norm(pred_i[index] - target_i[index])))
        result[int(horizon)] = _mean_or_none(values)
    return result


def sparse_anchor_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    mask: Any | None = None,
    horizons: Sequence[int] = (1, 3, 5, 10, 20),
    prefix: str = "",
) -> dict[str, float | None]:
    """Compute the standard sparse human-anchor metric dictionary."""

    metrics: dict[str, float | None] = {
        f"{prefix}ade": ade(pred, target, mask),
        f"{prefix}fde": fde(pred, target, mask),
        f"{prefix}discrete_frechet": discrete_frechet(pred, target, mask),
        f"{prefix}symmetric_hausdorff": symmetric_hausdorff(pred, target, mask),
        f"{prefix}endpoint_error": endpoint_error(pred, target, mask),
        f"{prefix}trajectory_length_error": trajectory_length_error(pred, target, mask),
        f"{prefix}smoothness": trajectory_smoothness(pred, mask),
    }
    for horizon, value in error_by_horizon(pred, target, horizons, mask).items():
        metrics[f"{prefix}horizon_{horizon}_error"] = value
    return metrics
