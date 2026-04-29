"""Read-only SurgWMBench final-layout validation command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.surgwmbench import (
    DATASET_VERSION,
    _coord_from_item,
    _frame_local_index,
    _frame_path_value,
    _parse_image_size,
    load_json,
    read_jsonl_manifest,
    resolve_dataset_path,
)


def _add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def _interpolation_files(entry: dict[str, Any], annotation: dict[str, Any]) -> dict[str, str]:
    files = annotation.get("interpolation_files") or entry.get("interpolation_files")
    if not isinstance(files, dict):
        raise ValueError("interpolation_files must be a mapping")
    return {str(key): str(value) for key, value in files.items()}


def _frame_path(dataset_root: Path, frames_dir: Path, frame: Any, fallback_idx: int) -> Path:
    local_idx = _frame_local_index(frame, fallback_idx)
    value = _frame_path_value(frame)
    if value is None:
        return frames_dir / f"{local_idx:06d}.jpg"
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    root_path = resolve_dataset_path(dataset_root, candidate)
    if root_path.exists():
        return root_path
    frame_dir_path = resolve_dataset_path(frames_dir, value)
    if frame_dir_path is None:
        return frames_dir / value
    return frame_dir_path


def _validate_interpolation(
    *,
    interpolation_path: Path,
    method: str,
    num_frames: int,
    image_size_hw: tuple[int, int],
    human_anchors: list[dict[str, Any]],
    errors: list[str],
) -> None:
    try:
        interpolation = load_json(interpolation_path)
    except Exception as exc:
        _add_error(errors, f"{interpolation_path}: cannot read interpolation file: {exc}")
        return
    coordinates = interpolation.get("coordinates") if isinstance(interpolation, dict) else None
    if not isinstance(coordinates, list):
        _add_error(errors, f"{interpolation_path}: missing coordinates[]")
        return
    if len(coordinates) != num_frames:
        _add_error(
            errors, f"{interpolation_path}: method {method} has {len(coordinates)} coordinates, expected {num_frames}"
        )

    by_local_idx: dict[int, dict[str, Any]] = {}
    for fallback_idx, coord in enumerate(coordinates):
        if not isinstance(coord, dict):
            _add_error(errors, f"{interpolation_path}: coordinate {fallback_idx} is not an object")
            continue
        by_local_idx[int(coord.get("local_frame_idx", fallback_idx))] = coord

    for anchor_pos, anchor in enumerate(human_anchors):
        local_idx = int(anchor.get("local_frame_idx", anchor_pos))
        coord = by_local_idx.get(local_idx)
        if coord is None:
            _add_error(
                errors, f"{interpolation_path}: missing coordinate for human anchor local_frame_idx={local_idx}"
            )
            continue
        if str(coord.get("source")).lower() != "human":
            _add_error(errors, f"{interpolation_path}: anchor local_frame_idx={local_idx} source is not 'human'")
        if float(coord.get("confidence", 0.0)) != 1.0:
            _add_error(errors, f"{interpolation_path}: anchor local_frame_idx={local_idx} confidence is not 1.0")
        if float(coord.get("label_weight", 0.0)) != 1.0:
            _add_error(errors, f"{interpolation_path}: anchor local_frame_idx={local_idx} label_weight is not 1.0")

        anchor_px, _ = _coord_from_item(anchor, image_size_hw)
        coord_px, _ = _coord_from_item(coord, image_size_hw)
        if not np.allclose(np.asarray(anchor_px), np.asarray(coord_px), atol=1e-5):
            _add_error(
                errors,
                f"{interpolation_path}: anchor local_frame_idx={local_idx} coordinate does not match human label",
            )


def validate_surgwmbench(
    dataset_root: str | Path,
    manifest: str | Path,
    interpolation_method: str | None = None,
    check_files: bool = False,
    num_samples: int | None = None,
) -> list[str]:
    dataset_root = Path(dataset_root).expanduser()
    manifest_path = Path(manifest).expanduser()
    manifest_path = manifest_path if manifest_path.is_absolute() else dataset_root / manifest_path
    errors: list[str] = []

    try:
        entries = read_jsonl_manifest(manifest_path)
    except Exception as exc:
        return [f"{manifest_path}: cannot read manifest: {exc}"]
    if num_samples is not None:
        entries = entries[: max(num_samples, 0)]

    for entry_idx, entry in enumerate(entries):
        prefix = f"manifest entry {entry_idx}"
        if entry.get("dataset_version") != DATASET_VERSION:
            _add_error(
                errors, f"{prefix}: dataset_version={entry.get('dataset_version')!r}, expected {DATASET_VERSION!r}"
            )

        annotation_path = resolve_dataset_path(dataset_root, entry.get("annotation_path"))
        if annotation_path is None:
            _add_error(errors, f"{prefix}: missing annotation_path")
            continue
        if not annotation_path.exists():
            _add_error(errors, f"{prefix}: annotation not found: {annotation_path}")
            continue

        try:
            annotation = load_json(annotation_path)
        except Exception as exc:
            _add_error(errors, f"{annotation_path}: cannot read annotation: {exc}")
            continue
        if not isinstance(annotation, dict):
            _add_error(errors, f"{annotation_path}: annotation is not an object")
            continue
        if annotation.get("dataset_version", entry.get("dataset_version")) != DATASET_VERSION:
            _add_error(errors, f"{annotation_path}: dataset_version is not {DATASET_VERSION!r}")

        frames = annotation.get("frames")
        if not isinstance(frames, list):
            _add_error(errors, f"{annotation_path}: missing frames[]")
            frames = []
        human_anchors = annotation.get("human_anchors")
        if not isinstance(human_anchors, list):
            _add_error(errors, f"{annotation_path}: missing human_anchors[]")
            human_anchors = []
        if len(human_anchors) != 20:
            _add_error(errors, f"{annotation_path}: expected 20 human anchors, got {len(human_anchors)}")

        sampled_indices = annotation.get("sampled_indices", entry.get("sampled_indices"))
        if not isinstance(sampled_indices, list):
            _add_error(errors, f"{annotation_path}: sampled_indices is not a list")
            sampled_indices = []
        if len(sampled_indices) != 20:
            _add_error(errors, f"{annotation_path}: expected 20 sampled_indices, got {len(sampled_indices)}")

        try:
            image_size_hw = _parse_image_size(annotation.get("image_size"))
        except Exception as exc:
            _add_error(errors, f"{annotation_path}: invalid image_size: {exc}")
            image_size_hw = (1, 1)

        num_frames = int(entry.get("num_frames", annotation.get("num_frames", len(frames))))
        if len(frames) and len(frames) != num_frames:
            _add_error(
                errors, f"{annotation_path}: frames[] length {len(frames)} does not match num_frames={num_frames}"
            )

        if check_files:
            source_video_path = resolve_dataset_path(
                dataset_root, annotation.get("source_video_path", entry.get("source_video_path"))
            )
            if source_video_path is not None and not source_video_path.exists():
                _add_error(errors, f"{prefix}: source video not found: {source_video_path}")
            frames_dir = resolve_dataset_path(dataset_root, entry.get("frames_dir"))
            if frames_dir is None or not frames_dir.exists():
                _add_error(errors, f"{prefix}: frames_dir not found: {frames_dir}")
            elif frames:
                for fallback_idx, frame in enumerate(frames):
                    frame_path = _frame_path(dataset_root, frames_dir, frame, fallback_idx)
                    if not frame_path.exists():
                        _add_error(errors, f"{annotation_path}: frame not found: {frame_path}")

        try:
            files = _interpolation_files(entry, annotation)
        except Exception as exc:
            _add_error(errors, f"{annotation_path}: invalid interpolation_files: {exc}")
            continue
        selected_method = interpolation_method or entry.get("default_interpolation_method") or "linear"
        methods = sorted(files)
        if selected_method not in files:
            _add_error(errors, f"{annotation_path}: selected interpolation method {selected_method!r} not listed")
        for method in methods:
            path = resolve_dataset_path(dataset_root, files[method])
            if path is None or not path.exists():
                _add_error(errors, f"{annotation_path}: interpolation file not found for {method}: {path}")
                continue
            _validate_interpolation(
                interpolation_path=path,
                method=method,
                num_frames=num_frames,
                image_size_hw=image_size_hw,
                human_anchors=human_anchors,
                errors=errors,
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--interpolation-method", default=None)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    errors = validate_surgwmbench(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        interpolation_method=args.interpolation_method,
        check_files=args.check_files,
        num_samples=args.num_samples,
    )
    if errors:
        print(f"SurgWMBench validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("SurgWMBench validation passed.")


if __name__ == "__main__":
    main()
