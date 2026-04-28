from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INTERPOLATION_METHODS = ("linear", "pchip", "akima", "cubic_spline")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((48, 64, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _anchor_indices(num_frames: int) -> list[int]:
    return [round(i * (num_frames - 1) / 19) for i in range(20)]


def _coord_px(local_frame_idx: int, offset: float = 0.0) -> list[float]:
    return [10.0 + local_frame_idx * 1.5 + offset, 5.0 + local_frame_idx + offset]


def _coord_norm(coord_px: list[float]) -> list[float]:
    return [coord_px[0] / 64.0, coord_px[1] / 48.0]


def make_surgwmbench_root(
    tmp_path: Path,
    *,
    bad_version: bool = False,
    missing_interpolation_method: str | None = None,
) -> Path:
    root = tmp_path / "SurgWMBench"
    (root / "videos" / "video_01").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "video_01" / "video_left.avi").write_bytes(b"synthetic")
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    for metadata_name in (
        "dataset_stats.json",
        "difficulty_rubric.json",
        "interpolation_config.json",
        "source_videos.json",
        "validation_report.json",
    ):
        _write_json(root / "metadata" / metadata_name, {})

    entries = []
    for clip_idx, (patient_id, trajectory_id, num_frames, difficulty) in enumerate(
        [
            ("patient_001", "traj_001", 25, "low"),
            ("patient_002", "traj_002", 23, None),
        ],
        start=1,
    ):
        version = "SurgWMBench-v0" if bad_version else "SurgWMBench"
        frames_dir_rel = f"clips/{patient_id}/{trajectory_id}/frames"
        annotation_rel = f"clips/{patient_id}/{trajectory_id}/annotation.json"
        sampled_indices = _anchor_indices(num_frames)
        sampled_set = set(sampled_indices)

        frames = []
        for local_idx in range(num_frames):
            frame_rel = f"{frames_dir_rel}/{local_idx:06d}.jpg"
            _write_image(root / frame_rel, value=(clip_idx * 20 + local_idx) % 255)
            frames.append(
                {
                    "local_frame_idx": local_idx,
                    "source_frame_idx": 1000 + local_idx,
                    "path": frame_rel,
                }
            )

        human_anchors = []
        for anchor_idx, local_idx in enumerate(sampled_indices):
            px = _coord_px(local_idx)
            human_anchors.append(
                {
                    "anchor_idx": anchor_idx,
                    "local_frame_idx": local_idx,
                    "source_frame_idx": 1000 + local_idx,
                    "old_frame_idx": anchor_idx,
                    "coord_px": px,
                    "coord_norm": _coord_norm(px),
                }
            )

        interpolation_files = {
            method: f"interpolations/{patient_id}/{trajectory_id}.{method}.json" for method in INTERPOLATION_METHODS
        }
        for method_idx, method in enumerate(INTERPOLATION_METHODS):
            coords = []
            for local_idx in range(num_frames):
                is_anchor = local_idx in sampled_set
                px = _coord_px(local_idx, offset=0.0 if is_anchor else method_idx * 0.25)
                coords.append(
                    {
                        "local_frame_idx": local_idx,
                        "coord_px": px,
                        "coord_norm": _coord_norm(px),
                        "source": "human" if is_anchor else "interpolated",
                        "anchor_idx": sampled_indices.index(local_idx) if is_anchor else None,
                        "confidence": 1.0 if is_anchor else 0.6,
                        "label_weight": 1.0 if is_anchor else 0.5,
                        "is_out_of_bounds": False,
                    }
                )
            _write_json(root / interpolation_files[method], {"coordinates": coords})

        if missing_interpolation_method is not None:
            missing = root / interpolation_files[missing_interpolation_method]
            if missing.exists():
                missing.unlink()

        annotation = {
            "dataset_version": version,
            "patient_id": patient_id,
            "source_video_id": "video_01",
            "source_video_path": "videos/video_01/video_left.avi",
            "trajectory_id": trajectory_id,
            "difficulty": difficulty,
            "num_frames": num_frames,
            "frames": frames,
            "human_anchors": human_anchors,
            "sampled_indices": sampled_indices,
            "interpolation_files": interpolation_files,
            "default_interpolation_method": "linear",
            "image_size": {"width": 64, "height": 48},
        }
        _write_json(root / annotation_rel, annotation)

        entries.append(
            {
                "dataset_version": version,
                "patient_id": patient_id,
                "source_video_id": "video_01",
                "source_video_path": "videos/video_01/video_left.avi",
                "trajectory_id": trajectory_id,
                "difficulty": difficulty,
                "num_frames": num_frames,
                "annotation_path": annotation_rel,
                "frames_dir": frames_dir_rel,
                "interpolation_files": interpolation_files,
                "default_interpolation_method": "linear",
                "num_human_anchors": 20,
                "sampled_indices": sampled_indices,
            }
        )

    manifest_text = "\n".join(json.dumps(entry) for entry in entries)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "all"):
        (root / "manifests" / f"{split}.jsonl").write_text(manifest_text)
    return root
