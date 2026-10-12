#!/usr/bin/env python3
"""Batch filter and merge every Dataset v2 object/gripper combination.

This is the non-interactive counterpart of ``filter_grasps_gui_v2.py``.
There are no command-line arguments: edit the variables in the configuration
section below and run the file.  Source scene files are read once per gripper,
split into compressed per-object temporary shards, filtered, and exported in
the same compact zero-pose format as the v2 GUI.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import dataset_v2_adapter as adapter
from grasp.mesh_geometry import load_obj_triangles, resolve_obj_path, transform_object_vertices
from merge_grasps import grasp_boxes, mesh_in_box, rotation_distance


# =============================================================================
# 실행 설정: argparse 없이 이 변수들만 수정한다.
# =============================================================================
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")
MODEL_ROOT = Path("/nas/ochansol/3d_model/peel3_scan_data_2026")
OUTPUT_SUBDIR = "grasp_data"
MESH_UNIT_SCALE = 0.01

# None이면 발견되는 250개 물체와 13개 그리퍼를 전부 처리한다.
# 일부만 시험할 때 문자열 목록으로 바꾼다.
OBJECT_ALLOWLIST: list[str] | None = None
GRIPPER_ALLOWLIST: list[str] | None = None

# ------------------------------- quality filter
# dataset_v2에는 외력/자세/접촉면 score가 없고 disturbed_object_count만 있다.
# target 자체가 움직인 1개는 제외한 collateral count로 다음 점수를 사용한다.
# disturbance_score = 1 / (1 + max(0, disturbed_object_count - 1))
USE_DISTURBANCE_SCORE_FILTER = True
MIN_DISTURBANCE_SCORE = 0.5

# None이면 별도 제한하지 않는다. 0이면 target 외 다른 물체를 움직이지 않은
# grasp만 남긴다. 점수 필터와 동시에 켜도 된다.
MAX_COLLATERAL_OBJECTS: int | None = None

# ------------------------------- empty bbox filter
USE_EMPTY_BBOX_FILTER = True
BOX_THICKNESS_M = 0.02
BOX_MARGIN_M = 0.002
MIN_OCCUPIED_BOXES = 1

# ------------------------------- NMS
USE_NMS = True
NMS_CENTER_DISTANCE_M = 0.015
NMS_ROTATION_DISTANCE_DEG = 20.0

# 필터 후 점수 내림차순으로 최대 N개만 저장한다. None이면 제한 없음.
MAX_GRASPS_PER_OBJECT: int | None = None

# 기존 JSON/NPZ가 있어도 현재 필터 설정으로 다시 쓴다.
OVERWRITE_EXISTING = True

# scene을 한 번만 읽기 위한 압축 임시 shard. 정상적으로 한 그리퍼를 끝내면
# 자동 삭제한다. 디버깅하려면 True로 바꾼다.
WORK_ROOT = MODEL_ROOT / ".batch_filter_merge_v2_work"
KEEP_TEMP_SHARDS = False

# 전체 실행 현황 파일
SUMMARY_PATH = MODEL_ROOT / "batch_filter_merge_v2_summary.json"
PRINT_EVERY_SCENES = 100


def _selected(all_names: list[str], allowlist: list[str] | None, label: str) -> list[str]:
    if allowlist is None:
        return all_names
    requested = list(dict.fromkeys(allowlist))
    missing = sorted(set(requested) - set(all_names), key=str.casefold)
    if missing:
        raise ValueError(f"존재하지 않는 {label}: {missing}")
    return requested


def _safe_name(value: str) -> str:
    return adapter._safe_name(value)


def _atomic_summary(payload: dict) -> None:
    adapter.atomic_json(SUMMARY_PATH, payload)


def _remove_work_directory(path: Path) -> None:
    """Remove only one validated child of the configured work root."""
    resolved_root = WORK_ROOT.resolve()
    resolved = path.resolve()
    if resolved.parent != resolved_root or not resolved.name:
        raise RuntimeError(f"임시 폴더 삭제 범위가 안전하지 않습니다: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def _hand_match_key(record: dict) -> tuple:
    def numbers(name: str) -> tuple[float, ...]:
        values = np.asarray(record.get(name, []), dtype=np.float64).reshape(-1)
        return tuple(np.round(values, 9).tolist())

    return (
        str(record.get("target_object", "")),
        str(record.get("preset_name", "")),
        numbers("target_points"),
        numbers("target_orientation"),
    )


def _hand_pregrasp_lookup(pregrasps: list[dict]) -> dict[tuple, list[dict]]:
    lookup: dict[tuple, list[dict]] = {}
    for record in pregrasps:
        lookup.setdefault(_hand_match_key(record), []).append(record)
    return lookup


def _match_hand_pregrasp_fast(
    output: dict,
    pregrasps: list[dict],
    lookup: dict[tuple, list[dict]],
) -> dict | None:
    exact = lookup.get(_hand_match_key(output), [])
    if len(exact) == 1:
        return exact[0]
    if exact:
        output_width = float(output.get("target_width", 0.0))
        return min(
            exact,
            key=lambda item: abs(float(item.get("target_width", 0.0)) - output_width),
        )
    return adapter._match_hand_pregrasp(output, pregrasps)


def _append_json_line(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def build_gripper_shards(
    gripper: str,
    selected_objects: set[str],
    work_dir: Path,
) -> tuple[dict[str, dict], Counter, dict]:
    """Read every source scene once and split converted grasps by object."""
    rows = adapter.usage_rows(DATASET_ROOT, gripper)
    finger_database = adapter.load_json(adapter.FINGER_INFO_PATH)
    hand_database = adapter.load_json(adapter.HAND_INFO_PATH)
    finger_info = finger_database.get(gripper)
    is_hand_gripper = gripper in hand_database

    work_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}
    object_configs: dict[str, dict] = {}
    counts: Counter = Counter()
    diagnostics: Counter = Counter()
    started = time.time()

    try:
        for row_index, row in enumerate(rows):
            relative = row["platform_path"]
            scene_id = row["scene_id"]
            scene_ref = f"{relative}/{scene_id}"
            base = DATASET_ROOT / relative
            conf_path = base / "conf" / f"{scene_id}.json"
            output_path = base / "output_grasp" / f"{scene_id}.json"
            if not conf_path.is_file() or not output_path.is_file():
                diagnostics["missing_source_file"] += 1
                continue

            try:
                conf = adapter.load_json(conf_path)
            except (OSError, ValueError, TypeError):
                diagnostics["source_read_error"] += 1
                continue
            class_to_object: dict[str, tuple[str, dict]] = {}
            for obj in conf.get("objects", []):
                object_name = adapter.object_name_from_conf(obj)
                if object_name in selected_objects:
                    class_to_object[str(obj.get("class", ""))] = (object_name, obj)
                    object_configs.setdefault(object_name, adapter.zero_pose_object_config(obj))
            if not class_to_object:
                continue
            try:
                raw_payload = adapter.load_json(output_path)
            except (OSError, ValueError, TypeError):
                diagnostics["source_read_error"] += 1
                continue
            raw_records = raw_payload if isinstance(raw_payload, list) else []
            if not raw_records:
                continue

            pregrasps: list[dict] = []
            hand_lookup: dict[tuple, list[dict]] = {}
            if is_hand_gripper:
                pre_path = base / "pre_grasp" / f"{scene_id}.json"
                if pre_path.is_file():
                    try:
                        pregrasps = adapter._flatten_pregrasp(adapter.load_json(pre_path))
                        hand_lookup = _hand_pregrasp_lookup(pregrasps)
                    except (OSError, ValueError, TypeError):
                        diagnostics["pregrasp_read_error"] += 1

            for output_index, raw in enumerate(raw_records):
                resolved = class_to_object.get(str(raw.get("target_object", "")))
                if resolved is None:
                    continue
                object_name, obj = resolved
                if raw.get("gripper_model", gripper) != gripper:
                    diagnostics["gripper_mismatch"] += 1
                    continue
                try:
                    if is_hand_gripper or str(raw.get("gripper_type", "")).lower() == "hand":
                        matched = _match_hand_pregrasp_fast(raw, pregrasps, hand_lookup)
                        if matched is None:
                            diagnostics["hand_pregrasp_unmatched"] += 1
                            continue
                        boxes_world = np.asarray(matched.get("grasp_bbox", []), dtype=np.float64)
                    else:
                        if finger_info is None:
                            diagnostics["missing_gripper_info"] += 1
                            continue
                        boxes_world = adapter._finger_boxes_world(raw, finger_info)
                    if (
                        boxes_world.ndim != 3
                        or boxes_world.shape[1:] != (4, 3)
                        or not len(boxes_world)
                        or not np.all(np.isfinite(boxes_world))
                    ):
                        diagnostics["invalid_bbox"] += 1
                        continue
                    converted = adapter._zero_pose_record(
                        raw,
                        obj,
                        boxes_world,
                        source_scene=scene_ref,
                        output_index=output_index,
                    )
                    compact = adapter.compact_v2_item(converted)
                    handle = handles.get(object_name)
                    if handle is None:
                        shard_path = work_dir / f"{_safe_name(object_name)}.jsonl.gz"
                        handle = gzip.open(shard_path, "wt", encoding="utf-8", compresslevel=3)
                        handles[object_name] = handle
                    _append_json_line(handle, compact)
                    counts[object_name] += 1
                except (KeyError, TypeError, ValueError, FloatingPointError):
                    diagnostics["record_conversion_error"] += 1

            if PRINT_EVERY_SCENES > 0 and (
                (row_index + 1) % PRINT_EVERY_SCENES == 0 or row_index + 1 == len(rows)
            ):
                elapsed = time.time() - started
                print(
                    f"[{gripper}] shard {row_index + 1}/{len(rows)} scenes · "
                    f"{sum(counts.values())} grasps · {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    return object_configs, counts, dict(diagnostics)


def _load_shard(path: Path) -> list[dict]:
    items = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                items.append(json.loads(line))
    return items


def _occupied_box_count(
    item: dict,
    mesh_vertices: np.ndarray,
    mesh_tree: cKDTree,
) -> int:
    count = 0
    for box in grasp_boxes(item):
        center = box.mean(axis=0)
        radius = (
            float(np.linalg.norm(box - center, axis=1).max())
            + max(0.0, BOX_THICKNESS_M) * 0.5
            + max(0.0, BOX_MARGIN_M)
        )
        indices = mesh_tree.query_ball_point(center, max(radius, 1.0e-9))
        if indices and mesh_in_box(
            mesh_vertices[indices],
            box,
            max(0.0, BOX_THICKNESS_M),
            max(0.0, BOX_MARGIN_M),
        ):
            count += 1
    return count


def _nms(items: list[dict]) -> list[dict]:
    center_threshold = max(0.0, float(NMS_CENTER_DISTANCE_M))
    rotation_threshold = max(0.0, float(NMS_ROTATION_DISTANCE_DEG))
    if center_threshold <= 0.0:
        return items
    kept: list[dict] = []
    cells: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    ordered = sorted(items, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for item in ordered:
        boxes = grasp_boxes(item)
        if not len(boxes):
            continue
        center = boxes.reshape(-1, 3).mean(axis=0)
        matrix = np.asarray(item.get("grasp_mat", np.eye(4)), dtype=np.float64)
        rotation = matrix[:3, :3] if matrix.shape == (4, 4) else np.eye(3)
        cell = tuple(np.floor(center / center_threshold).astype(np.int64).tolist())
        nearby = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nearby.extend(cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()))
        duplicate = any(
            np.linalg.norm(center - previous_center) <= center_threshold
            and rotation_distance(rotation, previous_rotation) <= rotation_threshold
            for previous_center, previous_rotation in nearby
        )
        if not duplicate:
            kept.append(item)
            cells.setdefault(cell, []).append((center, rotation))
    return kept


def filter_items(items: list[dict], obj_conf: dict) -> tuple[list[dict], dict]:
    counts = {"loaded": len(items)}
    if USE_DISTURBANCE_SCORE_FILTER:
        items = [
            item
            for item in items
            if float(item.get("disturbance_score", item.get("score", 0.0)))
            >= float(MIN_DISTURBANCE_SCORE)
        ]
    counts["after_disturbance_score"] = len(items)

    if MAX_COLLATERAL_OBJECTS is not None:
        items = [
            item
            for item in items
            if int(item.get("collateral_object_count", math.inf))
            <= int(MAX_COLLATERAL_OBJECTS)
        ]
    counts["after_collateral_count"] = len(items)

    if USE_EMPTY_BBOX_FILTER and items:
        obj_path = resolve_obj_path(obj_conf["usd_path"])
        vertices, _ = load_obj_triangles(obj_path)
        vertices = transform_object_vertices(
            vertices, obj_conf, MESH_UNIT_SCALE, apply_pose=False
        )
        tree = cKDTree(vertices)
        required = max(1, int(MIN_OCCUPIED_BOXES))
        items = [
            item
            for item in items
            if _occupied_box_count(item, vertices, tree) >= required
        ]
    counts["after_empty_bbox"] = len(items)

    if USE_NMS and items:
        items = _nms(items)
    counts["after_nms"] = len(items)

    items.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            int(item.get("collateral_object_count", 0)),
            str(item.get("scene_id", "")),
            int(item.get("source_output_index", 0)),
        )
    )
    if MAX_GRASPS_PER_OBJECT is not None:
        items = items[: max(0, int(MAX_GRASPS_PER_OBJECT))]
    counts["saved"] = len(items)
    return items, counts


def filter_metadata() -> dict:
    return {
        "disturbance_score": {
            "enabled": USE_DISTURBANCE_SCORE_FILTER,
            "minimum": MIN_DISTURBANCE_SCORE,
            "definition": "1 / (1 + max(0, disturbed_object_count - 1))",
        },
        "max_collateral_objects": MAX_COLLATERAL_OBJECTS,
        "empty_bbox": {
            "enabled": USE_EMPTY_BBOX_FILTER,
            "thickness_m": BOX_THICKNESS_M,
            "margin_m": BOX_MARGIN_M,
            "min_occupied_boxes": MIN_OCCUPIED_BOXES,
        },
        "nms": {
            "enabled": USE_NMS,
            "center_distance_m": NMS_CENTER_DISTANCE_M,
            "rotation_distance_deg": NMS_ROTATION_DISTANCE_DEG,
            "priority": "disturbance_score_descending",
        },
        "max_grasps_per_object": MAX_GRASPS_PER_OBJECT,
    }


def export_object_gripper(
    object_name: str,
    gripper: str,
    obj_conf: dict,
    shard_path: Path,
) -> dict:
    output_dir = MODEL_ROOT / object_name / OUTPUT_SUBDIR / gripper
    stem = f"{object_name}_{gripper}_merged_grasp_zero_pose"
    json_path = output_dir / f"{stem}.json"
    npz_path = output_dir / f"{stem}.npz"
    if not OVERWRITE_EXISTING and json_path.is_file() and npz_path.is_file():
        return {"status": "skipped_existing", "json": str(json_path), "npz": str(npz_path)}

    loaded = _load_shard(shard_path)
    items, counts = filter_items(loaded, obj_conf)
    source_scenes = sorted({str(item.get("scene_id", "")) for item in loaded})
    metadata = {
        "schema": "dataset_v2_object_grasp_compact",
        "schema_version": 1,
        "coordinate_frame": "object_zero_pose",
        "object": object_name,
        "gripper_model": gripper,
        "gripper_type": items[0].get("gripper_type", "") if items else (
            loaded[0].get("gripper_type", "") if loaded else ""
        ),
        "source": {
            "dataset_root": str(DATASET_ROOT.resolve()),
            "format": "environment/section/platform/output_grasp",
        },
        "filters": filter_metadata(),
        "object_config_zero_pose": copy.deepcopy(obj_conf),
        "source_scene_count": len(source_scenes),
        "count": len(items),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter.atomic_json(json_path, {"metadata": metadata, "data": items})
    adapter.save_v2_npz(npz_path, metadata, items)
    return {
        "status": "saved",
        "counts": counts,
        "json": str(json_path),
        "npz": str(npz_path),
    }


def validate_configuration() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)
    if not MODEL_ROOT.is_dir():
        raise FileNotFoundError(MODEL_ROOT)
    if not 0.0 <= float(MIN_DISTURBANCE_SCORE) <= 1.0:
        raise ValueError("MIN_DISTURBANCE_SCORE는 0~1이어야 합니다")
    if MAX_COLLATERAL_OBJECTS is not None and MAX_COLLATERAL_OBJECTS < 0:
        raise ValueError("MAX_COLLATERAL_OBJECTS는 None 또는 0 이상이어야 합니다")
    if MIN_OCCUPIED_BOXES < 1:
        raise ValueError("MIN_OCCUPIED_BOXES는 1 이상이어야 합니다")


def main() -> None:
    validate_configuration()
    objects = _selected(adapter.object_names(MODEL_ROOT), OBJECT_ALLOWLIST, "물체")
    grippers = _selected(adapter.gripper_names(DATASET_ROOT), GRIPPER_ALLOWLIST, "그리퍼")
    selected_objects = set(objects)
    print(
        f"Batch grasp merge 시작 · objects={len(objects)} · grippers={len(grippers)}\n"
        f"source={DATASET_ROOT}\noutput={MODEL_ROOT}",
        flush=True,
    )

    summary = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(DATASET_ROOT.resolve()),
        "model_root": str(MODEL_ROOT.resolve()),
        "objects": objects,
        "grippers": grippers,
        "filters": filter_metadata(),
        "results": {},
    }
    _atomic_summary(summary)

    for gripper_index, gripper in enumerate(grippers):
        print(f"\n[{gripper_index + 1}/{len(grippers)}] {gripper}", flush=True)
        work_dir = WORK_ROOT / _safe_name(gripper)
        _remove_work_directory(work_dir)
        try:
            object_configs, raw_counts, diagnostics = build_gripper_shards(
                gripper, selected_objects, work_dir
            )
            gripper_result = {
                "status": "filtering",
                "raw_count": int(sum(raw_counts.values())),
                "diagnostics": diagnostics,
                "objects": {},
            }
            summary["results"][gripper] = gripper_result
            _atomic_summary(summary)

            available_objects = [name for name in objects if raw_counts.get(name, 0) > 0]
            for object_index, object_name in enumerate(available_objects):
                shard_path = work_dir / f"{_safe_name(object_name)}.jsonl.gz"
                try:
                    result = export_object_gripper(
                        object_name,
                        gripper,
                        object_configs[object_name],
                        shard_path,
                    )
                except Exception as exc:
                    result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                gripper_result["objects"][object_name] = result
                if result["status"] == "saved":
                    counts = result["counts"]
                    print(
                        f"  [{object_index + 1}/{len(available_objects)}] {object_name}: "
                        f"{counts['loaded']} -> {counts['saved']}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{object_index + 1}/{len(available_objects)}] "
                        f"{object_name}: {result['status']}",
                        flush=True,
                    )
                if (object_index + 1) % 10 == 0:
                    _atomic_summary(summary)

            gripper_result["status"] = "completed"
            gripper_result["object_count"] = len(available_objects)
            _atomic_summary(summary)
            if not KEEP_TEMP_SHARDS:
                _remove_work_directory(work_dir)
        except Exception as exc:
            summary["results"][gripper] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _atomic_summary(summary)
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)

    summary["status"] = "completed"
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_summary(summary)
    print(f"\n전체 완료 · summary={SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
