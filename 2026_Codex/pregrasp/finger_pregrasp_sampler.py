"""Legacy-style two/three-finger pre-grasp sampling for the 2026 pipeline.

The sampler is Isaac-free.  It uses the same top-view world point cloud and
finger/palm depth-gap test as the original ``chansol/pre_grasp_point_sampler``.
Joint/width calibration coefficients are intentionally not used here. They
are required only later, when Isaac Lab converts a saved width into joint
targets and calculates the width-dependent Z hop.
"""

from __future__ import annotations

import math

import numpy as np


PREGRASP_GEOMETRY_KEYS = (
    "gripper_name",
    "type",
    "width",
    "height",
    "finger_thickness",
    "depth",
)


def has_pregrasp_geometry(gripper: dict) -> bool:
    return all(key in gripper for key in PREGRASP_GEOMETRY_KEYS)


def validate_pregrasp_geometry(gripper: dict) -> None:
    missing = [key for key in PREGRASP_GEOMETRY_KEYS if key not in gripper]
    if missing:
        raise ValueError(
            f"{gripper.get('gripper_name', '<unnamed>')} pregrasp geometry "
            f"is not ready: missing={missing}"
        )
    numeric_keys = ("width", "height", "finger_thickness", "depth")
    invalid = [
        key
        for key in numeric_keys
        if not isinstance(gripper.get(key), (int, float))
        or not np.isfinite(float(gripper[key]))
        or float(gripper[key]) <= 0.0
    ]
    if invalid:
        raise ValueError(
            f"{gripper.get('gripper_name', '<unnamed>')} has invalid positive "
            f"pregrasp geometry values: {invalid}"
        )


def grippers_by_family(database: dict) -> dict[str, list[dict]]:
    result = {"finger2": [], "finger3": []}
    for gripper in database.values():
        gripper_type = str(gripper.get("type", "")).lower()
        family = "finger2" if gripper_type.startswith("finger2") else (
            "finger3" if gripper_type.startswith("finger3") else None
        )
        if family is not None and has_pregrasp_geometry(gripper):
            result[family].append(gripper)
    return result


def _rotation_2d(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)


def _transform_polygon(polygon: np.ndarray, xy, degrees: float) -> np.ndarray:
    return polygon @ _rotation_2d(degrees).T + np.asarray(xy, dtype=float)


def _finger_and_palm_polygons(gripper: dict, width: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    height = float(gripper["height"])
    thickness = float(gripper["finger_thickness"])
    gripper_type = str(gripper["type"]).lower()

    finger = np.asarray(
        [
            [height / 2.0, -width / 2.0 - thickness],
            [height / 2.0, -width / 2.0],
            [-height / 2.0, -width / 2.0],
            [-height / 2.0, -width / 2.0 - thickness],
        ],
        dtype=float,
    )
    palm = np.asarray(
        [
            [height / 2.0, -width / 2.0],
            [height / 2.0, 0.0],
            [-height / 2.0, 0.0],
            [-height / 2.0, -width / 2.0],
        ],
        dtype=float,
    )

    if gripper_type.startswith("finger2"):
        mirrored_finger = finger.copy()
        mirrored_finger[:, 1] *= -1.0
        full_palm = np.asarray(
            [
                [height / 2.0, -width / 2.0],
                [height / 2.0, width / 2.0],
                [-height / 2.0, width / 2.0],
                [-height / 2.0, -width / 2.0],
            ],
            dtype=float,
        )
        return [finger, mirrored_finger], [full_palm]

    infos = gripper.get("finger_bbox_info")
    if not isinstance(infos, list) or len(infos) < 3:
        raise ValueError(
            f"{gripper.get('gripper_name')} needs finger_bbox_info for finger3 sampling"
        )

    fingers: list[np.ndarray] = []
    palms: list[np.ndarray] = []
    for info in infos:
        offset = np.asarray(info["pos"][:2], dtype=float)
        rotation = _rotation_2d(float(info["rot"]))
        fingers.append(finger @ rotation.T + offset)
        palms.append(palm @ rotation.T + offset)
    return fingers, palms


def _points_in_convex_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros(0, dtype=bool)
    edge = np.roll(polygon, -1, axis=0) - polygon
    rel = points_xy[:, None, :] - polygon[None, :, :]
    cross = edge[None, :, 0] * rel[:, :, 1] - edge[None, :, 1] * rel[:, :, 0]
    tolerance = 1.0e-10
    return np.all(cross >= -tolerance, axis=1) | np.all(cross <= tolerance, axis=1)


def _maximum_z(points: np.ndarray, polygon: np.ndarray) -> float:
    inside = _points_in_convex_polygon(points[:, :2], polygon)
    if not np.any(inside):
        return 0.0
    return float(np.nanmax(points[inside, 2]))


def _candidate_record(
    gripper: dict,
    target_object: str,
    candidate: np.ndarray,
    yaw_deg: float,
    width: float,
    scene_points: np.ndarray,
    scene_point_index=None,
    *,
    min_depth_gap: float,
    safety_margin: float,
) -> dict | None:
    finger_polygons, palm_polygons = _finger_and_palm_polygons(gripper, width)
    finger_polygons = [
        _transform_polygon(polygon, candidate[:2], yaw_deg)
        for polygon in finger_polygons
    ]
    palm_polygons = [
        _transform_polygon(polygon, candidate[:2], yaw_deg)
        for polygon in palm_polygons
    ]

    if scene_point_index is not None:
        def polygon_max(polygon):
            points = scene_point_index.points_in_convex_polygon(polygon)
            return 0.0 if len(points) == 0 else float(np.nanmax(points[:, 2]))
    else:
        all_polygons = finger_polygons + palm_polygons
        stacked = np.concatenate(all_polygons, axis=0)
        extent_mask = (
            (scene_points[:, 0] >= stacked[:, 0].min())
            & (scene_points[:, 0] <= stacked[:, 0].max())
            & (scene_points[:, 1] >= stacked[:, 1].min())
            & (scene_points[:, 1] <= stacked[:, 1].max())
        )
        local_points = scene_points[extent_mask]
        if len(local_points) == 0:
            return None

        def polygon_max(polygon):
            return _maximum_z(local_points, polygon)

    finger_max = max(polygon_max(polygon) for polygon in finger_polygons)
    palm_max = max(polygon_max(polygon) for polygon in palm_polygons)
    depth_gap = palm_max - finger_max
    if not np.isfinite(depth_gap) or depth_gap < float(min_depth_gap):
        return None

    gripper_depth = float(gripper["depth"])
    if depth_gap > gripper_depth * 0.9:
        target_z = palm_max + safety_margin - gripper_depth * 0.9
    else:
        target_z = max(palm_max + safety_margin - 0.04, finger_max + safety_margin)

    return {
        "gripper_model": gripper["gripper_name"],
        "gripper_type": gripper["type"],
        "target_object": target_object,
        "target_points": [
            round(float(candidate[0]), 6),
            round(float(candidate[1]), 6),
            round(float(target_z), 6),
        ],
        "target_orientation": [0.0, 0.0, float(yaw_deg)],
        "target_width": round(float(width), 6),
    }


def collect_finger_records(
    database: dict,
    class_candidates: dict[str, np.ndarray],
    scene_points: np.ndarray,
    rng: np.random.Generator,
    *,
    scene_point_index=None,
    yaw_step_deg: int = 10,
    width_ratios: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4),
    min_depth_gap: float = 0.01,
    safety_margin: float = 0.002,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Choose one geometry-ready gripper per finger family and collect records."""
    if yaw_step_deg <= 0:
        raise ValueError("yaw_step_deg must be positive")

    families = grippers_by_family(database)
    grouped: dict[str, list[dict]] = {}
    messages: list[str] = []

    for family in ("finger2", "finger3"):
        candidates_for_family = families[family]
        if not candidates_for_family:
            messages.append(
                f"{family}: skipped because no gripper has all pregrasp "
                f"geometry fields {PREGRASP_GEOMETRY_KEYS}"
            )
            continue

        gripper = candidates_for_family[int(rng.integers(len(candidates_for_family)))]
        validate_pregrasp_geometry(gripper)
        name = gripper["gripper_name"]
        grouped[name] = []
        widths = np.clip(
            np.asarray(width_ratios, dtype=float) * float(gripper["width"]),
            0.0,
            float(gripper["width"]),
        )
        widths = np.unique(np.round(widths, 6))[::-1]
        yaw_max = int(round(float(gripper.get("yaw_max", 180))))
        yaws = range(yaw_max, 0, -int(yaw_step_deg))

        for target_object, object_candidates in class_candidates.items():
            for width in widths:
                for yaw in yaws:
                    for candidate in object_candidates:
                        record = _candidate_record(
                            gripper,
                            target_object,
                            candidate,
                            float(yaw),
                            float(width),
                            scene_points,
                            scene_point_index,
                            min_depth_gap=min_depth_gap,
                            safety_margin=safety_margin,
                        )
                        if record is not None:
                            grouped[name].append(record)

        record_count = len(grouped[name])
        messages.append(f"{family}: selected {name}, records={record_count}")
        if record_count == 0:
            grouped.pop(name)

    return grouped, messages
