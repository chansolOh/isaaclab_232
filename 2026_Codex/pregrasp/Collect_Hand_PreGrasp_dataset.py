"""Collect hand and two/three-finger pre-grasp candidates.

This script is Isaac-free.  It reuses the saved hand bottom height maps and
creates pre-grasp JSON files beside the input scene data:

    <DATASET_ROOT>/pre_grasp/0000.json
"""

from __future__ import annotations

import ast
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand_pregrasp_heightmap as heightmap
import finger_pregrasp_sampler as finger_sampler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(
    "/nas/Dataset/Dataset_2026/dataset_v2/Home/"
    "MasterBedroom/beside_table_01"
)
GRIPPER_INFO_PATH = Path("/nas/ochansol/gripper_info/gripper_info_hand_2026.json")
FINGER_GRIPPER_INFO_PATH = Path(
    "/nas/ochansol/gripper_info/gripper_info_new_2026.json"
)
CAMERA_NAME = "top_view_camera"

ENABLE_HAND_GRIPPERS = True
ENABLE_FINGER_GRIPPERS = True
# RANDOM_GRIPPER_CANDIDATES = ("Robotiq_2f140", "Inspire-F1", "Hitbot_z_efg_100")
# RANDOM_GRIPPER_CANDIDATES = ("Agibot-Omnihand-Pro_right", "Inspire-F1_right", "Leap-Hand-V1_right")
RANDOM_GRIPPER_CANDIDATES = ("UON_3finger_gripper",)
# Direct execution without CLI arguments uses the candidate list above. Any
# CLI/server argument switches this off and samples from every usable hand and
# finger gripper in the two databases.
FILTER_GRIPPERS_BY_CANDIDATES = True
FINGER_YAW_STEP_DEG = 10
FINGER_WIDTH_RATIOS = (1.0, 0.8, 0.6, 0.4)
FINGER_MIN_DEPTH_GAP = 0.01

OUTPUT_FOLDER_NAME = "pre_grasp"
HEIGHTMAP_FOLDER_NAME = "heightmaps"

# Scene selection
#   None: discover and process every scene under depth/<CAMERA_NAME>.
#   3: process only scene 0003.
SCENE_NUMBER = 1

# True: keep an existing pre_grasp/<scene_id>.json and move to the next scene.
# False: regenerate and overwrite an existing output file.
SKIP_EXISTING_PREGRASP = True

SAFETY_MARGIN = 0.002
MIN_OBJECT_PIXELS = 50
MIN_OBJECT_WORLD_POINTS = 20
POINT_SAMPLE_NUM = 1500
POINT_SAMPLE_PIXEL_DIST = 40.0
RANDOM_SEED = None

# Optional preset grasp-BBox filter. Height is always calculated from the hand
# collision height map first. When enabled, the min/max scene-depth range is
# checked independently inside every preset BBox. Segmentation is not used.
# BBoxes in one contact_set are OR alternatives; every contact_set must pass.
ENABLE_BBOX_FILTER = True
BBOX_FILTER_MIN_DEPTH_RANGE = 0.01
# After placing the hand at the height-map target Z, scene geometry inside a
# passing BBox must rise this far above the hand height map's lowest point.
BBOX_FILTER_MIN_HEIGHT_ABOVE_GRIPPER_BOTTOM = 0.01
BBOX_FILTER_SPATIAL_CELL_SIZE = 0.02

# Same sweep style as the old sampler: yaw_max, yaw_max-10, ..., 10.
YAW_STEP_DEG = 30

DEBUG_VISUALIZE_SAMPLING = False
DEBUG_VISUALIZE_BBOX_2D = False
DEBUG_VISUALIZE_HEIGHTMAP = False
DEBUG_SCENE_IDS = None  # None means all scenes. Example: {"0000"}
DEBUG_SHOW_OBJECT_MASK = DEBUG_VISUALIZE_SAMPLING
DEBUG_POINT_SIZE = 28

IGNORED_CLASSES = {
    "BACKGROUND",
    "UNLABELLED",
    "floor",
    "pallet,prop_general",
}


def safe_path_component(name: str) -> str:
    component = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("._")
    if not component:
        raise ValueError(f"Cannot create a path component from {name!r}")
    return component


def heightmap_path(gripper_name: str, preset_name: str) -> Path:
    return (
        GRIPPER_INFO_PATH.parent
        / HEIGHTMAP_FOLDER_NAME
        / safe_path_component(gripper_name)
        / f"{safe_path_component(preset_name)}.npz"
    )


def scene_id_from_path(path: Path) -> str:
    return path.stem


def load_top_camera(scene_conf_path: Path) -> dict:
    with open(scene_conf_path, "r", encoding="utf-8") as stream:
        scene_config = json.load(stream)
    for camera in scene_config.get("cameras", []):
        if camera.get("name") == CAMERA_NAME:
            return camera
    raise KeyError(f"Camera {CAMERA_NAME!r} was not found in {scene_conf_path}")


def parse_rgba_key(key: str) -> tuple[int, int, int, int]:
    value = ast.literal_eval(key)
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"Invalid RGBA key: {key!r}")
    return tuple(int(channel) for channel in value)


def load_class_masks(segmentation_path: Path, mapping_path: Path) -> dict[str, np.ndarray]:
    image = np.asarray(Image.open(segmentation_path).convert("RGBA"))
    with open(mapping_path, "r", encoding="utf-8") as stream:
        mapping = json.load(stream)

    masks: dict[str, np.ndarray] = {}
    for rgba_text, info in mapping.items():
        class_name = info.get("class")
        if class_name in IGNORED_CLASSES:
            continue
        rgba = np.asarray(parse_rgba_key(rgba_text), dtype=np.uint8)
        mask = np.all(image == rgba, axis=-1)
        if int(mask.sum()) < MIN_OBJECT_PIXELS:
            continue
        if class_name in masks:
            masks[class_name] |= mask
        else:
            masks[class_name] = mask
    return masks


def pixels_to_world_points(
    rows: np.ndarray,
    columns: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_world_transform: np.ndarray,
) -> np.ndarray:
    z = depth[rows, columns]
    x = (columns - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (rows - intrinsic[1, 2]) / intrinsic[1, 1] * z
    camera_points = np.vstack((x, y, z, np.ones_like(z)))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return (camera_world_transform @ flip @ camera_points)[:3].T


def distance_based_pixel_sampling(
    pixels: np.ndarray,
    dist_th: float,
) -> np.ndarray:
    """Greedy image-space spacing similar to the previous sampler."""
    if pixels.size == 0:
        return pixels.reshape(0, 2)

    selected: list[np.ndarray] = []
    dist_sq = float(dist_th) ** 2
    for pixel in pixels:
        if not selected:
            selected.append(pixel)
            continue
        selected_pixels = np.asarray(selected, dtype=float)
        delta = selected_pixels - pixel.astype(float)
        if np.all(np.sum(delta * delta, axis=1) >= dist_sq):
            selected.append(pixel)
    return np.asarray(selected, dtype=int)


def sample_object_candidates(
    mask: np.ndarray,
    depth: np.ndarray,
    camera: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    if rows.size < MIN_OBJECT_WORLD_POINTS:
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=int), int(rows.size)

    pixels = np.column_stack((rows, columns))
    sample_count = min(POINT_SAMPLE_NUM, len(pixels))
    sampled_indices = rng.choice(len(pixels), sample_count, replace=False)
    sampled_pixels = distance_based_pixel_sampling(
        pixels[sampled_indices],
        POINT_SAMPLE_PIXEL_DIST,
    )
    if len(sampled_pixels) == 0:
        return np.empty((0, 3), dtype=float), sampled_pixels, int(rows.size)

    points = pixels_to_world_points(
        sampled_pixels[:, 0],
        sampled_pixels[:, 1],
        depth,
        np.asarray(camera["intrinsic_isaac"], dtype=float),
        np.asarray(camera["cam_poses"], dtype=float),
    )
    unique_points, unique_indices = np.unique(
        points.round(6),
        axis=0,
        return_index=True,
    )
    return unique_points, sampled_pixels[unique_indices], int(rows.size)


def visualize_sampling(
    scene_id: str,
    class_name: str,
    depth: np.ndarray,
    mask: np.ndarray,
    sampled_pixels: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    valid_depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        f"PreGrasp sampling | scene={scene_id} | object={class_name} | "
        f"samples={len(sampled_pixels)}"
    )

    depth_image = axes[0].imshow(valid_depth, cmap="viridis")
    axes[0].set_title("Top-view depth")
    axes[0].set_axis_off()
    fig.colorbar(depth_image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(valid_depth, cmap="gray")
    if DEBUG_SHOW_OBJECT_MASK:
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[..., 1] = 1.0
        overlay[..., 3] = mask.astype(float) * 0.28
        axes[1].imshow(overlay)
    if len(sampled_pixels) > 0:
        axes[1].scatter(
            sampled_pixels[:, 1],
            sampled_pixels[:, 0],
            s=DEBUG_POINT_SIZE,
            c="yellow",
            edgecolors="black",
            linewidths=0.7,
            label="sampled grasp XY",
        )
        axes[1].legend(loc="upper right")
    axes[1].set_title("Object mask + sampled points")
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


def world_points_to_image_pixels(
    world_points: np.ndarray,
    camera: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with the inverse of ``pixels_to_world_points``."""
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"world_points must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=bool)

    intrinsic = np.asarray(camera["intrinsic_isaac"], dtype=np.float64)
    camera_world_transform = np.asarray(camera["cam_poses"], dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_points = (flip @ np.linalg.inv(camera_world_transform) @ homogeneous.T).T
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > 1.0e-8)

    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = (
        camera_points[valid, 0] / depth[valid] * intrinsic[0, 0]
        + intrinsic[0, 2]
    )
    pixels[valid, 1] = (
        camera_points[valid, 1] / depth[valid] * intrinsic[1, 1]
        + intrinsic[1, 2]
    )
    return pixels, valid


def visualize_bbox_filter_2d(
    scene_id: str,
    class_name: str,
    gripper_name: str,
    preset_name: str,
    yaw: float,
    candidate: np.ndarray,
    depth: np.ndarray,
    camera: dict,
    grasp_bboxes: list[list[list[float]]] | np.ndarray,
    scene_point_index: "XYPointIndex",
    filter_passed: bool,
    filter_result: dict,
) -> None:
    """Show projected preset BBoxes and their selected depth points."""
    import matplotlib.pyplot as plt

    bboxes = np.asarray(grasp_bboxes, dtype=np.float64)
    valid_depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan)
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    colours = plt.get_cmap("tab10")

    for axis in axes:
        depth_image = axis.imshow(valid_depth, cmap="viridis")
        for bbox_index, bbox in enumerate(bboxes):
            set_id = filter_result["bbox_sets"][bbox_index]
            colour = colours((set_id - 1) % 10)
            bbox_pixels, bbox_valid = world_points_to_image_pixels(bbox, camera)
            if np.all(bbox_valid):
                closed = np.vstack((bbox_pixels, bbox_pixels[0]))
                axis.plot(
                    closed[:, 0],
                    closed[:, 1],
                    color=colour,
                    linewidth=2.0,
                    label=f"bbox {bbox_index} / set {set_id}",
                )
                centre = np.mean(bbox_pixels, axis=0)
                bbox_range = filter_result["bbox_depth_ranges"][bbox_index]
                height_above_bottom = filter_result[
                    "bbox_height_above_gripper_bottom"
                ][bbox_index]
                bbox_passed = filter_result["bbox_passed"][bbox_index]
                range_text = (
                    "empty"
                    if bbox_range is None
                    else f"range={bbox_range * 1000.0:.1f} mm"
                )
                height_text = (
                    ""
                    if height_above_bottom is None
                    else f"\nrise={height_above_bottom * 1000.0:.1f} mm"
                )
                axis.text(
                    centre[0],
                    centre[1],
                    f"B{bbox_index} S{set_id}: {'P' if bbox_passed else 'F'}\n"
                    f"{range_text}{height_text}",
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox={"facecolor": colour, "edgecolor": "none", "alpha": 0.8},
                )

            selected_points = scene_point_index.points_in_convex_polygon(bbox)
            selected_pixels, selected_valid = world_points_to_image_pixels(
                selected_points, camera
            )
            selected_pixels = selected_pixels[selected_valid]
            if len(selected_pixels):
                axis.scatter(
                    selected_pixels[:, 0],
                    selected_pixels[:, 1],
                    s=7,
                    color=colour,
                    alpha=0.55,
                )

        candidate_pixels, candidate_valid = world_points_to_image_pixels(
            np.asarray(candidate, dtype=np.float64).reshape(1, 3), camera
        )
        if candidate_valid[0]:
            axis.scatter(
                candidate_pixels[0, 0],
                candidate_pixels[0, 1],
                marker="*",
                s=180,
                c="red",
                edgecolors="white",
                linewidths=1.0,
                zorder=10,
                label="sampled grasp XY",
            )

    axes[0].set_title("Top-view depth + projected grasp BBoxes")
    axes[0].set_xlim(0, depth.shape[1])
    axes[0].set_ylim(depth.shape[0], 0)
    axes[0].set_axis_off()

    all_bbox_pixels, all_bbox_valid = world_points_to_image_pixels(
        bboxes.reshape(-1, 3), camera
    )
    all_bbox_pixels = all_bbox_pixels[all_bbox_valid]
    if len(all_bbox_pixels):
        padding = 30.0
        axes[1].set_xlim(
            max(0.0, float(all_bbox_pixels[:, 0].min()) - padding),
            min(float(depth.shape[1]), float(all_bbox_pixels[:, 0].max()) + padding),
        )
        axes[1].set_ylim(
            min(float(depth.shape[0]), float(all_bbox_pixels[:, 1].max()) + padding),
            max(0.0, float(all_bbox_pixels[:, 1].min()) - padding),
        )
    axes[1].set_title("BBox zoom + depth points used by filter")

    figure.suptitle(
        f"scene={scene_id} | object={class_name} | {gripper_name}/{preset_name} | "
        f"yaw={yaw:.1f} | {'PASS' if filter_passed else 'REJECT'} | "
        f"passed BBoxes={filter_result['passed_count']}/"
        f"{filter_result['bbox_count']} "
        f"| passed sets={filter_result['passed_set_count']}/"
        f"{filter_result['set_count']}"
    )
    figure.colorbar(depth_image, ax=axes, fraction=0.025, pad=0.02, label="Depth [m]")
    plt.show()


def visualize_heightmap_match(
    scene_id: str,
    class_name: str,
    gripper_name: str,
    preset_name: str,
    requested_yaw: float,
    grasp_xy: np.ndarray,
    archive: dict,
    scene_world_points: np.ndarray,
    safety_margin: float,
) -> None:
    """Show the exact height maps used to calculate the candidate TCP Z."""
    import matplotlib.pyplot as plt

    gripper_map, selected_yaw = heightmap.select_yaw_height_map(
        archive, requested_yaw
    )
    resolution = float(np.asarray(archive["resolution"]).item())
    x_min = float(np.asarray(archive["x_min"]).item())
    y_min = float(np.asarray(archive["y_min"]).item())
    x_max = x_min + (gripper_map.shape[1] - 1) * resolution
    y_max = y_min + (gripper_map.shape[0] - 1) * resolution
    scene_map = heightmap.scene_height_map_aligned_to_gripper(
        scene_world_points,
        float(grasp_xy[0]),
        float(grasp_xy[1]),
        gripper_map,
        x_min,
        y_min,
        resolution,
    )

    overlap = np.isfinite(scene_map) & np.isfinite(gripper_map)
    difference = np.full(gripper_map.shape, np.nan, dtype=np.float64)
    difference[overlap] = scene_map[overlap] - gripper_map[overlap]
    first_contact_z, target_z = heightmap.calculate_first_contact_tcp_height(
        scene_map, gripper_map, safety_margin
    )

    extent = [x_min, x_max, y_min, y_max]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    maps = (gripper_map, scene_map, difference)
    titles = (
        f"Gripper bottom height map\nselected yaw={selected_yaw:.1f} deg",
        "Aligned scene height map",
        "Scene - gripper\n(max = first contact TCP Z)",
    )
    colour_maps = ("viridis", "viridis", "coolwarm")

    for axis, map_data, title, colour_map in zip(
        axes, maps, titles, colour_maps
    ):
        image = axis.imshow(
            map_data,
            origin="lower",
            extent=extent,
            cmap=colour_map,
            interpolation="nearest",
        )
        axis.scatter(
            0.0,
            0.0,
            marker="+",
            s=90,
            c="red",
            linewidths=1.8,
            label="grasp center",
        )
        axis.set_title(title)
        axis.set_xlabel("TCP-relative X [m]")
        axis.set_ylabel("TCP-relative Y [m]")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Z [m]")

    contact_cells = np.argwhere(
        np.isfinite(difference)
        & np.isclose(difference, first_contact_z, rtol=0.0, atol=1.0e-10)
    )
    if len(contact_cells):
        contact_row, contact_column = contact_cells[0]
        contact_x = x_min + float(contact_column) * resolution
        contact_y = y_min + float(contact_row) * resolution
        axes[2].scatter(
            contact_x,
            contact_y,
            marker="*",
            s=190,
            c="yellow",
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
            label="first contact cell",
        )
        axes[2].legend(loc="best")

    figure.suptitle(
        f"scene={scene_id} | object={class_name} | {gripper_name}/{preset_name} | "
        f"requested yaw={requested_yaw:.1f} | first_contact_z={first_contact_z:.5f} m | "
        f"target_z={target_z:.5f} m (margin={safety_margin:.5f} m)"
    )
    plt.tight_layout()
    plt.show()


def yaw_sweep(gripper: dict) -> range:
    yaw_max = int(gripper.get("yaw_max", 360))
    return range(yaw_max, 0, -int(YAW_STEP_DEG))


def quat_wxyz_to_matrix(quaternion: list[float] | np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(pose: dict) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(pose["orientation_wxyz"])
    matrix[:3, 3] = np.asarray(pose["position"], dtype=np.float64)
    return matrix


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64))
    )
    return (matrix @ homogeneous.T).T[:, :3]


def bbox_corners(fingertip: dict, start_tf: np.ndarray) -> np.ndarray:
    bbox = fingertip.get("grasp_bbox", {})
    if "points" in bbox:
        corners = np.asarray(bbox["points"], dtype=np.float64)
        if corners.shape != (4, 3):
            raise ValueError(f"grasp_bbox.points must be 4x3, got {corners.shape}")
        if str(fingertip.get("frame", "world")).lower() == "gripper_base":
            corners = transform_points(start_tf, corners)
    elif "min" in bbox and "max" in bbox:
        minimum = np.asarray(bbox["min"], dtype=np.float64)
        maximum = np.asarray(bbox["max"], dtype=np.float64)
        z_value = float(min(minimum[2], maximum[2]))
        corners = np.asarray(
            [
                [minimum[0], minimum[1], z_value],
                [maximum[0], minimum[1], z_value],
                [maximum[0], maximum[1], z_value],
                [minimum[0], maximum[1], z_value],
            ],
            dtype=np.float64,
        )
        if str(fingertip.get("frame", "world")).lower() == "gripper_base":
            corners = transform_points(start_tf, corners)
    else:
        raise ValueError("grasp_bbox requires either points or min/max")

    if "world_min_z" in bbox:
        corners[:, 2] = float(bbox["world_min_z"])
    return corners


def yaw_matrix(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    return np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def preset_relative_bboxes(preset: dict) -> list[np.ndarray]:
    """Return preset BBoxes relative to their common min/max extent center.

    The preset maker stores each swept grasp BBox in the START base-local
    frame. Convert every BBox through the START base transform, then use the
    center of the global coordinate-wise min/max extent. This gives every BBox
    equal geometric treatment regardless of how many BBoxes the preset has.
    """
    start_tf = pose_matrix(preset["start_base_tf"])
    fingertips = preset.get("fingertip_points", {})
    if not isinstance(fingertips, dict) or not fingertips:
        raise ValueError(f"Preset {preset.get('name', '<unnamed>')!r} has no fingertip grasp_bbox")

    world_bboxes = [bbox_corners(fingertip, start_tf) for fingertip in fingertips.values()]
    bbox_extent_center = heightmap.preset_tcp_world_point(preset)
    return [bbox - bbox_extent_center for bbox in world_bboxes]


def preset_bbox_sets(preset: dict) -> list[int]:
    """Return contact-set IDs in the same order as ``preset_relative_bboxes``.

    Legacy fingertips without ``contact_set`` each become an independent set,
    so every selected fingertip remains required until the user groups them.
    """
    fingertips = preset.get("fingertip_points", {})
    if not isinstance(fingertips, dict) or not fingertips:
        raise ValueError(
            f"Preset {preset.get('name', '<unnamed>')!r} has no fingertip grasp_bbox"
        )
    raw_sets = [
        fingertip.get("contact_set") if isinstance(fingertip, dict) else None
        for fingertip in fingertips.values()
    ]
    explicit_sets = {
        raw_set
        for raw_set in raw_sets
        if isinstance(raw_set, int) and not isinstance(raw_set, bool) and raw_set >= 1
    }
    result: list[int] = []
    used_sets = set(explicit_sets)
    next_set = 1
    for raw_set in raw_sets:
        if isinstance(raw_set, int) and not isinstance(raw_set, bool) and raw_set >= 1:
            result.append(raw_set)
            continue
        while next_set in used_sets:
            next_set += 1
        result.append(next_set)
        used_sets.add(next_set)
        next_set += 1
    return result


def record_grasp_bbox(record: dict, preset: dict) -> list[list[list[float]]]:
    target_point = np.asarray(record["target_points"], dtype=np.float64)
    yaw = float(record.get("selected_heightmap_yaw", record.get("requested_yaw", 0.0)))
    yaw_rotation = yaw_matrix(yaw)
    return [
        (relative_bbox @ yaw_rotation.T + target_point).tolist()
        for relative_bbox in preset_relative_bboxes(preset)
    ]


class XYPointIndex:
    """Small uniform-grid index for repeated world-XY polygon queries."""

    def __init__(self, points: np.ndarray, cell_size: float):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {self.points.shape}")
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self.cell_size = float(cell_size)
        self.order = np.empty(0, dtype=np.int64)
        self.ranges: dict[int, tuple[int, int]] = {}
        self.min_ix = self.min_iy = 0
        self.max_ix = self.max_iy = -1
        self.y_cell_count = 0
        if len(self.points) == 0:
            return

        ix = np.floor(self.points[:, 0] / self.cell_size).astype(np.int64)
        iy = np.floor(self.points[:, 1] / self.cell_size).astype(np.int64)
        self.min_ix, self.max_ix = int(ix.min()), int(ix.max())
        self.min_iy, self.max_iy = int(iy.min()), int(iy.max())
        self.y_cell_count = self.max_iy - self.min_iy + 1
        keys = (ix - self.min_ix) * self.y_cell_count + (iy - self.min_iy)
        self.order = np.argsort(keys, kind="stable")
        sorted_keys = keys[self.order]
        unique_keys, starts, counts = np.unique(
            sorted_keys, return_index=True, return_counts=True
        )
        self.ranges = {
            int(key): (int(start), int(start + count))
            for key, start, count in zip(unique_keys, starts, counts)
        }

    def points_in_convex_polygon(self, polygon: np.ndarray) -> np.ndarray:
        polygon_xy = np.asarray(polygon, dtype=np.float64)[:, :2]
        if polygon_xy.shape != (4, 2):
            raise ValueError(f"grasp bbox polygon must be 4x2, got {polygon_xy.shape}")
        if len(self.points) == 0:
            return self.points

        minimum = polygon_xy.min(axis=0)
        maximum = polygon_xy.max(axis=0)
        query_min_ix, query_min_iy = np.floor(minimum / self.cell_size).astype(np.int64)
        query_max_ix, query_max_iy = np.floor(maximum / self.cell_size).astype(np.int64)
        query_min_ix = max(int(query_min_ix), self.min_ix)
        query_max_ix = min(int(query_max_ix), self.max_ix)
        query_min_iy = max(int(query_min_iy), self.min_iy)
        query_max_iy = min(int(query_max_iy), self.max_iy)
        if query_min_ix > query_max_ix or query_min_iy > query_max_iy:
            return self.points[:0]

        chunks: list[np.ndarray] = []
        for ix in range(query_min_ix, query_max_ix + 1):
            for iy in range(query_min_iy, query_max_iy + 1):
                key = (ix - self.min_ix) * self.y_cell_count + (iy - self.min_iy)
                bounds = self.ranges.get(int(key))
                if bounds is not None:
                    chunks.append(self.order[bounds[0]:bounds[1]])
        if not chunks:
            return self.points[:0]

        candidate_indices = np.concatenate(chunks)
        candidates = self.points[candidate_indices]
        xy = candidates[:, :2]
        aabb = (
            (xy[:, 0] >= minimum[0])
            & (xy[:, 0] <= maximum[0])
            & (xy[:, 1] >= minimum[1])
            & (xy[:, 1] <= maximum[1])
        )
        candidates = candidates[aabb]
        if len(candidates) == 0:
            return candidates

        edges = np.roll(polygon_xy, -1, axis=0) - polygon_xy
        relative = candidates[:, None, :2] - polygon_xy[None, :, :]
        cross = (
            edges[None, :, 0] * relative[:, :, 1]
            - edges[None, :, 1] * relative[:, :, 0]
        )
        inside = np.all(cross >= -1.0e-10, axis=1) | np.all(
            cross <= 1.0e-10, axis=1
        )
        return candidates[inside]


def hand_bbox_filter_passes(
    grasp_bboxes: list[list[list[float]]] | np.ndarray,
    bbox_sets: list[int] | np.ndarray,
    scene_point_index: XYPointIndex,
    gripper_lowest_world_z: float,
) -> tuple[bool, dict]:
    """Check scene depth and usable object height inside every preset BBox.

    Only top-view scene depth is used. An empty BBox fails only its own check;
    no target segmentation is required. BBoxes in one set are OR alternatives;
    all sets are combined with AND. A BBox passes only when both of these are
    true:

    1. Its scene-depth min/max range reaches ``BBOX_FILTER_MIN_DEPTH_RANGE``.
    2. Its highest scene point is at least
       ``BBOX_FILTER_MIN_HEIGHT_ABOVE_GRIPPER_BOTTOM`` above the lowest point
       of the placed gripper bottom height map.
    """
    bboxes = np.asarray(grasp_bboxes, dtype=np.float64)
    if bboxes.ndim != 3 or bboxes.shape[1:] != (4, 3) or len(bboxes) == 0:
        raise ValueError(f"grasp_bboxes must have shape (N, 4, 3), got {bboxes.shape}")
    sets = np.asarray(bbox_sets)
    if sets.shape != (len(bboxes),):
        raise ValueError(
            f"bbox_sets must contain one set ID per BBox, got {sets.shape} for {len(bboxes)}"
        )
    if not np.issubdtype(sets.dtype, np.integer) or np.any(sets < 1):
        raise ValueError(f"bbox_sets must contain positive integers, got {sets.tolist()}")
    sets = sets.astype(np.int64)

    bbox_point_counts: list[int] = []
    bbox_depth_min: list[float | None] = []
    bbox_depth_max: list[float | None] = []
    bbox_depth_ranges: list[float | None] = []
    bbox_depth_range_passed: list[bool] = []
    bbox_height_above_gripper_bottom: list[float | None] = []
    bbox_height_passed: list[bool] = []
    bbox_passed: list[bool] = []
    for bbox in bboxes:
        scene_points = scene_point_index.points_in_convex_polygon(bbox)
        bbox_point_counts.append(int(len(scene_points)))
        if len(scene_points) == 0:
            bbox_depth_min.append(None)
            bbox_depth_max.append(None)
            bbox_depth_ranges.append(None)
            bbox_depth_range_passed.append(False)
            bbox_height_above_gripper_bottom.append(None)
            bbox_height_passed.append(False)
            bbox_passed.append(False)
            continue

        depth_min = float(np.min(scene_points[:, 2]))
        depth_max = float(np.max(scene_points[:, 2]))
        depth_range = depth_max - depth_min
        depth_range_passed = depth_range >= float(BBOX_FILTER_MIN_DEPTH_RANGE)
        height_above_gripper_bottom = depth_max - float(gripper_lowest_world_z)
        height_passed = height_above_gripper_bottom >= float(
            BBOX_FILTER_MIN_HEIGHT_ABOVE_GRIPPER_BOTTOM
        )
        passed = depth_range_passed and height_passed
        bbox_depth_min.append(depth_min)
        bbox_depth_max.append(depth_max)
        bbox_depth_ranges.append(depth_range)
        bbox_depth_range_passed.append(depth_range_passed)
        bbox_height_above_gripper_bottom.append(height_above_gripper_bottom)
        bbox_height_passed.append(height_passed)
        bbox_passed.append(passed)

    passed_count = int(sum(bbox_passed))
    unique_sets = sorted(int(value) for value in np.unique(sets))
    set_passed = {
        set_id: any(
            bbox_passed[index]
            for index in range(len(bbox_passed))
            if int(sets[index]) == set_id
        )
        for set_id in unique_sets
    }
    passed_set_count = int(sum(set_passed.values()))
    return passed_set_count == len(unique_sets), {
        "bbox_count": int(len(bboxes)),
        "passed_count": passed_count,
        "bbox_sets": sets.tolist(),
        "set_count": len(unique_sets),
        "passed_set_count": passed_set_count,
        "set_passed": set_passed,
        "bbox_point_counts": bbox_point_counts,
        "bbox_depth_min": bbox_depth_min,
        "bbox_depth_max": bbox_depth_max,
        "bbox_depth_ranges": bbox_depth_ranges,
        "bbox_depth_range_passed": bbox_depth_range_passed,
        "gripper_lowest_world_z": float(gripper_lowest_world_z),
        "bbox_height_above_gripper_bottom": bbox_height_above_gripper_bottom,
        "bbox_height_passed": bbox_height_passed,
        "bbox_passed": bbox_passed,
    }


def load_heightmap_archives(database: dict) -> dict[str, list[tuple[dict, dict, dict, Path]]]:
    archives: dict[str, list[tuple[dict, dict, dict, Path]]] = {}
    for gripper_name, gripper in database.items():
        for preset in gripper.get("preset", []):
            preset_name = preset.get("name")
            archive_path = heightmap_path(gripper_name, preset_name)
            if not archive_path.exists():
                raise FileNotFoundError(
                    f"Missing height map for {gripper_name}/{preset_name}: {archive_path}"
                )
            archive = heightmap.load_height_map_archive(archive_path)
            heightmap.validate_height_map_archive_preset(
                archive, preset, archive_path=archive_path
            )
            archives.setdefault(gripper_name, []).append(
                (gripper, preset, archive, archive_path)
            )
    return archives


def select_scene_gripper(
    hand_database: dict,
    finger_database: dict,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Choose exactly one gripper for this pre-grasp run."""
    candidates: list[tuple[str, str]] = []
    if FILTER_GRIPPERS_BY_CANDIDATES:
        gripper_names = list(dict.fromkeys(RANDOM_GRIPPER_CANDIDATES))
    else:
        gripper_names = list(
            dict.fromkeys([*hand_database.keys(), *finger_database.keys()])
        )

    for gripper_name in gripper_names:
        if (
            ENABLE_HAND_GRIPPERS
            and gripper_name in hand_database
            and hand_database[gripper_name].get("preset")
            and all(
                heightmap_path(gripper_name, preset.get("name")).exists()
                for preset in hand_database[gripper_name].get("preset", [])
            )
        ):
            candidates.append(("hand", gripper_name))
        if (
            ENABLE_FINGER_GRIPPERS
            and gripper_name in finger_database
            and finger_sampler.has_pregrasp_geometry(finger_database[gripper_name])
        ):
            candidates.append(("finger", gripper_name))

    if not candidates:
        selection_source = (
            f"RANDOM_GRIPPER_CANDIDATES={RANDOM_GRIPPER_CANDIDATES}"
            if FILTER_GRIPPERS_BY_CANDIDATES
            else "the complete hand/finger databases"
        )
        raise ValueError(
            f"No usable gripper was found from {selection_source}"
        )
    gripper_kind, gripper_name = candidates[int(rng.integers(len(candidates)))]
    return gripper_kind, gripper_name


def save_grouped_records(output_path: Path, grouped_records: dict[str, list[dict]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(grouped_records) != 1:
        raise ValueError(
            "Each scene must save exactly one gripper group, got "
            f"{list(grouped_records.keys())}"
        )
    gripper_name, records = next(iter(grouped_records.items()))
    payload = {"gripper_model": gripper_name, "data": records}
    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp.{os.getpid()}"
    )
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4)
    os.replace(temporary_path, output_path)


def collect_scene(
    scene_id: str,
    hand_database: dict,
    finger_database: dict,
    rng: np.random.Generator,
) -> dict:
    depth_path = DATASET_ROOT / "depth" / CAMERA_NAME / f"{scene_id}.npy"
    conf_path = DATASET_ROOT / "conf" / f"{scene_id}.json"
    segmentation_path = DATASET_ROOT / "inst_seg" / CAMERA_NAME / f"{scene_id}.png"
    mapping_path = DATASET_ROOT / "inst_seg" / CAMERA_NAME / f"semantics_mapping_{scene_id}.json"

    if not conf_path.exists():
        return {"scene_id": scene_id, "status": "skipped", "reason": f"missing {conf_path}"}
    if not segmentation_path.exists() or not mapping_path.exists():
        return {"scene_id": scene_id, "status": "skipped", "reason": "missing segmentation"}

    camera = load_top_camera(conf_path)
    depth = np.load(depth_path)
    scene_points = heightmap.top_depth_to_world_points(
        depth,
        np.asarray(camera["intrinsic_isaac"], dtype=float),
        np.asarray(camera["cam_poses"], dtype=float),
    )
    scene_point_index = (
        XYPointIndex(scene_points, BBOX_FILTER_SPATIAL_CELL_SIZE)
        if ENABLE_BBOX_FILTER or DEBUG_VISUALIZE_BBOX_2D or ENABLE_FINGER_GRIPPERS
        else None
    )
    class_masks = load_class_masks(segmentation_path, mapping_path)
    selected_gripper_kind, selected_gripper_name = select_scene_gripper(
        hand_database, finger_database, rng
    )
    if selected_gripper_kind == "hand":
        database_archives = load_heightmap_archives(
            {selected_gripper_name: hand_database[selected_gripper_name]}
        )
        selected_finger_database = {}
    else:
        database_archives = {}
        selected_finger_database = {
            selected_gripper_name: finger_database[selected_gripper_name]
        }
    print(
        f"PreGrasp > selected_gripper:{selected_gripper_name} "
        f"kind:{selected_gripper_kind}",
        flush=True,
    )

    grouped_records: dict[str, list[dict]] = {}
    failures: list[str] = []
    object_count = 0
    bbox_filter_evaluated = 0
    bbox_filter_rejected = 0
    class_candidates: dict[str, np.ndarray] = {}

    for class_name, mask in class_masks.items():
        candidates, sampled_pixels, object_point_count = sample_object_candidates(
            mask, depth, camera, rng
        )
        if candidates.shape[0] < 1:
            failures.append(f"{class_name}: not enough valid depth points")
            continue

        class_candidates[class_name] = candidates

        if DEBUG_VISUALIZE_SAMPLING and (
            DEBUG_SCENE_IDS is None or scene_id in DEBUG_SCENE_IDS
        ):
            visualize_sampling(scene_id, class_name, depth, mask, sampled_pixels)

        object_count += 1

        for gripper_name, preset_entries in database_archives.items():
            for gripper, preset, archive, archive_path in preset_entries:
                for yaw in yaw_sweep(gripper):
                    selected_gripper_map, _ = heightmap.select_yaw_height_map(
                        archive, float(yaw)
                    )
                    finite_gripper_bottom = selected_gripper_map[
                        np.isfinite(selected_gripper_map)
                    ]
                    if finite_gripper_bottom.size == 0:
                        failures.append(
                            f"{class_name}/{gripper_name}/{preset.get('name')}/"
                            f"yaw={yaw:.3f}: selected gripper height map is empty"
                        )
                        continue
                    gripper_lowest_relative_z = float(
                        np.min(finite_gripper_bottom)
                    )
                    for candidate_index, candidate in enumerate(candidates):
                        try:
                            record = heightmap.make_pregrasp_record(
                                archive,
                                preset,
                                scene_points,
                                float(candidate[0]),
                                float(candidate[1]),
                                float(yaw),
                                SAFETY_MARGIN,
                                class_name,
                                gripper,
                            )

                            record["scene_index"] = int(scene_id)
                            record["source_scene_id"] = scene_id
                            record["source_heightmap"] = str(archive_path)
                            record["source_object_point_count"] = object_point_count
                            record["source_candidate_index"] = int(candidate_index)
                            record["source_sampled_candidate_count"] = int(len(candidates))
                            record["requested_yaw"] = float(yaw)
                            grasp_bbox = record_grasp_bbox(record, preset)
                            grasp_bbox_sets = preset_bbox_sets(preset)
                            if DEBUG_VISUALIZE_HEIGHTMAP and (
                                DEBUG_SCENE_IDS is None or scene_id in DEBUG_SCENE_IDS
                            ):
                                visualize_heightmap_match(
                                    scene_id=scene_id,
                                    class_name=class_name,
                                    gripper_name=gripper_name,
                                    preset_name=str(preset.get("name", "<unnamed>")),
                                    requested_yaw=float(yaw),
                                    grasp_xy=candidate[:2],
                                    archive=archive,
                                    scene_world_points=scene_points,
                                    safety_margin=SAFETY_MARGIN,
                                )
                            if ENABLE_BBOX_FILTER or DEBUG_VISUALIZE_BBOX_2D:
                                gripper_lowest_world_z = (
                                    float(record["target_points"][2])
                                    + gripper_lowest_relative_z
                                )
                                bbox_passed, bbox_filter_result = hand_bbox_filter_passes(
                                    grasp_bbox,
                                    grasp_bbox_sets,
                                    scene_point_index,
                                    gripper_lowest_world_z,
                                )
                                if DEBUG_VISUALIZE_BBOX_2D and (
                                    DEBUG_SCENE_IDS is None or scene_id in DEBUG_SCENE_IDS
                                ):
                                    visualize_bbox_filter_2d(
                                        scene_id=scene_id,
                                        class_name=class_name,
                                        gripper_name=gripper_name,
                                        preset_name=str(preset.get("name", "<unnamed>")),
                                        yaw=float(yaw),
                                        candidate=candidate,
                                        depth=depth,
                                        camera=camera,
                                        grasp_bboxes=grasp_bbox,
                                        scene_point_index=scene_point_index,
                                        filter_passed=bbox_passed,
                                        filter_result=bbox_filter_result,
                                    )
                                if ENABLE_BBOX_FILTER:
                                    bbox_filter_evaluated += 1
                                    if not bbox_passed:
                                        bbox_filter_rejected += 1
                                        continue
                            record["grasp_bbox"] = grasp_bbox
                            record["grasp_bbox_sets"] = grasp_bbox_sets
                            grouped_records.setdefault(gripper_name, []).append(record)
                        except Exception as error:
                            failures.append(
                                f"{class_name}/{gripper_name}/{preset.get('name')}/"
                                f"candidate={candidate_index}/yaw={yaw:.3f}: "
                                f"{type(error).__name__}: {error}"
                            )

    finger_messages: list[str] = []
    if selected_finger_database:
        finger_records, finger_messages = finger_sampler.collect_finger_records(
            selected_finger_database,
            class_candidates,
            scene_points,
            rng,
            scene_point_index=scene_point_index,
            yaw_step_deg=FINGER_YAW_STEP_DEG,
            width_ratios=FINGER_WIDTH_RATIOS,
            min_depth_gap=FINGER_MIN_DEPTH_GAP,
            safety_margin=SAFETY_MARGIN,
        )
        grouped_records.update(finger_records)
        for message in finger_messages:
            print(f"PreGrasp > {message}", flush=True)

    if selected_gripper_name not in grouped_records or not grouped_records[selected_gripper_name]:
        return {
            "scene_id": scene_id,
            "status": "skipped",
            "reason": f"no pre-grasp records for selected gripper {selected_gripper_name}",
            "objects": object_count,
            "records": 0,
            "selected_gripper_kind": selected_gripper_kind,
            "selected_gripper_name": selected_gripper_name,
            "bbox_filter_enabled": ENABLE_BBOX_FILTER,
            "bbox_filter_evaluated": bbox_filter_evaluated,
            "bbox_filter_rejected": bbox_filter_rejected,
            "finger_messages": finger_messages,
            "failures": failures,
        }

    output_path = DATASET_ROOT / OUTPUT_FOLDER_NAME / f"{scene_id}.json"
    save_grouped_records(output_path, grouped_records)
    record_count = sum(len(records) for records in grouped_records.values())
    return {
        "scene_id": scene_id,
        "status": "saved",
        "output": str(output_path),
        "objects": object_count,
        "records": record_count,
        "selected_gripper_kind": selected_gripper_kind,
        "selected_gripper_name": selected_gripper_name,
        "bbox_filter_enabled": ENABLE_BBOX_FILTER,
        "bbox_filter_evaluated": bbox_filter_evaluated,
        "bbox_filter_rejected": bbox_filter_rejected,
        "finger_messages": finger_messages,
        "failures": failures,
    }


def discover_scene_ids() -> list[str]:
    if SCENE_NUMBER is not None:
        return [f"{int(SCENE_NUMBER):04d}"]

    depth_dir = DATASET_ROOT / "depth" / CAMERA_NAME
    return [scene_id_from_path(path) for path in sorted(depth_dir.glob("*.npy"))]


def main() -> None:
    print("PreGrasp > App_start", flush=True)
    print("PreGrasp > START", flush=True)
    print(
        "PreGrasp > gripper_selection_mode:"
        + (
            "direct_candidates"
            if FILTER_GRIPPERS_BY_CANDIDATES
            else "cli_all_grippers_random"
        ),
        flush=True,
    )
    hand_database = (
        heightmap.load_hand_database(GRIPPER_INFO_PATH)
        if ENABLE_HAND_GRIPPERS
        else {}
    )
    if ENABLE_FINGER_GRIPPERS:
        with open(FINGER_GRIPPER_INFO_PATH, "r", encoding="utf-8") as stream:
            finger_database = json.load(stream)
    else:
        finger_database = {}
    if not isinstance(finger_database, dict):
        raise ValueError("Finger gripper database root must be a JSON object")
    rng = np.random.default_rng(RANDOM_SEED)
    summary = {
        "dataset_root": str(DATASET_ROOT),
        "gripper_info_path": str(GRIPPER_INFO_PATH),
        "finger_gripper_info_path": str(FINGER_GRIPPER_INFO_PATH),
        "random_gripper_candidates": list(RANDOM_GRIPPER_CANDIDATES),
        "gripper_selection_mode": (
            "direct_candidates"
            if FILTER_GRIPPERS_BY_CANDIDATES
            else "cli_all_grippers_random"
        ),
        "hand_grippers_enabled": ENABLE_HAND_GRIPPERS,
        "finger_grippers_enabled": ENABLE_FINGER_GRIPPERS,
        "camera_name": CAMERA_NAME,
        "safety_margin": SAFETY_MARGIN,
        "yaw_step_deg": YAW_STEP_DEG,
        "point_sample_num": POINT_SAMPLE_NUM,
        "point_sample_pixel_dist": POINT_SAMPLE_PIXEL_DIST,
        "bbox_filter_enabled": ENABLE_BBOX_FILTER,
        "bbox_filter_min_depth_range": BBOX_FILTER_MIN_DEPTH_RANGE,
        "bbox_filter_min_height_above_gripper_bottom": (
            BBOX_FILTER_MIN_HEIGHT_ABOVE_GRIPPER_BOTTOM
        ),
        "bbox_filter_rule": "all_contact_sets_and__any_bbox_within_set_or",
        "scenes": [],
    }

    for scene_id in discover_scene_ids():
        print(f"PreGrasp > SCENE:{scene_id}", flush=True)
        output_path = DATASET_ROOT / OUTPUT_FOLDER_NAME / f"{scene_id}.json"
        if SKIP_EXISTING_PREGRASP and output_path.exists():
            result = {
                "scene_id": scene_id,
                "status": "skipped",
                "reason": f"existing {output_path}",
                "output": str(output_path),
            }
            summary["scenes"].append(result)
            print(f"{scene_id}: skipped (existing {output_path})")
            continue

        result = collect_scene(scene_id, hand_database, finger_database, rng)

        summary["scenes"].append(result)
        if result["status"] == "saved":
            print(
                f"{scene_id}: saved {result['records']} records from "
                f"{result['objects']} objects; bbox_filter="
                f"{result.get('bbox_filter_rejected', 0)}/"
                f"{result.get('bbox_filter_evaluated', 0)} rejected -> "
                f"{result['output']}"
            )
        else:
            print(f"{scene_id}: skipped ({result['reason']})")

    summary_path = DATASET_ROOT / OUTPUT_FOLDER_NAME / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary_path = summary_path.with_suffix(
        f"{summary_path.suffix}.tmp.{os.getpid()}"
    )
    with open(temporary_summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=4)
    os.replace(temporary_summary_path, summary_path)
    print(f"summary: {summary_path}")
    print("PreGrasp > END", flush=True)


def configure_from_cli(argv: list[str] | None = None) -> None:
    """Apply optional server/CLI overrides while preserving top-level variables."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Hand pre-grasp dataset collector")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_root_path", type=Path, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--section_name", type=str, default=None)
    parser.add_argument("--platform_name", type=str, default=None)
    parser.add_argument(
        "--scene_start",
        type=int,
        default=None,
        help="Override the top-level SCENE_NUMBER only when explicitly provided.",
    )
    parser.add_argument("--gripper_info_path", type=Path, default=None)
    parser.add_argument("--finger_gripper_info_path", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate an existing pre_grasp/<scene>.json instead of skipping it.",
    )
    args = parser.parse_args(raw_argv)

    global DATASET_ROOT, GRIPPER_INFO_PATH, FINGER_GRIPPER_INFO_PATH
    global SCENE_NUMBER, SKIP_EXISTING_PREGRASP, FILTER_GRIPPERS_BY_CANDIDATES
    FILTER_GRIPPERS_BY_CANDIDATES = len(raw_argv) == 0
    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root.expanduser().resolve()
    elif args.output_root_path is not None:
        missing = [
            name
            for name, value in (
                ("env_name", args.env_name),
                ("section_name", args.section_name),
                ("platform_name", args.platform_name),
            )
            if not value
        ]
        if missing:
            parser.error(
                "--output_root_path requires --env_name, --section_name and "
                f"--platform_name (missing: {', '.join(missing)})"
            )
        DATASET_ROOT = (
            args.output_root_path.expanduser().resolve()
            / args.env_name
            / args.section_name
            / args.platform_name
        )

    if args.scene_start is not None:
        SCENE_NUMBER = int(args.scene_start)
    if args.gripper_info_path is not None:
        GRIPPER_INFO_PATH = args.gripper_info_path.expanduser().resolve()
    if args.finger_gripper_info_path is not None:
        FINGER_GRIPPER_INFO_PATH = args.finger_gripper_info_path.expanduser().resolve()
    if args.overwrite:
        SKIP_EXISTING_PREGRASP = False


if __name__ == "__main__":
    configure_from_cli()
    main()
