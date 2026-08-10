"""Collect hand pre-grasp candidates for one rendered dataset folder.

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(
    "/nas/Dataset/Dataset_2026/dataset_v2/Logistic_site/"
    "General_LogisticSite/conveyor_track_01"
)
GRIPPER_INFO_PATH = Path("/nas/ochansol/gripper_info/gripper_info_hand.json")
CAMERA_NAME = "top_view_camera"

OUTPUT_FOLDER_NAME = "pre_grasp"
HEIGHTMAP_FOLDER_NAME = "heightmaps"

# Scene selection
#   None: discover and process every scene under depth/<CAMERA_NAME>.
#   3: process only scene 0003.
SCENE_NUMBER = 0

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
# checked independently inside every preset BBox. Segmentation is not used;
# the candidate passes when at least two individual BBoxes pass.
ENABLE_BBOX_FILTER = True
BBOX_FILTER_MIN_DEPTH_RANGE = 0.01
BBOX_FILTER_MIN_PASS_COUNT = 2
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
            colour = colours(bbox_index % 10)
            bbox_pixels, bbox_valid = world_points_to_image_pixels(bbox, camera)
            if np.all(bbox_valid):
                closed = np.vstack((bbox_pixels, bbox_pixels[0]))
                axis.plot(
                    closed[:, 0],
                    closed[:, 1],
                    color=colour,
                    linewidth=2.0,
                    label=f"bbox {bbox_index}",
                )
                centre = np.mean(bbox_pixels, axis=0)
                bbox_range = filter_result["bbox_depth_ranges"][bbox_index]
                bbox_passed = filter_result["bbox_passed"][bbox_index]
                range_text = (
                    "empty"
                    if bbox_range is None
                    else f"{bbox_range * 1000.0:.1f} mm"
                )
                axis.text(
                    centre[0],
                    centre[1],
                    f"{bbox_index}: {'P' if bbox_passed else 'F'}\n{range_text}",
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
        f"(required={filter_result['required_count']})"
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
    scene_point_index: XYPointIndex,
) -> tuple[bool, dict]:
    """Check the min/max scene-depth range of each preset BBox independently.

    Only top-view scene depth is used. An empty BBox fails only its own check;
    no target segmentation is required. At least
    ``BBOX_FILTER_MIN_PASS_COUNT`` BBoxes must pass.
    """
    bboxes = np.asarray(grasp_bboxes, dtype=np.float64)
    if bboxes.ndim != 3 or bboxes.shape[1:] != (4, 3) or len(bboxes) == 0:
        raise ValueError(f"grasp_bboxes must have shape (N, 4, 3), got {bboxes.shape}")

    bbox_point_counts: list[int] = []
    bbox_depth_min: list[float | None] = []
    bbox_depth_max: list[float | None] = []
    bbox_depth_ranges: list[float | None] = []
    bbox_passed: list[bool] = []
    for bbox in bboxes:
        scene_points = scene_point_index.points_in_convex_polygon(bbox)
        bbox_point_counts.append(int(len(scene_points)))
        if len(scene_points) == 0:
            bbox_depth_min.append(None)
            bbox_depth_max.append(None)
            bbox_depth_ranges.append(None)
            bbox_passed.append(False)
            continue

        depth_min = float(np.min(scene_points[:, 2]))
        depth_max = float(np.max(scene_points[:, 2]))
        depth_range = depth_max - depth_min
        passed = depth_range >= float(BBOX_FILTER_MIN_DEPTH_RANGE)
        bbox_depth_min.append(depth_min)
        bbox_depth_max.append(depth_max)
        bbox_depth_ranges.append(depth_range)
        bbox_passed.append(passed)

    passed_count = int(sum(bbox_passed))
    required_count = int(BBOX_FILTER_MIN_PASS_COUNT)
    return passed_count >= required_count, {
        "bbox_count": int(len(bboxes)),
        "required_count": required_count,
        "passed_count": passed_count,
        "bbox_point_counts": bbox_point_counts,
        "bbox_depth_min": bbox_depth_min,
        "bbox_depth_max": bbox_depth_max,
        "bbox_depth_ranges": bbox_depth_ranges,
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


def save_grouped_records(output_path: Path, grouped_records: dict[str, list[dict]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"gripper_model": gripper_name, "data": records}
        for gripper_name, records in grouped_records.items()
    ]
    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp.{os.getpid()}"
    )
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4)
    os.replace(temporary_path, output_path)


def collect_scene(
    scene_id: str,
    database_archives: dict[str, list[tuple[dict, dict, dict, Path]]],
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
        if ENABLE_BBOX_FILTER or DEBUG_VISUALIZE_BBOX_2D
        else None
    )
    class_masks = load_class_masks(segmentation_path, mapping_path)

    grouped_records: dict[str, list[dict]] = {}
    failures: list[str] = []
    object_count = 0
    bbox_filter_evaluated = 0
    bbox_filter_rejected = 0

    for class_name, mask in class_masks.items():
        candidates, sampled_pixels, object_point_count = sample_object_candidates(
            mask, depth, camera, rng
        )
        if candidates.shape[0] < 1:
            failures.append(f"{class_name}: not enough valid depth points")
            continue

        if DEBUG_VISUALIZE_SAMPLING and (
            DEBUG_SCENE_IDS is None or scene_id in DEBUG_SCENE_IDS
        ):
            visualize_sampling(scene_id, class_name, depth, mask, sampled_pixels)

        object_count += 1

        for gripper_name, preset_entries in database_archives.items():
            for gripper, preset, archive, archive_path in preset_entries:
                for yaw in yaw_sweep(gripper):
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
                                bbox_passed, bbox_filter_result = hand_bbox_filter_passes(
                                    grasp_bbox,
                                    scene_point_index,
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
                            grouped_records.setdefault(gripper_name, []).append(record)
                        except Exception as error:
                            failures.append(
                                f"{class_name}/{gripper_name}/{preset.get('name')}/"
                                f"candidate={candidate_index}/yaw={yaw:.3f}: "
                                f"{type(error).__name__}: {error}"
                            )

    output_path = DATASET_ROOT / OUTPUT_FOLDER_NAME / f"{scene_id}.json"
    save_grouped_records(output_path, grouped_records)
    record_count = sum(len(records) for records in grouped_records.values())
    return {
        "scene_id": scene_id,
        "status": "saved",
        "output": str(output_path),
        "objects": object_count,
        "records": record_count,
        "bbox_filter_enabled": ENABLE_BBOX_FILTER,
        "bbox_filter_evaluated": bbox_filter_evaluated,
        "bbox_filter_rejected": bbox_filter_rejected,
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
    database = heightmap.load_hand_database(GRIPPER_INFO_PATH)
    archives = load_heightmap_archives(database)
    rng = np.random.default_rng(RANDOM_SEED)
    summary = {
        "dataset_root": str(DATASET_ROOT),
        "gripper_info_path": str(GRIPPER_INFO_PATH),
        "camera_name": CAMERA_NAME,
        "safety_margin": SAFETY_MARGIN,
        "yaw_step_deg": YAW_STEP_DEG,
        "point_sample_num": POINT_SAMPLE_NUM,
        "point_sample_pixel_dist": POINT_SAMPLE_PIXEL_DIST,
        "bbox_filter_enabled": ENABLE_BBOX_FILTER,
        "bbox_filter_min_depth_range": BBOX_FILTER_MIN_DEPTH_RANGE,
        "bbox_filter_min_pass_count": BBOX_FILTER_MIN_PASS_COUNT,
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

        result = collect_scene(scene_id, archives, rng)
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
    parser = argparse.ArgumentParser(description="Hand pre-grasp dataset collector")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_root_path", type=Path, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--section_name", type=str, default=None)
    parser.add_argument("--platform_name", type=str, default=None)
    parser.add_argument("--scene_start", type=int, default=None)
    parser.add_argument("--gripper_info_path", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate an existing pre_grasp/<scene>.json instead of skipping it.",
    )
    args = parser.parse_args(argv)

    global DATASET_ROOT, GRIPPER_INFO_PATH, SCENE_NUMBER, SKIP_EXISTING_PREGRASP
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
    if args.overwrite:
        SKIP_EXISTING_PREGRASP = False


if __name__ == "__main__":
    configure_from_cli()
    main()
