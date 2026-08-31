"""Analyze and visualize scene-level pre-grasp gripper usage.

Expected dataset layout::

    <dataset_root>/<environment>/<section>/<platform>/
        conf/<scene>.json
        pre_grasp/<scene>.json

Each pre-grasp scene must contain exactly one gripper group. Statistics are
therefore counted per scene, not by the number of candidate records stored in
the scene JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Editable configuration for direct execution
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")
OUTPUT_DIR = DATASET_ROOT / "pregrasp_statistics"
HAND_GRIPPER_INFO_PATH = Path(
    "/nas/ochansol/gripper_info/gripper_info_hand_2026.json"
)
FINGER_GRIPPER_INFO_PATH = Path(
    "/nas/ochansol/gripper_info/gripper_info_new_2026.json"
)
# True: save each graph and also open it with matplotlib.
# False: save PNG files only (useful on a headless server).
SHOW_PLOTS = True
ANNOTATE_HEATMAP_MAX_CELLS = 400


@dataclass(frozen=True)
class SceneUsage:
    environment: str
    section: str
    platform: str
    platform_path: str
    scene_id: str
    gripper_model: str
    gripper_type_raw: str
    gripper_family: str
    candidate_record_count: int
    conf_exists: bool
    pregrasp_path: str


@dataclass(frozen=True)
class ScanError:
    pregrasp_path: str
    error_type: str
    message: str


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_gripper_type_lookup(paths: list[Path]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            print(f"[WARN] gripper info not found: {path}", flush=True)
            continue
        database = load_json(path)
        if not isinstance(database, dict):
            print(f"[WARN] gripper info root is not an object: {path}", flush=True)
            continue
        for key, info in database.items():
            if not isinstance(info, dict):
                continue
            model = str(info.get("gripper_name") or key).strip()
            raw_type = str(info.get("type", "")).strip()
            if model and raw_type:
                lookup[model] = raw_type
            if str(key).strip() and raw_type:
                lookup[str(key).strip()] = raw_type
    return lookup


def normalize_gripper_family(raw_type: str) -> str:
    value = str(raw_type).strip().lower().replace("-", "").replace("_", "")
    if value == "hand" or "hand" in value:
        return "hand"
    if value.startswith("finger2") or value in {"2finger", "2f"}:
        return "finger2"
    if value.startswith("finger3") or value in {"3finger", "3f"}:
        return "finger3"
    return "unknown"


def split_dataset_location(dataset_root: Path, pregrasp_path: Path) -> tuple[str, str, str]:
    relative = pregrasp_path.relative_to(dataset_root)
    parts = relative.parts
    try:
        pregrasp_index = parts.index("pre_grasp")
    except ValueError as error:
        raise ValueError("path does not contain a pre_grasp directory") from error
    if pregrasp_index < 3:
        raise ValueError(
            "expected <environment>/<section>/<platform>/pre_grasp layout"
        )
    return (
        parts[pregrasp_index - 3],
        parts[pregrasp_index - 2],
        parts[pregrasp_index - 1],
    )


def one_gripper_group(payload, path: Path) -> dict:
    if isinstance(payload, dict):
        groups = [payload]
    elif isinstance(payload, list):
        groups = [item for item in payload if isinstance(item, dict)]
    else:
        raise TypeError(f"JSON root must be an object or list, got {type(payload).__name__}")

    if len(groups) != 1:
        raise ValueError(
            f"scene must contain exactly one gripper group, found {len(groups)}"
        )
    group = groups[0]
    records = group.get("data", [])
    if not isinstance(records, list):
        raise TypeError("gripper group data must be a list")

    declared_models = {
        str(value).strip()
        for value in [group.get("gripper_model")]
        + [record.get("gripper_model") for record in records if isinstance(record, dict)]
        if value is not None and str(value).strip()
    }
    if len(declared_models) != 1:
        raise ValueError(
            f"scene must resolve to one gripper model, found {sorted(declared_models)}"
        )
    group = dict(group)
    group["gripper_model"] = declared_models.pop()
    group["data"] = records
    return group


def infer_group_type(group: dict, type_lookup: dict[str, str]) -> str:
    record_types = {
        str(record.get("gripper_type")).strip()
        for record in group["data"]
        if isinstance(record, dict)
        and record.get("gripper_type") is not None
        and str(record.get("gripper_type")).strip()
    }
    if len(record_types) > 1:
        raise ValueError(f"one scene contains mixed gripper types: {sorted(record_types)}")
    if record_types:
        return record_types.pop()
    return type_lookup.get(group["gripper_model"], "unknown")


def discover_pregrasp_files(dataset_root: Path) -> list[Path]:
    # This bounded glob avoids traversing large RGB/depth/pointcloud trees.
    return sorted(
        path
        for path in dataset_root.glob("*/*/*/pre_grasp/*.json")
        if path.name != "summary.json"
    )


def discover_conf_keys(dataset_root: Path) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for path in dataset_root.glob("*/*/*/conf/*.json"):
        relative = path.relative_to(dataset_root)
        if len(relative.parts) < 5:
            continue
        keys.add((relative.parts[0], relative.parts[1], relative.parts[2], path.stem))
    return keys


def scan_dataset(
    dataset_root: Path, type_lookup: dict[str, str]
) -> tuple[list[SceneUsage], list[ScanError], set[tuple[str, str, str, str]]]:
    conf_keys = discover_conf_keys(dataset_root)
    paths = discover_pregrasp_files(dataset_root)
    usages: list[SceneUsage] = []
    errors: list[ScanError] = []

    print(f"[INFO] discovered pre_grasp files: {len(paths)}", flush=True)
    for path in tqdm(paths, desc="Reading pre_grasp", unit="scene"):
        try:
            environment, section, platform = split_dataset_location(dataset_root, path)
            group = one_gripper_group(load_json(path), path)
            raw_type = infer_group_type(group, type_lookup)
            scene_key = (environment, section, platform, path.stem)
            usages.append(
                SceneUsage(
                    environment=environment,
                    section=section,
                    platform=platform,
                    platform_path=f"{environment}/{section}/{platform}",
                    scene_id=path.stem,
                    gripper_model=group["gripper_model"],
                    gripper_type_raw=raw_type,
                    gripper_family=normalize_gripper_family(raw_type),
                    candidate_record_count=len(group["data"]),
                    conf_exists=scene_key in conf_keys,
                    pregrasp_path=str(path),
                )
            )
        except Exception as error:
            errors.append(
                ScanError(
                    pregrasp_path=str(path),
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
    return usages, errors, conf_keys


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ratio_rows(
    usages: list[SceneUsage], location_fields: tuple[str, ...], category_field: str
) -> list[dict]:
    location_totals: Counter[tuple[str, ...]] = Counter()
    category_totals: Counter[str] = Counter()
    counts: Counter[tuple[tuple[str, ...], str]] = Counter()
    for usage in usages:
        location = tuple(getattr(usage, field) for field in location_fields)
        category = str(getattr(usage, category_field))
        location_totals[location] += 1
        category_totals[category] += 1
        counts[(location, category)] += 1

    rows: list[dict] = []
    for (location, category), count in sorted(counts.items()):
        row = {field: value for field, value in zip(location_fields, location)}
        row.update(
            {
                category_field: category,
                "scene_count": count,
                "ratio_within_location_percent": round(
                    count / location_totals[location] * 100.0, 6
                ),
                "ratio_of_category_usage_percent": round(
                    count / category_totals[category] * 100.0, 6
                ),
            }
        )
        rows.append(row)
    return rows


def overall_rows(usages: list[SceneUsage], category_field: str) -> list[dict]:
    counts = Counter(str(getattr(usage, category_field)) for usage in usages)
    total = len(usages)
    return [
        {
            category_field: category,
            "scene_count": count,
            "ratio_percent": round(count / total * 100.0, 6) if total else 0.0,
        }
        for category, count in counts.most_common()
    ]


def coverage_rows(
    usages: list[SceneUsage], conf_keys: set[tuple[str, str, str, str]]
) -> tuple[list[dict], list[dict]]:
    pregrasp_keys = {
        (usage.environment, usage.section, usage.platform, usage.scene_id)
        for usage in usages
    }
    conf_by_platform: Counter[tuple[str, str, str]] = Counter(
        key[:3] for key in conf_keys
    )
    pregrasp_by_platform: Counter[tuple[str, str, str]] = Counter(
        key[:3] for key in pregrasp_keys if key in conf_keys
    )
    coverage = []
    for location, conf_count in sorted(conf_by_platform.items()):
        generated = pregrasp_by_platform[location]
        coverage.append(
            {
                "environment": location[0],
                "section": location[1],
                "platform": location[2],
                "platform_path": "/".join(location),
                "conf_scene_count": conf_count,
                "pregrasp_scene_count": generated,
                "missing_scene_count": conf_count - generated,
                "coverage_percent": round(generated / conf_count * 100.0, 6),
            }
        )
    missing = [
        {
            "environment": key[0],
            "section": key[1],
            "platform": key[2],
            "scene_id": key[3],
        }
        for key in sorted(conf_keys - pregrasp_keys)
    ]
    return coverage, missing


def model_color_map(models: list[str]) -> dict[str, tuple[float, float, float, float]]:
    color_map = plt.get_cmap("tab20" if len(models) <= 20 else "hsv")
    denominator = max(1, len(models) - 1)
    return {model: color_map(index / denominator) for index, model in enumerate(models)}


def save_overall_bar(
    rows: list[dict], category_field: str, title: str, output_path: Path
) -> None:
    if not rows:
        return
    ordered = list(reversed(rows))
    labels = [row[category_field] for row in ordered]
    counts = [row["scene_count"] for row in ordered]
    ratios = [row["ratio_percent"] for row in ordered]
    figure, axis = plt.subplots(figsize=(12, max(4.5, len(rows) * 0.48)))
    bars = axis.barh(labels, counts, color=plt.get_cmap("tab20").colors[: len(rows)])
    for bar, count, ratio in zip(bars, counts, ratios):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2.0,
            f"  {count} ({ratio:.1f}%)",
            va="center",
            fontsize=9,
        )
    axis.set_title(title)
    axis.set_xlabel("Scene count")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(figure)


def percentage_matrix(
    usages: list[SceneUsage], location_field: str
) -> tuple[list[str], list[str], np.ndarray]:
    locations = sorted({str(getattr(row, location_field)) for row in usages})
    models = sorted({row.gripper_model for row in usages})
    location_index = {value: index for index, value in enumerate(locations)}
    model_index = {value: index for index, value in enumerate(models)}
    matrix = np.zeros((len(locations), len(models)), dtype=np.float64)
    for row in usages:
        matrix[location_index[str(getattr(row, location_field))], model_index[row.gripper_model]] += 1
    totals = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0) * 100.0
    return locations, models, matrix


def save_stacked_ratio(
    usages: list[SceneUsage], location_field: str, title: str, output_path: Path
) -> None:
    if not usages:
        return
    locations, models, matrix = percentage_matrix(usages, location_field)
    colors = model_color_map(models)
    figure, axis = plt.subplots(figsize=(14, max(5.0, len(locations) * 0.5)))
    y = np.arange(len(locations))
    left = np.zeros(len(locations), dtype=np.float64)
    for model_index, model in enumerate(models):
        values = matrix[:, model_index]
        axis.barh(y, values, left=left, label=model, color=colors[model])
        left += values
    axis.set_yticks(y, labels=locations)
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Gripper usage within location [%]")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(figure)


def save_heatmap(
    usages: list[SceneUsage], location_field: str, title: str, output_path: Path
) -> None:
    if not usages:
        return
    locations, models, matrix = percentage_matrix(usages, location_field)
    figure, axis = plt.subplots(
        figsize=(max(9.0, len(models) * 0.75), max(5.0, len(locations) * 0.48))
    )
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")
    axis.set_xticks(np.arange(len(models)), labels=models, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(locations)), labels=locations)
    axis.set_title(title)
    axis.set_xlabel("Gripper model")
    axis.set_ylabel("Location")
    if matrix.size <= ANNOTATE_HEATMAP_MAX_CELLS:
        for row_index, column_index in np.argwhere(matrix > 0.0):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 55.0 else "black",
            )
    figure.colorbar(image, ax=axis, label="Usage within location [%]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(figure)


def build_statistics(
    usages: list[SceneUsage],
    errors: list[ScanError],
    conf_keys: set[tuple[str, str, str, str]],
) -> dict:
    overall_model = overall_rows(usages, "gripper_model")
    overall_family = overall_rows(usages, "gripper_family")
    environment_model = ratio_rows(usages, ("environment",), "gripper_model")
    environment_family = ratio_rows(usages, ("environment",), "gripper_family")
    platform_model = ratio_rows(
        usages,
        ("environment", "section", "platform", "platform_path"),
        "gripper_model",
    )
    platform_family = ratio_rows(
        usages,
        ("environment", "section", "platform", "platform_path"),
        "gripper_family",
    )
    coverage, missing = coverage_rows(usages, conf_keys)
    covered_conf_count = sum(usage.conf_exists for usage in usages)
    return {
        "overview": {
            "dataset_conf_scene_count": len(conf_keys),
            "valid_pregrasp_scene_count": len(usages),
            "pregrasp_with_matching_conf_count": covered_conf_count,
            "missing_pregrasp_scene_count": len(missing),
            "invalid_pregrasp_file_count": len(errors),
            "coverage_percent": round(
                covered_conf_count / len(conf_keys) * 100.0, 6
            ) if conf_keys else 0.0,
        },
        "overall_by_gripper_model": overall_model,
        "overall_by_gripper_family": overall_family,
        "by_environment_gripper_model": environment_model,
        "by_environment_gripper_family": environment_family,
        "by_platform_gripper_model": platform_model,
        "by_platform_gripper_family": platform_family,
        "coverage_by_platform": coverage,
        "missing_pregrasp_scenes": missing,
    }


def save_outputs(
    output_dir: Path,
    usages: list[SceneUsage],
    errors: list[ScanError],
    statistics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scene_gripper_usage.csv", [asdict(row) for row in usages])
    write_csv(output_dir / "scan_errors.csv", [asdict(row) for row in errors])

    table_names = (
        "overall_by_gripper_model",
        "overall_by_gripper_family",
        "by_environment_gripper_model",
        "by_environment_gripper_family",
        "by_platform_gripper_model",
        "by_platform_gripper_family",
        "coverage_by_platform",
        "missing_pregrasp_scenes",
    )
    for table_name in table_names:
        write_csv(output_dir / f"{table_name}.csv", statistics[table_name])

    with (output_dir / "pregrasp_statistics.json").open("w", encoding="utf-8") as stream:
        json.dump(statistics, stream, ensure_ascii=False, indent=2)

    save_overall_bar(
        statistics["overall_by_gripper_model"],
        "gripper_model",
        "Overall pre-grasp usage by gripper model",
        output_dir / "overall_gripper_model_usage.png",
    )
    save_overall_bar(
        statistics["overall_by_gripper_family"],
        "gripper_family",
        "Overall pre-grasp usage by gripper family",
        output_dir / "overall_gripper_family_usage.png",
    )
    save_stacked_ratio(
        usages,
        "environment",
        "Gripper model ratio by environment",
        output_dir / "environment_gripper_ratio.png",
    )
    save_stacked_ratio(
        usages,
        "platform_path",
        "Gripper model ratio by platform",
        output_dir / "platform_gripper_ratio.png",
    )
    save_heatmap(
        usages,
        "environment",
        "Gripper usage heatmap by environment",
        output_dir / "environment_gripper_heatmap.png",
    )
    save_heatmap(
        usages,
        "platform_path",
        "Gripper usage heatmap by platform",
        output_dir / "platform_gripper_heatmap.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset pre-grasp usage")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PLOTS,
        help="Display plots interactively (use --no-show on a headless server)",
    )
    return parser.parse_args()


def main() -> None:
    global SHOW_PLOTS
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "pregrasp_statistics"
    )
    SHOW_PLOTS = bool(args.show)

    print(f"[INFO] dataset root: {dataset_root}", flush=True)
    print(f"[INFO] output dir: {output_dir}", flush=True)
    type_lookup = load_gripper_type_lookup(
        [HAND_GRIPPER_INFO_PATH, FINGER_GRIPPER_INFO_PATH]
    )
    usages, errors, conf_keys = scan_dataset(dataset_root, type_lookup)
    statistics = build_statistics(usages, errors, conf_keys)
    save_outputs(output_dir, usages, errors, statistics)

    overview = statistics["overview"]
    print("\n=== PreGrasp statistics ===")
    print(f"Dataset conf scenes       : {overview['dataset_conf_scene_count']}")
    print(f"Valid pre_grasp scenes    : {overview['valid_pregrasp_scene_count']}")
    print(f"Missing pre_grasp scenes  : {overview['missing_pregrasp_scene_count']}")
    print(f"Invalid pre_grasp files   : {overview['invalid_pregrasp_file_count']}")
    print(f"Coverage                  : {overview['coverage_percent']:.2f}%")
    print("\nGripper model usage:")
    for row in statistics["overall_by_gripper_model"]:
        print(
            f"  {row['gripper_model']}: {row['scene_count']} scenes "
            f"({row['ratio_percent']:.2f}%)"
        )
    print(f"\n[INFO] reports and plots saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
