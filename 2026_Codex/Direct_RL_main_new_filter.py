"""Grasp collection entry point with optional hand robot-platform collision filtering."""

from __future__ import annotations

import json
import os
import sys
import csv
import copy
import ast
from pathlib import Path


FINGER_GRIPPER_INFO_PATH = Path(
    "/nas/ochansol/gripper_info/gripper_info_new_2026.json"
)
HAND_GRIPPER_INFO_PATH = Path("/nas/ochansol/gripper_info/gripper_info_hand_2026.json")
OBJECT_NAME_CSV_PATH = Path(
    "/nas/ochansol/3d_model/peel3_scan_data_2026/2026_objects_cat_attr.csv"
)
CODEX_DIR = Path(__file__).resolve().parent
LEGACY_CHANSOL_DIR = CODEX_DIR.parent / "chansol"
HEADLESS = True
DEBUG_DRAW_PREGRASP = not HEADLESS
pre_grasp_start_index = 0
HAND_CONTACT_MAX_DATA_COUNT_PER_PRIM = 32
MINIMUM_GRASP_SUCCESSES = 5
MAX_ADDITIONAL_COLLECTION_ATTEMPTS = 3

if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))
if str(LEGACY_CHANSOL_DIR) not in sys.path:
    sys.path.append(str(LEGACY_CHANSOL_DIR))


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _rigid_usd_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_rigid{path.suffix}")


def _load_class_name_aliases(csv_path: Path) -> dict[str, list[str]]:
    """Map dataset Class_name values such as obj_133 to real object folder names.

    The 2024/2025 CSV is expected to have Object_name already normalized to
    the folder name that exists on NAS.
    """
    if not csv_path.exists():
        print(f"Grasp > object-name CSV not found, skip remap: {csv_path}")
        return {}

    aliases: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            class_name = str(row.get("Class_name", "")).strip()
            if not class_name:
                continue
            object_name = str(row.get("Object_name", "")).strip()
            if object_name:
                aliases[class_name] = [object_name]
    return aliases


def _candidate_remap_paths(original_path: Path, aliases: list[str]) -> list[Path]:
    candidates: list[Path] = []
    path_text = str(original_path)
    original_stem = original_path.stem
    original_parent_name = original_path.parent.parent.name if original_path.parent.name == "edited" else original_path.parent.name

    for alias in aliases:
        replacements = []
        for source_name in (original_stem, original_parent_name):
            if source_name and source_name not in replacements:
                replacements.append(source_name)
        for source_name in replacements:
            if source_name not in path_text:
                continue
            candidate = Path(path_text.replace(source_name, alias))
            if candidate not in candidates:
                candidates.append(candidate)

        # Common NAS layout:
        #   /nas/.../peel3_scan_data_2025/<object>/edited/<object>.usd
        if original_path.parent.name == "edited":
            candidate = original_path.parent.parent.parent / alias / "edited" / f"{alias}.usd"
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _resolve_object_usd_path(obj: dict, aliases_by_class: dict[str, list[str]]) -> str:
    original = Path(obj["usd_path"])
    if _rigid_usd_path(original).exists():
        return str(original)

    class_name = str(obj.get("class", "")).strip()
    aliases = aliases_by_class.get(class_name, [])
    for candidate in _candidate_remap_paths(original, aliases):
        if _rigid_usd_path(candidate).exists():
            print(
                "Grasp > remap object usd "
                f"class={class_name} from={original} to={candidate}"
            )
            return str(candidate)

    return str(original)


def _resolve_conf_object_usd_paths(conf_data: dict, csv_path: Path) -> dict:
    resolved = copy.deepcopy(conf_data)
    aliases_by_class = _load_class_name_aliases(csv_path)
    for obj in resolved.get("objects", []):
        if "usd_path" not in obj:
            continue
        obj["usd_path"] = _resolve_object_usd_path(obj, aliases_by_class)
    return resolved


def _object_asset_prim_name(usd_path: str) -> str:
    """Infer the child prim name created under /objXX from an object USD path."""
    stem = Path(usd_path).stem
    return stem.removesuffix("_rigid")


def _object_frame_transformer_setup_with_asset_names(scene, obj_conf: list[dict]) -> None:
    """Track objects by their real USD child prim while keeping Class_name labels.

    Some older dataset conf files use Class_name values such as ``obj_133`` but
    the referenced USD contains a child prim named after the actual object
    folder, e.g. ``plastic_wrapped_bar_soap``.  The action policies should still
    see the frame name ``obj_133`` because pre-grasp records target that class,
    while the frame transformer must point at the real prim path.
    """
    from isaaclab.sensors import FrameTransformerCfg

    target_frames = []
    for index in range(10):
        item = getattr(scene, f"obj{index:02d}", None)
        if item is None:
            continue
        asset_name = _object_asset_prim_name(item.spawn.usd_path)
        target_frames.append(
            FrameTransformerCfg.FrameCfg(
                name=item.class_name,
                prim_path=f"/World/envs/env_.*/obj{index:02d}/{asset_name}",
            )
        )
    scene.transformer.target_frames = target_frames


def _configure_hand_contact_separation_filters(
    scene, obj_conf: list[dict]
) -> None:
    """Expose detailed hand-object contacts, including PhysX separation.

    The hand sensor bodies are still selected exclusively from the configured
    hand contact-sensor prim paths. These filters select the scene rigid bodies
    against which detailed contact records are requested.
    """
    filter_paths = []
    for index, obj in enumerate(obj_conf[:10]):
        asset_name = _object_asset_prim_name(obj["usd_path"])
        filter_paths.append(
            f"{{ENV_REGEX_NS}}/obj{index:02d}/{asset_name}"
        )
    if not filter_paths:
        raise ValueError("Hand contact-separation checking requires scene objects")
    scene.contact_sensor.filter_prim_paths_expr = []
    configured_sensor_count = 0
    for index in range(64):
        sensor = getattr(scene, f"penetration_contact_sensor_{index:02d}", None)
        if sensor is None:
            break
        sensor.filter_prim_paths_expr = list(filter_paths)
        sensor.max_contact_data_count_per_prim = (
            HAND_CONTACT_MAX_DATA_COUNT_PER_PRIM
        )
        configured_sensor_count += 1
    if configured_sensor_count == 0:
        raise RuntimeError("No per-fingertip penetration contact sensors were configured")


def _select_pre_grasp_group(payload) -> dict:
    if isinstance(payload, dict):
        group = payload
    elif isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                "Pre-grasp JSON must contain exactly one gripper group. "
                "Regenerate pre_grasp with the 2026 random single-gripper collector."
            )
        group = payload[0]
    else:
        raise TypeError("Pre-grasp JSON root must be an object or a list")

    if not isinstance(group, dict) or not isinstance(group.get("data"), list):
        raise ValueError("Selected pre-grasp group must contain a data list")
    if not group["data"]:
        raise ValueError("Selected pre-grasp group contains no grasp candidates")
    return group


def _detect_gripper_type(pre_grasp_group: dict) -> str:
    """Detect hand/finger from the selected pre-grasp, never from preset names."""
    records = pre_grasp_group["data"][pre_grasp_start_index:]
    declared_types = {
        str(record["gripper_type"]).strip().lower()
        for record in records
        if record.get("gripper_type") is not None
    }
    if len(declared_types) > 1:
        raise ValueError(
            f"One pre-grasp group contains mixed gripper types: {sorted(declared_types)}"
        )
    if declared_types:
        return declared_types.pop()

    # Compatibility fallback for older finger pre-grasp files which did not
    # copy gripper_type into every record.
    gripper_name = pre_grasp_group["gripper_model"]
    finger_database = _load_json(FINGER_GRIPPER_INFO_PATH)
    hand_database = _load_json(HAND_GRIPPER_INFO_PATH)
    if gripper_name in finger_database:
        return str(finger_database[gripper_name]["type"]).lower()
    if gripper_name in hand_database:
        return str(hand_database[gripper_name]["type"]).lower()
    raise KeyError(f"Unknown gripper model in pre-grasp: {gripper_name!r}")


def _load_gripper_info(pre_grasp_group: dict, gripper_type: str) -> dict:
    gripper_name = pre_grasp_group["gripper_model"]
    database_path = (
        HAND_GRIPPER_INFO_PATH if gripper_type == "hand" else FINGER_GRIPPER_INFO_PATH
    )
    database = _load_json(database_path)
    if gripper_name not in database:
        raise KeyError(f"{gripper_name!r} was not found in {database_path}")
    gripper_info = database[gripper_name]
    configured_type = str(gripper_info.get("type", "")).lower()
    if configured_type != gripper_type:
        raise ValueError(
            f"Pre-grasp type {gripper_type!r} does not match gripper config "
            f"type {configured_type!r}"
        )
    return gripper_info


def _inject_missing_hand_preset_metadata(
    pre_grasp_group: dict, gripper_info: dict
) -> int:
    """Refresh transition/contact metadata and strip obsolete VIA fields."""
    from pregrasp.hand_pregrasp_heightmap import preset_contact_sensor_sets

    presets = {
        str(preset.get("name")): preset
        for preset in gripper_info.get("preset", [])
        if isinstance(preset, dict) and preset.get("name")
    }
    updated = 0
    for record in pre_grasp_group.get("data", []):
        preset = presets.get(str(record.get("preset_name")))
        if not isinstance(preset, dict):
            continue
        record_updated = False
        target_base_tf = record.get("target_base_tf")
        if isinstance(target_base_tf, dict) and "via" in target_base_tf:
            target_base_tf.pop("via", None)
            record_updated = True
        preset_transition = preset.get("transition")
        if isinstance(preset_transition, dict):
            record_transition = record.setdefault("transition", {})
            if isinstance(record_transition, dict):
                for key in (
                    "duration_sec",
                    "interpolation",
                ):
                    if key in preset_transition and key not in record_transition:
                        record_transition[key] = copy.deepcopy(preset_transition[key])
                        record_updated = True
                for obsolete_key in ("via_time_sec", "via_time_ratio"):
                    if obsolete_key in record_transition:
                        record_transition.pop(obsolete_key, None)
                        record_updated = True
        current_contact_sets = preset_contact_sensor_sets(preset)
        if record.get("contact_sensor_sets") != current_contact_sets:
            record["contact_sensor_sets"] = current_contact_sets
            record_updated = True
        updated += int(record_updated)
    return updated


def _save_output(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=4)
    os.replace(temporary_path, path)


def _load_debug_instance_colors(root_path: Path, scene_num: int) -> dict[str, list[float]]:
    """Load the top-view instance colors as normalized RGBA debug colors."""
    mapping_path = (
        root_path
        / "inst_seg"
        / "top_view_camera"
        / f"semantics_mapping_{scene_num:04d}.json"
    )
    if not mapping_path.exists():
        print(f"Grasp > debug instance-color mapping not found: {mapping_path}")
        return {}

    mapping = _load_json(mapping_path)
    result: dict[str, list[float]] = {}
    for raw_color, semantic in mapping.items():
        if not isinstance(semantic, dict):
            continue
        class_name = str(semantic.get("class", "")).strip()
        if not class_name or class_name == "UNLABELLED":
            continue
        try:
            rgba = tuple(int(value) for value in ast.literal_eval(raw_color))
        except (ValueError, SyntaxError, TypeError):
            continue
        if len(rgba) not in (3, 4):
            continue
        if len(rgba) == 3:
            rgba = (*rgba, 255)
        result[class_name] = [
            max(0.0, min(1.0, float(value) / 255.0)) for value in rgba
        ]
    return result


def main(root_path, scene_num):
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Grasp collection in Isaac Lab")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = HEADLESS
    debug_draw_pregrasp = DEBUG_DRAW_PREGRASP and not args_cli.headless

    root_path = Path(root_path)
    pre_grasp_file = root_path / "pre_grasp" / f"{scene_num:04d}.json"
    conf_file = root_path / "conf" / f"{scene_num:04d}.json"
    output_file = root_path / "output_grasp" / f"{scene_num:04d}.json"

    pre_grasp_group = _select_pre_grasp_group(_load_json(pre_grasp_file))
    conf_data = _resolve_conf_object_usd_paths(_load_json(conf_file), OBJECT_NAME_CSV_PATH)
    if debug_draw_pregrasp:
        conf_data["debug_instance_colors"] = _load_debug_instance_colors(
            root_path, scene_num
        )
    gripper_type = _detect_gripper_type(pre_grasp_group)
    gripper_info = _load_gripper_info(pre_grasp_group, gripper_type)
    is_hand = gripper_type == "hand"
    if is_hand:
        injected_metadata_count = _inject_missing_hand_preset_metadata(
            pre_grasp_group, gripper_info
        )
        if injected_metadata_count:
            print(
                "Grasp > injected current preset metadata into "
                f"{injected_metadata_count} legacy pre-grasp record(s)"
            )
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    env = None
    try:
        print("Grasp > App_start")
        print(
            f"Grasp > gripper={pre_grasp_group['gripper_model']} "
            f"type={gripper_type} policy={'hand preset' if is_hand else 'calibrated finger'}"
        )
        if DEBUG_DRAW_PREGRASP and args_cli.headless:
            print("Grasp > debug draw disabled in headless mode")
        sys.stdout.flush()

        import time
        import torch

        print("Grasp > loading environment modules")
        sys.stdout.flush()
        if is_hand:
            import codex_cfgs.Robot_EnvCfg_hand_platform_filter as cs_robot
        else:
            import codex_cfgs.Robot_EnvCfg_finger_platform_filter as cs_robot
        from cfgs.scene_cfg import (
            platform_setup,
            scene_obj_setup,
        )
        print("Grasp > environment modules loaded")
        sys.stdout.flush()

        output_list = _load_json(output_file) if output_file.exists() else []
        if not isinstance(output_list, list):
            raise TypeError(f"Existing output is not a list: {output_file}")

        env_cfg = cs_robot.RobotEnvCfg()
        if is_hand:
            cs_robot.Set_RobotEnvCFG(
                env_cfg, gripper_info, pre_grasp_group["data"][pre_grasp_start_index:]
            )
        else:
            cs_robot.Set_RobotEnvCFG(env_cfg, gripper_info)

        scene_obj_setup(env_cfg.scene, conf_data["objects"])
        platform_setup(env_cfg.scene, conf_data["platform"])
        _object_frame_transformer_setup_with_asset_names(
            env_cfg.scene, conf_data["objects"]
        )
        if is_hand:
            _configure_hand_contact_separation_filters(
                env_cfg.scene, conf_data["objects"]
            )

        print("Grasp > creating environment")
        sys.stdout.flush()
        env = cs_robot.RobotEnv(
            cfg=env_cfg,
            pre_grasp_data=pre_grasp_group["data"][pre_grasp_start_index:],
            conf_data=conf_data,
            debug=debug_draw_pregrasp,
        )

        collection_attempt = 1
        output_list_tmp = []
        obs, _ = env.reset()
        print("Grasp > START")
        print(f"Grasp > SCENE:{scene_num}")
        sys.stdout.flush()
        old_time = time.time()
        minimum_successes = min(
            MINIMUM_GRASP_SUCCESSES,
            len(pre_grasp_group["data"][pre_grasp_start_index:]),
        )
        max_collection_attempts = 1 + MAX_ADDITIONAL_COLLECTION_ATTEMPTS

        while simulation_app.is_running():
            with torch.inference_mode():
                operation_tensor = torch.ones((env.cfg.envs), device=env.device)
                obs, rew, term, trunc, info = env.step(operation_tensor)
                if obs["policy"].sum() != 0:
                    continue

                output_list_tmp += env.act_pol.output_list
                print("output_list_tmp : ", len(output_list_tmp))
                if is_hand:
                    print(
                        "Grasp > penetration_summary "
                        f"failures={env.act_pol.penetration_failure_count} "
                        "deepest_separation="
                        f"{env.act_pol.deepest_contact_separation:.6f}"
                    )
                if len(output_list_tmp) < minimum_successes:
                    if collection_attempt < max_collection_attempts:
                        print(
                            "Grasp > insufficient successes "
                            f"({len(output_list_tmp)}/{minimum_successes}); "
                            f"retry {collection_attempt}/"
                            f"{MAX_ADDITIONAL_COLLECTION_ATTEMPTS}"
                        )
                        env.factory_reset()
                        print("Grasp > factory_reset")
                        collection_attempt += 1
                        continue
                    print(
                        "Grasp > retry limit reached; pass scene with "
                        f"{len(output_list_tmp)}/{minimum_successes} successes"
                    )

                output_list += output_list_tmp
                sorted_output_list = sorted(
                    output_list, key=lambda item: item.get("target_object", "")
                )
                _save_output(output_file, sorted_output_list)
                print(f"Grasp > saved:{output_file}")
                break

        print("Grasp > data_gen_time : ", time.time() - old_time)
        print("Grasp > END")
        sys.stdout.flush()
    except BaseException:
        # SimulationApp.close() may terminate Kit before Python gets a chance
        # to render an unhandled exception.  Print it while Kit is still alive
        # so startup failures are never mistaken for a normal early exit.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        if env is not None:
            try:
                if hasattr(env, "act_pol") and hasattr(env.act_pol, "_clear_debug_draw"):
                    env.act_pol._clear_debug_draw()
                env.close()
            except SystemExit:
                if not active_exception:
                    raise
        try:
            simulation_app.close()
        except SystemExit:
            if not active_exception:
                raise


if __name__ == "__main__":
    main(root_path="", scene_num=0)
