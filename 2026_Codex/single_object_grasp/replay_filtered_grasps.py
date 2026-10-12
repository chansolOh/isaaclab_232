#!/usr/bin/env python3
"""Inspect the exact filter-GUI result one grasp at a time in Isaac Lab.

Workflow
--------
1. In filter_grasps_gui.py, apply filters and press "Isaac 검수 목록 저장".
2. Run this file.  By default it opens the most recently saved manifest.
3. Replay each source pregrasp, then mark it accepted/rejected with A/X.
4. Back in the GUI, press "Isaac 검수 결과 불러오기" before final merge.

Each entry keeps its original scene pose and source pregrasp.  The object reset
pose is switched before each replay, so grasps from every scene share one
simulator without changing the originally collected physics geometry.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# 실행 설정: argparse 없이 이 부분만 수정한다.
# DATASET_PATH = <root>/<object>/<gripper>. None이면 DATASET_ROOT 아래에서
# GUI가 가장 최근에 저장한 manifest를 자동으로 선택한다.
# -----------------------------------------------------------------------------
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen")
DATASET_PATH: Path | None = None
GRASP_DIR_NAME = "output_grasp"
# GUI의 "필터 결과 Replay 시작"은 환경변수로 정확한 manifest를
# 넘긴다. 단독 실행시 특정 파일을 고정하려면 아래를 Path로 바꾼다.
_MANIFEST_FROM_GUI = os.environ.get("ISAAC_REPLAY_MANIFEST", "").strip()
MANIFEST_PATH: Path | None = (
    Path(_MANIFEST_FROM_GUI) if _MANIFEST_FROM_GUI else None
)

FINGER_INFO = Path("/nas/ochansol/gripper_info/gripper_info_new_2026.json")
HAND_INFO = Path("/nas/ochansol/gripper_info/gripper_info_hand_2026.json")

START_INDEX = 0
END_INDEX: int | None = None
STRIDE = 1
AUTO_ADVANCE_AFTER_DECISION = True
SAVE_AFTER_EACH_DECISION = True
RESUME_AT_FIRST_UNREVIEWED = True

DEVICE = "cpu"  # "cpu", "cuda", "cuda:0" 등
SEED = 42
SIM_DT = 1.0 / 800.0
DECIMATION = 4
ENABLE_CCD = True
PHYSX_SOLVE_ARTICULATION_CONTACT_LAST = False
GPU_PHYSX_BUFFER_SCALE = 4
GPU_MAX_NUM_PARTITIONS = 16

CONTACT_OFFSET_MM = 1.0
REST_OFFSET_MM = 0.1
CONTACT_PENETRATION_THRESHOLD_MM = 1.0
OVERRIDE_GRIPPER_COLLISION_OFFSETS = True
ENABLE_OBJECT_CONTACT_SENSOR = True
CONTACT_MAX_DATA_COUNT_PER_PRIM = 4096
PRE_STRESS_OBJECT_MOTION_THRESHOLD = 0.1
CONTACT_LOST_DURATION_S = 0.10
ROOT_MAX_LINEAR_SPEED = 0.25
ROOT_MAX_ANGULAR_SPEED_DEG = 90.0
GRIPPER_INITIAL_PARK_Z = 2.0
PRINT_CONTACT_SEPARATION = False

# None이면 저장된 quality.finger_z_hop_enabled를 사용한다.
APPLY_FINGER_Z_HOP = None

CAMERA_EYE = (0.45, -0.45, 0.30)
CAMERA_LOOKAT = (0.0, 0.0, 0.02)
DRAW_GRASP_DATA = True
GRASP_LINE_WIDTH = 5.0
GRASP_POINT_SIZE = 13.0
NORMAL_LINE_SCALE = 0.08
FRAME_AXIS_SCALE = 0.035
SHOW_GRASP_FRAME = True
DEBUG = True


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def manifest_suffix() -> str:
    label = GRASP_DIR_NAME.removeprefix("output_grasp").strip("_")
    return "" if not label else f"_{label}"


def resolve_manifest_path() -> Path:
    if MANIFEST_PATH is not None:
        selected = Path(MANIFEST_PATH).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"MANIFEST_PATH가 없습니다: {selected}")
        return selected

    pattern = f"*_filtered_replay_manifest{manifest_suffix()}.json"
    if DATASET_PATH is not None:
        dataset_path = Path(DATASET_PATH).expanduser().resolve()
        expected = dataset_path / (
            f"{dataset_path.parent.name}_filtered_replay_manifest"
            f"{manifest_suffix()}.json"
        )
        if expected.is_file():
            return expected.resolve()
        matches = list(dataset_path.glob(pattern))
        search_description = str(dataset_path)
    else:
        dataset_root = Path(DATASET_ROOT).expanduser().resolve()
        # Dataset layout is exactly <root>/<object>/<gripper>/<manifest>.
        # Avoid a recursive walk through conf/output files on NAS.
        matches = list(dataset_root.glob(f"*/*/{pattern}"))
        search_description = str(dataset_root)

    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Isaac replay manifest가 없습니다: {search_description}\n"
            "filter_grasps_gui.py에서 'Isaac 검수 목록 저장'을 먼저 누르세요."
        )
    # GUI save is atomic, so mtime reliably identifies the last pressed save.
    return max(matches, key=lambda path: path.stat().st_mtime_ns).resolve()


def normalized_device(value: str) -> str:
    device = str(value).strip().lower()
    valid_cuda = device == "cuda" or (
        device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    )
    if device != "cpu" and not valid_cuda:
        raise ValueError("DEVICE must be 'cpu', 'cuda', or 'cuda:<index>'")
    return device


def score_color(score: float) -> tuple[float, float, float, float]:
    score = min(1.0, max(0.0, float(score)))
    blue, green, yellow = (0.0, 0.25, 1.0), (0.0, 0.95, 0.2), (1.0, 0.9, 0.0)
    if score < 0.5:
        amount, first, second = score / 0.5, blue, green
    else:
        amount, first, second = (score - 0.5) / 0.5, green, yellow
    return tuple((1.0 - amount) * first[i] + amount * second[i] for i in range(3)) + (1.0,)


def draw_saved_grasp(draw, grasp: dict, decision: str, score_field: str) -> None:
    draw.clear_lines()
    draw.clear_points()
    if not DRAW_GRASP_DATA:
        return
    raw_boxes = grasp.get("grasp_boxes")
    if not raw_boxes:
        raw_box = grasp.get("grasp_box", [])
        raw_boxes = [raw_box]
    valid = (
        isinstance(raw_boxes, list)
        and raw_boxes
        and all(
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(point, list) and len(point) == 3 for point in box)
            for box in raw_boxes
        )
    )
    if not valid:
        print("Replay draw > invalid grasp_boxes/grasp_box; skipped", flush=True)
        return

    boxes = [
        [tuple(float(value) for value in point) for point in box]
        for box in raw_boxes
    ]
    red = (1.0, 0.12, 0.05, 1.0)
    selected_color = score_color(grasp.get(score_field, grasp.get("score", 1.0)))
    starts, ends, colors, widths = [], [], [], []
    for box in boxes:
        starts.extend([box[0], box[1], box[2], box[3]])
        ends.extend([box[1], box[2], box[3], box[0]])
        colors.extend([red, selected_color, red, red])
        widths.extend([float(GRASP_LINE_WIDTH)] * 4)

    points = [point for box in boxes for point in box]
    center = tuple(
        sum(point[axis] for point in points) / len(points) for axis in range(3)
    )
    normal = grasp.get("normal")
    if isinstance(normal, list) and len(normal) == 3:
        normal = tuple(float(value) for value in normal)
        length = math.sqrt(sum(value * value for value in normal))
        if length > 1.0e-9:
            unit = tuple(value / length for value in normal)
            starts.append(center)
            ends.append(
                tuple(
                    center[axis] + unit[axis] * float(NORMAL_LINE_SCALE)
                    for axis in range(3)
                )
            )
            colors.append((1.0, 0.0, 0.75, 1.0))
            widths.append(float(GRASP_LINE_WIDTH))

    matrix = grasp.get("grasp_mat")
    if (
        SHOW_GRASP_FRAME
        and isinstance(matrix, list)
        and len(matrix) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in matrix)
    ):
        origin = tuple(float(matrix[axis][3]) for axis in range(3))
        axis_colors = (
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.35, 1.0, 1.0),
        )
        for column in range(3):
            starts.append(origin)
            ends.append(
                tuple(
                    origin[row]
                    + float(matrix[row][column]) * float(FRAME_AXIS_SCALE)
                    for row in range(3)
                )
            )
            colors.append(axis_colors[column])
            widths.append(max(2.0, float(GRASP_LINE_WIDTH) - 1.0))

    draw.draw_lines(starts, ends, colors, widths)
    target = grasp.get("target_points", center)
    if not (isinstance(target, list) and len(target) == 3):
        target = center
    target = tuple(float(value) for value in target)
    marker_color = {
        "accept": (0.1, 1.0, 0.2, 1.0),
        "reject": (1.0, 0.05, 0.05, 1.0),
    }.get(decision, (1.0, 0.85, 0.0, 1.0))
    draw.draw_points([target], [marker_color], [float(GRASP_POINT_SIZE)])


manifest_path = resolve_manifest_path()
manifest = load_json(manifest_path)
metadata = manifest.get("metadata", {})
if metadata.get("schema") != "single_object_filtered_replay_manifest":
    raise ValueError(f"Unsupported manifest schema: {metadata.get('schema')!r}")
if metadata.get("coordinate_frame") != "source_scene_world_per_entry":
    raise ValueError("Manifest must contain per-entry source-scene coordinates")
manifest_dataset_path = Path(
    metadata.get("dataset_path", manifest_path.parent)
).expanduser().resolve()
all_entries = manifest.get("data", [])
if not isinstance(all_entries, list) or not all_entries:
    raise ValueError(f"No filtered grasps in {manifest_path}")
scenes = manifest.get("scenes", {})
if not isinstance(scenes, dict) or not scenes:
    raise ValueError("Manifest has no source scene configurations")

end = len(all_entries) if END_INDEX is None else min(int(END_INDEX), len(all_entries))
source_indices = list(range(int(START_INDEX), end, max(1, int(STRIDE))))
entries = [all_entries[index] for index in source_indices]
for index, entry in zip(source_indices, entries):
    if not isinstance(entry.get("grasp"), dict):
        raise ValueError(f"manifest data[{index}] has no grasp")
    if not isinstance(entry.get("pregrasp"), dict):
        raise ValueError(f"manifest data[{index}] has no pregrasp")
    if not entry.get("review_id"):
        raise ValueError(f"manifest data[{index}] has no review_id")
    scene_id = str(entry.get("source", {}).get("scene_id", ""))
    if not isinstance(scenes.get(scene_id), dict):
        raise ValueError(f"manifest data[{index}] references unknown scene {scene_id!r}")

first_scene_id = str(entries[0]["source"]["scene_id"])
first_conf = scenes[first_scene_id]
if len(first_conf.get("objects", [])) != 1:
    raise ValueError("The first manifest scene must contain exactly one object")
object_config = first_conf["objects"][0]

finger_database = load_json(FINGER_INFO)
hand_database = load_json(HAND_INFO)
gripper_name = str(metadata.get("gripper_model", metadata.get("gripper_directory", "")))
if gripper_name in finger_database:
    gripper = finger_database[gripper_name]
elif gripper_name in hand_database:
    gripper = hand_database[gripper_name]
else:
    raise ValueError(f"Unknown gripper: {gripper_name}")

review_path = manifest_path.with_name(f"{manifest_path.stem}_review.json")
decisions: dict[str, str] = {}
manifest_ids = {str(entry["review_id"]) for entry in all_entries}
if review_path.is_file():
    saved = load_json(review_path)
    saved_meta = saved.get("metadata", {})
    if saved_meta.get("schema") == "single_object_filtered_replay_review":
        decisions = {
            str(key): str(value)
            for key, value in saved.get("decisions", {}).items()
            if str(key) in manifest_ids and value in ("accept", "reject")
        }


def save_review(last_review_id: str | None) -> None:
    payload = {
        "metadata": {
            "schema": "single_object_filtered_replay_review",
            "schema_version": 1,
            "manifest_path": str(manifest_path),
            "dataset_path": str(manifest_dataset_path),
            "grasp_directory": str(metadata.get("grasp_directory", GRASP_DIR_NAME)),
            "object": metadata.get("object"),
            "gripper_model": gripper_name,
            "last_review_id": last_review_id,
            "manifest_count": len(all_entries),
            "reviewed_count": len(decisions),
        },
        "decisions": decisions,
    }
    atomic_json(review_path, payload)


device = normalized_device(DEVICE)
launcher = AppLauncher(headless=False, device=device, enable_cameras=False)
simulation_app = launcher.app
env = None
keyboard_subscription = None
input_interface = None
debug_draw = None
state = None

try:
    import carb
    import omni.appwindow
    import torch
    from isaacsim.util.debug_draw import _debug_draw

    from grasp import env as grasp_env
    from grasp.scene_cfg import configure_object, orientation_wxyz

    cfg = grasp_env.RobotEnvCfg()
    cfg.seed = int(SEED)
    cfg.sim.device = device
    cfg.sim.use_fabric = device != "cpu"
    cfg.sim.dt = float(SIM_DT)
    cfg.decimation = int(DECIMATION)
    cfg.sim.render_interval = int(DECIMATION)
    cfg.sim.physx.enable_ccd = bool(ENABLE_CCD)
    cfg.sim.physx.solve_articulation_contact_last = bool(
        PHYSX_SOLVE_ARTICULATION_CONTACT_LAST
    )
    if device.startswith("cuda"):
        scale = int(GPU_PHYSX_BUFFER_SCALE)
        cfg.sim.physx.gpu_max_rigid_contact_count = (2**23) * scale
        cfg.sim.physx.gpu_max_rigid_patch_count = (5 * 2**15) * scale
        cfg.sim.physx.gpu_found_lost_pairs_capacity = (2**21) * scale
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = (2**25) * scale
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = (2**21) * scale
        cfg.sim.physx.gpu_collision_stack_size = (2**26) * scale
        cfg.sim.physx.gpu_heap_capacity = (2**26) * scale
        cfg.sim.physx.gpu_temp_buffer_capacity = (2**24) * scale
        cfg.sim.physx.gpu_max_num_partitions = int(GPU_MAX_NUM_PARTITIONS)
        cfg.sim.physx.gpu_max_soft_body_contacts = (2**20) * scale
        cfg.sim.physx.gpu_max_particle_contacts = (2**20) * scale

    cfg.contact_offset = float(CONTACT_OFFSET_MM) * 0.001
    cfg.rest_offset = float(REST_OFFSET_MM) * 0.001
    cfg.contact_penetration_threshold = float(CONTACT_PENETRATION_THRESHOLD_MM) * 0.001
    cfg.override_gripper_collision_offsets = bool(OVERRIDE_GRIPPER_COLLISION_OFFSETS)
    cfg.enable_object_contact_sensor = bool(ENABLE_OBJECT_CONTACT_SENSOR)
    cfg.penetration_backend = "contact_sensor"
    cfg.contact_max_data_count_per_prim = int(CONTACT_MAX_DATA_COUNT_PER_PRIM)
    cfg.pre_stress_object_motion_threshold = float(PRE_STRESS_OBJECT_MOTION_THRESHOLD)
    cfg.contact_lost_duration = float(CONTACT_LOST_DURATION_S)
    cfg.root_max_linear_speed = float(ROOT_MAX_LINEAR_SPEED)
    cfg.root_max_angular_speed = math.radians(float(ROOT_MAX_ANGULAR_SPEED_DEG))
    cfg.print_contact_separation = bool(PRINT_CONTACT_SEPARATION)
    cfg.envs = 1
    cfg.scene.num_envs = 1
    cfg.viewer.eye = tuple(float(value) for value in CAMERA_EYE)
    cfg.viewer.lookat = tuple(float(value) for value in CAMERA_LOOKAT)
    cfg.viewer.origin_type = "world"
    configure_object(cfg.scene, [object_config])

    first_grasp = entries[0]["grasp"]
    first_hop = first_grasp.get("quality", {}).get("finger_z_hop_enabled", False)
    cfg.apply_finger_z_hop = (
        bool(first_hop) if APPLY_FINGER_Z_HOP is None else bool(APPLY_FINGER_Z_HOP)
    )
    grasp_env.configure_gripper(cfg, gripper, [entries[0]["pregrasp"]])
    cfg.robot_cfg.init_state.pos = (0.0, 0.0, float(GRIPPER_INITIAL_PARK_Z))

    env = grasp_env.RobotEnv(
        cfg=cfg,
        pre_grasp_data=[entries[0]["pregrasp"]],
        conf_data=first_conf,
        debug=DEBUG,
        hold_completed=True,
    )

    state = {
        "index": 0,
        "pending": 0,
        "running": False,
        "quit": False,
        "sort_field": "score",
        "sort_label": "total score",
    }

    def request_index(index: int) -> None:
        state["pending"] = int(index) % len(entries)

    def current_id() -> str:
        return str(entries[int(state["index"])]["review_id"])

    def replay_sort_value(entry: dict, field: str) -> float:
        grasp = entry["grasp"]
        if field == "contact_area_score":
            raw_area = grasp.get("quality", {}).get("grasp_contact_area_mm2")
            if raw_area is not None:
                return float(raw_area)
        return float(grasp.get(field, float("-inf")))

    sort_keys = {
        "0": ("score", "total score"),
        "1": ("force_score", "force score"),
        "2": ("pregrasp_pose_score", "pregrasp pose score"),
        "3": ("stress_pose_score", "stress pose score"),
        "4": ("contact_area_score", "contact area score"),
    }

    def sort_replays(field: str, label: str) -> None:
        combined = sorted(
            zip(entries, source_indices),
            key=lambda pair: (
                replay_sort_value(pair[0], field),
                float(pair[0]["grasp"].get("score", 0.0)),
            ),
            reverse=True,
        )
        entries[:] = [pair[0] for pair in combined]
        source_indices[:] = [pair[1] for pair in combined]
        state["sort_field"] = field
        state["sort_label"] = label
        print(f"Replay sort > {label} descending; restarting at index 0", flush=True)
        request_index(0)

    def mark_current(decision: str) -> None:
        identifier = current_id()
        if decision:
            decisions[identifier] = decision
        else:
            decisions.pop(identifier, None)
        if SAVE_AFTER_EACH_DECISION:
            save_review(identifier)
        draw_saved_grasp(
            debug_draw,
            entries[int(state["index"])]["grasp"],
            decisions.get(identifier, ""),
            state["sort_field"],
        )
        print(
            f"Review > {identifier} = {decision or 'hold'} "
            f"({len(decisions)}/{len(all_entries)} reviewed)",
            flush=True,
        )
        if AUTO_ADVANCE_AFTER_DECISION:
            current = int(state["index"])
            next_unreviewed = next(
                (
                    index
                    for index in list(range(current + 1, len(entries)))
                    + list(range(0, current))
                    if str(entries[index]["review_id"]) not in decisions
                ),
                None,
            )
            if next_unreviewed is None:
                print("Review > all selected grasps have decisions", flush=True)
            else:
                request_index(next_unreviewed)

    def on_keyboard_event(event, *_) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = getattr(event.input, "name", str(event.input).split(".")[-1]).upper()
        number_key = next(
            (
                number
                for number in sort_keys
                if key in {
                    number,
                    f"KEY_{number}",
                    f"KEY{number}",
                    f"NUMPAD_{number}",
                    f"NUMPAD{number}",
                    f"KP_{number}",
                }
            ),
            None,
        )
        if number_key is not None:
            sort_replays(*sort_keys[number_key])
        elif key in {"N", "RIGHT", "SPACE", "ENTER"}:
            request_index(int(state["index"]) + 1)
        elif key in {"P", "LEFT", "BACKSPACE"}:
            request_index(int(state["index"]) - 1)
        elif key == "R":
            request_index(int(state["index"]))
        elif key == "A":
            mark_current("accept")
        elif key == "X":
            mark_current("reject")
        elif key == "U":
            mark_current("")
        elif key == "S":
            save_review(current_id())
            print(f"Review saved > {review_path}", flush=True)
        elif key in {"Q", "ESCAPE"}:
            state["quit"] = True
        return True

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(
        keyboard, on_keyboard_event
    )
    debug_draw = _debug_draw.acquire_debug_draw_interface()

    def load_replay(index: int) -> None:
        entry = entries[index]
        grasp = entry["grasp"]
        pregrasp = entry["pregrasp"]
        scene_id = str(entry["source"]["scene_id"])
        scene_conf = scenes[scene_id]
        identifier = str(entry["review_id"])
        saved_hop = grasp.get("quality", {}).get("finger_z_hop_enabled", False)
        env.cfg.apply_finger_z_hop = (
            bool(saved_hop)
            if APPLY_FINGER_Z_HOP is None
            else bool(APPLY_FINGER_Z_HOP)
        )
        objects = scene_conf.get("objects", [])
        if len(objects) != 1:
            raise ValueError(f"{identifier}: scene_conf must contain one object")
        scene_object = objects[0]
        object_position = torch.tensor(
            scene_object["translate"], dtype=torch.float, device=env.device
        )
        object_orientation = torch.tensor(
            orientation_wxyz(scene_object.get("orient", [0.0, 0.0, 0.0])),
            dtype=torch.float,
            device=env.device,
        )
        env.obj00.data.default_root_state[0, :3] = object_position
        env.obj00.data.default_root_state[0, 3:7] = object_orientation
        env.pre_grasp_data = [pregrasp]
        env.conf_data = scene_conf
        env.factory_reset()
        with torch.inference_mode():
            env.reset()

        normal = grasp.get("normal")
        if isinstance(normal, list) and len(normal) == 3:
            direction = torch.tensor(normal, dtype=torch.float, device=env.device)
            direction /= torch.linalg.vector_norm(direction).clamp_min(1.0e-8)
            env.policy.force_direction[0] = direction
        draw_saved_grasp(
            debug_draw,
            grasp,
            decisions.get(identifier, ""),
            state["sort_field"],
        )

        state["index"] = index
        state["pending"] = None
        state["running"] = True
        source = entry.get("source", {})
        quality = grasp.get("quality", {})
        decision = decisions.get(identifier, "hold")
        print("", flush=True)
        print(
            "Replay > "
            f"selected={index + 1}/{len(entries)} manifest={source_indices[index] + 1}/{len(all_entries)} "
            f"id={identifier} decision={decision} sort={state['sort_label']} "
            f"scene={source.get('scene_id')} output={source.get('output_index')} "
            f"source_pregrasp={source.get('source_pregrasp_index')} "
            f"score={float(grasp.get('score', 0.0)):.6f} "
            f"(force={float(grasp.get('force_score', 0.0)):.4f}, "
            f"area={float(grasp.get('contact_area_score', 0.0)):.4f}, "
            f"pregrasp_pose={float(grasp.get('pregrasp_pose_score', 0.0)):.4f}, "
            f"stress_pose={float(grasp.get('stress_pose_score', 0.0)):.4f})",
            flush=True,
        )
        print(
            "Replay > saved "
            f"result={quality.get('result')} force={quality.get('max_test_force_n')}N "
            f"penetration={quality.get('maximum_contact_penetration_mm')}mm "
            f"direction={grasp.get('normal')}",
            flush=True,
        )

    print(f"Replay manifest > {manifest_path}", flush=True)
    print(f"Replay review   > {review_path}", flush=True)
    print(
        "Keyboard > N/Right/Space=next, P/Left=previous, R=replay, "
        "A=accept, X=reject, U=hold, S=save, 0=total score, "
        "1=force, 2=pregrasp pose, 3=stress pose, 4=contact area, Q=quit",
        flush=True,
    )
    print(
        "Draw > red=bbox, magenta=force normal, RGB=grasp frame, "
        "point: yellow=hold green=accept red=reject",
        flush=True,
    )
    initial_index = 0
    if RESUME_AT_FIRST_UNREVIEWED:
        initial_index = next(
            (
                index
                for index, entry in enumerate(entries)
                if str(entry["review_id"]) not in decisions
            ),
            0,
        )
    load_replay(initial_index)

    while simulation_app.is_running() and not state["quit"]:
        if state["pending"] is not None:
            load_replay(int(state["pending"]))
        if state["running"]:
            with torch.inference_mode():
                env.step(torch.ones((1, 1), device=env.device))
            if bool(env.replay_finished[0]):
                state["running"] = False
                actual = (
                    env.policy.attempt_history[-1]
                    if env.policy.attempt_history
                    else {"result": "timeout"}
                )
                print(
                    "Replay > actual "
                    f"result={actual.get('result')} "
                    f"penetration={actual.get('maximum_contact_penetration_mm')}mm "
                    f"translation_error={actual.get('relative_translation_error_m')}m "
                    f"contact_area={float(actual.get('grasp_contact_area_m2', 0.0)) * 1.0e6:.4f}mm^2; "
                    "final pose held",
                    flush=True,
                )
        else:
            env.sim.render()

finally:
    if state is not None and entries:
        try:
            save_review(str(entries[int(state["index"])]["review_id"]))
        except Exception as exc:
            print(f"Review final save failed > {exc}", flush=True)
    if debug_draw is not None:
        debug_draw.clear_lines()
        debug_draw.clear_points()
    if input_interface is not None and keyboard_subscription is not None:
        try:
            input_interface.unsubscribe_to_keyboard_events(
                keyboard, keyboard_subscription
            )
        except Exception:
            pass
    if env is not None:
        env.close()
    simulation_app.close()
