#!/usr/bin/env python3
"""Replay collected grasps one at a time in the original Isaac Lab physics."""

from __future__ import annotations

import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# 실행 설정: argparse 없이 이 부분만 수정한다.
# -----------------------------------------------------------------------------
ROOT = Path(
    "/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/"
    "black_pepper_shaker/"
    "Robotiq_2f140"#"UON_3finger_gripper"
)
SCENE = 0

FINGER_INFO = Path("/nas/ochansol/gripper_info/gripper_info_new_2026.json")
HAND_INFO = Path("/nas/ochansol/gripper_info/gripper_info_hand_2026.json")

# output_grasp JSON에 정렬되어 저장된 순서를 사용한다.
START_GRASP_INDEX = 0
END_GRASP_INDEX = None
STRIDE = 1

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
MAX_RELATIVE_TRANSLATION_M = 0.010
MAX_RELATIVE_ROTATION_DEG = 20.0
CONTACT_LOST_DURATION_S = 0.10
ROOT_MAX_LINEAR_SPEED = 0.25
ROOT_MAX_ANGULAR_SPEED_DEG = 90.0
GRIPPER_INITIAL_PARK_Z = 2.0
PRINT_CONTACT_SEPARATION = False

# None이면 각 저장 grasp의 quality.finger_z_hop_enabled 값을 사용한다.
APPLY_FINGER_Z_HOP = None

CAMERA_EYE = (0.45, -0.45, 0.30)
CAMERA_LOOKAT = (0.0, 0.0, 0.02)
DRAW_GRASP_DATA = True
GRASP_LINE_WIDTH = 5.0
GRASP_POINT_SIZE = 12.0
NORMAL_LINE_SCALE = 0.08
FRAME_AXIS_SCALE = 0.035
SHOW_GRASP_FRAME = True
DEBUG = True


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def select_pregrasp_group(payload) -> dict:
    if isinstance(payload, dict):
        group = payload
    elif isinstance(payload, list) and len(payload) == 1:
        group = payload[0]
    else:
        raise ValueError("pre_grasp JSON must contain exactly one gripper group")
    if not isinstance(group.get("data"), list) or not group["data"]:
        raise ValueError("pre_grasp group has no data")
    return group


def normalized_device(value: str) -> str:
    device = str(value).strip().lower()
    valid_cuda = device == "cuda" or (
        device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    )
    if device != "cpu" and not valid_cuda:
        raise ValueError("DEVICE must be 'cpu', 'cuda', or 'cuda:<index>'")
    return device


def rotation_score_color(score: float) -> tuple[float, float, float, float]:
    """Match the blue-green-yellow color used by grasp_data_viz.py."""
    score = min(1.0, max(0.0, float(score)))
    blue = (0.0, 0.25, 1.0)
    green = (0.0, 0.95, 0.2)
    yellow = (1.0, 0.9, 0.0)
    if score < 0.5:
        amount = score / 0.5
        first, second = blue, green
    else:
        amount = (score - 0.5) / 0.5
        first, second = green, yellow
    rgb = tuple((1.0 - amount) * first[i] + amount * second[i] for i in range(3))
    return rgb + (1.0,)


def draw_saved_grasp(draw, grasp: dict, score_field: str = "pose_score") -> None:
    """Draw the saved Open3D grasp geometry in the Isaac Sim viewport."""
    draw.clear_lines()
    draw.clear_points()
    if not DRAW_GRASP_DATA:
        return

    raw_boxes = grasp.get("grasp_boxes")
    if raw_boxes is None:
        raw_box = grasp.get("grasp_box", [])
        raw_boxes = [raw_box]
    valid_boxes = (
        isinstance(raw_boxes, list)
        and len(raw_boxes) > 0
        and all(
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(point, list) and len(point) == 3 for point in box)
            for box in raw_boxes
        )
    )
    if not valid_boxes:
        print("Replay draw > invalid grasp_boxes/grasp_box; skipped", flush=True)
        return
    boxes = [
        [tuple(float(value) for value in point) for point in box]
        for box in raw_boxes
    ]
    red = (1.0, 0.12, 0.05, 1.0)
    score_color = rotation_score_color(
        grasp.get(score_field, grasp.get("score", 1.0))
    )

    # Draw every finger/fingertip contact rectangle. The second edge carries
    # the active sort-score color and the remaining edges are red.
    starts, ends, colors, widths = [], [], [], []
    for box in boxes:
        starts.extend([box[0], box[1], box[2], box[3]])
        ends.extend([box[1], box[2], box[3], box[0]])
        colors.extend([red, score_color, red, red])
        widths.extend([float(GRASP_LINE_WIDTH)] * 4)

    all_points = [point for box in boxes for point in box]
    center = tuple(
        sum(point[axis] for point in all_points) / len(all_points)
        for axis in range(3)
    )
    normal = grasp.get("normal")
    if isinstance(normal, list) and len(normal) == 3:
        normal = tuple(float(value) for value in normal)
        length = math.sqrt(sum(value * value for value in normal))
        if length > 1.0e-9:
            normal = tuple(value / length for value in normal)
            starts.append(center)
            ends.append(
                tuple(
                    center[axis] + normal[axis] * float(NORMAL_LINE_SCALE)
                    for axis in range(3)
                )
            )
            colors.append((1.0, 0.0, 0.75, 1.0))
            widths.append(float(GRASP_LINE_WIDTH))

    grasp_matrix = grasp.get("grasp_mat")
    if (
        SHOW_GRASP_FRAME
        and isinstance(grasp_matrix, list)
        and len(grasp_matrix) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in grasp_matrix)
    ):
        origin = tuple(float(grasp_matrix[axis][3]) for axis in range(3))
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
                    + float(grasp_matrix[row][column]) * float(FRAME_AXIS_SCALE)
                    for row in range(3)
                )
            )
            colors.append(axis_colors[column])
            widths.append(max(2.0, float(GRASP_LINE_WIDTH) - 1.0))

    draw.draw_lines(starts, ends, colors, widths)
    target = grasp.get("target_points", center)
    if isinstance(target, list) and len(target) == 3:
        target = tuple(float(value) for value in target)
    else:
        target = center
    draw.draw_points(
        [target],
        [(1.0, 0.85, 0.0, 1.0)],
        [float(GRASP_POINT_SIZE)],
    )


scene_id = f"{int(SCENE):04d}"
conf_path = ROOT / "conf" / f"{scene_id}.json"
pregrasp_path = ROOT / "pre_grasp" / f"{scene_id}.json"
grasp_path = ROOT / "output_grasp" / f"{scene_id}.json"

conf = load_json(conf_path)
pregrasp_group = select_pregrasp_group(load_json(pregrasp_path))
all_pregrasps = pregrasp_group["data"]
all_grasps = load_json(grasp_path)
if not isinstance(all_grasps, list) or not all_grasps:
    raise ValueError(f"No collected grasps in {grasp_path}")

end_index = len(all_grasps) if END_GRASP_INDEX is None else int(END_GRASP_INDEX)
selected_output_indices = list(
    range(
        int(START_GRASP_INDEX),
        min(end_index, len(all_grasps)),
        max(1, int(STRIDE)),
    )
)
selected_grasps = [all_grasps[index] for index in selected_output_indices]
if not selected_grasps:
    raise ValueError("Selected grasp range is empty")

replay_pregrasps = []
for output_index, grasp in enumerate(selected_grasps, start=int(START_GRASP_INDEX)):
    if "source_pregrasp_index" not in grasp:
        raise KeyError(f"output grasp {output_index} has no source_pregrasp_index")
    source_index = int(grasp["source_pregrasp_index"])
    if not 0 <= source_index < len(all_pregrasps):
        raise IndexError(
            f"output grasp {output_index} source_pregrasp_index={source_index} "
            f"is outside 0..{len(all_pregrasps) - 1}"
        )
    replay_pregrasps.append(all_pregrasps[source_index])

finger_database = load_json(FINGER_INFO)
hand_database = load_json(HAND_INFO)
gripper_name = str(pregrasp_group["gripper_model"])
if gripper_name in finger_database:
    gripper = finger_database[gripper_name]
elif gripper_name in hand_database:
    gripper = hand_database[gripper_name]
else:
    raise ValueError(f"Unknown gripper: {gripper_name}")

device = normalized_device(DEVICE)
launcher = AppLauncher(headless=False, device=device, enable_cameras=False)
simulation_app = launcher.app
env = None
keyboard = None
keyboard_subscription = None
input_interface = None
debug_draw = None

try:
    import carb
    import omni.appwindow
    import torch
    from isaacsim.util.debug_draw import _debug_draw

    from grasp import env as grasp_env
    from grasp.scene_cfg import configure_object

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
    cfg.contact_penetration_threshold = (
        float(CONTACT_PENETRATION_THRESHOLD_MM) * 0.001
    )
    cfg.override_gripper_collision_offsets = bool(
        OVERRIDE_GRIPPER_COLLISION_OFFSETS
    )
    cfg.enable_object_contact_sensor = bool(ENABLE_OBJECT_CONTACT_SENSOR)
    cfg.penetration_backend = "contact_sensor"
    cfg.contact_max_data_count_per_prim = int(CONTACT_MAX_DATA_COUNT_PER_PRIM)
    cfg.pre_stress_object_motion_threshold = float(
        PRE_STRESS_OBJECT_MOTION_THRESHOLD
    )
    cfg.max_relative_translation = float(MAX_RELATIVE_TRANSLATION_M)
    cfg.max_relative_rotation_deg = float(MAX_RELATIVE_ROTATION_DEG)
    cfg.contact_lost_duration = float(CONTACT_LOST_DURATION_S)
    cfg.root_max_linear_speed = float(ROOT_MAX_LINEAR_SPEED)
    cfg.root_max_angular_speed = math.radians(float(ROOT_MAX_ANGULAR_SPEED_DEG))
    cfg.print_contact_separation = bool(PRINT_CONTACT_SEPARATION)
    cfg.envs = 1
    cfg.scene.num_envs = 1
    cfg.viewer.eye = tuple(float(value) for value in CAMERA_EYE)
    cfg.viewer.lookat = tuple(float(value) for value in CAMERA_LOOKAT)
    cfg.viewer.origin_type = "world"
    configure_object(cfg.scene, conf.get("objects", []))

    first_hop = selected_grasps[0].get("quality", {}).get(
        "finger_z_hop_enabled", False
    )
    cfg.apply_finger_z_hop = (
        bool(first_hop) if APPLY_FINGER_Z_HOP is None else bool(APPLY_FINGER_Z_HOP)
    )
    grasp_env.configure_gripper(cfg, gripper, [replay_pregrasps[0]])
    cfg.robot_cfg.init_state.pos = (0.0, 0.0, float(GRIPPER_INITIAL_PARK_Z))

    env = grasp_env.RobotEnv(
        cfg=cfg,
        pre_grasp_data=[replay_pregrasps[0]],
        conf_data=conf,
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
        state["pending"] = int(index) % len(selected_grasps)

    sort_keys = {
        "1": ("force_score", "force score"),
        "2": ("pregrasp_pose_score", "pregrasp pose score"),
        "3": ("stress_pose_score", "stress pose score"),
        "4": ("contact_area_score", "contact area score"),
    }

    def replay_sort_value(grasp: dict, field: str) -> float:
        if field == "contact_area_score":
            # Older files used a 100 mm^2 saturation threshold, leaving many
            # contact_area_score values tied at 1. Sort by the preserved raw
            # footprint whenever it is available.
            raw_area = grasp.get("quality", {}).get("grasp_contact_area_mm2")
            if raw_area is not None:
                return float(raw_area)
        return float(grasp.get(field, float("-inf")))

    def sort_replays(field: str, label: str) -> None:
        order = sorted(
            range(len(selected_grasps)),
            key=lambda index: (
                replay_sort_value(selected_grasps[index], field),
                float(selected_grasps[index].get("score", 0.0)),
            ),
            reverse=True,
        )
        selected_grasps[:] = [selected_grasps[index] for index in order]
        replay_pregrasps[:] = [replay_pregrasps[index] for index in order]
        selected_output_indices[:] = [
            selected_output_indices[index] for index in order
        ]
        state["sort_field"] = field
        state["sort_label"] = label
        print(
            f"Replay sort > {label} descending; restarting at index 0",
            flush=True,
        )
        request_index(0)

    def on_keyboard_event(event, *_) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = getattr(event.input, "name", str(event.input).split(".")[-1]).upper()
        number_key = None
        for number in sort_keys:
            if key in {
                number,
                f"KEY_{number}",
                f"KEY{number}",
                f"NUMPAD_{number}",
                f"NUMPAD{number}",
                f"KP_{number}",
            }:
                number_key = number
                break
        if number_key is not None:
            sort_replays(*sort_keys[number_key])
        elif key in {"N", "RIGHT", "SPACE", "ENTER"}:
            request_index(state["index"] + 1)
        elif key in {"P", "LEFT", "BACKSPACE"}:
            request_index(state["index"] - 1)
        elif key == "R":
            request_index(state["index"])
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
        grasp = selected_grasps[index]
        pregrasp = replay_pregrasps[index]
        saved_hop = grasp.get("quality", {}).get("finger_z_hop_enabled", False)
        env.cfg.apply_finger_z_hop = (
            bool(saved_hop)
            if APPLY_FINGER_Z_HOP is None
            else bool(APPLY_FINGER_Z_HOP)
        )
        env.pre_grasp_data = [pregrasp]
        env.factory_reset()
        with torch.inference_mode():
            env.reset()

        normal = grasp.get("normal")
        if isinstance(normal, list) and len(normal) == 3:
            direction = torch.tensor(normal, dtype=torch.float, device=env.device)
            direction /= torch.linalg.vector_norm(direction).clamp_min(1.0e-8)
            env.policy.force_direction[0] = direction
        draw_saved_grasp(debug_draw, grasp, state["sort_field"])

        state["index"] = index
        state["pending"] = None
        state["running"] = True
        quality = grasp.get("quality", {})
        output_index = selected_output_indices[index]
        print("", flush=True)
        print(
            "Replay > "
            f"grasp={output_index + 1}/{len(all_grasps)} "
            f"selected={index + 1}/{len(selected_grasps)} "
            f"sort={state['sort_label']} "
            f"source_pregrasp={grasp['source_pregrasp_index']} "
            f"score={float(grasp.get('score', 0.0)):.6f} "
            f"(force={float(grasp.get('force_score', 0.0)):.4f}, "
            f"area={float(grasp.get('contact_area_score', 0.0)):.4f}, "
            f"pregrasp_pose={float(grasp.get('pregrasp_pose_score', 0.0)):.4f}, "
            f"stress_pose={float(grasp.get('stress_pose_score', 0.0)):.4f})",
            flush=True,
        )
        print(
            "Replay > saved "
            f"result={quality.get('result')} "
            f"force={quality.get('max_test_force_n')}N "
            f"penetration={quality.get('maximum_contact_penetration_mm')}mm "
            f"direction={grasp.get('normal')}",
            flush=True,
        )

    print(
        "Replay keyboard > N/Right/Space=next, P/Left=previous, "
        "R=replay current, 1=force, 2=pregrasp pose, 3=stress pose, "
        "4=contact area, Q/Esc=quit",
        flush=True,
    )
    print(
        "Replay draw > red=grasp sides, score-color=center edge, "
        "magenta=stored force normal, RGB=grasp frame, yellow=target point",
        flush=True,
    )
    load_replay(0)

    while simulation_app.is_running() and not state["quit"]:
        if state["pending"] is not None:
            load_replay(int(state["pending"]))

        if state["running"]:
            with torch.inference_mode():
                actions = torch.ones((1, 1), device=env.device)
                env.step(actions)
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
                    f"translation_error={actual.get('relative_translation_error_m')}m; "
                    f"contact_area={float(actual.get('grasp_contact_area_m2', 0.0)) * 1.0e6:.4f}mm^2; "
                    "final pose held",
                    flush=True,
                )
        else:
            # Refresh viewport/keyboard while SimulationContext temporarily
            # disables physics playback, preserving the held final state.
            env.sim.render()

finally:
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
