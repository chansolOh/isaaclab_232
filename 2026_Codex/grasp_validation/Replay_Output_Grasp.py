"""Replay one saved output_grasp record forever in a single Isaac Lab env.

This process is controlled by ``grasp_bbox_viewer_validation.py`` through a
small JSON command file.  A command for another grasp in the same scene resets
the existing environment.  Scene/gripper changes are handled by restarting
this process from the viewer because those changes require rebuilding the USD
stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


CODEX_DIR = Path(__file__).resolve().parents[1]
LEGACY_CHANSOL_DIR = CODEX_DIR.parent / "chansol"
HEADLESS = False
REPLAY_RESET_DELAY_SEC = 0.35
CONTROL_POLL_SEC = 0.10

if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))
if str(LEGACY_CHANSOL_DIR) not in sys.path:
    sys.path.append(str(LEGACY_CHANSOL_DIR))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_status(path: Path | None, **payload) -> None:
    if path is None:
        return
    payload["updated_at"] = time.time()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def flatten_pre_grasps(payload) -> list[tuple[str, dict]]:
    groups = payload if isinstance(payload, list) else [payload]
    records: list[tuple[str, dict]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        model = str(group.get("gripper_model", ""))
        for record in group.get("data", []):
            if isinstance(record, dict):
                records.append((model or str(record.get("gripper_model", "")), record))
    return records


def angular_error(lhs, rhs) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape != rhs_arr.shape:
        return 1.0e6
    delta = (lhs_arr - rhs_arr + 180.0) % 360.0 - 180.0
    return float(np.linalg.norm(delta))


def record_match_score(output: dict, model: str, candidate: dict) -> float:
    output_model = str(output.get("gripper_model", ""))
    candidate_model = model or str(candidate.get("gripper_model", ""))
    if output_model and candidate_model and output_model != candidate_model:
        return math.inf
    if output.get("target_object") != candidate.get("target_object"):
        return math.inf
    if output.get("preset_name") and output.get("preset_name") != candidate.get("preset_name"):
        return math.inf

    score = 0.0
    try:
        score += float(
            np.linalg.norm(
                np.asarray(output["target_points"], dtype=np.float64)
                - np.asarray(candidate["target_points"], dtype=np.float64)
            )
        )
    except (KeyError, TypeError, ValueError):
        score += 1.0e3
    if "target_orientation" in output and "target_orientation" in candidate:
        score += angular_error(output["target_orientation"], candidate["target_orientation"]) * 1.0e-4
    try:
        score += abs(float(output.get("target_width", 0.0)) - float(candidate.get("target_width", 0.0)))
    except (TypeError, ValueError):
        score += 1.0
    return score


def fallback_grasp_bbox(record: dict) -> list[list[list[float]]]:
    """Supply the hand policy's required field without inventing output data."""
    center = np.asarray(record.get("target_points", [0.0, 0.0, 0.0]), dtype=np.float64)
    radius = 0.002
    bbox = np.asarray(
        [
            [-radius, -radius, 0.0],
            [-radius, radius, 0.0],
            [radius, radius, 0.0],
            [radius, -radius, 0.0],
        ]
    ) + center
    return [bbox.tolist()]


def resolve_replay_record(root_path: Path, scene_num: int, grasp_index: int) -> tuple[dict, dict]:
    output_path = root_path / "output_grasp" / f"{scene_num:04d}.json"
    output_records = load_json(output_path)
    if not isinstance(output_records, list):
        raise TypeError(f"output_grasp root must be a list: {output_path}")
    if not 0 <= grasp_index < len(output_records):
        raise IndexError(f"grasp index {grasp_index} is outside 0..{len(output_records) - 1}")

    output = copy.deepcopy(output_records[grasp_index])
    matched = None
    match_score = math.inf
    pre_grasp_path = root_path / "pre_grasp" / f"{scene_num:04d}.json"
    if pre_grasp_path.exists():
        for model, candidate in flatten_pre_grasps(load_json(pre_grasp_path)):
            score = record_match_score(output, model, candidate)
            if score < match_score:
                match_score = score
                matched = candidate

    replay = copy.deepcopy(matched if matched is not None else output)
    # The replay must use the exact execution values that survived validation
    # and were written to output_grasp.
    for key in (
        "target_points",
        "target_orientation",
        "target_width",
        "target_object",
        "gripper_model",
        "gripper_type",
        "preset_name",
        "target_base_tf",
        "target_joint_pos",
        "joint_unit",
        "transition",
    ):
        if key in output:
            replay[key] = copy.deepcopy(output[key])

    if str(replay.get("gripper_type", "")).lower() == "hand" and "grasp_bbox" not in replay:
        replay["grasp_bbox"] = fallback_grasp_bbox(replay)

    metadata = {
        "output_path": str(output_path),
        "grasp_index": grasp_index,
        "matched_pre_grasp": matched is not None,
        "matched_grasp_bbox": matched is not None and "grasp_bbox" in matched,
        "match_score": None if not math.isfinite(match_score) else match_score,
        "output_record": output,
    }
    return replay, metadata


def quaternion_to_matrix_wxyz(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_xyz_degrees_to_matrix(euler) -> np.ndarray:
    x, y, z = np.radians(np.asarray(euler, dtype=np.float64))
    rx = np.asarray([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.asarray([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.asarray([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


class OutputGraspDebugDraw:
    """Draw the selected output record in the simulator's world coordinates."""

    def __init__(self):
        import carb
        from isaacsim.util.debug_draw import _debug_draw

        self.carb = carb
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def clear(self) -> None:
        self.draw.clear_points()
        self.draw.clear_lines()

    def lines(self, starts, ends, color, thickness=3.0) -> None:
        starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        self.draw.draw_lines(
            [self.carb.Float3(*point.tolist()) for point in starts],
            [self.carb.Float3(*point.tolist()) for point in ends],
            [color] * len(starts),
            [float(thickness)] * len(starts),
        )

    def axes(self, position, rotation, length=0.06) -> None:
        position = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        starts = np.tile(position, (3, 1))
        # Rotation columns are the local X/Y/Z axes expressed in world space.
        ends = starts + rotation.T * float(length)
        colors = [
            self.carb.ColorRgba(1.0, 0.1, 0.1, 1.0),
            self.carb.ColorRgba(0.1, 1.0, 0.1, 1.0),
            self.carb.ColorRgba(0.1, 0.4, 1.0, 1.0),
        ]
        self.draw.draw_lines(
            [self.carb.Float3(*point.tolist()) for point in starts],
            [self.carb.Float3(*point.tolist()) for point in ends],
            colors,
            [4.0, 4.0, 4.0],
        )

    def draw_record(self, record: dict, metadata: dict) -> None:
        self.clear()
        yellow = self.carb.ColorRgba(1.0, 0.85, 0.0, 1.0)
        cyan = self.carb.ColorRgba(0.0, 1.0, 1.0, 1.0)
        magenta = self.carb.ColorRgba(1.0, 0.0, 1.0, 1.0)

        target = np.asarray(record["target_points"], dtype=np.float64)
        cross = 0.025
        starts = target + np.asarray([[-cross, 0, 0], [0, -cross, 0], [0, 0, -cross]])
        ends = target + np.asarray([[cross, 0, 0], [0, cross, 0], [0, 0, cross]])
        self.lines(starts, ends, yellow, thickness=5.0)

        poses = record.get("target_base_tf", {})
        start_pose = poses.get("start")
        end_pose = poses.get("end")
        if start_pose and end_pose:
            start_pos = np.asarray(start_pose["position"], dtype=np.float64)
            end_pos = np.asarray(end_pose["position"], dtype=np.float64)
            self.lines([start_pos], [end_pos], magenta, thickness=5.0)
            self.axes(start_pos, quaternion_to_matrix_wxyz(start_pose["orientation_wxyz"]))
            self.axes(end_pos, quaternion_to_matrix_wxyz(end_pose["orientation_wxyz"]))
        else:
            self.axes(target, euler_xyz_degrees_to_matrix(record.get("target_orientation", [0, 0, 0])))

        bbox_data = record.get("grasp_bbox")
        matched = bool(metadata.get("matched_grasp_bbox"))
        if bbox_data is not None and matched:
            for bbox in np.asarray(bbox_data, dtype=np.float64):
                self.lines(bbox[[0, 1, 2, 3]], bbox[[1, 2, 3, 0]], cyan, thickness=4.0)


def parse_control(path: Path, previous_request_id: int) -> dict | None:
    try:
        command = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    request_id = int(command.get("request_id", -1))
    if request_id <= previous_request_id:
        return None
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one output_grasp record in Isaac Lab")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--scene_num", required=True, type=int)
    parser.add_argument("--grasp_index", required=True, type=int)
    parser.add_argument("--control_file", type=Path)
    parser.add_argument("--status_file", type=Path)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    status_file = args.status_file
    scene_num = int(args.scene_num)
    grasp_index = int(args.grasp_index)
    replay_record, metadata = resolve_replay_record(root_path, scene_num, grasp_index)

    # AppLauncher owns Kit arguments; the verifier deliberately remains visible.
    sys.argv = [sys.argv[0]]
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args = launcher_parser.parse_args([])
    launcher_args.headless = HEADLESS
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app
    env = None

    try:
        import torch

        import Direct_RL_main_new_filter as grasp_main
        from cfgs.scene_cfg import platform_setup, scene_obj_setup

        conf_path = root_path / "conf" / f"{scene_num:04d}.json"
        conf_data = grasp_main._resolve_conf_object_usd_paths(
            load_json(conf_path), grasp_main.OBJECT_NAME_CSV_PATH
        )
        group = {
            "gripper_model": replay_record["gripper_model"],
            "data": [replay_record],
        }
        gripper_type = grasp_main._detect_gripper_type(group)
        gripper_info = grasp_main._load_gripper_info(group, gripper_type)
        is_hand = gripper_type == "hand"

        if is_hand:
            import codex_cfgs.Robot_EnvCfg_hand_platform_filter as robot_module
        else:
            import cfgs.Robot_EnvCfg as robot_module

        env_cfg = robot_module.RobotEnvCfg()
        env_cfg.envs = 1
        env_cfg.scene.num_envs = 1
        if is_hand:
            robot_module.Set_RobotEnvCFG(env_cfg, gripper_info, [replay_record])
        else:
            robot_module.Set_RobotEnvCFG(env_cfg, gripper_info)

        scene_obj_setup(env_cfg.scene, conf_data["objects"])
        platform_setup(env_cfg.scene, conf_data["platform"])
        grasp_main._object_frame_transformer_setup_with_asset_names(env_cfg.scene, conf_data["objects"])

        env = robot_module.RobotEnv(
            cfg=env_cfg,
            pre_grasp_data=[replay_record],
            conf_data=conf_data,
            # Legacy finger policies generate their 3D approach bbox here.
            # Hand debug drawing is enabled only when the true pre_grasp bbox
            # was matched; the output-only fallback bbox must not be presented
            # as real validation geometry.
            debug=(not is_hand or bool(metadata["matched_grasp_bbox"])),
        )
        env.reset()
        debug_draw = OutputGraspDebugDraw()
        debug_draw.draw_record(replay_record, metadata)

        request_id = -1
        cycle_count = 0
        last_reset_time = time.monotonic()
        last_control_poll = 0.0
        save_status(
            status_file,
            state="running",
            scene_num=scene_num,
            grasp_index=grasp_index,
            gripper_model=replay_record["gripper_model"],
            matched_pre_grasp=metadata["matched_pre_grasp"],
            matched_grasp_bbox=metadata["matched_grasp_bbox"],
            cycle_count=cycle_count,
        )
        print(
            f"Replay > scene={scene_num:04d} grasp={grasp_index} "
            f"gripper={replay_record['gripper_model']} matched_pregrasp={metadata['matched_pre_grasp']}"
        )
        sys.stdout.flush()

        while simulation_app.is_running():
            now = time.monotonic()
            if args.control_file is not None and now - last_control_poll >= CONTROL_POLL_SEC:
                last_control_poll = now
                command = parse_control(args.control_file, request_id)
                if command is not None:
                    request_id = int(command["request_id"])
                    requested_scene = int(command["scene_num"])
                    requested_index = int(command["grasp_index"])
                    requested_model = str(command.get("gripper_model", ""))
                    if requested_scene != scene_num or (
                        requested_model and requested_model != replay_record["gripper_model"]
                    ):
                        save_status(
                            status_file,
                            state="restart_required",
                            scene_num=scene_num,
                            grasp_index=grasp_index,
                            requested_scene=requested_scene,
                            requested_grasp_index=requested_index,
                        )
                    elif requested_index != grasp_index:
                        replay_record, metadata = resolve_replay_record(
                            root_path, scene_num, requested_index
                        )
                        env.pre_grasp_data = [replay_record]
                        # env.step() creates/updates Isaac Lab state buffers in
                        # inference mode.  Resetting those buffers outside the
                        # same context causes PyTorch's "inference tensor"
                        # in-place update error when switching a grasp.
                        with torch.inference_mode():
                            env.factory_reset()
                        debug_draw.draw_record(replay_record, metadata)
                        grasp_index = requested_index
                        cycle_count = 0
                        last_reset_time = now
                        save_status(
                            status_file,
                            state="running",
                            scene_num=scene_num,
                            grasp_index=grasp_index,
                            gripper_model=replay_record["gripper_model"],
                            matched_pre_grasp=metadata["matched_pre_grasp"],
                            matched_grasp_bbox=metadata["matched_grasp_bbox"],
                            cycle_count=cycle_count,
                        )
                        print(f"Replay > switched grasp={grasp_index}")
                        sys.stdout.flush()

            with torch.inference_mode():
                action = torch.ones((1,), device=env.device)
                obs, _rew, _term, _trunc, _info = env.step(action)
                if obs["policy"].sum() == 0 and now - last_reset_time >= REPLAY_RESET_DELAY_SEC:
                    cycle_count += 1
                    env.factory_reset()
                    debug_draw.draw_record(replay_record, metadata)
                    last_reset_time = now
                    save_status(
                        status_file,
                        state="running",
                        scene_num=scene_num,
                        grasp_index=grasp_index,
                        gripper_model=replay_record["gripper_model"],
                        matched_pre_grasp=metadata["matched_pre_grasp"],
                        matched_grasp_bbox=metadata["matched_grasp_bbox"],
                        cycle_count=cycle_count,
                    )
    except BaseException as error:
        save_status(
            status_file,
            state="error",
            scene_num=scene_num,
            grasp_index=grasp_index,
            message=str(error),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        try:
            simulation_app.close()
        except BaseException:
            pass


if __name__ == "__main__":
    main()
