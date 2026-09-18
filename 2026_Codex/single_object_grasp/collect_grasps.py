
from __future__ import annotations


def main(
        object_name: str,
        Gripper_name: str,
        scene_num: int,
        root_path: str = "/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen",
):

    from collections import Counter
    import json
    import math
    import os
    import sys
    import time
    from pathlib import Path

    from isaaclab.app import AppLauncher


    # -----------------------------------------------------------------------------
    # 실행 설정: 이 부분만 수정해서 사용한다.
    # ROOT는 <output-root>/<object>/<gripper> 폴더다.
    # -----------------------------------------------------------------------------
    ROOT = Path(root_path) / object_name / Gripper_name
    SCENE = scene_num

    FINGER_INFO = Path("/nas/ochansol/gripper_info/gripper_info_new_2026.json")
    HAND_INFO = Path("/nas/ochansol/gripper_info/gripper_info_hand_2026.json")

    ENVS = 200
    SEED = 42
    START_INDEX = 0
    MINIMUM_RECORDS = 2
    RETRIES = 0
    OVERWRITE = False
    DEBUG = False
    DEBUG_GRASP_LINE_WIDTH = 5.0
    DEBUG_VECTOR_LENGTH = 0.08

    # Physics는 SIM_DT 간격으로 진행하고 policy/action은 DECIMATION step마다 갱신한다.
    SIM_DT = 1.0 / 800.0
    DECIMATION = 4
    ENABLE_CCD = True
    PHYSX_SOLVE_ARTICULATION_CONTACT_LAST = False
    # GPU PhysX buffer를 Isaac Lab 기본값보다 일괄 확대한다. 기존 수집기와 같은
    # 4배를 기본으로 쓰며, CPU PhysX에서는 적용하지 않는다.
    GPU_PHYSX_BUFFER_SCALE = 4
    GPU_MAX_NUM_PARTITIONS = 16
    # 접촉 관련 길이는 여기서 mm로 입력하고, Isaac/PhysX에 넘길 때만 m로 변환한다.
    CONTACT_OFFSET_MM = 1.0
    REST_OFFSET_MM = 0.1
    # False면 기존 수집기처럼 gripper USD의 collider offset을 그대로 사용한다.
    OVERRIDE_GRIPPER_COLLISION_OFFSETS = True
    # 허용 관통 깊이(mm). 예: 1.0은 1 mm, 0.08은 0.08 mm이다.
    CONTACT_PENETRATION_THRESHOLD_MM = 1
    # object rigid body를 sensor source로 삼아 gripper 모든 link와의 접촉을
    # 단일 상세 센서에서 확인한다. False면 gripper-side 상세 센서로 되돌아간다.
    ENABLE_OBJECT_CONTACT_SENSOR = True
    # 추후 Warp 기반 검사로 교체할 때 policy/저장 형식은 유지하고 이 backend만
    # 교체할 수 있도록 penetration 판정 경계를 분리해 둔다.
    PENETRATION_BACKEND = "contact_sensor"
    # APPROACH/CLOSE 중 물체가 초기 pose에서 이 이상 이동하면 transient
    # contact report 유무와 관계없이 충돌 실패로 처리한다.
    PRE_STRESS_OBJECT_MOTION_THRESHOLD = 0.1
    # 외력 시험 중 contact가 이 시간 이상 연속으로 사라질 때만 놓침 실패다.
    # 위치와 회전 변화는 score에만 반영하며 성공/실패 조건으로 사용하지 않는다.
    CONTACT_LOST_DURATION_S = 0.10
    CONTACT_MAX_DATA_COUNT_PER_PRIM = 4096
    PRINT_CONTACT_SEPARATION = False
    CONTACT_SEPARATION_PRINT_DELTA_MM = 1.0
    ROOT_MAX_LINEAR_SPEED = 0.25
    ROOT_MAX_ANGULAR_SPEED_DEG = 90.0
    # PhysX가 scene을 처음 만들 때 원점의 물체와 gripper collider가 겹치지
    # 않도록, 첫 reset 전 gripper가 대기할 로컬 Z 위치다.
    GRIPPER_INITIAL_PARK_Z = 2.0
    # False면 닫히는 동안 joint_to_z calibration으로 base를 들어 올리지 않는다.
    # 같은 pregrasp A/B에서 False가 더 작은 최악 관통값을 보였다.
    APPLY_FINGER_Z_HOP = True

    # "attempt": 기존처럼 시도한 pregrasp pose와 target_width를 저장한다.
    # "grasp_complete_restored": 시도한 width는 유지하고 외력 직전 pose를
    # 닫힘 동안 움직인 물체의 초기 pose로 되돌린 좌표에 저장한다.
    GRASP_RECORD_MODE = "grasp_complete_restored"
    # None이면 mode와 관계없이 output_grasp를 사용한다.
    OUTPUT_GRASP_FOLDER: str | None = None

    HEADLESS = True
    # "cuda:0" = GPU PhysX, "cpu" = CPU PhysX.
    # AppLauncher와 SimulationCfg 양쪽에 동일하게 적용된다.
    DEVICE = "cpu"


    def load_json(path: Path):
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)


    def select_group(payload) -> dict:
        if isinstance(payload, dict):
            group = payload
        elif isinstance(payload, list) and len(payload) == 1:
            group = payload[0]
        else:
            raise ValueError("pre_grasp JSON must contain exactly one gripper group")
        if not isinstance(group.get("data"), list) or not group["data"]:
            raise ValueError("pre_grasp group has no data")
        return group


    def atomic_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(temporary, path)


    scene_id = f"{SCENE:04d}"
    normalized_record_mode = str(GRASP_RECORD_MODE).strip().lower()
    if normalized_record_mode not in {"attempt", "grasp_complete_restored"}:
        raise ValueError(
            "GRASP_RECORD_MODE must be 'attempt' or "
            "'grasp_complete_restored'"
        )
    output_folder = OUTPUT_GRASP_FOLDER or "output_grasp"
    pregrasp_path = ROOT / "pre_grasp" / f"{scene_id}.json"
    conf_path = ROOT / "conf" / f"{scene_id}.json"
    output_path = ROOT / output_folder / f"{scene_id}.json"
    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(f"output already exists: {output_path}; set OVERWRITE=True")
    group = select_group(load_json(pregrasp_path))
    records = group["data"][START_INDEX:]
    if not records:
        raise ValueError("START_INDEX is past the end of pre-grasp data")
    finger_database = load_json(FINGER_INFO)
    hand_database = load_json(HAND_INFO)
    gripper_name = group["gripper_model"]
    if gripper_name in finger_database:
        gripper = finger_database[gripper_name]
    elif gripper_name in hand_database:
        gripper = hand_database[gripper_name]
    else:
        raise ValueError(f"gripper {gripper_name!r} is absent from both info files")
    conf = load_json(conf_path)

    normalized_device = str(DEVICE).strip().lower()
    valid_cuda_device = normalized_device == "cuda" or (
        normalized_device.startswith("cuda:")
        and normalized_device.removeprefix("cuda:").isdigit()
    )
    if normalized_device != "cpu" and not valid_cuda_device:
        raise ValueError("DEVICE must be 'cpu', 'cuda', or 'cuda:<index>'")
    launcher = AppLauncher(
        headless=HEADLESS, device=normalized_device, enable_cameras=False
    )
    simulation_app = launcher.app
    env = None
    try:
        import torch

        from grasp import env as grasp_env
        from grasp.scene_cfg import configure_object

        cfg = grasp_env.RobotEnvCfg()
        cfg.seed = int(SEED)
        cfg.sim.device = normalized_device
        # Isaac Lab 2.3.2의 batched Fabric 연산은 CUDA 전용이다.
        cfg.sim.use_fabric = normalized_device != "cpu"
        if SIM_DT <= 0.0 or DECIMATION < 1:
            raise ValueError("SIM_DT must be positive and DECIMATION must be at least 1")
        if GPU_PHYSX_BUFFER_SCALE < 1:
            raise ValueError("GPU_PHYSX_BUFFER_SCALE must be at least 1")
        if GPU_MAX_NUM_PARTITIONS not in (1, 2, 4, 8, 16, 32):
            raise ValueError("GPU_MAX_NUM_PARTITIONS must be a power of two from 1 to 32")
        if not (CONTACT_OFFSET_MM > max(0.0, REST_OFFSET_MM)):
            raise ValueError("CONTACT_OFFSET_MM must exceed both zero and REST_OFFSET_MM")
        if CONTACT_PENETRATION_THRESHOLD_MM < 0.0:
            raise ValueError("CONTACT_PENETRATION_THRESHOLD_MM must be non-negative")
        if CONTACT_LOST_DURATION_S <= 0.0:
            raise ValueError("CONTACT_LOST_DURATION_S must be positive")
        if CONTACT_MAX_DATA_COUNT_PER_PRIM < 1:
            raise ValueError("CONTACT_MAX_DATA_COUNT_PER_PRIM must be at least 1")
        if ROOT_MAX_LINEAR_SPEED <= 0.0 or ROOT_MAX_ANGULAR_SPEED_DEG <= 0.0:
            raise ValueError("Root motion speed limits must be positive")
        cfg.sim.dt = float(SIM_DT)
        cfg.decimation = int(DECIMATION)
        cfg.sim.render_interval = int(DECIMATION)
        cfg.sim.physx.enable_ccd = bool(ENABLE_CCD)
        cfg.sim.physx.solve_articulation_contact_last = bool(
            PHYSX_SOLVE_ARTICULATION_CONTACT_LAST
        )
        if normalized_device.startswith("cuda"):
            scale = int(GPU_PHYSX_BUFFER_SCALE)
            # Isaac Lab 2.3.2 PhysxCfg defaults, enlarged as one coherent set.
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
        millimetres_to_metres = 0.001
        contact_penetration_threshold_m = (
            float(CONTACT_PENETRATION_THRESHOLD_MM) * millimetres_to_metres
        )
        cfg.contact_offset = float(CONTACT_OFFSET_MM) * millimetres_to_metres
        cfg.rest_offset = float(REST_OFFSET_MM) * millimetres_to_metres
        cfg.override_gripper_collision_offsets = bool(
            OVERRIDE_GRIPPER_COLLISION_OFFSETS
        )
        cfg.contact_penetration_threshold = contact_penetration_threshold_m
        cfg.enable_object_contact_sensor = bool(ENABLE_OBJECT_CONTACT_SENSOR)
        cfg.penetration_backend = str(PENETRATION_BACKEND)
        cfg.pre_stress_object_motion_threshold = float(
            PRE_STRESS_OBJECT_MOTION_THRESHOLD
        )
        cfg.contact_lost_duration = float(CONTACT_LOST_DURATION_S)
        cfg.contact_max_data_count_per_prim = int(CONTACT_MAX_DATA_COUNT_PER_PRIM)
        cfg.print_contact_separation = bool(PRINT_CONTACT_SEPARATION)
        cfg.contact_separation_print_delta = (
            float(CONTACT_SEPARATION_PRINT_DELTA_MM) * millimetres_to_metres
        )
        cfg.root_max_linear_speed = float(ROOT_MAX_LINEAR_SPEED)
        cfg.root_max_angular_speed = math.radians(float(ROOT_MAX_ANGULAR_SPEED_DEG))
        cfg.apply_finger_z_hop = bool(APPLY_FINGER_Z_HOP)
        cfg.grasp_record_mode = normalized_record_mode
        cfg.draw_successful_grasps = bool(DEBUG and not HEADLESS)
        cfg.debug_grasp_line_width = float(DEBUG_GRASP_LINE_WIDTH)
        cfg.debug_vector_length = float(DEBUG_VECTOR_LENGTH)
        cfg.envs = ENVS
        cfg.scene.num_envs = ENVS
        configure_object(cfg.scene, conf.get("objects", []))
        grasp_env.configure_gripper(cfg, gripper, records)
        cfg.robot_cfg.init_state.pos = (0.0, 0.0, float(GRIPPER_INITIAL_PARK_Z))
        print(
            "Grasp > contact_units "
            f"offset={CONTACT_OFFSET_MM:.4f}mm "
            f"rest={REST_OFFSET_MM:.4f}mm "
            f"penetration_threshold={CONTACT_PENETRATION_THRESHOLD_MM:.4f}mm "
            f"print_delta={CONTACT_SEPARATION_PRINT_DELTA_MM:.4f}mm",
            flush=True,
        )
        print(f"Grasp > record_mode={normalized_record_mode}", flush=True)
        env = grasp_env.RobotEnv(
            cfg=cfg,
            pre_grasp_data=records,
            conf_data=conf,
            debug=DEBUG and not HEADLESS,
        )
        started = time.time()
        collected_by_source: dict[int, dict] = {}
        attempt_history = []
        minimum = min(MINIMUM_RECORDS, len(records))
        for attempt in range(RETRIES + 1):
            # Isaac Lab state buffers are updated under inference mode below;
            # reset must use the same mode on subsequent attempts.
            with torch.inference_mode():
                obs, _ = env.reset()
            print(
                f"Grasp > scene={scene_id} gripper={gripper_name} "
                f"attempt={attempt + 1}/{RETRIES + 1}",
                flush=True,
            )
            while simulation_app.is_running():
                with torch.inference_mode():
                    actions = torch.ones((env.num_envs, 1), device=env.device)
                    obs, _, _, _, _ = env.step(actions)
                if int(obs["policy"].sum()) == 0:
                    break
            for item in env.policy.output_list:
                item["source_pregrasp_index"] += START_INDEX
                source_index = int(item["source_pregrasp_index"])
                previous = collected_by_source.get(source_index)
                if previous is None or float(item.get("score", 0.0)) > float(
                    previous.get("score", 0.0)
                ):
                    collected_by_source[source_index] = item
            current_history = env.policy.attempt_history
            for item in current_history:
                item["source_pregrasp_index"] += START_INDEX
            attempt_history.extend(dict(item, attempt=attempt + 1) for item in current_history)
            print(f"Grasp > collected={len(collected_by_source)}", flush=True)
            close_separations = [
                float(item["minimum_close_contact_separation_m"])
                for item in current_history
                if item.get("minimum_close_contact_separation_m") is not None
            ]
            if close_separations:
                print(
                    "Grasp > close_contact_diagnostics "
                    f"measured={len(close_separations)}/{len(current_history)} "
                    f"minimum={min(close_separations) * 1000.0:.4f}mm "
                    f"penetrating={sum(value < 0.0 for value in close_separations)} "
                    f"over_threshold={sum(value < -contact_penetration_threshold_m for value in close_separations)}",
                    flush=True,
                )
            failure_counts = Counter(
                item["result"] for item in current_history if item["result"] != "completed"
            )
            if failure_counts:
                print(f"Grasp > failures={dict(failure_counts)}", flush=True)
                failed = [item for item in current_history if item["result"] != "completed"]
                worst_position = max(item["relative_translation_error_m"] for item in failed)
                worst_rotation = max(item["relative_rotation_error_deg"] for item in failed)
                print(
                    "Grasp > failure_diagnostics "
                    f"max_relative_translation={worst_position:.5f}m "
                    f"max_relative_rotation={worst_rotation:.2f}deg",
                    flush=True,
                )
            if len(collected_by_source) >= minimum or attempt >= RETRIES:
                break
            env.factory_reset()
        collected = list(collected_by_source.values())
        collected.sort(key=lambda item: (-float(item.get("score", 0.0)), item["target_object"]))
        atomic_json(output_path, collected)
        atomic_json(output_path.with_suffix(".attempts.json"), attempt_history)
        print(
            f"Grasp > saved={output_path} records={len(collected)} "
            f"seconds={time.time() - started:.2f}",
            flush=True,
        )
    except BaseException:
        # Kit can translate startup/runtime failures into SystemExit. Print the
        # original control-flow exception before SimulationApp.close hides it.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        if env is not None:
            try:
                env.close()
            except SystemExit:
                if not active_exception:
                    raise
        try:
            simulation_app.close()
        except SystemExit:
            if not active_exception:
                raise
