"""Re-run saved output grasps three times with CPU PhysX.

Only ``conf/<scene>.json`` and ``output_grasp/<scene>.json`` are consumed.
The original records that pass every attempt are copied without schema changes
to ``output_grasp_verified/<scene>.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import Direct_RL_main_new_filter as collector


HEADLESS = True
VERIFICATION_REPEATS = 3
VERIFICATION_NUM_ENVS = 150


def _load_records(path: Path) -> list[dict]:
    payload = collector._load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"output_grasp root must be a list: {path}")
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise TypeError(f"output_grasp[{index}] must be an object: {path}")
    return payload


def _save_json_atomic(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=4)
    os.replace(temporary, path)


def _verification_records(records: list[dict]) -> list[dict]:
    result = []
    for index, original in enumerate(records):
        record = copy.deepcopy(original)
        record["_verification_id"] = index
        result.append(record)
    return result


def _validate_one_gripper(records: list[dict]) -> dict:
    models = {str(item.get("gripper_model", "")).strip() for item in records}
    if "" in models:
        raise KeyError("Every output_grasp record needs gripper_model")
    if len(models) != 1:
        raise ValueError(f"One scene must contain one gripper model, got {sorted(models)}")
    return {"gripper_model": models.pop(), "data": records}


def _run_until_complete(simulation_app, env, torch) -> set[int]:
    while simulation_app.is_running():
        with torch.inference_mode():
            operation = torch.ones((env.cfg.envs,), device=env.device)
            obs, _rew, _term, _trunc, _info = env.step(operation)
        if int(obs["policy"].sum().item()) == 0:
            return {
                int(record["_verification_id"])
                for record in env.act_pol.output_list
                if "_verification_id" in record
            }
    raise RuntimeError("Isaac Sim stopped before grasp verification completed")


def main(root_path, scene_num):
    root = Path(root_path)
    scene_num = int(scene_num)
    output_path = root / "output_grasp" / f"{scene_num:04d}.json"
    verified_path = root / "output_grasp_verified" / f"{scene_num:04d}.json"
    conf_path = root / "conf" / f"{scene_num:04d}.json"

    original_records = _load_records(output_path)
    if not original_records:
        _save_json_atomic(verified_path, [])
        print(f"GraspVerif > no records; saved empty result: {verified_path}")
        return

    replay_records = _verification_records(original_records)
    group = _validate_one_gripper(replay_records)
    gripper_type = collector._detect_gripper_type(group)
    gripper_info = collector._load_gripper_info(group, gripper_type)
    conf_data = collector._resolve_conf_object_usd_paths(
        collector._load_json(conf_path), collector.OBJECT_NAME_CSV_PATH
    )

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="CPU output-grasp verifier")
    AppLauncher.add_app_launcher_args(parser)
    launcher_args = parser.parse_args([])
    launcher_args.headless = HEADLESS
    launcher_args.device = "cpu"
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app
    env = None

    try:
        import torch

        if gripper_type == "hand":
            import codex_cfgs.Robot_EnvCfg_hand_platform_filter as robot_module
        else:
            import codex_cfgs.Robot_EnvCfg_finger_platform_filter as robot_module

        from cfgs.scene_cfg import platform_setup, scene_obj_setup

        env_cfg = robot_module.RobotEnvCfg()
        num_envs = min(VERIFICATION_NUM_ENVS, len(replay_records))
        env_cfg.envs = num_envs
        env_cfg.scene.num_envs = num_envs
        # Both the simulation backend and Isaac Lab tensor device must remain
        # on CPU.  This is intentionally independent of collection settings.
        env_cfg.sim.device = "cpu"

        if gripper_type == "hand":
            robot_module.Set_RobotEnvCFG(env_cfg, gripper_info, replay_records)
        else:
            robot_module.Set_RobotEnvCFG(env_cfg, gripper_info)

        scene_obj_setup(env_cfg.scene, conf_data["objects"])
        platform_setup(env_cfg.scene, conf_data["platform"])
        collector._object_frame_transformer_setup_with_asset_names(
            env_cfg.scene, conf_data["objects"]
        )
        if gripper_type == "hand":
            collector._configure_hand_contact_separation_filters(
                env_cfg.scene, conf_data["objects"]
            )

        print(
            "GraspVerif > creating CPU PhysX environment "
            f"scene={scene_num:04d} gripper={group['gripper_model']} "
            f"records={len(replay_records)} envs={num_envs}"
        )
        env = robot_module.RobotEnv(
            cfg=env_cfg,
            pre_grasp_data=replay_records,
            conf_data=conf_data,
            debug=False,
        )

        passed_all = set(range(len(original_records)))
        for attempt in range(1, VERIFICATION_REPEATS + 1):
            # Environment construction may perform internal warm-up resets.
            # Start every measured attempt from a fresh policy cursor so those
            # initialization resets can never consume a verification record.
            with torch.inference_mode():
                env.factory_reset()
            passed_this_attempt = _run_until_complete(simulation_app, env, torch)
            passed_all.intersection_update(passed_this_attempt)
            print(
                f"GraspVerif > attempt={attempt}/{VERIFICATION_REPEATS} "
                f"passed={len(passed_this_attempt)}/{len(original_records)} "
                f"passed_all={len(passed_all)}"
            )
            sys.stdout.flush()

        verified_records = [
            copy.deepcopy(record)
            for index, record in enumerate(original_records)
            if index in passed_all
        ]
        _save_json_atomic(verified_path, verified_records)
        print(
            f"GraspVerif > saved={verified_path} "
            f"verified={len(verified_records)}/{len(original_records)}"
        )
        sys.stdout.flush()
    except BaseException:
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


if __name__ == "__main__":
    raise SystemExit(
        "Run through Grasp_arg_new_filter.py grasp_verif so the dataset path is selected."
    )
