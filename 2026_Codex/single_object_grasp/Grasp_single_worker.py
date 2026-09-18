"""TUI client worker. Job values are supplied by Generator_client via env."""

import os
import sys


object_name = os.environ["GRASP_OBJECT_NAME"]
gripper_name = os.environ["GRASP_GRIPPER_NAME"]
scene_num = int(os.environ["GRASP_SCENE_NUM"])
output_root = os.environ["GRASP_OUTPUT_ROOT"]

# Isaac Lab owns any arguments it adds internally; this worker exposes no CLI.
sys.argv = [sys.argv[0]]

import collect_grasps


collect_grasps.main(
    object_name=object_name,
    Gripper_name=gripper_name,
    scene_num=scene_num,
    root_path=output_root,
)
