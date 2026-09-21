"""Argument wrapper for hand grasp collection with robot-platform collision filtering."""

import argparse
import os
import sys


parser = argparse.ArgumentParser(description="Grasp Generator (finger + hand)")
parser.add_argument(
    "--root_path",
    default="/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen",
    type=str,
)
parser.add_argument("--object_name", default="black_pepper_shaker", type=str)
parser.add_argument("--Gripper_name", default="Inspire-F1_right", type=str)
parser.add_argument("--scene_start", default=0, type=int)

args = parser.parse_args()


# Isaac Lab's launcher owns the remaining CLI parsing, matching Grasp_arg.py.
sys.argv = [sys.argv[0]]
 
import collect_grasps as DRm

DRm.main(
    object_name=args.object_name,
    Gripper_name=args.Gripper_name,
    scene_num=args.scene_start,
    root_path=args.root_path,
)
