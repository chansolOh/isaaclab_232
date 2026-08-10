"""Argument wrapper for hand grasp collection with robot-platform collision filtering."""

import argparse
import os
import sys


parser = argparse.ArgumentParser(description="Grasp Generator (finger + hand)")
parser.add_argument(
    "--root_path",
    default="/nas/Dataset/Dataset_2026/dataset_v2",
    type=str,
)
parser.add_argument("--env_name", default="Logistic_site", type=str)
parser.add_argument("--section_name", default="General_LogisticSite", type=str)
parser.add_argument("--platform_name", default="conveyor_track_01", type=str)
parser.add_argument("--scene_start", default=0, type=int)
parser.add_argument("--pre_grasp_index", default=0, type=int)
args = parser.parse_args()
root_path = os.path.join(
    args.root_path,
    args.env_name,
    args.section_name,
    args.platform_name,
)

# Isaac Lab's launcher owns the remaining CLI parsing, matching Grasp_arg.py.
sys.argv = [sys.argv[0]]

import Direct_RL_main_new_filter as DRm

DRm.main(
    root_path=root_path,
    scene_num=args.scene_start,
    pre_grasp_index=args.pre_grasp_index,
)
