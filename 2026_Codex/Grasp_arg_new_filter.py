"""Argument wrapper for grasp collection and output-grasp verification."""

import argparse
import os
import sys


parser = argparse.ArgumentParser(description="Grasp Generator / Verifier (finger + hand)")
parser.add_argument(
    "command",
    nargs="?",
    default="grasp",
    choices=("grasp", "grasp_verif"),
    help="grasp: collect from pre_grasp; grasp_verif: verify output_grasp on CPU",
)
parser.add_argument(
    "--root_path",
    default="/nas/Dataset/Dataset_2026/dataset_v2",
    type=str,
)
parser.add_argument("--env_name", default="Manufactory", type=str)
parser.add_argument("--section_name", default="Seongju_Melon_Processing_Facility", type=str)
parser.add_argument("--platform_name", default="coating_machine_01", type=str)
parser.add_argument("--scene_start", default=7, type=int)
parser.add_argument("--pre_grasp_index", default=0, type=int, help=argparse.SUPPRESS)
args = parser.parse_args()
root_path = os.path.join(
    args.root_path,
    args.env_name,
    args.section_name,
    args.platform_name,
)

# Isaac Lab's launcher owns the remaining CLI parsing, matching Grasp_arg.py.
sys.argv = [sys.argv[0]]
 
if args.command == "grasp_verif":
    import Verify_Output_Grasp_CPU as runner
else:
    import Direct_RL_main_new_filter as runner

runner.main(root_path=root_path, scene_num=args.scene_start)
