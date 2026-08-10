import argparse
import os

parser = argparse.ArgumentParser(description="Grasp Generator")
parser.add_argument("--root_path",       default="/nas/Dataset/Dataset_2025/dataset_v1",help="", type=str)
parser.add_argument("--env_name",       default="Logistic_site",              help="", type=str)
parser.add_argument("--section_name",   default="FOODnamoo",  help="", type=str)
parser.add_argument("--platform_name",  default="roller_conveyor_04",              help="", type=str)
parser.add_argument("--scene_start",     default=1042,                           help="", type=int)
parser.add_argument("--pre_grasp_index",     default=1,                           help="", type=int)


args = parser.parse_args()



root_path = os.path.join(args.root_path, args.env_name, args.section_name, args.platform_name)
scene_num = args.scene_start
pre_grasp_index = args.pre_grasp_index
import sys
sys.argv = [sys.argv[0]]  # Clear the command line arguments to avoid conflicts with the main script



import Direct_RL_main as DRm
DRm.main( 
        root_path = root_path,
        scene_num    = scene_num,
        pre_grasp_index = pre_grasp_index)

