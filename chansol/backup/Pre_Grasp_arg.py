import argparse


parser = argparse.ArgumentParser(description="pre-Grasp Generator")
parser.add_argument("--output_root_path",       default="/nas/Dataset/Dataset_2025/test",              help="", type=str)
parser.add_argument("--env_name",       default="Manufactory",              help="", type=str)
parser.add_argument("--section_name",   default="FOODnamoo_poultry_plant",  help="", type=str)
parser.add_argument("--platform_name",  default="catch_table",              help="", type=str)
parser.add_argument("--scene_start",     default=0,                           help="", type=int)



args = parser.parse_args()


import pre_grasp_point_sampler as PGS

PGS.main(root_path=args.output_root_path,
         env=args.env_name,
         section=args.section_name,
         platform=args.platform_name,
         scene_num=args.scene_start)
