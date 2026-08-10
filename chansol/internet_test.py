import time
programe_init_time  = time.time()
import argparse
from isaaclab.app import AppLauncher



parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--scene_num", type=int, default=0, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


debug = False
args_cli.headless = True



app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print(f"Program processing time : {time.time()-programe_init_time}")