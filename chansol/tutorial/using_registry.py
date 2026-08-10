import argparse
from isaaclab.app import AppLauncher



parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app




import Register.chansol_robot as cs_robot
import torch





def main():

    env_cfg = cs_robot.ChansolRobotEnvCfg()
    env = cs_robot.ChansolRobotEnv(cfg=env_cfg)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
