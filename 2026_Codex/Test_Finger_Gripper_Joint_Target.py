"""Interactive one-gripper joint-target test without a dataset scene.

The script spawns one articulation, drives only the selected master joint, and
leaves every follower to the mimic constraints authored in the USD.  Type a
target angle in the terminal while Isaac Sim is running to change the command.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_GRIPPER_INFO = Path(
    "/nas/ochansol/gripper_info/gripper_info_new_2026.json"
)


parser = argparse.ArgumentParser(description="Interactive finger-gripper joint test")
parser.add_argument("--gripper", default="OnRobot_RG6_v1_2", type=str)
parser.add_argument("--gripper-info", default=str(DEFAULT_GRIPPER_INFO), type=str)
parser.add_argument(
    "--master-joint",
    default=None,
    type=str,
    help="Master joint name; defaults to the first joint_cfg entry.",
)
parser.add_argument("--initial-deg", default=0.0, type=float)
parser.add_argument("--physics-hz", default=120.0, type=float)
parser.add_argument("--stiffness", default=None, type=float)
parser.add_argument("--damping", default=None, type=float)
parser.add_argument("--effort-limit", default=None, type=float)
parser.add_argument("--position-iterations", default=20, type=int)
parser.add_argument("--velocity-iterations", default=3, type=int)
parser.add_argument("--dump-mimic", action="store_true")
parser.add_argument("--dump-usd-structure", action="store_true")
parser.add_argument("--dump-contact-sensor", action="store_true")
parser.add_argument(
    "--drive-mimic-followers",
    action="store_true",
    help="Add matching auxiliary drives to the three authored mimic followers.",
)
parser.add_argument(
    "--remove-mimic",
    action="store_true",
    help="Remove the three mimic APIs before physics starts (comparison test).",
)
parser.add_argument(
    "--solve-articulation-contact-last",
    action="store_true",
    help="Enable only PhysX solve_articulation_contact_last on the base config.",
)
parser.add_argument(
    "--enable-ccd",
    action="store_true",
    help="Enable only PhysX CCD on the base config.",
)
parser.add_argument(
    "--min-velocity-iterations",
    default=None,
    type=int,
    help="Override the PhysX-scene minimum velocity iteration count.",
)
parser.add_argument(
    "--match-grasp-physics",
    action="store_true",
    help="Use the grasp collector's 1/700 s PhysX configuration.",
)
parser.add_argument(
    "--floating-root",
    action="store_true",
    help="Disable the USD world anchor and write a fixed root pose every step.",
)
parser.add_argument(
    "--kinematic-root-body",
    action="store_true",
    help="Make the gripper's body root kinematic after disabling its world anchor.",
)
parser.add_argument("--write-fixed-root-pose", action="store_true")
parser.add_argument("--root-offset-x", default=0.0, type=float)
parser.add_argument(
    "--print-every",
    default=120,
    type=int,
    help="Periodically print all measured joint positions; 0 disables it.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from codex_cfgs.grasp_physics_settings import make_grasp_simulation_cfg


def _optional_number(value):
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def _load_gripper_info() -> tuple[dict, str]:
    path = Path(args_cli.gripper_info).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        database = json.load(stream)
    if args_cli.gripper not in database:
        raise KeyError(f"{args_cli.gripper!r} was not found in {path}")

    info = database[args_cli.gripper]
    joint_cfg = info.get("joint_cfg", {})
    if not joint_cfg:
        raise KeyError(f"{args_cli.gripper} has no joint_cfg")
    master_name = args_cli.master_joint or next(iter(joint_cfg))
    if master_name not in joint_cfg:
        raise KeyError(
            f"Master joint {master_name!r} is not in joint_cfg={list(joint_cfg)}"
        )
    return info, master_name


def _input_worker(commands: queue.Queue[str]) -> None:
    while True:
        try:
            command = input("target-deg> ").strip()
        except EOFError:
            command = "quit"
        commands.put(command)
        if command.lower() in {"q", "quit", "exit"}:
            return


def _format_joint_positions(robot: Articulation) -> str:
    positions = robot.data.joint_pos[0].detach().cpu().tolist()
    return "  ".join(
        f"{name}={math.degrees(float(value)):+.3f} deg"
        for name, value in zip(robot.joint_names, positions)
    )


def _disable_world_anchor(robot_root_path: str) -> None:
    from pxr import Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    root = stage.GetPrimAtPath(robot_root_path)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if bool(body0) != bool(body1):
            joint.CreateJointEnabledAttr(False).Set(False)


def _set_root_body_kinematic(robot_root_path: str) -> None:
    from pxr import Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    root = get_current_stage().GetPrimAtPath(robot_root_path)
    rigid_bodies = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    child_bodies = set()
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetBody0Rel().GetTargets():
            child_bodies.update(joint.GetBody1Rel().GetTargets())
    candidates = [prim for prim in rigid_bodies if prim.GetPath() not in child_bodies]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one root rigid body, found {[str(p.GetPath()) for p in candidates]}"
        )
    body = candidates[0]
    UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True).Set(True)
    print(f"kinematic root body: {body.GetPath()}")


def _dump_mimic_properties(robot_root_path: str) -> None:
    from pxr import Usd
    from isaaclab.sim.utils.stage import get_current_stage

    root = get_current_stage().GetPrimAtPath(robot_root_path)
    for prim in Usd.PrimRange(root):
        names = [prop.GetName() for prop in prim.GetProperties()]
        mimic_names = [name for name in names if "mimic" in name.lower()]
        if not mimic_names:
            continue
        print(f"mimic prim: {prim.GetPath()}")
        for name in mimic_names:
            prop = prim.GetProperty(name)
            value = prop.Get() if hasattr(prop, "Get") else prop.GetTargets()
            print(f"  {name}={value}")


def _dump_usd_structure(robot_root_path: str) -> None:
    from pxr import Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    root = get_current_stage().GetPrimAtPath(robot_root_path)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"articulation root: {prim.GetPath()} ({prim.GetTypeName()})")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid = UsdPhysics.RigidBodyAPI(prim)
            print(
                f"rigid body: {prim.GetPath()} ({prim.GetTypeName()}) "
                f"kinematic={rigid.GetKinematicEnabledAttr().Get()}"
            )
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        line = (
            f"joint: {prim.GetPath()} ({prim.GetTypeName()}) "
            f"enabled={joint.GetJointEnabledAttr().Get()} "
            f"body0={joint.GetBody0Rel().GetTargets()} "
            f"body1={joint.GetBody1Rel().GetTargets()}"
        )
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute = UsdPhysics.RevoluteJoint(prim)
            line += (
                f" axis={revolute.GetAxisAttr().Get()} "
                f"limits=({revolute.GetLowerLimitAttr().Get()},"
                f"{revolute.GetUpperLimitAttr().Get()})"
            )
        print(line)


def _dump_contact_sensor_matching(robot_root_path: str, relative_pattern: str) -> None:
    """Print the same parent/leaf matching used by IsaacLab ContactSensor."""
    from pxr import PhysxSchema, UsdPhysics
    import isaaclab.sim as sim_utils

    full_pattern = f"{robot_root_path}/{relative_pattern.strip('/')}"
    parent_pattern, leaf_pattern = full_pattern.rsplit("/", 1)
    parents = sim_utils.find_matching_prims(parent_pattern)
    matches = sim_utils.find_matching_prims(full_pattern)
    print(f"contact input pattern: {full_pattern}")
    print(f"contact parent pattern: {parent_pattern}")
    print(f"contact matched parents ({len(parents)}):")
    for prim in parents:
        print(f"  {prim.GetPath()}")
    print(f"contact matched leaves ({len(matches)}):")
    for prim in matches:
        print(
            f"  {prim.GetPath()} "
            f"rigid_body={prim.HasAPI(UsdPhysics.RigidBodyAPI)} "
            f"contact_report={prim.HasAPI(PhysxSchema.PhysxContactReportAPI)}"
        )

    if not parents:
        return
    template_pattern = f"{parents[0].GetPath()}/{leaf_pattern}"
    template_matches = sim_utils.find_matching_prims(template_pattern)
    reporter_names = [
        prim.GetName()
        for prim in template_matches
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
    ]
    print(f"contact template pattern: {template_pattern}")
    print(f"contact template reporter names: {reporter_names}")


def _remove_mimic_apis(robot_root_path: str, joint_names: set[str] | None = None) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    root = get_current_stage().GetPrimAtPath(robot_root_path)
    removed = 0
    for prim in Usd.PrimRange(root):
        if joint_names is not None and prim.GetName() not in joint_names:
            continue
        if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotX") and prim.RemoveAPI(
            PhysxSchema.PhysxMimicJointAPI, "rotX"
        ):
            removed += 1
        if joint_names is None or prim.GetName() in joint_names:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(300.0)
            drive.CreateDampingAttr(10.0)
            drive.CreateMaxForceAttr(600.0)
    print(f"removed mimic APIs: {removed}")


def main() -> None:
    gripper_info, master_name = _load_gripper_info()
    master_cfg = gripper_info["joint_cfg"][master_name]

    actuators = {
        "master": ImplicitActuatorCfg(
            joint_names_expr=[master_name],
            effort_limit_sim=(
                args_cli.effort_limit
                if args_cli.effort_limit is not None
                else _optional_number(master_cfg.get("effort_limit"))
            ),
            velocity_limit_sim=_optional_number(master_cfg.get("velocity_limit")),
            stiffness=(
                args_cli.stiffness
                if args_cli.stiffness is not None
                else _optional_number(master_cfg.get("stiffness"))
            ),
            damping=(
                args_cli.damping
                if args_cli.damping is not None
                else _optional_number(master_cfg.get("damping"))
            ),
        )
    }
    if args_cli.drive_mimic_followers:
        actuators["mimic_followers"] = ImplicitActuatorCfg(
            joint_names_expr=["R_joint", "L_finger_joint", "R_finger_joint"],
            effort_limit_sim=600.0,
            velocity_limit_sim=None,
            stiffness=300.0,
            damping=10.0,
        )
    sim_cfg = (
        make_grasp_simulation_cfg()
        if args_cli.match_grasp_physics
        else SimulationCfg(dt=1.0 / float(args_cli.physics_hz), device=args_cli.device)
    )
    if not args_cli.match_grasp_physics:
        sim_cfg.physx.solve_articulation_contact_last = (
            args_cli.solve_articulation_contact_last
        )
        sim_cfg.physx.enable_ccd = args_cli.enable_ccd
        if args_cli.min_velocity_iterations is not None:
            sim_cfg.physx.min_velocity_iteration_count = (
                args_cli.min_velocity_iterations
            )
    sim_cfg.device = args_cli.device
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((0.65, 0.65, 0.45), (0.0, 0.0, 0.15))

    light_cfg = sim_utils.DomeLightCfg(
        intensity=2500.0, color=(0.8, 0.8, 0.8)
    )
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=gripper_info["usd_path"],
            activate_contact_sensors=args_cli.dump_contact_sensor,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=not args_cli.floating_root,
                enabled_self_collisions=False,
                solver_position_iteration_count=args_cli.position_iterations,
                solver_velocity_iteration_count=args_cli.velocity_iterations,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                name: float(value)
                for name, value in gripper_info.get("init_joint_pos", {}).items()
            }
        ),
        actuators=actuators,
    )
    robot = Articulation(robot_cfg)
    contact_sensor = None
    if args_cli.dump_contact_sensor:
        contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=(
                    f"/World/Gripper/"
                    f"{gripper_info['contact_sensor_prim_path'].strip('/')}"
                ),
                update_period=0.0,
                history_length=10,
            )
        )
    if args_cli.remove_mimic:
        _remove_mimic_apis("/World/Gripper")
    if args_cli.dump_mimic:
        _dump_mimic_properties("/World/Gripper")
    if args_cli.dump_usd_structure:
        _dump_usd_structure("/World/Gripper")
    if args_cli.dump_contact_sensor:
        _dump_contact_sensor_matching(
            "/World/Gripper", gripper_info["contact_sensor_prim_path"]
        )
    if args_cli.floating_root:
        _disable_world_anchor("/World/Gripper")
    if args_cli.kinematic_root_body:
        _set_root_body_kinematic("/World/Gripper")

    sim.reset()
    robot.reset()
    if contact_sensor is not None:
        print(
            "contact sensor PhysX view: "
            f"num_envs={contact_sensor._num_envs} "
            f"num_bodies={contact_sensor.num_bodies} "
            f"body_names={contact_sensor.body_names} "
            f"net_forces_shape={tuple(contact_sensor.data.net_forces_w.shape)}"
        )
    sim_dt = sim.get_physics_dt()
    held_root_pose = robot.data.default_root_state[:, :7].clone()
    held_root_pose[:, 0] += float(args_cli.root_offset_x)
    zero_root_velocity = torch.zeros_like(robot.data.default_root_state[:, 7:])

    master_ids, _ = robot.find_joints(master_name)
    if len(master_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one {master_name!r}, found indices {master_ids}"
        )
    master_id = int(master_ids[0])
    commanded_joint_ids = [master_id]
    commanded_joint_scales = [1.0]
    if args_cli.drive_mimic_followers:
        follower_scales = {
            "R_joint": 1.0,
            "L_finger_joint": -1.0,
            "R_finger_joint": -1.0,
        }
        for name, scale in follower_scales.items():
            ids, _ = robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"Expected one {name}, found {ids}")
            commanded_joint_ids.append(int(ids[0]))
            commanded_joint_scales.append(scale)
    target_deg = float(args_cli.initial_deg)
    target_rad = math.radians(target_deg)

    commands: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_input_worker, args=(commands,), daemon=True).start()

    print("\nInteractive gripper test ready")
    print(f"  gripper : {args_cli.gripper}")
    print(f"  USD     : {gripper_info['usd_path']}")
    print(f"  master  : {master_name} (index {master_id})")
    print(f"  joints  : {robot.joint_names}")
    print("Commands: <degrees>, status, reset, quit")
    print("Only the master receives a drive target; USD mimic drives the followers.\n")

    frame = 0
    running = True
    while simulation_app.is_running() and running:
        while True:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                break

            lowered = command.lower()
            if lowered in {"q", "quit", "exit"}:
                running = False
                break
            if lowered == "status":
                print(
                    f"target {master_name}={target_deg:+.3f} deg | "
                    + _format_joint_positions(robot)
                    + " | root="
                    + str(robot.data.root_pose_w[0].detach().cpu().tolist())
                )
                continue
            if lowered == "reset":
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = torch.zeros_like(robot.data.default_joint_vel)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                target_deg = float(args_cli.initial_deg)
                target_rad = math.radians(target_deg)
                print(f"reset complete; target={target_deg:+.3f} deg")
                continue
            if not command:
                continue
            try:
                target_deg = float(command)
            except ValueError:
                print(f"Unknown command: {command!r}")
                continue
            target_rad = math.radians(target_deg)
            print(f"new target: {master_name}={target_deg:+.3f} deg")

        if not running:
            break

        target = torch.tensor(
            [[target_rad * scale for scale in commanded_joint_scales]],
            dtype=torch.float,
            device=robot.device,
        )
        robot.set_joint_position_target(target, joint_ids=commanded_joint_ids)
        if args_cli.floating_root or args_cli.write_fixed_root_pose:
            robot.write_root_pose_to_sim(held_root_pose)
            robot.write_root_velocity_to_sim(zero_root_velocity)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        frame += 1
        if args_cli.print_every > 0 and frame % args_cli.print_every == 0:
            print(
                f"target {master_name}={target_deg:+.3f} deg | "
                + _format_joint_positions(robot)
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
