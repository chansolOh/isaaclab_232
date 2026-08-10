import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaacsim.core.utils.rotations import quat_to_euler_angles

@configclass
class SceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    platform = AssetBaseCfg(
        prim_path="/World/envs/env_.*/platform", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    obj00 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj00", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj01 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj01", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj02 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj02", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj03 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj03", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj04 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj04", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj05 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj05", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj06 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj06", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj07 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj07", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj08 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj08", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )
    obj09 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/obj09", spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        )
    )

    contact_sensor = ContactSensorCfg(
        prim_path="", update_period=0.0, history_length=10, debug_vis=True,        
    )

    fixed_prim = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/fixed_prim",
        spawn = sim_utils.CuboidCfg(
            size=(0.1,0.1,0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            visible=False,
        ),
    )

    transformer = FrameTransformerCfg(prim_path=fixed_prim.prim_path, debug_vis=False)


def platform_setup(scene : SceneCfg, platform_conf):

    scene.platform.init_state.pos = platform_conf["translate"]
    scene.platform.init_state.rot = platform_conf["orient"]
    scene.platform.spawn.scale = platform_conf["scale"]
    scene.platform.spawn.usd_path = platform_conf["usd_path"]



def scene_obj_setup(scene : SceneCfg, obj_conf):
    for i in range(10):
        if len(obj_conf)>i:
            item = getattr(scene, f"obj{i:02d}")
            # item.spawn.usd_path = obj_conf[i]["usd_path"].replace("peel3_scan_data_2025","peel3_scan_data_2025_rigid") if "_2024" not in obj_conf[i]["usd_path"] else \
            #     obj_conf[i]["usd_path"].replace("peel3_scan_data_2024","peel3_scan_data_2024_rigid")
            item.spawn.usd_path = obj_conf[i]["usd_path"].replace(".usd", "_rigid.usd")
            item.class_name     = obj_conf[i]["class"]
            item.pos            = obj_conf[i]["translate"]
            item.quat           = obj_conf[i]["orient"]
            item.rot            = quat_to_euler_angles(obj_conf[i]["orient"])
            item.scale          = obj_conf[i]["scale"]
        else :
            setattr(scene, f"obj{i:02d}", None)

def object_frame_transformer_setup(scene : SceneCfg, target_obj_class_name_list):
    for i in range(10):
        if getattr(scene, f"obj{i:02d}") is not None:
            item = getattr(scene, f"obj{i:02d}")
            for target_obj_class_name in target_obj_class_name_list:
                if item.class_name == target_obj_class_name:
                    frame = getattr(scene, f"transformer")
                    try:
                        frame.target_frames.append(FrameTransformerCfg.FrameCfg(prim_path= f"/World/envs/env_.*/obj{i:02d}/{item.class_name}"))
                    except:
                        frame.target_frames = [FrameTransformerCfg.FrameCfg(prim_path= f"/World/envs/env_.*/obj{i:02d}/{item.class_name}")]

                








    # for i in range(10):
    #     if len(obj_conf)>i:
    #         item = getattr(scene, f"obj{i:02d}")
    #         if item.class_name == target_obj_class_name:
    #             getattr(scene, f"object_contact_sensor{i:02d}").prim_path = f"/World/envs/env_.*/obj{i:02d}/{item.class_name}"
    #         else:
    #             setattr(scene, f"object_contact_sensor{i:02d}", None)
    #     else :
    #         setattr(scene, f"object_contact_sensor{i:02d}", None)
        






    # contact_sensor_left = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/Body/gripper/onrobot_2fg_14/.*", update_period=0.0, history_length=6, debug_vis=True,        
    # )

    # chansol_robot = CRC.CHANSOL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # # robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Body/gripper/onrobot_2fg_14/Left", update_period=0.0, history_length=6, debug_vis=True
    # )