import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

@configclass
class SceneCfg(InteractiveSceneCfg):

    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    # )

    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    obj00 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/obj00",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale = (0.1,0.1,0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
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
    for i in range(1):
        if len(obj_conf)>i:
            item = getattr(scene, f"obj{i:02d}")
            # item.spawn.usd_path = obj_conf[i]["usd_path"].replace("peel3_scan_data_2025","peel3_scan_data_2025_rigid") if "_2024" not in obj_conf[i]["usd_path"] else \
            #     obj_conf[i]["usd_path"].replace("peel3_scan_data_2024","peel3_scan_data_2024_rigid")
            item.spawn.usd_path = obj_conf[i]["usd_path"].replace(".usd", "_rigid.usd")
            item.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
            item.class_name     = obj_conf[i]["class"]
            item.spawn.scale    = obj_conf[i]["scale"]
            item.init_state.pos = obj_conf[i]["translate"]
            item.init_state.rot = euler_angles_to_quat(obj_conf[i]["orient"], degrees=True)
        else :
            setattr(scene, f"obj{i:02d}", None)

def object_frame_transformer_setup(scene : SceneCfg, target_obj_class_name_list):
    for i in range(1):
        if getattr(scene, f"obj{i:02d}") is not None:
            item = getattr(scene, f"obj{i:02d}")
            for target_obj_class_name in target_obj_class_name_list:
                if item.class_name == target_obj_class_name:
                    frame = getattr(scene, f"transformer")
                    try:
                        frame.target_frames.append(FrameTransformerCfg.FrameCfg(prim_path= f"/World/envs/env_.*/obj{i:02d}/{item.class_name}"))
                    except:
                        frame.target_frames = [FrameTransformerCfg.FrameCfg(prim_path= f"/World/envs/env_.*/obj{i:02d}/{item.class_name}")]

                
