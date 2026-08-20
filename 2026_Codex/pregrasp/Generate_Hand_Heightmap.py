"""Generate preset-specific hand bottom height maps inside Isaac Sim 5.1.

Edit the configuration variables below, then run this file with the Isaac Sim
Python environment.  Every gripper and preset in the JSON is processed.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEADLESS = True

GRIPPER_INFO_PATH = "/nas/ochansol/gripper_info/gripper_info_hand_2026.json"

GRIPPER_ROOT_PRIM_PATH = "/World/PreGraspHeightMap/Hand"
TCP_PRIM_PATH = "/World/PreGraspHeightMap/TCP"

# Grid resolution in metres. 0.002 means 2 mm.
HEIGHT_MAP_RESOLUTION = 0.002
SETTLE_STEPS = 3

# Created beside GRIPPER_INFO_PATH:
#   heightmaps/<gripper name>/<preset name>.npz
HEIGHT_MAP_FOLDER_NAME = "heightmaps"


from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": HEADLESS})

import sys
from pathlib import Path
import re

import numpy as np
import omni.usd
from isaacsim.core.api import World


sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand_pregrasp_heightmap as heightmap


def safe_path_component(name: str) -> str:
    """Convert a JSON name into one safe directory/file component."""
    component = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("._")
    if not component:
        raise ValueError(f"Cannot create a path component from {name!r}")
    return component


def height_map_output_path(gripper_name: str, preset_name: str) -> Path:
    info_path = Path(GRIPPER_INFO_PATH).expanduser().resolve()
    return (
        info_path.parent
        / HEIGHT_MAP_FOLDER_NAME
        / safe_path_component(gripper_name)
        / f"{safe_path_component(preset_name)}.npz"
    )


def main() -> None:
    database = heightmap.load_hand_database(GRIPPER_INFO_PATH)
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    completed: list[Path] = []
    failures: list[tuple[str, str, Exception]] = []

    try:
        for gripper_name, gripper in database.items():
            presets = gripper.get("preset", [])
            if not presets:
                print(f"Skip gripper without presets: {gripper_name}")
                continue

            for preset in presets:
                preset_name = preset.get("name", "<unnamed>")
                print(f"Generating: gripper={gripper_name}, preset={preset_name}")
                try:
                    setup = heightmap.instantiate_preset_start_gripper(
                        stage,
                        gripper,
                        preset,
                        GRIPPER_ROOT_PRIM_PATH,
                        TCP_PRIM_PATH,
                        replace_existing=True,
                    )
                    world.reset()
                    heightmap.synchronize_articulation_start_pose(
                        setup["articulation_root_path"], preset
                    )
                    for _ in range(max(1, SETTLE_STEPS)):
                        world.step(render=False)

                    output_path = heightmap.build_archive_from_current_stage(
                        stage,
                        GRIPPER_ROOT_PRIM_PATH,
                        TCP_PRIM_PATH,
                        height_map_output_path(gripper_name, preset_name),
                        gripper_name,
                        preset,
                        HEIGHT_MAP_RESOLUTION,
                    )
                    archive = heightmap.load_height_map_archive(output_path)
                    heightmap.validate_height_map_archive_preset(
                        archive, preset, archive_path=output_path
                    )
                    maps = archive["bottom_height_maps"]
                    completed.append(output_path)
                    print(
                        f"Saved: {output_path} | maps={maps.shape[0]}, "
                        f"grid={maps.shape[1]}x{maps.shape[2]}, "
                        f"valid_cells={int(np.isfinite(maps).sum())}"
                    )
                except Exception as error:
                    failures.append((gripper_name, preset_name, error))
                    print(
                        f"Failed: gripper={gripper_name}, preset={preset_name}: "
                        f"{type(error).__name__}: {error}"
                    )
    finally:
        world.stop()

    print(
        f"Height-map generation finished: {len(completed)} saved, "
        f"{len(failures)} failed"
    )
    if failures:
        details = "; ".join(
            f"{gripper}/{preset}: {error}" for gripper, preset, error in failures
        )
        raise RuntimeError(f"Some height maps could not be generated: {details}")


try:
    main()
finally:
    simulation_app.close()
