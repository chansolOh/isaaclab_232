"""Central physics tuning values for hand and finger grasp collection."""

from isaaclab.sim import PhysxCfg, RigidBodyPropertiesCfg, SimulationCfg


# Simulation timing
SIM_DT = 1.0 / 700.0
RENDER_INTERVAL =4

# /physicsScene PhysX solver settings
PHYSX_SOLVER_TYPE = 1  # 0: PGS, 1: TGS
PHYSX_MIN_POSITION_ITERATION_COUNT = 1
PHYSX_MAX_POSITION_ITERATION_COUNT = 255
PHYSX_MIN_VELOCITY_ITERATION_COUNT = 1
PHYSX_MAX_VELOCITY_ITERATION_COUNT = 255
PHYSX_SOLVE_ARTICULATION_CONTACT_LAST = True
PHYSX_ENABLE_CCD = True

# Per-articulation solver settings. The physics-scene minimum values above
# clamp these upward during simulation, but they remain here for easy tuning.
ARTICULATION_POSITION_ITERATION_COUNT = 20
ARTICULATION_VELOCITY_ITERATION_COUNT = 3

# Dynamic scene-object limits.  Scenes are settled with CPU PhysX during
# generation, while grasp collection uses GPU PhysX.  Small differences in
# convex contact penetration can otherwise create a very large separation
# impulse on the first GPU physics step.
SCENE_OBJECT_MAX_DEPENETRATION_VELOCITY = 0.5
SCENE_OBJECT_MAX_LINEAR_VELOCITY = 2.5
SCENE_OBJECT_MAX_ANGULAR_VELOCITY = 300.0
SCENE_OBJECT_LINEAR_DAMPING = 0.7

# GPU PhysX buffer settings
GPU_MAX_RIGID_PATCH_COUNT = 2**18 * 2**4
GPU_TEMP_BUFFER_CAPACITY = 2**24 * 2**2
GPU_MAX_RIGID_CONTACT_COUNT = 2**23 * 2**2
GPU_HEAP_CAPACITY = 2**26 * 2**2
GPU_FOUND_LOST_PAIRS_CAPACITY = 2**21 * 2**2
GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY = 2**25 * 2**2
GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 2**21 * 2**2


def make_scene_object_rigid_props() -> RigidBodyPropertiesCfg:
    """Return GPU-safe limits for dataset objects, excluding robot/platform."""
    return RigidBodyPropertiesCfg(
        max_depenetration_velocity=SCENE_OBJECT_MAX_DEPENETRATION_VELOCITY,
        max_linear_velocity=SCENE_OBJECT_MAX_LINEAR_VELOCITY,
        max_angular_velocity=SCENE_OBJECT_MAX_ANGULAR_VELOCITY,
        linear_damping=SCENE_OBJECT_LINEAR_DAMPING,
    )


def configure_scene_object_rigid_props(scene_cfg) -> None:
    """Attach the scene-object limits before InteractiveScene spawns assets."""
    for index in range(10):
        object_cfg = getattr(scene_cfg, f"obj{index:02d}", None)
        if object_cfg is not None:
            object_cfg.spawn.rigid_props = make_scene_object_rigid_props()


def make_grasp_physx_cfg() -> PhysxCfg:
    return PhysxCfg(
        solver_type=PHYSX_SOLVER_TYPE,
        solve_articulation_contact_last=PHYSX_SOLVE_ARTICULATION_CONTACT_LAST,
        enable_ccd=PHYSX_ENABLE_CCD,
        min_position_iteration_count=PHYSX_MIN_POSITION_ITERATION_COUNT,
        max_position_iteration_count=PHYSX_MAX_POSITION_ITERATION_COUNT,
        min_velocity_iteration_count=PHYSX_MIN_VELOCITY_ITERATION_COUNT,
        max_velocity_iteration_count=PHYSX_MAX_VELOCITY_ITERATION_COUNT,
        gpu_max_rigid_patch_count=GPU_MAX_RIGID_PATCH_COUNT,
        gpu_temp_buffer_capacity=GPU_TEMP_BUFFER_CAPACITY,
        gpu_max_rigid_contact_count=GPU_MAX_RIGID_CONTACT_COUNT,
        gpu_heap_capacity=GPU_HEAP_CAPACITY,
        gpu_found_lost_pairs_capacity=GPU_FOUND_LOST_PAIRS_CAPACITY,
        gpu_found_lost_aggregate_pairs_capacity=(
            GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY
        ),
        gpu_total_aggregate_pairs_capacity=GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY,
    )


def make_grasp_simulation_cfg() -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        render_interval=RENDER_INTERVAL,
        physx=make_grasp_physx_cfg(),
    )
