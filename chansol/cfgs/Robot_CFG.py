# import json

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets.articulation import ArticulationCfg





# def Get_ArticulationCfg_empty() -> ArticulationCfg :
    
#     return ArticulationCfg(
#             spawn=sim_utils.UsdFileCfg(
#                 usd_path="",
#                 activate_contact_sensors=False,
#                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                     disable_gravity=False,
#                     max_depenetration_velocity=5.0,
#                 ),
#                 articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#                     enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
#                 ),
#                 # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#             ),
#             init_state=ArticulationCfg.InitialStateCfg(
#                 joint_pos={},
#             ),
#             actuators={}
#         )





# Custom_onrobot = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/nas/ochansol/isaac/sanjabu/Robot/Custom_onrobot/Custom_onrobot.usd",
#         activate_contact_sensors=False,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=5.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
#         ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             "lift_joint": -0.5,
#             "right_joint": 0.0,
#             "left_joint": 0.0,
#         },
#     ),
#     actuators={
#         "Right_finger": ImplicitActuatorCfg(
#             joint_names_expr=["right_joint"],
#             effort_limit=50.0,
#             # velocity_limit=2.175,
#             stiffness=2000.0,
#             damping=150.0,
#         ),
#         "Left_figner": ImplicitActuatorCfg(
#             joint_names_expr=["left_joint"],
#             effort_limit=50.0,
#             # velocity_limit=2.61,
#             stiffness=2000.0,
#             damping=150.0,
#         ),
#         "Lift": ImplicitActuatorCfg(
#             joint_names_expr=["lift_joint"],
#             effort_limit=200.0,
#             # velocity_limit=0.2,
#             stiffness=1000,
#             damping=130,
#         ),
#     },
# )



# Custom_GEP2016IO = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/nas/ochansol/isaac/sanjabu/Robot/Custom_GEP2016IO/Custom_GEP2016IO.usd",
#         activate_contact_sensors=False,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=5.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
#         ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             "lift_joint": -0.5,
#             "right_joint": 0.0,
#             "left_joint": 0.0,
#         },
#     ),
#     actuators={
#         "Right_finger": ImplicitActuatorCfg(
#             joint_names_expr=["right_joint"],
#             effort_limit=50.0,
#             # velocity_limit=2.175,
#             stiffness=2000.0,
#             damping=150.0,
#         ),
#         "Left_figner": ImplicitActuatorCfg(
#             joint_names_expr=["left_joint"],
#             effort_limit=50.0,
#             # velocity_limit=2.61,
#             stiffness=2000.0,
#             damping=150.0,
#         ),
#         "Lift": ImplicitActuatorCfg(
#             joint_names_expr=["lift_joint"],
#             effort_limit=200.0,
#             # velocity_limit=0.2,
#             stiffness=1000,
#             damping=130,
#         ),
#     },
# )