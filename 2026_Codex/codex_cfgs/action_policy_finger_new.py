"""Calibrated two/three-finger policy for the 2026 grasp collector.

Stages are deliberately shared with the hand workflow:

0. approach the pre-grasp root pose from above,
1. close the physical finger joints while applying calibrated root Z-hop,
2. keep the gripper fixed and lower the platform,
3. evaluate and finish.

There is no post-grasp random/external-force stress test.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import torch
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

from .finger_gripper_calibration import (
    close_joint_degrees,
    evaluate_polynomial,
    uses_direct_prismatic_width,
    validate_calibration,
)


class FingerActionPolicy:
    def __init__(
        self,
        env_num,
        pre_grasp_data,
        conf_data,
        gripper_info,
        joint_names,
        joint_index,
        joint_pos,
        step_dt,
        device,
        contact_sensor,
        contact_sensor_rows_by_env,
        frame_transformer,
        env_origin,
        debug=False,
        width_command_duration_sec=0.25,
        width_ready_timeout_sec=1.0,
        width_ready_revolute_error_rad=math.radians(2.0),
        width_ready_prismatic_error_m=0.001,
        approach_duration_sec=0.35,
        approach_height=0.10,
        close_timeout_sec=1.5,
        platform_drop_height=0.70,
        target_drop_failure_distance=0.20,
        platform_settle_duration_sec=1.0,
    ):
        self.env_num = int(env_num)
        self.pre_grasp_data = pre_grasp_data
        self.conf_data = conf_data
        self.obj_conf_data = conf_data["objects"]
        self.debug_instance_colors = conf_data.get("debug_instance_colors", {})
        self.gripper_info = gripper_info
        self.type = str(gripper_info["type"]).lower()
        self.joint_names = list(joint_names)
        self.joint_index = joint_index
        self.joint_pos = joint_pos
        self.step_dt = float(step_dt)
        self.device = device
        self.contact_sensor = contact_sensor
        self.contact_sensor_rows_by_env = contact_sensor_rows_by_env
        self.frame_transformer = frame_transformer
        self.env_origin = env_origin
        self.debug = bool(debug)
        self.width_debug = os.environ.get("FINGER_WIDTH_DEBUG", "0") == "1"

        self.direct_prismatic_width = uses_direct_prismatic_width(gripper_info)
        validate_calibration(gripper_info)
        if "total_length" not in gripper_info:
            raise KeyError(
                f"{gripper_info.get('gripper_name')} needs total_length for base-TF approach"
            )
        self.total_length = float(gripper_info["total_length"])
        self.gripper_height = float(gripper_info["height"])
        self.max_width = float(gripper_info["width"])
        # joint_cfg contains the one actuator joint whose USD mimic relation
        # drives the remaining physical finger joints.
        self.calibration_joint_names = list(self.joint_names)
        if len(self.calibration_joint_names) != 1:
            raise ValueError(
                "FingerActionPolicy requires exactly one mimic actuator joint; "
                f"got {self.calibration_joint_names}"
            )
        self.calibration_policy_indices = torch.tensor(
            [0],
            dtype=torch.long,
            device=device,
        )

        self.approach_steps = max(1, int(round(approach_duration_sec / self.step_dt)))
        self.approach_height = float(approach_height)
        self.width_command_steps = max(
            1, int(round(width_command_duration_sec / self.step_dt))
        )
        self.width_ready_timeout_steps = max(
            self.width_command_steps,
            int(round(width_ready_timeout_sec / self.step_dt)),
        )
        self.width_ready_error = float(
            width_ready_prismatic_error_m
            if self.direct_prismatic_width
            else width_ready_revolute_error_rad
        )
        self.close_timeout_steps = max(1, int(round(close_timeout_sec / self.step_dt)))
        self.close_min_steps = max(3, int(round(0.08 / self.step_dt)))
        self.platform_settle_steps = max(
            1, int(round(platform_settle_duration_sec / self.step_dt))
        )
        self.platform_drop_height = float(platform_drop_height)
        self.target_drop_failure_distance = float(target_drop_failure_distance)
        if self.platform_drop_height <= 0.0:
            raise ValueError("platform_drop_height must be positive")
        if self.target_drop_failure_distance <= 0.0:
            raise ValueError("target_drop_failure_distance must be positive")
        self.grasp_blocked_error = 0.005
        self.grasp_stall_delta = 0.0002
        self.grasp_stall_count_threshold = 5
        self.lift_success_threshold = 0.02

        self.total_grasp_num = len(pre_grasp_data)
        self.current_grasp_num = 0
        self.output_list: list[dict] = []
        self._printed_first_z_debug = False
        self._printed_first_point_debug = False

        self._prepare_pre_grasps()

        joint_count = len(self.joint_names)
        self.stage_num = torch.zeros(self.env_num, dtype=torch.long, device=device)
        self.stage_step = torch.zeros(self.env_num, dtype=torch.long, device=device)
        self.action_enable = torch.ones(self.env_num, dtype=torch.long, device=device)
        self.recorded = torch.zeros(self.env_num, dtype=torch.bool, device=device)
        self.grasp_fail_idx = torch.zeros(self.env_num, dtype=torch.bool, device=device)
        self.assigned_pregrasp = torch.full(
            (self.env_num,), -1, dtype=torch.long, device=device
        )
        self.joint_command_dirty = torch.zeros(
            self.env_num, dtype=torch.bool, device=device
        )
        self.target_joint = torch.zeros(
            (self.env_num, joint_count), dtype=torch.float, device=device
        )
        self.open_joint = torch.zeros_like(self.target_joint)
        self.close_joint = torch.zeros_like(self.target_joint)
        self.approach_start_joint = torch.zeros_like(self.target_joint)
        self.width_ready = torch.zeros(
            self.env_num, dtype=torch.bool, device=device
        )
        self.root_pose = torch.zeros((self.env_num, 7), dtype=torch.float, device=device)
        self.root_pose[:, 3] = 1.0
        self.base_position = torch.zeros((self.env_num, 3), dtype=torch.float, device=device)
        self.base_quaternion = torch.zeros((self.env_num, 4), dtype=torch.float, device=device)
        self.base_quaternion[:, 0] = 1.0
        self.open_z = torch.zeros(self.env_num, dtype=torch.float, device=device)
        self.target_class = torch.zeros(self.env_num, dtype=torch.long, device=device)
        self.target_points = torch.zeros((self.env_num, 3), dtype=torch.float, device=device)
        self.target_yaw_deg = torch.zeros(self.env_num, dtype=torch.float, device=device)
        self.target_width = torch.zeros(self.env_num, dtype=torch.float, device=device)
        self.platform_drop_offset = torch.zeros(self.env_num, dtype=torch.float, device=device)
        self.close_error_previous = torch.zeros(self.env_num, dtype=torch.float, device=device)
        self.close_error_valid = torch.zeros(self.env_num, dtype=torch.bool, device=device)
        self.close_stall_count = torch.zeros(self.env_num, dtype=torch.long, device=device)

        self.object_pos_org = torch.tensor(
            [item["translate"] for item in self.obj_conf_data],
            dtype=torch.float,
            device=device,
        )
        self.top_cam_config = next(
            camera
            for camera in self.conf_data["cameras"]
            if camera.get("name") == "top_view_camera"
        )

        self._debug_carb = None
        self._debug_draw = None
        if self.debug:
            try:
                import carb
                from isaacsim.util.debug_draw import _debug_draw

                self._debug_carb = carb
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as error:
                print(f"FingerActionPolicy > debug draw disabled: {error}")
                self.debug = False

    def _prepare_pre_grasps(self) -> None:
        class_to_index = {
            item["class"]: index for index, item in enumerate(self.obj_conf_data)
        }
        positions, quaternions, widths, classes, yaws = [], [], [], [], []
        for record in self.pre_grasp_data:
            record_type = str(record.get("gripper_type", self.type)).lower()
            if not record_type.startswith("finger"):
                raise ValueError(f"Finger policy received {record_type!r} pre-grasp")
            position = list(record["target_points"])
            position[2] += self.total_length
            orientation = list(record["target_orientation"])
            orientation[0] += 180.0
            positions.append(position)
            quaternions.append(euler_angles_to_quat(orientation, degrees=True))
            widths.append(float(record["target_width"]))
            classes.append(class_to_index[record["target_object"]])
            yaws.append(float(record["target_orientation"][2]))

        self.total_base_position = torch.tensor(positions, dtype=torch.float, device=self.device)
        self.total_base_quaternion = torch.tensor(
            np.asarray(quaternions), dtype=torch.float, device=self.device
        )
        self.total_width = torch.tensor(widths, dtype=torch.float, device=self.device).clamp(
            0.0, self.max_width
        )
        self.total_class = torch.tensor(classes, dtype=torch.long, device=self.device)
        self.total_yaw_deg = torch.tensor(yaws, dtype=torch.float, device=self.device)

    def _initial_joint_vector(self, count: int) -> torch.Tensor:
        values = [float(self.gripper_info.get("init_joint_pos", {}).get(name, 0.0)) for name in self.joint_names]
        return torch.tensor(values, dtype=torch.float, device=self.device).repeat(count, 1)

    def _calibrated_open_joint(self, widths: torch.Tensor) -> torch.Tensor:
        joint_count = len(self.calibration_joint_names)
        if self.direct_prismatic_width:
            # Legacy finger2: target_width is the full gap, while the single
            # master prismatic joint moves one side and its USD mimic moves the
            # other. This is the previous close +/- width/2 relationship.
            close = float(self.gripper_info["close_joint_deg"])
            opened = float(self.gripper_info["open_joint_deg"])
            direction = 1.0 if opened >= close else -1.0
            return (close + direction * widths[:, None] * 0.5).repeat(
                1, joint_count
            )

        # Polynomial output is already the signed physical joint angle. Do not
        # apply the close direction a second time.
        joint_degrees = evaluate_polynomial(
            self.gripper_info["width_to_joint_coeffs"], widths
        )
        return joint_degrees[:, None].repeat(1, joint_count) * (
            math.pi / 180.0
        )

    def _calibrated_close_joint(self, count: int) -> torch.Tensor:
        if self.direct_prismatic_width:
            return torch.full(
                (count, len(self.calibration_joint_names)),
                float(self.gripper_info["close_joint_deg"]),
                dtype=torch.float,
                device=self.device,
            )
        degrees = close_joint_degrees(
            self.gripper_info, self.calibration_joint_names
        )
        return torch.tensor(degrees, dtype=torch.float, device=self.device).repeat(count, 1) * (
            math.pi / 180.0
        )

    def _joint_z(self, policy_joint_positions: torch.Tensor) -> torch.Tensor:
        if self.direct_prismatic_width:
            return torch.zeros(
                policy_joint_positions.shape[0],
                dtype=policy_joint_positions.dtype,
                device=policy_joint_positions.device,
            )
        joint_degrees = policy_joint_positions[:, 0] * (180.0 / math.pi)
        return evaluate_polynomial(
            self.gripper_info["joint_to_z_coeffs"], joint_degrees
        )

    def _set_disabled(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.action_enable[env_ids] = 0
        self.stage_num[env_ids] = 4
        self.stage_step[env_ids] = 0
        self.assigned_pregrasp[env_ids] = -1
        self.joint_command_dirty[env_ids] = False
        self.platform_drop_offset[env_ids] = 0.0

    def reset(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        remaining = self.total_grasp_num - self.current_grasp_num

        if remaining <= 0:
            self._set_disabled(env_ids)
            self._refresh_pregrasp_points()
            return

        active_count = min(len(env_ids), remaining)
        active_envs = env_ids[:active_count]
        self._set_disabled(env_ids[active_count:])
        source_ids = torch.arange(
            self.current_grasp_num,
            self.current_grasp_num + active_count,
            dtype=torch.long,
            device=self.device,
        )

        self.action_enable[active_envs] = 1
        self.stage_num[active_envs] = 0
        self.stage_step[active_envs] = 0
        self.recorded[active_envs] = False
        self.grasp_fail_idx[active_envs] = False
        self.assigned_pregrasp[active_envs] = source_ids
        self.platform_drop_offset[active_envs] = 0.0
        self.close_error_previous[active_envs] = 0.0
        self.close_error_valid[active_envs] = False
        self.close_stall_count[active_envs] = 0
        self.width_ready[active_envs] = False

        self.base_position[active_envs] = self.total_base_position[source_ids]
        self.base_quaternion[active_envs] = self.total_base_quaternion[source_ids]
        self.target_class[active_envs] = self.total_class[source_ids]
        self.target_width[active_envs] = self.total_width[source_ids]
        self.target_yaw_deg[active_envs] = self.total_yaw_deg[source_ids]
        self.target_points[active_envs] = self.total_base_position[source_ids]
        self.target_points[active_envs, 2] -= self.total_length

        open_joint = self._initial_joint_vector(active_count)
        close_joint = open_joint.clone()
        open_joint[:, self.calibration_policy_indices] = self._calibrated_open_joint(
            self.total_width[source_ids]
        )
        close_joint[:, self.calibration_policy_indices] = self._calibrated_close_joint(
            active_count
        )
        self.open_joint[active_envs] = open_joint
        self.close_joint[active_envs] = close_joint
        approach_start_joint = self._initial_joint_vector(active_count)
        self.approach_start_joint[active_envs] = approach_start_joint
        # Keep the USD mimic chain at its mutually consistent default pose
        # during reset. Stage 0 smoothly moves its sole actuator joint to the
        # requested opening instead of teleporting it.
        self.target_joint[active_envs] = approach_start_joint
        self.open_z[active_envs] = self._joint_z(open_joint)

        # ``joint_to_z`` is the increase in gripper length from the fully-open
        # calibration pose.  Since the gripper faces downward (X +180 deg),
        # raise the root by that amount so the grasp plane remains at the
        # pre-grasp Z even when an episode starts from a narrower width.
        self.base_position[active_envs, 2] += self.open_z[active_envs]

        self.root_pose[active_envs, :3] = self.base_position[active_envs]
        self.root_pose[active_envs, 2] += self.approach_height
        self.root_pose[active_envs, 3:7] = self.base_quaternion[active_envs]
        if not self._printed_first_z_debug and len(active_envs):
            env_id = int(active_envs[0].item())
            source_id = int(source_ids[0].item())
            source = self.pre_grasp_data[source_id]
            print(
                "FingerActionPolicy > z_debug "
                f"gripper={self.gripper_info.get('gripper_name')} "
                f"pregrasp_z={float(source['target_points'][2]):.6f} "
                f"total_length={self.total_length:.6f} "
                f"base_z={float(self.base_position[env_id, 2]):.6f} "
                f"approach_z={float(self.root_pose[env_id, 2]):.6f} "
                f"open_z={float(self.open_z[env_id]):.6f}"
            )
            self._printed_first_z_debug = True
        self.joint_command_dirty[active_envs] = True
        self.current_grasp_num += active_count
        print(f"current grasp num : {self.current_grasp_num}")
        self._refresh_pregrasp_points()

    def step(self) -> torch.Tensor:
        active = self.action_enable == 1
        stage0 = torch.where(active & (self.stage_num == 0))[0]
        if len(stage0):
            width_preparing = stage0[~self.width_ready[stage0]]
            if len(width_preparing):
                progress = (
                    (self.stage_step[width_preparing] + 1).float()
                    / self.width_command_steps
                ).clamp(0.0, 1.0)
                smooth = progress * progress * (3.0 - 2.0 * progress)
                # Hold the root at the safe approach height while opening to
                # the candidate width. Only the master joint is commanded;
                # the rest of the chain follows through USD mimic constraints.
                self.root_pose[width_preparing, :3] = self.base_position[
                    width_preparing
                ]
                self.root_pose[width_preparing, 2] += self.approach_height
                self.root_pose[width_preparing, 3:7] = self.base_quaternion[
                    width_preparing
                ]
                self.target_joint[width_preparing] = (
                    self.approach_start_joint[width_preparing]
                    + (
                        self.open_joint[width_preparing]
                        - self.approach_start_joint[width_preparing]
                    )
                    * smooth[:, None]
                )
                self.joint_command_dirty[width_preparing] = True
                self.stage_step[width_preparing] += 1

                current = self.joint_pos[width_preparing][:, self.joint_index]

                error = torch.linalg.vector_norm(
                    self.open_joint[width_preparing] - current, dim=1
                )
                if self.width_debug and len(width_preparing):
                    env_id = int(width_preparing[0])
                    if int(self.stage_step[env_id]) % 10 == 0:
                        print(
                            "FingerActionPolicy > width_debug "
                            f"env={env_id} source={int(self.assigned_pregrasp[env_id])} "
                            f"width={float(self.target_width[env_id]):.6f} "
                            f"step={int(self.stage_step[env_id])} "
                            f"command={self.target_joint[env_id].detach().cpu().tolist()} "
                            f"current={current[0].detach().cpu().tolist()} "
                            f"error={float(error[0]):.6f}"
                        )
                command_finished = (
                    self.stage_step[width_preparing] >= self.width_command_steps
                )
                reached = command_finished & (error <= self.width_ready_error)
                timed_out = (
                    self.stage_step[width_preparing]
                    >= self.width_ready_timeout_steps
                )

                ready_envs = width_preparing[reached]
                if len(ready_envs):
                    self.width_ready[ready_envs] = True
                    self.stage_step[ready_envs] = 0
                    self.target_joint[ready_envs] = self.open_joint[ready_envs]
                    self.joint_command_dirty[ready_envs] = True

                failed_envs = width_preparing[timed_out & ~reached]
                if len(failed_envs):
                    self.grasp_fail_idx[failed_envs] = True
                    self.stage_num[failed_envs] = 3
                    self.stage_step[failed_envs] = 0

        # Z descent begins only after the actual master joint is sufficiently
        # close to the requested opening width.
        stage0 = torch.where(
            active & (self.stage_num == 0) & self.width_ready
        )[0]
        if len(stage0):
            progress = ((self.stage_step[stage0] + 1).float() / self.approach_steps).clamp(0.0, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            self.root_pose[stage0, :3] = self.base_position[stage0]
            self.root_pose[stage0, 2] += self.approach_height * (1.0 - smooth)
            self.root_pose[stage0, 3:7] = self.base_quaternion[stage0]
            self.target_joint[stage0] = self.open_joint[stage0]
            self.stage_step[stage0] += 1
            done = stage0[self.stage_step[stage0] >= self.approach_steps]
            if len(done):
                self.stage_num[done] = 1
                self.stage_step[done] = 0
                self.target_joint[done] = self.close_joint[done]
                self.joint_command_dirty[done] = True

        stage1 = torch.where(active & (self.stage_num == 1))[0]
        if len(stage1):
            current = self.joint_pos[stage1][:, self.joint_index]
            z_hop = self._joint_z(current) - self.open_z[stage1]
            self.root_pose[stage1, :3] = self.base_position[stage1]
            self.root_pose[stage1, 2] += z_hop
            self.root_pose[stage1, 3:7] = self.base_quaternion[stage1]
            self.target_joint[stage1] = self.close_joint[stage1]
            self.stage_step[stage1] += 1

        stage2 = torch.where(active & (self.stage_num == 2))[0]
        if len(stage2):
            current = self.joint_pos[stage2][:, self.joint_index]
            z_hop = self._joint_z(current) - self.open_z[stage2]
            self.root_pose[stage2, :3] = self.base_position[stage2]
            self.root_pose[stage2, 2] += z_hop
            self.root_pose[stage2, 3:7] = self.base_quaternion[stage2]
            # Keep the closed gripper fixed and move the support platform down
            # by the full configured distance immediately. The remaining stage
            # steps are only a settling/evaluation wait.
            self.platform_drop_offset[stage2] = -self.platform_drop_height
            self.stage_step[stage2] += 1
            done = stage2[
                self.stage_step[stage2] >= self.platform_settle_steps
            ]
            if len(done):
                self.stage_num[done] = 3
                self.stage_step[done] = 0

        return self.target_joint

    def _target_height_delta(self, env_ids: torch.Tensor):
        object_positions = self.frame_transformer.data.target_pos_source[env_ids]
        rows = torch.arange(len(env_ids), device=self.device)
        target_positions = object_positions[rows, self.target_class[env_ids]]
        original = self.object_pos_org[self.target_class[env_ids]]
        platform_drop = -self.platform_drop_offset[env_ids].clamp(max=0.0)
        height_delta = target_positions[:, 2] - (original[:, 2] - platform_drop)
        return height_delta, object_positions, platform_drop

    def _advance_closing(self) -> torch.Tensor:
        done = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        idx = torch.where((self.action_enable == 1) & (self.stage_num == 1))[0]
        if len(idx) == 0:
            return done
        current = self.joint_pos[idx][:, self.joint_index]
        error = torch.linalg.vector_norm(
            self.close_joint[idx] - current, dim=1
        )
        delta = torch.abs(error - self.close_error_previous[idx])
        stalled = self.close_error_valid[idx] & (delta < self.grasp_stall_delta) & (
            error > self.grasp_blocked_error
        )
        self.close_stall_count[idx] = torch.where(
            stalled,
            self.close_stall_count[idx] + 1,
            torch.zeros_like(self.close_stall_count[idx]),
        )
        self.close_error_previous[idx] = error
        self.close_error_valid[idx] = True

        waited = self.stage_step[idx] >= self.close_min_steps
        blocked = error > self.grasp_blocked_error
        stable_block = self.close_stall_count[idx] >= self.grasp_stall_count_threshold
        timeout = self.stage_step[idx] >= self.close_timeout_steps
        grasped = waited & blocked & (stable_block | timeout)
        empty = timeout & ~blocked

        grasped_envs = idx[grasped]
        if len(grasped_envs):
            self.stage_num[grasped_envs] = 2
            self.stage_step[grasped_envs] = 0
        failed_envs = idx[empty]
        if len(failed_envs):
            self.grasp_fail_idx[failed_envs] = True
            self.stage_num[failed_envs] = 3
            done[failed_envs] = True
        return done

    def _mark_contact_failures(self) -> torch.Tensor:
        done = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        idx = torch.where(
            (self.action_enable == 1)
            & ((self.stage_num == 2) | (self.stage_num == 3))
            & ~self.recorded
        )[0]
        if len(idx) == 0:
            return done

        sensor_rows = self.contact_sensor_rows_by_env[idx]
        history = self.contact_sensor.data.net_forces_w_history[sensor_rows]
        # Shape: (logical env, wildcard parent, history, body, xyz). Any one
        # finger contact contributes to the environment-level contact total.
        peak_contact = torch.amax(history.abs(), dim=2).sum(dim=(1, 2, 3))
        failed = idx[peak_contact < 0.2]
        if len(failed):
            self.grasp_fail_idx[failed] = True
            self.stage_num[failed] = 3
            self.stage_step[failed] = 0
            done[failed] = True
        return done

    def _mark_target_drop_failures(self) -> torch.Tensor:
        """Fail immediately if the target falls far below its initial world Z."""
        done = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        idx = torch.where(
            (self.action_enable == 1)
            & ((self.stage_num == 2) | (self.stage_num == 3))
            & ~self.recorded
            & ~self.grasp_fail_idx
        )[0]
        if len(idx) == 0:
            return done

        object_positions = self.frame_transformer.data.target_pos_source[idx]
        target_positions = object_positions[
            torch.arange(len(idx), device=self.device), self.target_class[idx]
        ]
        original_target_z = self.object_pos_org[self.target_class[idx], 2]
        drop_distance = original_target_z - target_positions[:, 2]
        failed = drop_distance >= self.target_drop_failure_distance
        failed_envs = idx[failed]
        if len(failed_envs) == 0:
            return done

        self.grasp_fail_idx[failed_envs] = True
        self.stage_num[failed_envs] = 3
        self.stage_step[failed_envs] = 0
        done[failed_envs] = True
        if self.debug:
            for local_index, env_id in zip(
                torch.where(failed)[0].detach().cpu().tolist(),
                failed_envs.detach().cpu().tolist(),
            ):
                print(
                    "FingerActionPolicy > target_dropped "
                    f"env={env_id} "
                    f"drop_distance={float(drop_distance[local_index]):.5f} "
                    f"threshold={self.target_drop_failure_distance:.5f}"
                )
        return done

    def _grasp_bboxes_world(self, env_ids: torch.Tensor) -> torch.Tensor:
        rectangles = []
        for env_id in env_ids.tolist():
            half_height = self.gripper_height / 2.0
            # Grasp bbox represents the attempted grasp region.  Keep the
            # opening width stored in pre-grasp instead of reconstructing a
            # smaller/restored bbox from the joints after closing.
            attempted_width = float(self.target_width[env_id])
            if self.type.startswith("finger2"):
                local_boxes = np.asarray(
                    [[
                        [half_height, -attempted_width / 2.0, 0.0],
                        [-half_height, -attempted_width / 2.0, 0.0],
                        [-half_height, attempted_width / 2.0, 0.0],
                        [half_height, attempted_width / 2.0, 0.0],
                    ]],
                    dtype=np.float32,
                )
            else:
                base = np.asarray(
                    [
                        [half_height, -attempted_width / 2.0, 0.0],
                        [-half_height, -attempted_width / 2.0, 0.0],
                        [-half_height, 0.0, 0.0],
                        [half_height, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )
                local_boxes = []
                for info in self.gripper_info["finger_bbox_info"]:
                    radians = math.radians(float(info["rot"]))
                    cosine, sine = math.cos(radians), math.sin(radians)
                    rotation = np.asarray(
                        [[cosine, -sine], [sine, cosine]], dtype=np.float32
                    )
                    box = base.copy()
                    box[:, :2] = box[:, :2] @ rotation.T + np.asarray(info["pos"][:2])
                    local_boxes.append(box)
                local_boxes = np.asarray(local_boxes, dtype=np.float32)

            yaw = math.radians(float(self.target_yaw_deg[env_id]))
            rotation = np.asarray(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
                dtype=np.float32,
            )
            local_boxes[:, :, :2] = local_boxes[:, :, :2] @ rotation.T
            local_boxes += self.target_points[env_id].detach().cpu().numpy()[None, None, :]
            rectangles.append(local_boxes)
        return torch.tensor(np.asarray(rectangles), dtype=torch.float, device=self.device)

    def _bbox_3d_to_2d(self, bbox: np.ndarray) -> np.ndarray:
        intrinsic = np.asarray(self.top_cam_config["intrinsic_isaac"], dtype=float)
        camera_world = np.asarray(self.top_cam_config["cam_poses"], dtype=float)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        world_to_camera = np.linalg.inv(camera_world @ flip)
        homogeneous = np.concatenate(
            (bbox.reshape(-1, 3), np.ones((bbox.shape[0] * bbox.shape[1], 1))), axis=1
        ).T
        camera_points = (world_to_camera @ homogeneous)[:3]
        pixels = intrinsic @ camera_points
        pixels = (pixels[:2] / pixels[2:3]).T
        return pixels.reshape(bbox.shape[0], bbox.shape[1], 2)

    @staticmethod
    def _bbox_summary(bbox_2d: np.ndarray, angle_rad: float) -> dict:
        rounded = np.rint(bbox_2d).astype(np.int32)
        first = bbox_2d[0]
        return {
            "bbox": rounded.tolist(),
            "center": np.rint(bbox_2d.mean(axis=(0, 1))).astype(np.int32).tolist(),
            "width": int(round(float(np.linalg.norm(first[1] - first[2])))),
            "height": int(round(float(np.linalg.norm(first[0] - first[1])))),
            "angle": round(float(angle_rad), 6),
        }

    def _draw_success_bboxes(self, bboxes: torch.Tensor, env_ids: torch.Tensor) -> None:
        if not self.debug or self._debug_draw is None:
            return
        for local_index, env_id in enumerate(env_ids.tolist()):
            origin = self.env_origin[env_id].detach().cpu().numpy()
            source_id = int(self.assigned_pregrasp[env_id])
            color = self._debug_color_for_source(
                source_id, fallback=(0.0, 1.0, 0.1, 1.0)
            )
            for box in bboxes[local_index].detach().cpu().numpy() + origin[None, :]:
                starts = box[[0, 1, 2, 3]]
                ends = box[[1, 2, 3, 0]]
                self._debug_draw.draw_lines(
                    [self._debug_carb.Float3(*point.tolist()) for point in starts],
                    [self._debug_carb.Float3(*point.tolist()) for point in ends],
                    [color] * 4,
                    [4.0] * 4,
                )

    def _debug_color_for_source(self, source_id: int, fallback):
        if 0 <= source_id < len(self.pre_grasp_data):
            target_object = self.pre_grasp_data[source_id].get("target_object")
            rgba = self.debug_instance_colors.get(target_object)
            if isinstance(rgba, list) and len(rgba) == 4:
                return self._debug_carb.ColorRgba(*rgba)
        return self._debug_carb.ColorRgba(*fallback)

    def _refresh_pregrasp_points(self) -> None:
        """Remove points from finished episodes and redraw currently active ones once."""
        if not self.debug or self._debug_draw is None:
            return

        # DebugDraw cannot remove an individual point. Clear only the point
        # layer, then restore points belonging to episodes that are still active.
        # Lines (including successful grasp bboxes) are intentionally preserved.
        self._debug_draw.clear_points()
        env_ids = torch.where(
            (self.action_enable == 1) & (self.assigned_pregrasp >= 0)
        )[0]
        if len(env_ids) == 0:
            return
        points = self.target_points[env_ids] + self.env_origin[env_ids]
        if not self._printed_first_point_debug:
            first_point = points[0].detach().cpu().numpy()
            print(
                "FingerActionPolicy > pregrasp_point_debug "
                f"xyz={[round(float(value), 6) for value in first_point.tolist()]} "
                f"env_id={int(env_ids[0].item())}"
            )
            self._printed_first_point_debug = True
        self._debug_draw.draw_points(
            [self._debug_carb.Float3(*point.tolist()) for point in points.detach().cpu().numpy()],
            [
                self._debug_color_for_source(
                    int(self.assigned_pregrasp[env_id]),
                    fallback=(1.0, 0.9, 0.0, 1.0),
                )
                for env_id in env_ids.detach().cpu().tolist()
            ],
            [40.0] * len(env_ids),
        )

    def _record_successes(self, env_ids: torch.Tensor) -> torch.Tensor:
        done = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        if len(env_ids) == 0:
            return done
        height_delta, object_positions, platform_drop = self._target_height_delta(env_ids)
        expected_platform_z = (
            self.object_pos_org[None, :, 2] - platform_drop[:, None]
        )
        object_platform_separation = object_positions[:, :, 2] - expected_platform_z
        non_target_mask = torch.ones_like(
            object_platform_separation, dtype=torch.bool
        )
        non_target_mask.scatter_(
            1, self.target_class[env_ids].unsqueeze(1), False
        )
        collateral_grasp_mask = non_target_mask & (
            object_platform_separation > self.lift_success_threshold
        )
        collateral_grasped = collateral_grasp_mask.any(dim=1)
        success = (
            ~self.grasp_fail_idx[env_ids]
            & (height_delta > self.lift_success_threshold)
            & ~collateral_grasped
        )
        success_envs = env_ids[success]

        compensated = object_positions.clone()
        compensated[:, :, 2] += platform_drop[:, None]
        disturbed = (
            torch.linalg.vector_norm(
                compensated - self.object_pos_org[None, :, :], dim=2
            )
            > self.lift_success_threshold
        ).sum(dim=1)

        if len(success_envs):
            bboxes = self._grasp_bboxes_world(success_envs)
            self._draw_success_bboxes(bboxes, success_envs)
            success_rows = torch.where(success)[0]
            for output_index, env_id in enumerate(success_envs.tolist()):
                source_id = int(self.assigned_pregrasp[env_id])
                source = self.pre_grasp_data[source_id]
                bbox_2d = self._bbox_3d_to_2d(bboxes[output_index].detach().cpu().numpy())
                self.output_list.append(
                    {
                        "bbox_2d": self._bbox_summary(
                            bbox_2d, -math.radians(float(source["target_orientation"][2]))
                        ),
                        "target_points": copy.deepcopy(source["target_points"]),
                        "target_orientation": copy.deepcopy(source["target_orientation"]),
                        "target_width": copy.deepcopy(source["target_width"]),
                        "target_object": source["target_object"],
                        "gripper_model": self.gripper_info["gripper_name"],
                        "gripper_type": self.gripper_info["type"],
                        "disturbed_object_count": int(disturbed[success_rows[output_index]]),
                    }
                )

        failed_envs = env_ids[~success]
        if len(failed_envs):
            self.grasp_fail_idx[failed_envs] = True
        self.recorded[env_ids] = True
        done[env_ids] = True
        return done

    def get_done_idx(self) -> torch.Tensor:
        done = self._advance_closing()
        done |= self._mark_target_drop_failures()
        done |= self._mark_contact_failures()
        evaluate = torch.where(
            (self.action_enable == 1)
            & (self.stage_num == 3)
            & ~self.recorded
            & ~self.grasp_fail_idx
        )[0]
        done |= self._record_successes(evaluate)
        failed = torch.where(
            (self.action_enable == 1)
            & (self.stage_num == 3)
            & self.grasp_fail_idx
        )[0]
        if len(failed):
            self.recorded[failed] = True
            done[failed] = True
        return done

    def _clear_debug_draw(self) -> None:
        if self._debug_draw is None:
            return
        self._debug_draw.clear_points()
        self._debug_draw.clear_lines()
