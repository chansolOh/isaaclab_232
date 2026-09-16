"""Batched platform-free approach, grasp and direct stress-test policy."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from isaacsim.core.utils.rotations import euler_angles_to_quat

from . import calibration


APPROACH = 0
CLOSE = 1
STRESS = 2
DONE = 3
DISABLED = 4

FORCE_SCORE_WEIGHT = 0.3
PREGRASP_POSE_SCORE_WEIGHT = 0.4
STRESS_POSE_SCORE_WEIGHT = 0.2
CENTER_IN_POSE_WEIGHT = 0.5
ROTATION_IN_POSE_WEIGHT = 0.5
CENTER_SCORE_RANGE_M = 0.010
ROTATION_SCORE_RANGE_RAD = math.pi
CONTACT_AREA_SCORE_WEIGHT = 0.30
CONTACT_AREA_SCORE_RANGE_MM2 = 500.0
# 1.0 keeps each component linear. Values > 1 emphasize already-high scores;
# a logarithm is intentionally not used because log(total_score) cannot change rank.
SCORE_COMPONENT_POWER = 1.5


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((*vector.shape[:-1], 1), dtype=vector.dtype, device=vector.device)
    pure = torch.cat((zeros, vector), dim=-1)
    return quaternion_multiply(
        quaternion_multiply(quaternion, pure), quaternion_conjugate(quaternion)
    )[..., 1:]


def quaternion_slerp(
    start: torch.Tensor, end: torch.Tensor, amount: torch.Tensor
) -> torch.Tensor:
    start, end = normalize_quaternion(start), normalize_quaternion(end)
    dot = torch.sum(start * end, dim=-1, keepdim=True)
    end = torch.where(dot < 0.0, -end, end)
    dot = torch.abs(dot).clamp(max=1.0)
    linear = normalize_quaternion(start + amount * (end - start))
    theta = torch.acos(dot)
    spherical = (
        torch.sin(theta * (1.0 - amount)) / torch.sin(theta).clamp_min(1e-8) * start
        + torch.sin(theta * amount) / torch.sin(theta).clamp_min(1e-8) * end
    )
    return torch.where(dot > 0.9995, linear, spherical)


def quaternion_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(
        torch.sum(normalize_quaternion(first) * normalize_quaternion(second), dim=-1)
    ).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = normalize_quaternion(quaternion).unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


class GraspPolicy:
    """One state machine shared by calibrated fingers and preset hands."""

    def __init__(
        self,
        *,
        env_num: int,
        pre_grasp_data: list[dict],
        conf_data: dict,
        gripper_info: dict,
        joint_names: list[str],
        joint_index: torch.Tensor,
        joint_pos: torch.Tensor,
        step_dt: float,
        device: str,
        env_origin: torch.Tensor,
        object_initial_pos: torch.Tensor,
        object_initial_quat: torch.Tensor,
        approach_height: float = 0.12,
        approach_duration: float = 0.8,
        close_timeout: float = 1.5,
        stress_duration: float = 1.2,
        grasp_blocked_error: float = 0.003,
        max_relative_translation: float = 0.010,
        max_relative_rotation_deg: float = 20.0,
        min_contact_force: float = 0.12,
        contact_lost_duration: float = 0.10,
        contact_penetration_threshold: float = 0.005,
        pre_stress_object_motion_threshold: float = 0.005,
        apply_finger_z_hop: bool = False,
    ):
        if not pre_grasp_data:
            raise ValueError("pre_grasp_data is empty")
        if len(conf_data.get("objects", [])) != 1:
            raise ValueError("GraspPolicy requires exactly one scene object")
        self.env_num = int(env_num)
        self.records = pre_grasp_data
        self.conf_data = conf_data
        self.gripper = gripper_info
        self.gripper_type = str(gripper_info.get("type", "")).lower()
        self.is_hand = self.gripper_type == "hand"
        self.joint_names = list(joint_names)
        self.joint_index = joint_index
        self.joint_pos = joint_pos
        self.step_dt = float(step_dt)
        self.device = device
        self.env_origin = env_origin
        self.object_initial_pos = object_initial_pos.clone()
        self.object_initial_quat = normalize_quaternion(object_initial_quat.clone())
        self.approach_height = float(approach_height)
        self.approach_steps = max(1, round(approach_duration / self.step_dt))
        self.close_timeout_steps = max(1, round(close_timeout / self.step_dt))
        self.stress_steps = max(1, round(stress_duration / self.step_dt))
        self.grasp_blocked_error = float(grasp_blocked_error)
        self.max_relative_translation = float(max_relative_translation)
        self.max_relative_rotation = math.radians(float(max_relative_rotation_deg))
        self.min_contact_force = float(min_contact_force)
        self.contact_lost_steps = max(
            1, round(float(contact_lost_duration) / self.step_dt)
        )
        self.contact_penetration_threshold = float(contact_penetration_threshold)
        if self.contact_penetration_threshold < 0.0:
            raise ValueError("contact_penetration_threshold must be non-negative")
        self.pre_stress_object_motion_threshold = float(
            pre_stress_object_motion_threshold
        )
        if self.pre_stress_object_motion_threshold < 0.0:
            raise ValueError(
                "pre_stress_object_motion_threshold must be non-negative"
            )
        self.apply_finger_z_hop = bool(apply_finger_z_hop)
        self.output_list: list[dict] = []
        self.attempt_history: list[dict] = []
        self.total_grasp_num = len(pre_grasp_data)
        self.current_grasp_num = 0

        if not self.is_hand:
            calibration.validate(gripper_info)
            self.direct_prismatic = calibration.direct_prismatic(gripper_info)
            self.total_length = float(gripper_info["total_length"])
            self.max_width = float(gripper_info["width"])
        self._prepare_records()
        self._allocate()

    def _allocate(self) -> None:
        e, j = self.env_num, len(self.joint_names)
        kwargs = {"device": self.device}
        self.stage_num = torch.full((e,), DISABLED, dtype=torch.long, **kwargs)
        self.stage_step = torch.zeros(e, dtype=torch.long, **kwargs)
        self.action_enable = torch.ones(e, dtype=torch.long, **kwargs)
        self.assigned_pregrasp = torch.full((e,), -1, dtype=torch.long, **kwargs)
        self.recorded = torch.zeros(e, dtype=torch.bool, **kwargs)
        self.failed_reason = [""] * e
        self.target_joint = torch.zeros((e, j), dtype=torch.float, **kwargs)
        self.open_joint = torch.zeros_like(self.target_joint)
        self.close_joint = torch.zeros_like(self.target_joint)
        self.start_joint = torch.zeros_like(self.target_joint)
        self.end_joint = torch.zeros_like(self.target_joint)
        self.root_pose = torch.zeros((e, 7), dtype=torch.float, **kwargs)
        self.root_pose[:, 3] = 1.0
        self.start_pos = torch.zeros((e, 3), dtype=torch.float, **kwargs)
        self.end_pos = torch.zeros_like(self.start_pos)
        self.start_quat = torch.zeros((e, 4), dtype=torch.float, **kwargs)
        self.end_quat = torch.zeros_like(self.start_quat)
        self.transition_steps = torch.ones(e, dtype=torch.long, **kwargs)
        self.smooth_transition = torch.ones(e, dtype=torch.bool, **kwargs)
        self.open_z = torch.zeros(e, dtype=torch.float, **kwargs)
        self.applied_z_hop = torch.zeros(e, dtype=torch.float, **kwargs)
        self.close_error_previous = torch.zeros(e, dtype=torch.float, **kwargs)
        self.close_error_valid = torch.zeros(e, dtype=torch.bool, **kwargs)
        self.close_stall_count = torch.zeros(e, dtype=torch.long, **kwargs)
        self.close_contact_seen = torch.zeros(e, dtype=torch.bool, **kwargs)
        self.contact_lost_count = torch.zeros(e, dtype=torch.long, **kwargs)
        self.reference_rel_pos = torch.zeros((e, 3), dtype=torch.float, **kwargs)
        self.reference_rel_quat = torch.zeros((e, 4), dtype=torch.float, **kwargs)
        self.reference_rel_quat[:, 0] = 1.0
        self.relative_translation_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.relative_rotation_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.last_contact_force = torch.zeros(e, dtype=torch.float, **kwargs)
        self.max_test_force = torch.zeros(e, dtype=torch.float, **kwargs)
        self.force_direction = torch.zeros((e, 3), dtype=torch.float, **kwargs)
        self.force_direction[:, 2] = -1.0
        self.stress_survival = torch.zeros(e, dtype=torch.float, **kwargs)
        self.minimum_contact_separation = torch.full(
            (e,), torch.inf, dtype=torch.float, **kwargs
        )
        self.minimum_close_contact_separation = torch.full(
            (e,), torch.inf, dtype=torch.float, **kwargs
        )
        self.maximum_pre_stress_object_motion = torch.zeros(
            e, dtype=torch.float, **kwargs
        )
        self.pregrasp_rotation_error = torch.zeros(
            e, dtype=torch.float, **kwargs
        )
        self.stress_rotation_error = torch.zeros(
            e, dtype=torch.float, **kwargs
        )
        self.pregrasp_center_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.stress_center_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.stress_start_object_pos = torch.zeros(
            (e, 3), dtype=torch.float, **kwargs
        )
        self.stress_start_object_quat = torch.zeros(
            (e, 4), dtype=torch.float, **kwargs
        )
        self.stress_start_object_quat[:, 0] = 1.0
        self.grasp_contact_area = torch.zeros(e, dtype=torch.float, **kwargs)

    def _prepare_records(self) -> None:
        starts, ends, start_quat, end_quat = [], [], [], []
        start_joint, end_joint, transition, smooth = [], [], [], []
        for record in self.records:
            if record.get("target_object") != self.conf_data["objects"][0]["class"]:
                raise ValueError(
                    f"Pre-grasp target {record.get('target_object')!r} does not match scene object"
                )
            if self.is_hand:
                base = record.get("target_base_tf", {})
                joints = record.get("target_joint_pos", {})
                for phase in ("start", "end"):
                    if phase not in base or phase not in joints:
                        raise KeyError(f"Hand pre-grasp needs target_base_tf/joint_pos.{phase}")
                starts.append(base["start"]["position"])
                ends.append(base["end"]["position"])
                start_quat.append(base["start"]["orientation_wxyz"])
                end_quat.append(base["end"]["orientation_wxyz"])
                start_joint.append(self._hand_joint_vector(record, "start"))
                end_joint.append(self._hand_joint_vector(record, "end"))
                setting = record.get("transition", {})
                transition.append(max(1, round(float(setting.get("duration_sec", 1.0)) / self.step_dt)))
                smooth.append(str(setting.get("interpolation", "smoothstep")).lower() == "smoothstep")
            else:
                position = list(record["target_points"])
                position[2] += self.total_length
                orientation = list(record["target_orientation"])
                orientation[0] += 180.0
                quaternion = euler_angles_to_quat(orientation, degrees=True)
                starts.append(position)
                ends.append(position)
                start_quat.append(quaternion)
                end_quat.append(quaternion)
                start_joint.append([0.0] * len(self.joint_names))
                end_joint.append([0.0] * len(self.joint_names))
                transition.append(1)
                smooth.append(True)
        tensor = lambda value, dtype=torch.float: torch.tensor(value, dtype=dtype, device=self.device)
        self.total_start_pos = tensor(starts)
        self.total_end_pos = tensor(ends)
        self.total_start_quat = normalize_quaternion(tensor(np.asarray(start_quat)))
        self.total_end_quat = normalize_quaternion(tensor(np.asarray(end_quat)))
        self.total_start_joint = tensor(start_joint)
        self.total_end_joint = tensor(end_joint)
        self.total_transition_steps = tensor(transition, torch.long)
        self.total_smooth = tensor(smooth, torch.bool)

    def _hand_joint_vector(self, record: dict, phase: str) -> list[float]:
        values = record["target_joint_pos"][phase]
        missing = [name for name in self.joint_names if name not in values]
        if missing:
            raise KeyError(f"Hand {phase} pose missing joints: {missing}")
        result = [float(values[name]) for name in self.joint_names]
        unit = str(record.get("joint_unit", "rad")).lower()
        if unit.startswith("deg"):
            result = [math.radians(value) for value in result]
        elif not unit.startswith("rad"):
            raise ValueError(f"Unsupported joint unit: {unit}")
        return result

    def _initial_joint(self, count: int) -> torch.Tensor:
        values = [float(self.gripper.get("init_joint_pos", {}).get(name, 0.0)) for name in self.joint_names]
        return torch.tensor(values, dtype=torch.float, device=self.device).repeat(count, 1)

    def _finger_open_joint(self, widths: torch.Tensor) -> torch.Tensor:
        if self.direct_prismatic:
            closed = float(self.gripper["close_joint_deg"])
            opened = float(self.gripper["open_joint_deg"])
            direction = 1.0 if opened >= closed else -1.0
            return closed + direction * widths[:, None] * 0.5
        degrees = calibration.polynomial(self.gripper["width_to_joint_coeffs"], widths)
        return degrees[:, None] * (math.pi / 180.0)

    def _finger_close_joint(self, count: int) -> torch.Tensor:
        value = float(self.gripper["close_joint_deg"])
        if not self.direct_prismatic:
            value = math.radians(value)
        return torch.full((count, len(self.joint_names)), value, device=self.device)

    def _finger_joint_z(self, joint: torch.Tensor) -> torch.Tensor:
        if self.direct_prismatic:
            return torch.zeros(len(joint), dtype=torch.float, device=self.device)
        degrees = joint[:, 0] * (180.0 / math.pi)
        return calibration.polynomial(self.gripper["joint_to_z_coeffs"], degrees)

    def _disable(self, env_ids: torch.Tensor) -> None:
        if not len(env_ids):
            return
        self.stage_num[env_ids] = DISABLED
        self.stage_step[env_ids] = 0
        self.action_enable[env_ids] = 0
        self.assigned_pregrasp[env_ids] = -1

    def reset(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        remaining = self.total_grasp_num - self.current_grasp_num
        if remaining <= 0:
            self._disable(env_ids)
            return
        count = min(len(env_ids), remaining)
        active = env_ids[:count]
        self._disable(env_ids[count:])
        source = torch.arange(
            self.current_grasp_num,
            self.current_grasp_num + count,
            dtype=torch.long,
            device=self.device,
        )
        self.action_enable[active] = 1
        self.assigned_pregrasp[active] = source
        self.stage_num[active] = APPROACH
        self.stage_step[active] = 0
        self.recorded[active] = False
        self.close_error_valid[active] = False
        self.close_stall_count[active] = 0
        self.close_contact_seen[active] = False
        self.contact_lost_count[active] = 0
        self.stress_survival[active] = 0.0
        self.max_test_force[active] = 0.0
        self.relative_translation_error[active] = 0.0
        self.relative_rotation_error[active] = 0.0
        self.minimum_contact_separation[active] = torch.inf
        self.minimum_close_contact_separation[active] = torch.inf
        self.maximum_pre_stress_object_motion[active] = 0.0
        self.pregrasp_rotation_error[active] = 0.0
        self.stress_rotation_error[active] = 0.0
        self.pregrasp_center_error[active] = 0.0
        self.stress_center_error[active] = 0.0
        self.stress_start_object_pos[active] = 0.0
        self.stress_start_object_quat[active] = 0.0
        self.stress_start_object_quat[active, 0] = 1.0
        self.grasp_contact_area[active] = 0.0
        self.applied_z_hop[active] = 0.0
        for env_id in active.tolist():
            self.failed_reason[env_id] = ""
        direction = torch.randn((count, 3), dtype=torch.float, device=self.device)
        direction[:, 2] = -torch.abs(direction[:, 2])
        self.force_direction[active] = direction / torch.linalg.vector_norm(
            direction, dim=1, keepdim=True
        ).clamp_min(1e-8)

        self.start_pos[active] = self.total_start_pos[source]
        self.end_pos[active] = self.total_end_pos[source]
        self.start_quat[active] = self.total_start_quat[source]
        self.end_quat[active] = self.total_end_quat[source]
        self.transition_steps[active] = self.total_transition_steps[source]
        self.smooth_transition[active] = self.total_smooth[source]
        if self.is_hand:
            self.start_joint[active] = self.total_start_joint[source]
            self.end_joint[active] = self.total_end_joint[source]
            self.open_joint[active] = self.start_joint[active]
            self.close_joint[active] = self.end_joint[active]
        else:
            widths = torch.tensor(
                [float(self.records[index]["target_width"]) for index in source.tolist()],
                dtype=torch.float,
                device=self.device,
            ).clamp(0.0, self.max_width)
            initial = self._initial_joint(count)
            self.start_joint[active] = initial
            self.open_joint[active] = self._finger_open_joint(widths)
            self.close_joint[active] = self._finger_close_joint(count)
            self.end_joint[active] = self.close_joint[active]
            self.open_z[active] = self._finger_joint_z(self.open_joint[active])
            self.start_pos[active, 2] += self.open_z[active]
            self.end_pos[active] = self.start_pos[active]
        self.target_joint[active] = self.start_joint[active]
        self.root_pose[active, :3] = self.start_pos[active]
        self.root_pose[active, 2] += self.approach_height
        self.root_pose[active, 3:7] = self.start_quat[active]
        self.current_grasp_num += count
        print(f"GraspPolicy > assigned={self.current_grasp_num}/{self.total_grasp_num}")

    def step(self) -> torch.Tensor:
        active = self.action_enable == 1
        approaching = torch.where(active & (self.stage_num == APPROACH))[0]
        if len(approaching):
            progress = ((self.stage_step[approaching] + 1) / self.approach_steps).clamp(0.0, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            self.root_pose[approaching, :3] = self.start_pos[approaching]
            self.root_pose[approaching, 2] += self.approach_height * (1.0 - smooth)
            self.root_pose[approaching, 3:7] = self.start_quat[approaching]
            self.target_joint[approaching] = torch.lerp(
                self.start_joint[approaching], self.open_joint[approaching], smooth[:, None]
            )
            self.stage_step[approaching] += 1
            done = approaching[self.stage_step[approaching] >= self.approach_steps]
            self.stage_num[done] = CLOSE
            self.stage_step[done] = 0
            # Submit the final finger close target once at the transition.
            # The persistent PhysX drive keeps this target afterwards.
            if not self.is_hand and len(done):
                self.target_joint[done] = self.close_joint[done]

        closing = torch.where(active & (self.stage_num == CLOSE))[0]
        if len(closing):
            if self.is_hand:
                progress = ((self.stage_step[closing] + 1) / self.transition_steps[closing]).clamp(0.0, 1.0)
                smooth = progress * progress * (3.0 - 2.0 * progress)
                amount = torch.where(self.smooth_transition[closing], smooth, progress)
                self.root_pose[closing, :3] = torch.lerp(
                    self.start_pos[closing], self.end_pos[closing], amount[:, None]
                )
                self.root_pose[closing, 3:7] = quaternion_slerp(
                    self.start_quat[closing], self.end_quat[closing], amount[:, None]
                )
                self.target_joint[closing] = torch.lerp(
                    self.start_joint[closing], self.end_joint[closing], amount[:, None]
                )
            else:
                if self.apply_finger_z_hop:
                    current = self.joint_pos[closing][:, self.joint_index]
                    z_hop = self._finger_joint_z(current) - self.open_z[closing]
                else:
                    z_hop = torch.zeros(
                        len(closing), dtype=torch.float, device=self.device
                    )
                self.applied_z_hop[closing] = z_hop
                self.root_pose[closing, :3] = self.end_pos[closing]
                self.root_pose[closing, 2] += z_hop
                self.root_pose[closing, 3:7] = self.end_quat[closing]
            self.stage_step[closing] += 1

        stressing = torch.where(active & (self.stage_num == STRESS))[0]
        if len(stressing):
            self.root_pose[stressing, :3] = self.end_pos[stressing]
            self.root_pose[stressing, 3:7] = self.end_quat[stressing]
            self.target_joint[stressing] = self.end_joint[stressing]
            self.stage_step[stressing] += 1
        return self.target_joint

    def _relative_pose(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_inv = quaternion_conjugate(normalize_quaternion(robot_quat))
        relative_pos = quaternion_rotate(robot_inv, object_pos - robot_pos)
        relative_quat = quaternion_multiply(robot_inv, normalize_quaternion(object_quat))
        return relative_pos, normalize_quaternion(relative_quat)

    def _start_stress(
        self,
        env_ids: torch.Tensor,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        contact_area: torch.Tensor,
    ) -> None:
        if not len(env_ids):
            return
        rel_pos, rel_quat = self._relative_pose(
            object_pos[env_ids], object_quat[env_ids], robot_pos[env_ids], robot_quat[env_ids]
        )
        self.reference_rel_pos[env_ids] = rel_pos
        self.reference_rel_quat[env_ids] = rel_quat
        grasped_object_pos = object_pos[env_ids]
        grasped_object_quat = normalize_quaternion(object_quat[env_ids])
        self.stress_start_object_pos[env_ids] = grasped_object_pos
        self.stress_start_object_quat[env_ids] = grasped_object_quat
        self.pregrasp_center_error[env_ids] = torch.linalg.vector_norm(
            grasped_object_pos - self.object_initial_pos[env_ids], dim=1
        )
        self.pregrasp_rotation_error[env_ids] = quaternion_angle(
            grasped_object_quat, self.object_initial_quat[env_ids]
        )
        self.stress_center_error[env_ids] = 0.0
        self.stress_rotation_error[env_ids] = 0.0
        self.grasp_contact_area[env_ids] = contact_area[env_ids]
        # Pin the stress test to the pose PhysX actually reached. Root motion
        # is speed-limited, so this can differ slightly from the command pose.
        self.end_pos[env_ids] = robot_pos[env_ids] - self.env_origin[env_ids]
        self.end_quat[env_ids] = robot_quat[env_ids]
        self.root_pose[env_ids, :3] = self.end_pos[env_ids]
        self.root_pose[env_ids, 3:7] = self.end_quat[env_ids]
        self.end_joint[env_ids] = self.target_joint[env_ids]
        self.stage_num[env_ids] = STRESS
        self.stage_step[env_ids] = 0
        self.contact_lost_count[env_ids] = 0

    def _finish_without_record(self, env_ids: torch.Tensor, reason: str) -> None:
        env_ids = env_ids[~self.recorded[env_ids]]
        for env_id in env_ids.tolist():
            self.failed_reason[env_id] = reason
            self.attempt_history.append(
                {
                    "source_pregrasp_index": int(self.assigned_pregrasp[env_id]),
                    "result": reason,
                    "failure_stage": int(self.stage_num[env_id]),
                    "failure_stage_step": int(self.stage_step[env_id]),
                    "relative_translation_error_m": float(
                        self.relative_translation_error[env_id]
                    ),
                    "relative_rotation_error_deg": math.degrees(
                        float(self.relative_rotation_error[env_id])
                    ),
                    "contact_force_n": float(self.last_contact_force[env_id]),
                    "minimum_contact_separation_m": self._minimum_separation_value(
                        env_id
                    ),
                    "minimum_close_contact_separation_m": (
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_contact_penetration_mm": self._penetration_mm(
                        self._minimum_separation_value(env_id)
                    ),
                    "maximum_close_contact_penetration_mm": self._penetration_mm(
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_pre_stress_object_motion_m": round(
                        float(self.maximum_pre_stress_object_motion[env_id]), 7
                    ),
                    "finger_z_hop_m": round(float(self.applied_z_hop[env_id]), 7),
                    "finger_z_hop_enabled": self.apply_finger_z_hop,
                }
            )
        self.stage_num[env_ids] = DONE
        self.stage_step[env_ids] = 0
        self.recorded[env_ids] = True

    def observe(
        self,
        *,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        contact_force: torch.Tensor,
        minimum_contact_separation: torch.Tensor,
        contact_area: torch.Tensor,
        applied_force: torch.Tensor,
    ) -> torch.Tensor:
        """Advance contact-dependent stages and return episode-done mask."""
        self.joint_pos = joint_pos
        self.last_contact_force = contact_force
        self.max_test_force = torch.maximum(self.max_test_force, applied_force)
        finite_separation = torch.isfinite(minimum_contact_separation)
        closing_separation = (
            (self.action_enable == 1)
            & (self.stage_num == CLOSE)
            & finite_separation
        )
        self.minimum_close_contact_separation = torch.where(
            closing_separation,
            torch.minimum(
                self.minimum_close_contact_separation,
                minimum_contact_separation,
            ),
            self.minimum_close_contact_separation,
        )
        self.close_contact_seen |= (
            (self.action_enable == 1)
            & (self.stage_num == CLOSE)
            & (contact_force >= self.min_contact_force)
        )
        self.minimum_contact_separation = torch.where(
            finite_separation,
            torch.minimum(
                self.minimum_contact_separation, minimum_contact_separation
            ),
            self.minimum_contact_separation,
        )
        penetrating = torch.where(
            (self.action_enable == 1)
            & finite_separation
            & (
                minimum_contact_separation
                < -self.contact_penetration_threshold
            )
        )[0]
        self._finish_without_record(penetrating, "penetration")
        pre_stress = (
            (self.action_enable == 1)
            & ((self.stage_num == APPROACH) | (self.stage_num == CLOSE))
        )
        object_motion = torch.linalg.vector_norm(
            object_pos - self.object_initial_pos, dim=1
        )
        self.maximum_pre_stress_object_motion = torch.where(
            pre_stress,
            torch.maximum(self.maximum_pre_stress_object_motion, object_motion),
            self.maximum_pre_stress_object_motion,
        )
        moved_before_stress = torch.where(
            pre_stress
            & (object_motion > self.pre_stress_object_motion_threshold)
        )[0]
        self._finish_without_record(
            moved_before_stress, "pre_stress_object_motion"
        )
        closing = torch.where((self.action_enable == 1) & (self.stage_num == CLOSE))[0]
        if len(closing):
            current = joint_pos[closing][:, self.joint_index]
            error = torch.linalg.vector_norm(self.close_joint[closing] - current, dim=1)
            if self.is_hand:
                motion_done = self.stage_step[closing] >= self.transition_steps[closing]
                timed_out = self.stage_step[closing] >= (
                    self.transition_steps[closing] + self.close_timeout_steps
                )
                ready = motion_done & ((error < 0.03) | timed_out)
                # Net contact force can be zero for one frame when opposing
                # fingertip forces cancel. The direct stress test verifies
                # retention, so do not reject from this aggregate value alone.
                grasped = ready
                empty = torch.zeros_like(ready)
            else:
                delta = torch.abs(error - self.close_error_previous[closing])
                stalled = self.close_error_valid[closing] & (delta < 2.0e-4) & (
                    error > self.grasp_blocked_error
                )
                self.close_stall_count[closing] = torch.where(
                    stalled,
                    self.close_stall_count[closing] + 1,
                    torch.zeros_like(self.close_stall_count[closing]),
                )
                self.close_error_previous[closing] = error
                self.close_error_valid[closing] = True
                waited = self.stage_step[closing] >= max(5, int(0.08 / self.step_dt))
                timed_out = self.stage_step[closing] >= self.close_timeout_steps
                blocked = error > self.grasp_blocked_error
                contact_confirmed = self.close_contact_seen[closing]
                grasped = waited & blocked & contact_confirmed & (
                    (self.close_stall_count[closing] >= 4) | timed_out
                )
                # A mimic/limited articulation can stall a few milliradians
                # before its target even with no object present.  Joint error
                # alone must never start the external-force test.
                empty = timed_out & (~blocked | ~contact_confirmed)
            self._start_stress(
                closing[grasped],
                object_pos,
                object_quat,
                robot_pos,
                robot_quat,
                contact_area,
            )
            self._finish_without_record(closing[empty], "empty_close")

        tested = torch.where(
            (self.action_enable == 1) & (self.stage_num == STRESS)
        )[0]
        if len(tested):
            rel_pos, rel_quat = self._relative_pose(
                object_pos[tested], object_quat[tested], robot_pos[tested], robot_quat[tested]
            )
            position_error = torch.linalg.vector_norm(
                rel_pos - self.reference_rel_pos[tested], dim=1
            )
            rotation_error = quaternion_angle(rel_quat, self.reference_rel_quat[tested])
            self.relative_translation_error[tested] = position_error
            self.relative_rotation_error[tested] = rotation_error
            self.stress_center_error[tested] = torch.linalg.vector_norm(
                object_pos[tested] - self.stress_start_object_pos[tested], dim=1
            )
            self.stress_rotation_error[tested] = quaternion_angle(
                object_quat[tested], self.stress_start_object_quat[tested]
            )
            lost_contact = contact_force[tested] < self.min_contact_force
            self.contact_lost_count[tested] = torch.where(
                lost_contact,
                self.contact_lost_count[tested] + 1,
                torch.zeros_like(self.contact_lost_count[tested]),
            )
            drifted = (
                (position_error > self.max_relative_translation)
                | (rotation_error > self.max_relative_rotation)
            )
            contact_lost = (
                self.contact_lost_count[tested] >= self.contact_lost_steps
            )
            stress_envs = tested
            if len(stress_envs):
                progress = (self.stage_step[stress_envs].float() / self.stress_steps).clamp(0, 1)
                self.stress_survival[stress_envs] = progress
                failed_contact = stress_envs[contact_lost]
                failed_drift = stress_envs[drifted & ~contact_lost]
                completed = stress_envs[
                    ~drifted
                    & ~contact_lost
                    & (self.stage_step[stress_envs] >= self.stress_steps)
                ]
                self._record(failed_contact, "contact_lost")
                self._record(failed_drift, "stress_drop")
                self.stress_survival[completed] = 1.0
                self._record(completed, "completed")

        return (self.action_enable == 1) & (self.stage_num == DONE)

    def _grasp_box(self, source: dict) -> np.ndarray:
        center = np.asarray(source["target_points"], dtype=np.float64)
        yaw = math.radians(float(source["target_orientation"][2]))
        x_axis = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        y_axis = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0])
        if self.is_hand and source.get("grasp_bbox"):
            points = np.asarray(source["grasp_bbox"], dtype=np.float64).reshape(-1, 3)
            relative = points - center
            half_x = max(0.005, float(np.max(np.abs(relative @ x_axis))))
            half_y = max(0.005, float(np.max(np.abs(relative @ y_axis))))
        else:
            half_x = float(self.gripper["height"]) * 0.5
            half_y = float(source["target_width"]) * 0.5
        return np.asarray(
            [
                center + half_x * x_axis - half_y * y_axis,
                center - half_x * x_axis - half_y * y_axis,
                center - half_x * x_axis + half_y * y_axis,
                center + half_x * x_axis + half_y * y_axis,
            ]
        )

    def _minimum_separation_value(self, env_id: int) -> float | None:
        value = float(self.minimum_contact_separation[env_id])
        return round(value, 7) if math.isfinite(value) else None

    def _minimum_close_separation_value(self, env_id: int) -> float | None:
        value = float(self.minimum_close_contact_separation[env_id])
        return round(value, 7) if math.isfinite(value) else None

    @staticmethod
    def _penetration_mm(separation_m: float | None) -> float | None:
        if separation_m is None:
            return None
        return round(max(0.0, -float(separation_m)) * 1000.0, 4)

    def _record(self, env_ids: torch.Tensor, result: str) -> None:
        for env_id in env_ids.tolist():
            if self.recorded[env_id]:
                continue
            source_id = int(self.assigned_pregrasp[env_id])
            source = self.records[source_id]
            pregrasp_rotation_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.pregrasp_rotation_error[env_id])
                    / ROTATION_SCORE_RANGE_RAD,
                ),
            )
            stress_rotation_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.stress_rotation_error[env_id])
                    / ROTATION_SCORE_RANGE_RAD,
                ),
            )
            survival = min(
                1.0,
                max(0.0, float(self.stress_survival[env_id])),
            )
            pregrasp_center_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.pregrasp_center_error[env_id])
                    / CENTER_SCORE_RANGE_M,
                ),
            )
            stress_center_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.stress_center_error[env_id])
                    / CENTER_SCORE_RANGE_M,
                ),
            )
            pregrasp_pose_score = (
                CENTER_IN_POSE_WEIGHT * pregrasp_center_score
                + ROTATION_IN_POSE_WEIGHT * pregrasp_rotation_score
            )
            stress_pose_score = (
                CENTER_IN_POSE_WEIGHT * stress_center_score
                + ROTATION_IN_POSE_WEIGHT * stress_rotation_score
            )
            contact_area_score = min(
                1.0,
                max(
                    0.0,
                    float(self.grasp_contact_area[env_id]) * 1.0e6
                    / CONTACT_AREA_SCORE_RANGE_MM2,
                ),
            )
            score_weight_sum = (
                FORCE_SCORE_WEIGHT
                + PREGRASP_POSE_SCORE_WEIGHT
                + STRESS_POSE_SCORE_WEIGHT
                + CONTACT_AREA_SCORE_WEIGHT
            )
            if score_weight_sum <= 0.0:
                raise ValueError("At least one score weight must be positive")
            score = (
                FORCE_SCORE_WEIGHT * survival**SCORE_COMPONENT_POWER
                + PREGRASP_POSE_SCORE_WEIGHT
                * pregrasp_pose_score**SCORE_COMPONENT_POWER
                + STRESS_POSE_SCORE_WEIGHT
                * stress_pose_score**SCORE_COMPONENT_POWER
                + CONTACT_AREA_SCORE_WEIGHT
                * contact_area_score**SCORE_COMPONENT_POWER
            ) / score_weight_sum
            # Compatibility field for readers that still visualize rotation only.
            rotation_score = (
                PREGRASP_POSE_SCORE_WEIGHT * pregrasp_rotation_score
                + STRESS_POSE_SCORE_WEIGHT * stress_rotation_score
            ) / (
                PREGRASP_POSE_SCORE_WEIGHT
                + STRESS_POSE_SCORE_WEIGHT
            )
            pose_score = (
                PREGRASP_POSE_SCORE_WEIGHT * pregrasp_pose_score
                + STRESS_POSE_SCORE_WEIGHT * stress_pose_score
            ) / (PREGRASP_POSE_SCORE_WEIGHT + STRESS_POSE_SCORE_WEIGHT)
            quaternion = self.end_quat[env_id]
            matrix = torch.eye(4, dtype=torch.float, device=self.device)
            matrix[:3, :3] = quaternion_to_matrix(quaternion)
            matrix[:3, 3] = torch.tensor(
                source["target_points"], dtype=torch.float, device=self.device
            )
            record = {
                "grasp_box": self._grasp_box(source).round(7).tolist(),
                "grasp_mat": matrix.detach().cpu().numpy().round(7).tolist(),
                "target_points": copy.deepcopy(source["target_points"]),
                "target_orientation": copy.deepcopy(source["target_orientation"]),
                "target_width": copy.deepcopy(source.get("target_width", 0.0)),
                "target_object": source["target_object"],
                "source_pregrasp_index": source_id,
                "gripper_model": self.gripper["gripper_name"],
                "gripper_type": self.gripper["type"],
                "score": round(score, 6),
                "rotation_score": round(rotation_score, 6),
                "pose_score": round(pose_score, 6),
                "force_score": round(survival, 6),
                "contact_area_score": round(contact_area_score, 6),
                "pregrasp_pose_score": round(pregrasp_pose_score, 6),
                "stress_pose_score": round(stress_pose_score, 6),
                "pregrasp_center_score": round(pregrasp_center_score, 6),
                "stress_center_score": round(stress_center_score, 6),
                "pregrasp_rotation_score": round(pregrasp_rotation_score, 6),
                "stress_rotation_score": round(stress_rotation_score, 6),
                "normal": self.force_direction[env_id].detach().cpu().numpy().round(7).tolist(),
                "quality": {
                    "result": result,
                    "test_sequence": "close_then_stress",
                    "score_weights": {
                        "force": FORCE_SCORE_WEIGHT,
                        "pregrasp_pose": PREGRASP_POSE_SCORE_WEIGHT,
                        "stress_pose": STRESS_POSE_SCORE_WEIGHT,
                        "contact_area": CONTACT_AREA_SCORE_WEIGHT,
                    },
                    "score_component_power": SCORE_COMPONENT_POWER,
                    "pose_component_weights": {
                        "center": CENTER_IN_POSE_WEIGHT,
                        "rotation": ROTATION_IN_POSE_WEIGHT,
                    },
                    "contact_area_score_range_mm2": CONTACT_AREA_SCORE_RANGE_MM2,
                    "contact_area_estimator": "physx_manifold_footprint",
                    "grasp_contact_area_m2": round(
                        float(self.grasp_contact_area[env_id]), 9
                    ),
                    "grasp_contact_area_mm2": round(
                        float(self.grasp_contact_area[env_id]) * 1.0e6, 4
                    ),
                    "center_score_range_m": CENTER_SCORE_RANGE_M,
                    "rotation_score_range_deg": math.degrees(
                        ROTATION_SCORE_RANGE_RAD
                    ),
                    "stress_survival": round(survival, 6),
                    "max_test_force_n": round(float(self.max_test_force[env_id]), 6),
                    "relative_translation_error_m": round(
                        float(self.relative_translation_error[env_id]), 7
                    ),
                    "relative_rotation_error_deg": round(
                        math.degrees(float(self.relative_rotation_error[env_id])), 5
                    ),
                    "pregrasp_rotation_error_deg": round(
                        math.degrees(
                            float(self.pregrasp_rotation_error[env_id])
                        ),
                        5,
                    ),
                    "stress_rotation_error_deg": round(
                        math.degrees(
                            float(self.stress_rotation_error[env_id])
                        ),
                        5,
                    ),
                    "pregrasp_center_error_m": round(
                        float(self.pregrasp_center_error[env_id]), 7
                    ),
                    "stress_center_error_m": round(
                        float(self.stress_center_error[env_id]), 7
                    ),
                    "contact_force_n": round(float(self.last_contact_force[env_id]), 6),
                    "minimum_contact_separation_m": self._minimum_separation_value(
                        env_id
                    ),
                    "minimum_close_contact_separation_m": (
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_contact_penetration_mm": self._penetration_mm(
                        self._minimum_separation_value(env_id)
                    ),
                    "maximum_close_contact_penetration_mm": self._penetration_mm(
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_pre_stress_object_motion_m": round(
                        float(self.maximum_pre_stress_object_motion[env_id]), 7
                    ),
                    "finger_z_hop_m": round(float(self.applied_z_hop[env_id]), 7),
                    "finger_z_hop_enabled": self.apply_finger_z_hop,
                },
            }
            for key in (
                "preset_name", "target_base_tf", "target_joint_pos", "joint_unit",
                "transition", "selected_heightmap_yaw", "first_contact_z",
                "safety_margin", "grasp_bbox_sets", "contact_sensor_sets",
            ):
                if key in source:
                    record[key] = copy.deepcopy(source[key])
            # output_grasp is a success dataset. Failed stress attempts belong
            # only in attempt_history and must not become low-score grasps.
            if result == "completed":
                self.output_list.append(record)
            self.attempt_history.append(
                {
                    "source_pregrasp_index": source_id,
                    "result": result,
                    "score": score,
                    "force_score": survival,
                    "contact_area_score": contact_area_score,
                    "grasp_contact_area_m2": float(
                        self.grasp_contact_area[env_id]
                    ),
                    "pregrasp_pose_score": pregrasp_pose_score,
                    "stress_pose_score": stress_pose_score,
                    "pregrasp_center_score": pregrasp_center_score,
                    "stress_center_score": stress_center_score,
                    "pregrasp_rotation_score": pregrasp_rotation_score,
                    "stress_rotation_score": stress_rotation_score,
                    "relative_translation_error_m": float(
                        self.relative_translation_error[env_id]
                    ),
                    "relative_rotation_error_deg": math.degrees(
                        float(self.relative_rotation_error[env_id])
                    ),
                    "pregrasp_rotation_error_deg": math.degrees(
                        float(self.pregrasp_rotation_error[env_id])
                    ),
                    "stress_rotation_error_deg": math.degrees(
                        float(self.stress_rotation_error[env_id])
                    ),
                    "pregrasp_center_error_m": float(
                        self.pregrasp_center_error[env_id]
                    ),
                    "stress_center_error_m": float(
                        self.stress_center_error[env_id]
                    ),
                    "contact_force_n": float(self.last_contact_force[env_id]),
                    "minimum_contact_separation_m": self._minimum_separation_value(
                        env_id
                    ),
                    "minimum_close_contact_separation_m": (
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_contact_penetration_mm": self._penetration_mm(
                        self._minimum_separation_value(env_id)
                    ),
                    "maximum_close_contact_penetration_mm": self._penetration_mm(
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_pre_stress_object_motion_m": float(
                        self.maximum_pre_stress_object_motion[env_id]
                    ),
                    "finger_z_hop_m": float(self.applied_z_hop[env_id]),
                    "finger_z_hop_enabled": self.apply_finger_z_hop,
                }
            )
            self.recorded[env_id] = True
            self.stage_num[env_id] = DONE
            self.stage_step[env_id] = 0
            self.failed_reason[env_id] = result
