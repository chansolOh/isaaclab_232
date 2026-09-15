"""Finger gripper width/joint/Z calibration helpers."""

from __future__ import annotations

from collections.abc import Iterable


CALIBRATION_KEYS = (
    "joint_to_z_coeffs",
    "joint_to_width_coeffs",
    "width_to_joint_coeffs",
)


def direct_prismatic(gripper: dict) -> bool:
    return str(gripper.get("type", "")).strip().lower() == "finger2"


def joint_names(gripper: dict) -> list[str]:
    names = list(gripper.get("joint_cfg", {}))
    family = str(gripper.get("type", "")).lower()
    if not family.startswith(("finger2", "finger3")):
        raise ValueError(f"Unsupported finger gripper type: {family}")
    if len(names) != 1:
        raise ValueError(
            f"{gripper.get('gripper_name')} must expose one driven mimic joint; got {names}"
        )
    return names


def validate(gripper: dict) -> None:
    joint_names(gripper)
    if direct_prismatic(gripper):
        missing = [key for key in ("open_joint_deg", "close_joint_deg") if key not in gripper]
    else:
        missing = [
            key
            for key in CALIBRATION_KEYS
            if not isinstance(gripper.get(key), list) or len(gripper[key]) != 4
        ]
    if missing:
        raise ValueError(f"{gripper.get('gripper_name')} calibration missing/invalid: {missing}")


def polynomial(coefficients: Iterable[float], value):
    result = value * 0.0
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result

