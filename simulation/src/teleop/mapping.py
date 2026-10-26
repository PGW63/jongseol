"""리더암 정규화값 <-> MuJoCo 관절 각도(rad) 변환."""

from teleop.constants import LEADER_RANGE_ARM, LEADER_RANGE_GRIPPER, MUJOCO_JOINT_RANGE


def _ranges(joint: str, gripper_closed: float) -> tuple[tuple[float, float], tuple[float, float]]:
    if joint == "gripper":
        return LEADER_RANGE_GRIPPER, (gripper_closed, MUJOCO_JOINT_RANGE[joint][1])
    return LEADER_RANGE_ARM, MUJOCO_JOINT_RANGE[joint]


def normalized_to_rad(joint: str, value: float, gripper_closed: float) -> float:
    """정규화값(팔 -100~100, 그리퍼 0~100)을 MuJoCo 관절 범위에 선형으로 옮긴다."""
    (lo, hi), (mj_lo, mj_hi) = _ranges(joint, gripper_closed)
    return mj_lo + (value - lo) / (hi - lo) * (mj_hi - mj_lo)


def rad_to_normalized(joint: str, rad: float, gripper_closed: float) -> float:
    """normalized_to_rad 의 역변환."""
    (lo, hi), (mj_lo, mj_hi) = _ranges(joint, gripper_closed)
    return lo + (rad - mj_lo) / (mj_hi - mj_lo) * (hi - lo)
