"""SO-101 리더/팔로워가 같이 쓰는 상수."""

import sys
from pathlib import Path


def default_port(name: str) -> str:
    """scripts/auto_rules_mapping* 이 만들어 주는 고정 포트 이름.

    ubuntu 는 udev 규칙으로 /dev/<name>, macOS 는 udev 가 없어서 ~/dev/<name> 심볼릭 링크.
    """
    if sys.platform == "darwin":
        return str(Path.home() / "dev" / name)
    return f"/dev/{name}"


# 표준 관절 순서
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINT_NAMES = JOINT_NAMES[:-1]

# 리더암이 내보내는 정규화 범위 (FeetechMotorsBus norm mode)
LEADER_RANGE_ARM = (-100.0, 100.0)  # RANGE_M100_100: 팔 관절
LEADER_RANGE_GRIPPER = (0.0, 100.0)  # RANGE_0_100: 그리퍼

# MuJoCo 관절 범위 (rad). models/so101/so101.xml 액추에이터의 ctrlrange 와 같아야 한다.
MUJOCO_JOINT_RANGE = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.17453, 1.74533),
}
