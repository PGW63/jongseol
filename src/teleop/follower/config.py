"""실제 SO-101 팔로워암 설정."""

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower import SOFollowerConfig

from teleop.constants import default_port


# type 이름은 LeRobot 내장 "so101_follower" 와 겹치지 않게 한다.
# SOFollowerConfig 를 상속하므로 LeRobot 의 SOFollower 에 그대로 넘길 수 있다.
@RobotConfig.register_subclass("so101_real_follower")
@dataclass
class SO101FollowerConfig(RobotConfig, SOFollowerConfig):
    """실제 SO-101 팔로워암 (Feetech STS3215)."""

    # 고정 포트 이름. ubuntu: /dev/so101_follower, macOS: ~/dev/so101_follower
    port: str = default_port("so101_follower")
    # 리더와 같은 단위를 써야 한다. False = 정규화값, True = 도(degree)
    use_degrees: bool = False
