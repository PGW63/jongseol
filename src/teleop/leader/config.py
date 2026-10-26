"""실제 SO-101 리더암 설정 (실제 팔로워와 짝)."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

from teleop.constants import default_port


# type 이름은 LeRobot 내장 "so101_leader" 와 겹치지 않게 한다.
# SOLeaderConfig 를 상속하므로 LeRobot 의 SOLeader 에 그대로 넘길 수 있다.
@TeleoperatorConfig.register_subclass("so101_leader_arm")
@dataclass
class SO101LeaderConfig(TeleoperatorConfig, SOLeaderConfig):
    """실제 SO-101 리더암 (Feetech STS3215)."""

    # 고정 포트 이름. ubuntu: /dev/so101_leader, macOS: ~/dev/so101_leader
    port: str = default_port("so101_leader")
    # False = 정규화값(-100~100, 그리퍼 0~100), True = 도(degree)
    use_degrees: bool = False
