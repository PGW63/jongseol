"""MuJoCo 씬의 SO-101 을 움직이는 리더암.

LeRobot 의 SOLeader 로 정규화값을 읽고 MuJoCo 관절 목표(rad)로 바꿔 내보낸다.
"""

from lerobot.teleoperators.so_leader import SOLeader

from teleop.leader_mj.config import SO101LeaderMjConfig
from teleop.mapping import normalized_to_rad


class SO101LeaderMj(SOLeader):
    """SO-101 리더암 (Feetech STS3215, 5 DOF + 그리퍼).

    get_action() 은 {shoulder_pan.pos, ..., gripper.pos} 를 MuJoCo rad 로 돌려준다.
    방금 읽은 정규화값은 last_normalized 에 남는다.
    """

    # name 은 SOLeader 의 "so_leader" 를 그대로 둔다. 캘리브레이션 폴더 이름이라서,
    # 바꾸면 lerobot-calibrate --teleop.type=so101_leader 로 만든 파일을 못 찾는다.
    config_class = SO101LeaderMjConfig

    def __init__(self, config: SO101LeaderMjConfig):
        if config.use_degrees:
            raise ValueError("SO101LeaderMj 는 정규화값만 지원합니다 (use_degrees=False)")
        super().__init__(config)
        self.last_normalized: dict[str, float] = {}

    @property
    def is_calibrated(self) -> bool:
        # Feetech 버스는 가끔 깨진 응답("Incorrect status packet!")을 돌려준다.
        # LeRobot 의 캘리브레이션 확인은 재시도가 없어서 한 번 깨지면 연결 자체가 실패하므로 여기서 다시 읽는다.
        for attempt in range(self.config.num_read_retries + 1):
            try:
                return self.bus.is_calibrated
            except ConnectionError:
                if attempt == self.config.num_read_retries:
                    raise
        return False

    def get_action(self) -> dict[str, float]:
        raw = super().get_action()
        self.last_normalized = {key.removesuffix(".pos"): val for key, val in raw.items()}
        return {
            f"{joint}.pos": normalized_to_rad(joint, val, self.config.gripper_closed_safe)
            for joint, val in self.last_normalized.items()
        }
