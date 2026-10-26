"""MuJoCo 씬의 SO-101 을 움직이는 리더암 설정 (팔로워 없이 리더만 연결)."""

from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

from teleop.constants import default_port

# simulation/ (src/teleop/leader_mj/config.py 기준)
SIM_ROOT = Path(__file__).resolve().parents[3]


# SOLeaderConfig 를 상속하므로 LeRobot 의 SOLeader 에 그대로 넘길 수 있다.
@TeleoperatorConfig.register_subclass("so101_leader_mj")
@dataclass
class SO101LeaderMjConfig(TeleoperatorConfig, SOLeaderConfig):
    """리더암 정규화값을 MuJoCo 관절 목표(rad)로 바꿔 내보내는 리더 설정."""

    # 고정 포트 이름. ubuntu: /dev/so101_leader_mj, macOS: ~/dev/so101_leader_mj
    port: str = default_port("so101_leader_mj")
    # False = 정규화값(-100~100, 그리퍼 0~100). MuJoCo 매핑은 정규화값만 지원한다.
    use_degrees: bool = False
    # 읽기가 깨졌을 때 다시 시도하는 횟수. 녹화 중에 한 번 깨졌다고 세션이 끝나지 않게 넉넉히 둔다.
    num_read_retries: int = 5
    # 캘리브레이션 파일 이름 (lerobot-calibrate 의 --teleop.id 와 같아야 한다)
    id: str | None = "so101_leader_mj"

    # 리더를 연결할 씬 (실행 위치와 무관하게 simulation/models 기준)
    xml_path: Path = SIM_ROOT / "models" / "scene" / "scene_trolley.xml"
    # 씬에서 SO-101 관절/액추에이터 이름 앞에 붙는 접두사 (attach prefix)
    joint_prefix: str = "so101_"

    # 환경을 초기화할 때 물체를 흩뜨리는 범위 파일 (상대경로면 실행 위치 기준). None 이면 항상 원래 자리.
    randomize_path: Path | None = None
    randomize_seed: int | None = None

    # 리더 시리얼 읽기가 안정적인 주기 (Hz)
    fps: int = 20

    # 그리퍼 "완전히 닫힘" 을 매핑할 값 (rad). ctrlrange 하한(-0.17453)까지 보내면
    # 양쪽 조가 서로 충돌해 그리퍼가 튕겨 열리므로 그보다 조금 위에서 멈춘다.
    gripper_closed_safe: float = -0.1
