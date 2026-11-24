"""헤드캠(top 카메라)이 로봇 베이스에서 어디에 달려 있는지 (headCam_to_Base.py 의 결과).

카메라 축은 OpenCV 관례다: x 오른쪽, y 아래, z 보는 방향. 길이는 m, 쿼터니언은 w x y z.

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HeadCamExtrinsicConfig:
    # False 면 아래 값은 비어 있다
    calibrated: bool = True
    # 4x4. 카메라 좌표의 점을 로봇 베이스 좌표로 옮긴다
    T_base_cam: list[list[float]] | None = field(default_factory=lambda: [[0.5352725037417156, -0.7335870237906478, 0.41872834303884454, 0.018204304247540208],
         [-0.8254257580758761, -0.5595161363227033, 0.07492670484799993, 0.07208824133166156],
         [0.17932000625401703, -0.38573536488179877, -0.9050152284003655, 0.48503195347302464],
         [0.0, 0.0, 0.0, 1.0]])
    # 베이스 기준 카메라 위치
    position: list[float] | None = field(default_factory=lambda: [0.018204304247540208, 0.07208824133166156, 0.48503195347302464])
    # 베이스 기준 카메라 방향
    quat_wxyz: list[float] | None = field(default_factory=lambda: [-0.13298603217880334, 0.8659970941730671, -0.4500629368032728, 0.1726473314162597])
    # 트롤리 원점 꼭짓점 기준 카메라 위치 (상판 윗면이 Z=0)
    trolley_position: list[float] | None = field(default_factory=lambda: [0.04920430424754021, 0.23458824133166156, 0.4874319534730247])
    # 트롤리 기준 카메라 방향 (베이스와 방향은 같고 위치만 다르다)
    trolley_quat_wxyz: list[float] | None = field(default_factory=lambda: [-0.13298603217880334, 0.8659970941730671, -0.4500629368032728, 0.1726473314162597])
    # 씬 XML 의 <camera pos=...> 에 넣는 값
    mujoco_pos: list[float] | None = field(default_factory=lambda: [0.04920430424754021, 0.23458824133166156, 1.2174319534730247])
    # 씬 XML 의 <camera xyaxes=...> 에 넣는 값
    mujoco_xyaxes: list[float] | None = field(default_factory=lambda: [0.5352725037417156, -0.8254257580758761, 0.17932000625401703, 0.7335870237906478,
         0.5595161363227033, 0.38573536488179877])
    # 씬 XML 의 <camera fovy=...> 에 넣는 값
    mujoco_fovy: float | None = 47.69972589023009
    # 평균 낸 표본 수
    num_samples: int = 5
    # 재투영 오차 평균 (px)
    reprojection_px: float | None = 0.8672662615776062
