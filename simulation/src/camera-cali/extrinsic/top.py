"""top 카메라가 로봇의 base 링크에서 어디에 달려 있는지 (extrinsic_cali.py top 의 결과).

카메라 축은 OpenCV 관례다: x 오른쪽, y 아래, z 보는 방향. 길이는 m, 쿼터니언은 w x y z.
link 가 base 가 아니면 그 링크(MuJoCo body so101_<link>)와 같이 움직인다: T_base_cam = FK(q) @ T_link_cam.

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopCamExtrinsicConfig:
    # 기준 링크. base 면 고정 카메라
    link: str = 'base'
    # False 면 아래 값은 비어 있다
    calibrated: bool = True
    # 4x4. 카메라 좌표의 점을 링크 좌표로 옮긴다
    T_link_cam: list[list[float]] = field(default_factory=lambda: [[0.7049708416673188, -0.6599735775316468, 0.25971328298520147, 0.023712292449994656],
         [-0.7081464095269884, -0.634703390090763, 0.3093222741435337, -0.08239058512663233],
         [-0.03930362671442364, -0.40197821280185325, -0.9148053024331031, 0.5414690531189557],
         [0.0, 0.0, 0.0, 1.0]])
    # 링크 기준 카메라 위치
    position: list[float] = field(default_factory=lambda: [0.023712292449994656, -0.08239058512663233, 0.5414690531189557])
    # 링크 기준 카메라 방향
    quat_wxyz: list[float] = field(default_factory=lambda: [-0.1971434434260069, 0.9020088045844099, -0.3791869824621558, 0.061088554554722174])
    # 아래 <camera> 를 넣을 곳 (worldbody 또는 so101.xml 의 body)
    mujoco_body: str = 'worldbody (scene_trolley.xml)'
    # <camera pos=...>
    mujoco_pos: list[float] = field(default_factory=lambda: [0.054712292449994655, 0.08010941487336767, 1.2738690531189558])
    # <camera xyaxes=...>
    mujoco_xyaxes: list[float] = field(default_factory=lambda: [0.7049708416673188, -0.7081464095269884, -0.03930362671442364, 0.6599735775316468,
         0.634703390090763, 0.40197821280185325])
    # <camera fovy=...> (intrinsic 의 값)
    mujoco_fovy: float | None = 40.73036388070773
    # 평균 낸 표본 수
    num_samples: int = 51
    # 쓴 보드 자리
    placements: list[str] = field(default_factory=lambda: ['g1'])
    # 표본들의 위치가 평균에서 벗어난 정도 RMS (mm)
    spread_mm: float = 41.68138240498655
    # 표본들의 방향이 평균에서 벗어난 정도 RMS (도)
    spread_deg: float = 6.69138696963138
    # 보드 재투영 오차 평균 (px)
    reprojection_px: float = 19.043747602773788
