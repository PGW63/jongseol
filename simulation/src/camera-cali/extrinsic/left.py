"""left 카메라가 로봇의 base 링크에서 어디에 달려 있는지 (extrinsic_cali.py left 의 결과).

카메라 축은 OpenCV 관례다: x 오른쪽, y 아래, z 보는 방향. 길이는 m, 쿼터니언은 w x y z.
link 가 base 가 아니면 그 링크(MuJoCo body so101_<link>)와 같이 움직인다: T_base_cam = FK(q) @ T_link_cam.

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeftCamExtrinsicConfig:
    # 기준 링크. base 면 고정 카메라
    link: str = 'base'
    # False 면 아래 값은 비어 있다
    calibrated: bool = True
    # 4x4. 카메라 좌표의 점을 링크 좌표로 옮긴다
    T_link_cam: list[list[float]] = field(default_factory=lambda: [[-0.7743820563223732, -0.4267496776780748, 0.4671371784043679, 0.13154553284167197],
         [-0.630907066331099, 0.46497886113958603, -0.6210885060495165, 0.1755458712239045],
         [0.04784040655576563, -0.7756799412741014, -0.6293106738373238, 0.23273462076835175],
         [0.0, 0.0, 0.0, 1.0]])
    # 링크 기준 카메라 위치
    position: list[float] = field(default_factory=lambda: [0.13154553284167197, 0.1755458712239045, 0.23273462076835175])
    # 링크 기준 카메라 방향
    quat_wxyz: list[float] = field(default_factory=lambda: [0.12378017912805195, -0.31222978572493904, 0.8468576608998828, -0.41233861126065513])
    # 아래 <camera> 를 넣을 곳 (worldbody 또는 so101.xml 의 body)
    mujoco_body: str = 'worldbody (scene_trolley.xml)'
    # <camera pos=...>
    mujoco_pos: list[float] = field(default_factory=lambda: [0.16254553284167197, 0.3380458712239045, 0.9651346207683518])
    # <camera xyaxes=...>
    mujoco_xyaxes: list[float] = field(default_factory=lambda: [-0.7743820563223732, -0.630907066331099, 0.04784040655576563, 0.4267496776780748,
         -0.46497886113958603, 0.7756799412741014])
    # <camera fovy=...> (intrinsic 의 값)
    mujoco_fovy: float | None = 59.17347323709556
    # 평균 낸 표본 수
    num_samples: int = 5
    # 쓴 보드 자리
    placements: list[str] = field(default_factory=lambda: ['b1'])
    # 표본들의 위치가 평균에서 벗어난 정도 RMS (mm)
    spread_mm: float = 0.061398945905032275
    # 표본들의 방향이 평균에서 벗어난 정도 RMS (도)
    spread_deg: float = 0.0170254189679254
    # 보드 재투영 오차 평균 (px)
    reprojection_px: float = 0.3433592259883881
