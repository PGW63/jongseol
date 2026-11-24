"""wrist 카메라가 로봇의 gripper 링크에서 어디에 달려 있는지 (extrinsic_cali.py wrist 의 결과).

카메라 축은 OpenCV 관례다: x 오른쪽, y 아래, z 보는 방향. 길이는 m, 쿼터니언은 w x y z.
link 가 base 가 아니면 그 링크(MuJoCo body so101_<link>)와 같이 움직인다: T_base_cam = FK(q) @ T_link_cam.

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WristCamExtrinsicConfig:
    # 기준 링크. base 면 고정 카메라
    link: str = 'gripper'
    # False 면 아래 값은 비어 있다
    calibrated: bool = True
    # 4x4. 카메라 좌표의 점을 링크 좌표로 옮긴다
    T_link_cam: list[list[float]] = field(default_factory=lambda: [[-0.0842633933204832, -0.8231865483106587, -0.5614833810777488, 0.052433569441105485],
         [-0.9962196843238592, 0.08153931390723455, 0.029961322621129948, -0.0014460393255961567],
         [0.02111921191208399, 0.561885439362794, -0.8269454225765396, -0.07242286189283592],
         [0.0, 0.0, 0.0, 1.0]])
    # 링크 기준 카메라 위치
    position: list[float] = field(default_factory=lambda: [0.052433569441105485, -0.0014460393255961567, -0.07242286189283592])
    # 링크 기준 카메라 방향
    quat_wxyz: list[float] = field(default_factory=lambda: [-0.20635557783242223, -0.6444266279703263, 0.7058236553495953, 0.20962982662106355])
    # 아래 <camera> 를 넣을 곳 (worldbody 또는 so101.xml 의 body)
    mujoco_body: str = '<body name="gripper"> (so101.xml)'
    # <camera pos=...>
    mujoco_pos: list[float] = field(default_factory=lambda: [0.052433569441105485, -0.0014460393255961567, -0.07242286189283592])
    # <camera xyaxes=...>
    mujoco_xyaxes: list[float] = field(default_factory=lambda: [-0.0842633933204832, -0.9962196843238592, 0.02111921191208399, 0.8231865483106587,
         -0.08153931390723455, -0.561885439362794])
    # <camera fovy=...> (intrinsic 의 값)
    mujoco_fovy: float | None = 58.71725784638814
    # 평균 낸 표본 수
    num_samples: int = 57
    # 쓴 보드 자리
    placements: list[str] = field(default_factory=lambda: ['b1'])
    # 표본들의 위치가 평균에서 벗어난 정도 RMS (mm)
    spread_mm: float = 11.944314724715015
    # 표본들의 방향이 평균에서 벗어난 정도 RMS (도)
    spread_deg: float = 5.2968073563181335
    # 보드 재투영 오차 평균 (px)
    reprojection_px: float = 0.3106314277178363
