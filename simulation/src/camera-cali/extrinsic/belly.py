"""belly 카메라가 로봇의 shoulder 링크에서 어디에 달려 있는지 (extrinsic_cali.py belly 의 결과).

카메라 축은 OpenCV 관례다: x 오른쪽, y 아래, z 보는 방향. 길이는 m, 쿼터니언은 w x y z.
link 가 base 가 아니면 그 링크(MuJoCo body so101_<link>)와 같이 움직인다: T_base_cam = FK(q) @ T_link_cam.

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BellyCamExtrinsicConfig:
    # 기준 링크. base 면 고정 카메라
    link: str = 'shoulder'
    # False 면 아래 값은 비어 있다
    calibrated: bool = True
    # 4x4. 카메라 좌표의 점을 링크 좌표로 옮긴다
    T_link_cam: list[list[float]] = field(default_factory=lambda: [[-0.056228562336616, 0.042349233019463244, -0.9975193688546715, -0.05218049695632101],
         [-0.998409364006664, -0.0065218965287398446, 0.05600184576849478, 0.017797929894080657],
         [-0.004134082893114094, 0.9990815819182798, 0.042648587670651425, 0.03480170862175448],
         [0.0, 0.0, 0.0, 1.0]])
    # 링크 기준 카메라 위치
    position: list[float] = field(default_factory=lambda: [-0.05218049695632101, 0.017797929894080657, 0.03480170862175448])
    # 링크 기준 카메라 방향
    quat_wxyz: list[float] = field(default_factory=lambda: [-0.4949490198003469, -0.4763519566773799, 0.5017614169446535, 0.5256897960147238])
    # 아래 <camera> 를 넣을 곳 (worldbody 또는 so101.xml 의 body)
    mujoco_body: str = '<body name="shoulder"> (so101.xml)'
    # <camera pos=...>
    mujoco_pos: list[float] = field(default_factory=lambda: [-0.05218049695632101, 0.017797929894080657, 0.03480170862175448])
    # <camera xyaxes=...>
    mujoco_xyaxes: list[float] = field(default_factory=lambda: [-0.056228562336616, -0.998409364006664, -0.004134082893114094, -0.042349233019463244,
         0.0065218965287398446, -0.9990815819182798])
    # <camera fovy=...> (intrinsic 의 값)
    mujoco_fovy: float | None = 41.06613812964992
    # 평균 낸 표본 수
    num_samples: int = 5
    # 쓴 보드 자리
    placements: list[str] = field(default_factory=lambda: ['b1'])
    # 표본들의 위치가 평균에서 벗어난 정도 RMS (mm)
    spread_mm: float = 2.4806502561174493
    # 표본들의 방향이 평균에서 벗어난 정도 RMS (도)
    spread_deg: float = 0.7526794278312274
    # 보드 재투영 오차 평균 (px)
    reprojection_px: float = 0.18895160257816315
