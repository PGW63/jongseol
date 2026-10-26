"""헤드캠(top 카메라)이 로봇 베이스에서 어디에 달려 있는지 (headCam_to_Base.py 의 결과).

카메라 축은 OpenCV 관례다: x 오른쪽, y 아래, z 보는 방향. 길이는 m, 쿼터니언은 w x y z.

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HeadCamExtrinsicConfig:
    # False 면 아래 값은 비어 있다
    calibrated: bool = False
    # 4x4. 카메라 좌표의 점을 로봇 베이스 좌표로 옮긴다
    T_base_cam: list[list[float]] | None = None
    # 베이스 기준 카메라 위치
    position: list[float] | None = None
    # 베이스 기준 카메라 방향
    quat_wxyz: list[float] | None = None
    # 트롤리 원점 꼭짓점 기준 카메라 위치 (상판 윗면이 Z=0)
    trolley_position: list[float] | None = None
    # 트롤리 기준 카메라 방향 (베이스와 방향은 같고 위치만 다르다)
    trolley_quat_wxyz: list[float] | None = None
    # 씬 XML 의 <camera pos=...> 에 넣는 값
    mujoco_pos: list[float] | None = None
    # 씬 XML 의 <camera xyaxes=...> 에 넣는 값
    mujoco_xyaxes: list[float] | None = None
    # 씬 XML 의 <camera fovy=...> 에 넣는 값
    mujoco_fovy: float | None = None
    # 평균 낸 표본 수
    num_samples: int = 0
    # 재투영 오차 평균 (px)
    reprojection_px: float | None = None
