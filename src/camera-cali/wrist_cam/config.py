"""wrist 카메라 설정과 내부 파라미터 (intrinsic_cali.py wrist 의 결과).

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WristCamConfig:
    # 장치 경로(udev 이름) 또는 카메라 번호
    port: str = '1'
    # 캘리브레이션한 해상도. 다른 해상도로 쓰면 camera_matrix 가 맞지 않는다
    width: int = 640
    # 캘리브레이션한 해상도
    height: int = 480
    # False 면 아래 값은 비어 있다
    calibrated: bool = False
    # 3x3 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (px)
    camera_matrix: list[list[float]] | None = None
    # OpenCV 왜곡 계수 [k1, k2, p1, p2, k3]
    dist_coeffs: list[float] | None = None
    # 재투영 오차 RMS (px). 1 보다 크면 다시 찍는 게 좋다
    rms_px: float | None = None
    # 캘리브레이션에 쓴 이미지 수
    num_images: int = 0
    # 수직 화각 (도). MuJoCo 카메라의 fovy 에 넣는 값
    fovy_deg: float | None = None
    # 쓴 체커보드: 안쪽 꼭짓점 가로 수
    board_cols: int | None = None
    # 쓴 체커보드: 안쪽 꼭짓점 세로 수
    board_rows: int | None = None
    # 쓴 체커보드: 한 칸의 변 길이 (m)
    board_square: float | None = None
