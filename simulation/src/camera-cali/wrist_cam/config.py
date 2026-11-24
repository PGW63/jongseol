"""wrist 카메라 설정과 내부 파라미터 (charuco_intrinsic_cali.py wrist 의 결과).

캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WristCamConfig:
    # 장치 경로(udev 이름) 또는 카메라 번호
    port: str = 'uid:0x11000032e40234'
    # 캘리브레이션한 해상도. 다른 해상도로 쓰면 camera_matrix 가 맞지 않는다
    width: int = 640
    # 캘리브레이션한 해상도
    height: int = 480
    # False 면 아래 값은 비어 있다
    calibrated: bool = True
    # 3x3 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (px)
    camera_matrix: list[list[float]] | None = field(default_factory=lambda: [[425.9891839991416, 0.0, 315.2870614159798], [0.0, 426.6514114452565, 230.96955252622817],
         [0.0, 0.0, 1.0]])
    # OpenCV 왜곡 계수 [k1, k2, p1, p2, k3]
    dist_coeffs: list[float] | None = field(default_factory=lambda: [-0.40376738757682884, 0.1941585065376351, 0.0008671692911862154, -4.761505057447339e-05,
         -0.052396485130153776])
    # 재투영 오차 RMS (px). 1 보다 크면 다시 찍는 게 좋다
    rms_px: float | None = 0.20437402813736993
    # 캘리브레이션에 쓴 이미지 수
    num_images: int = 32
    # 수직 화각 (도). MuJoCo 카메라의 fovy 에 넣는 값
    fovy_deg: float | None = 58.71725784638814
    # 쓴 체커보드: 안쪽 꼭짓점 가로 수
    board_cols: int | None = None
    # 쓴 체커보드: 안쪽 꼭짓점 세로 수
    board_rows: int | None = None
    # 쓴 체커보드: 한 칸의 변 길이 (m)
    board_square: float | None = None
    # 쓴 ChArUco 보드: 가로 칸 수 (체커보드로 했으면 None)
    charuco_squares_x: int | None = 5
    # 쓴 ChArUco 보드: 세로 칸 수
    charuco_squares_y: int | None = 7
    # 쓴 ChArUco 보드: 한 칸의 변 길이 (m)
    charuco_square: float | None = 0.0368
    # 쓴 ChArUco 보드: ArUco 마커 한 변 길이 (m)
    charuco_marker: float | None = 0.02208
    # 쓴 ChArUco 보드: ArUco 사전 이름
    charuco_dict: str | None = 'DICT_6X6_250'
    # 쓴 ChArUco 보드: OpenCV 4.6 이전 배치(legacy pattern) 여부
    charuco_legacy: bool | None = False
