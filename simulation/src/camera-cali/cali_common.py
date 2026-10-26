"""intrinsic_cali.py 와 headCam_to_Base.py 가 같이 쓰는 것: 카메라 열기, 체커보드 찾기, 화면."""

import importlib
import os
import pprint
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


CAMERAS = ("top", "wrist")


def default_camera(name: str) -> str:
    """카메라의 고정 이름. ubuntu 는 udev 규칙으로 만든 /dev/so101_cam_<name>,
    macOS 는 카메라가 /dev 에 파일로 나오지 않아서 번호(top = 0, wrist = 1)로 연다."""
    return str(CAMERAS.index(name)) if sys.platform == "darwin" else f"/dev/so101_cam_{name}"


def _literal(value) -> str:
    """dataclass 기본값으로 쓸 파이썬 표현. 리스트는 default_factory 로 감싼다."""
    if isinstance(value, list):
        text = pprint.pformat(value, width=100, compact=True).replace("\n", "\n        ")
        return f"field(default_factory=lambda: {text})"
    return repr(value)


def _write_config(folder: str, cls: str, doc: str, fields: list[tuple[str, str, object, str]]) -> Path:
    """<folder>/config.py 를 통째로 다시 쓴다. fields: (이름, 타입, 값, 설명)."""
    lines = [f'"""{doc}', "", "캘리브레이션 스크립트가 통째로 다시 쓰는 파일이다. 손으로 고친 내용은 다음 캘리 때 사라진다.", '"""', "",
             "from dataclasses import dataclass, field", "", "", "@dataclass(frozen=True)", f"class {cls}:"]
    for name, typ, value, comment in fields:
        lines += [f"    # {comment}", f"    {name}: {typ} = {_literal(value)}"]
    path = HERE / folder / "config.py"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / folder / "__init__.py").write_text(
        f"from {folder}.config import {cls}\n\n__all__ = [\"{cls}\"]\n", encoding="utf-8")
    return path


def write_intrinsic_config(name: str, port: str, width: int, height: int, result: dict | None) -> Path:
    """<name>_cam/config.py. result 가 None 이면 아직 캘리브레이션하지 않은 기본 상태로 쓴다."""
    r = result or {}
    board = r.get("board", {})
    return _write_config(f"{name}_cam", f"{name.capitalize()}CamConfig", f"{name} 카메라 설정과 내부 파라미터 (intrinsic_cali.py {name} 의 결과).", [
        ("port", "str", port, "장치 경로(udev 이름) 또는 카메라 번호"),
        ("width", "int", width, "캘리브레이션한 해상도. 다른 해상도로 쓰면 camera_matrix 가 맞지 않는다"),
        ("height", "int", height, "캘리브레이션한 해상도"),
        ("calibrated", "bool", result is not None, "False 면 아래 값은 비어 있다"),
        ("camera_matrix", "list[list[float]] | None", r.get("camera_matrix"), "3x3 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (px)"),
        ("dist_coeffs", "list[float] | None", r.get("dist_coeffs"), "OpenCV 왜곡 계수 [k1, k2, p1, p2, k3]"),
        ("rms_px", "float | None", r.get("rms_px"), "재투영 오차 RMS (px). 1 보다 크면 다시 찍는 게 좋다"),
        ("num_images", "int", r.get("num_images", 0), "캘리브레이션에 쓴 이미지 수"),
        ("fovy_deg", "float | None", r.get("fovy_deg"), "수직 화각 (도). MuJoCo 카메라의 fovy 에 넣는 값"),
        ("board_cols", "int | None", board.get("cols"), "쓴 체커보드: 안쪽 꼭짓점 가로 수"),
        ("board_rows", "int | None", board.get("rows"), "쓴 체커보드: 안쪽 꼭짓점 세로 수"),
        ("board_square", "float | None", board.get("square"), "쓴 체커보드: 한 칸의 변 길이 (m)"),
    ])


def write_extrinsic_config(result: dict | None) -> Path:
    """extrinsic/config.py. result 가 None 이면 아직 캘리브레이션하지 않은 기본 상태로 쓴다."""
    r = result or {}
    return _write_config("extrinsic", "HeadCamExtrinsicConfig", "헤드캠(top 카메라)이 로봇 베이스에서 어디에 달려 있는지 (headCam_to_Base.py 의 결과).\n\n"
                         "카메라 축은 OpenCV 관례다: x 오른쪽, y 아래, z 보는 방향. 길이는 m, 쿼터니언은 w x y z.", [
        ("calibrated", "bool", result is not None, "False 면 아래 값은 비어 있다"),
        ("T_base_cam", "list[list[float]] | None", r.get("T_base_cam"), "4x4. 카메라 좌표의 점을 로봇 베이스 좌표로 옮긴다"),
        ("position", "list[float] | None", r.get("position"), "베이스 기준 카메라 위치"),
        ("quat_wxyz", "list[float] | None", r.get("quat_wxyz"), "베이스 기준 카메라 방향"),
        ("trolley_position", "list[float] | None", r.get("trolley_position"), "트롤리 원점 꼭짓점 기준 카메라 위치 (상판 윗면이 Z=0)"),
        ("trolley_quat_wxyz", "list[float] | None", r.get("trolley_quat_wxyz"), "트롤리 기준 카메라 방향 (베이스와 방향은 같고 위치만 다르다)"),
        ("mujoco_pos", "list[float] | None", r.get("mujoco_pos"), "씬 XML 의 <camera pos=...> 에 넣는 값"),
        ("mujoco_xyaxes", "list[float] | None", r.get("mujoco_xyaxes"), "씬 XML 의 <camera xyaxes=...> 에 넣는 값"),
        ("mujoco_fovy", "float | None", r.get("mujoco_fovy"), "씬 XML 의 <camera fovy=...> 에 넣는 값"),
        ("num_samples", "int", r.get("num_samples", 0), "평균 낸 표본 수"),
        ("reprojection_px", "float | None", r.get("reprojection_px"), "재투영 오차 평균 (px)"),
    ])


def load_camera_config(name: str):
    """<name>_cam/config.py 의 설정 객체. 파일이 비어 있으면 기본 상태로 먼저 만든다."""
    cls = f"{name.capitalize()}CamConfig"
    if cls not in (HERE / f"{name}_cam" / "config.py").read_text(encoding="utf-8"):
        write_intrinsic_config(name, default_camera(name), 640, 480, None)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    module = importlib.reload(importlib.import_module(f"{name}_cam.config"))
    return getattr(module, cls)()


def open_camera(spec: str, width: int, height: int) -> cv2.VideoCapture:
    """spec 이 숫자면 카메라 번호, 아니면 장치 경로 (udev 심볼릭 링크 포함)."""
    if spec.isdigit():
        cap = cv2.VideoCapture(int(spec))
    else:
        # udev 가 만든 이름은 /dev/videoN 을 가리키는 링크다. OpenCV 는 /dev/video* 꼴만 장치로 알아본다.
        cap = cv2.VideoCapture(os.path.realpath(spec), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없습니다: {spec}  (--camera 로 번호나 장치 경로를 지정하세요)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    got = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if got != (width, height):
        print(f"경고: {width}x{height} 을 요청했지만 카메라는 {got[0]}x{got[1]} 로 열렸습니다.")
    return cap


def parse_board(text: str) -> tuple[int, int]:
    """'9x6' -> (9, 6). 칸 수가 아니라 안쪽 꼭짓점 수 (가로 x 세로)."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise SystemExit(f"체커보드 크기는 '가로x세로' 안쪽 꼭짓점 수로 적습니다 (예: 9x6): {text}") from None
    return cols, rows


def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    """보드 좌표계의 꼭짓점 (m). 첫 꼭짓점이 원점, x 는 가로(cols), y 는 세로(rows), z = 0."""
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return pts


def find_board(rgb: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """체커보드 안쪽 꼭짓점을 찾아 (N, 2) 로 돌려준다. 못 찾으면 None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)


def load_images(spec: str) -> list[tuple[str, np.ndarray]]:
    """폴더나 파일 하나에서 RGB 이미지를 읽는다."""
    p = Path(spec)
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")) if p.is_dir() else [p]
    out = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            raise SystemExit(f"이미지를 읽을 수 없습니다: {f}")
        out.append((f.name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not out:
        raise SystemExit(f"이미지가 없습니다: {spec}")
    return out


class LiveView:
    """카메라 화면을 보여주고 키 입력을 받는 창.

    opencv 가 headless 판으로 깔려 있어 cv2.imshow 를 못 쓰므로 matplotlib 으로 그린다.
    """

    def __init__(self, title: str, width: int, height: int):
        # s, q 같은 키를 matplotlib 기본 단축키(저장, 종료)가 가로채지 않게 한다
        for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[key] = []
        self.fig, self.ax = plt.subplots(figsize=(9, 7.2))
        self.fig.canvas.manager.set_window_title(title)
        self.ax.axis("off")
        self.image = self.ax.imshow(np.zeros((height, width, 3), np.uint8))
        self.status = self.fig.suptitle("", fontsize=11)
        self.fig.tight_layout(rect=(0, 0, 1, 0.94))
        self.keys: list[str] = []
        self.fig.canvas.mpl_connect("key_press_event", lambda e: self.keys.append(e.key))
        plt.show(block=False)

    @property
    def open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def show(self, rgb: np.ndarray, status: str) -> list[str]:
        """한 장 그리고, 그 사이 눌린 키들을 돌려준다."""
        self.image.set_data(rgb)
        self.status.set_text(status)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.02)
        keys, self.keys = self.keys, []
        return keys
