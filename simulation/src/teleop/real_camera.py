"""MuJoCo 카메라 화면을 실제 카메라처럼 (캘리브레이션한 내부 파라미터 + 렌즈 왜곡) 그린다.

MuJoCo 는 왜곡 없는 핀홀 카메라만 그릴 수 있다. 그래서
    1. 실제 카메라의 모든 픽셀이 보는 방향(광선)을 캘리 값(camera_matrix, dist_coeffs)으로 미리 구해 두고,
    2. 그 광선을 다 담을 만큼 넓은 핀홀 화면을 그린 뒤,
    3. 픽셀마다 자기 광선이 닿는 자리를 핀홀 화면에서 집어 온다 (cv2.remap).
결과는 실제 카메라와 같은 해상도·초점거리·주점·왜곡을 가진다. 변환표는 처음에 한 번만 만든다.

캘리 값은 src/camera-cali/<name>_cam/config.py 에서 읽는다 (intrinsic_cali.py 의 결과).
"""

import importlib.util
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

CALI_DIR = Path(__file__).resolve().parents[1] / "camera-cali"
# 실제 렌즈 픽셀이 너무 바깥 광선(왜곡 모델이 뒤집히는 곳)을 가리키면 버린다
MAX_RAY_TAN = math.tan(math.radians(80))
# 광선을 다시 투영했을 때 이 픽셀보다 어긋나면 왜곡 모델이 그 자리에서 맞지 않는 것으로 보고 검게 둔다
MAX_REPROJ_PX = 0.5


def load_intrinsics(name: str):
    """camera-cali/<name>_cam/config.py 의 설정 객체 (top, wrist, belly)."""
    path = CALI_DIR / f"{name}_cam" / "config.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cam_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, f"{name.capitalize()}CamConfig")()
    if not cfg.calibrated:
        raise SystemExit(f"{name} 카메라 캘리브레이션 값이 없습니다 ({path}). intrinsic_cali.py {name} 를 먼저 하세요.")
    return cfg


class RealCamera:
    """MuJoCo 카메라 하나를 실제 카메라처럼 그린다. 렌더러를 만드므로 그리는 스레드에서 만들어야 한다."""

    def __init__(self, model: mujoco.MjModel, camera: str, calib_name: str):
        cfg = load_intrinsics(calib_name)
        K, dist = np.array(cfg.camera_matrix, float), np.array(cfg.dist_coeffs, float)
        self.width, self.height = cfg.width, cfg.height

        # 실제 화면 픽셀마다 왜곡을 걷어낸 광선 (x, y) = (X/Z, Y/Z)
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        pix = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)
        rays = cv2.undistortPointsIter(pix, K, dist, None, None, criteria).reshape(-1, 2)
        back, _ = cv2.projectPoints(np.column_stack([rays, np.ones(len(rays))]), np.zeros(3), np.zeros(3), K, dist)
        valid = (np.linalg.norm(back.reshape(-1, 2) - pix.reshape(-1, 2), axis=1) < MAX_REPROJ_PX)
        valid &= np.all(np.abs(rays) < MAX_RAY_TAN, axis=1)
        self.valid = valid.reshape(self.height, self.width)
        if not valid.all():
            rays = self._extend_radially(pix.reshape(-1, 2), rays, valid, K)

        # 모든 광선을 다 담는 핀홀 화면. 초점거리는 실제 카메라의 큰 쪽에 맞춰 가운데 해상도를 잃지 않게 하고,
        # 모델의 오프스크린 버퍼(<visual><global offwidth/offheight>)를 넘으면 줄인다
        half_w = np.abs(rays[:, 0]).max()
        half_h = np.abs(rays[:, 1]).max()
        gl = model.vis.global_
        f = min(max(K[0, 0], K[1, 1]), (gl.offwidth - 2) / (2 * half_w), (gl.offheight - 2) / (2 * half_h))
        rw, rh = 2 * math.ceil(f * half_w) + 2, 2 * math.ceil(f * half_h) + 2

        cam_id = model.camera(camera).id
        model.cam_fovy[cam_id] = math.degrees(2 * math.atan(rh / 2 / f))
        model.cam_sensorsize[cam_id] = 0  # fovy 로 그리게 한다 (sensorsize 가 있으면 MuJoCo 가 그쪽을 쓴다)
        self.camera = camera
        self.renderer = mujoco.Renderer(model, rh, rw)

        rays = rays.reshape(self.height, self.width, 2)
        self.map_x = (f * rays[..., 0] + (rw - 1) / 2).astype(np.float32)
        self.map_y = (f * rays[..., 1] + (rh - 1) / 2).astype(np.float32)

    @staticmethod
    def _extend_radially(pix: np.ndarray, rays: np.ndarray, valid: np.ndarray, K: np.ndarray) -> np.ndarray:
        """왜곡 모델이 맞지 않는 바깥 픽셀의 광선을, 안쪽의 (왜곡된 반지름 -> 광선 반지름) 관계를 직선으로 늘려 채운다.

        캘리 사진에 보드가 닿지 않은 화면 끝에서는 다항식 왜곡 모델이 뒤집혀 광선을 못 구한다.
        실제 화면에는 그 자리에도 뭔가 보이므로 검게 두지 않고, 방향은 그 픽셀 쪽 그대로 두고 반지름만 이어 붙인다.
        """
        xd = (pix[:, 0] - K[0, 2]) / K[0, 0]
        yd = (pix[:, 1] - K[1, 2]) / K[1, 1]
        rd = np.hypot(xd, yd)
        ru = np.hypot(rays[:, 0], rays[:, 1])
        # 유효한 픽셀에서 반지름 관계를 구간별 중앙값으로 잡고, 마지막 구간의 기울기로 바깥을 늘린다
        edges = np.linspace(0, rd[valid].max(), 41)
        centers, medians = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = valid & (rd >= lo) & (rd < hi)
            if sel.sum() > 20:
                centers.append(rd[sel].mean())
                medians.append(np.median(ru[sel]))
        centers, medians = np.array(centers), np.maximum.accumulate(medians)
        slope = max((medians[-1] - medians[-5]) / (centers[-1] - centers[-5]), 1.0)
        out = rays.copy()
        bad = ~valid
        r_new = np.where(rd[bad] <= centers[-1], np.interp(rd[bad], centers, medians),
                         medians[-1] + slope * (rd[bad] - centers[-1]))
        out[bad] = np.column_stack([xd[bad], yd[bad]]) / np.maximum(rd[bad], 1e-9)[:, None] * r_new[:, None]
        return out

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera)
        pinhole = self.renderer.render()
        return cv2.remap(pinhole, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def close(self) -> None:
        self.renderer.close()
