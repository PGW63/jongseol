"""ChArUco 보드로 카메라 외부 파라미터(extrinsic)를 구한다. 고정 캠(top, left)도, 팔에 달린 캠(belly, wrist)도.

    ../.venv/bin/python src/camera-cali/extrinsic_cali.py top   --square 0.0368 --placement p1
    ../.venv/bin/python src/camera-cali/extrinsic_cali.py left  --square 0.0368 --placement p1
    ../.venv/bin/python src/camera-cali/extrinsic_cali.py belly --square 0.0368 --placement p1
    ../.venv/bin/python src/camera-cali/extrinsic_cali.py wrist --square 0.0368 --placement p1
    ../.venv/bin/python src/camera-cali/extrinsic_cali.py top   --replay          # 다시 찍지 않고 계산만

무엇을 구하나
    카메라가 어느 링크(--link)에 달려 있는지 기준으로 T_link_cam 을 구한다.
        top, left  -> base      (고정)
        belly      -> shoulder  (shoulder_pan 과 같이 돈다)
        wrist      -> gripper   (손목 끝과 같이 움직인다)
    이름에 따라 기본값이 정해지고, 다른 이름은 --link 로 준다.

방법
    1. 보드를 단단한 판에 붙여 고정한다. 눕혀도, 세워도, 기울여도 된다. 찍는 동안 움직이면 안 된다.
    2. 그 자리(--placement)는 wrist 캠으로 먼저 잰다 (hand-eye):
           extrinsic_cali.py wrist --square 0.0368 --placement p1
       팔 자세를 크게 바꿔가며 (shoulder_pan, 손목 굽힘, 손목 돌림을 모두 섞어서) 보드를 15~20 장 찍는다 (s).
       팔이 흔들리면 안 되므로 자세를 잡고 h (토크로 붙잡기) -> s -> h (풀기).
       q 를 누르면 사진과 관절 값만으로 베이스 기준 보드 위치, gripper 기준 wrist 캠 위치,
       관절 영점 오차(shoulder_lift, elbow_flex, wrist_flex)를 한꺼번에 푼다. 꼭짓점을 찍을 필요가 없다.
       결과는 extrinsic/board_<placement>.json 에 남는다.
    3. 같은 자리로 다른 카메라를 돌린다. 보드 위치는 이미 아니까 카메라는 보드만 보면 된다.
         top, left : 3~5 장.
         belly     : shoulder_pan 을 여러 각도로 돌려가며 5~10 장 (그 순간 관절 값도 같이 읽는다).
       표본은 extrinsic/<name>_samples.json 에 쌓인다. 보드를 옮기고 --placement p2 로 2 를 다시 하고
       또 돌리면 두 자리의 표본을 같이 평균 낸다 (자리마다 결과가 얼마나 다른지가 정확도다).
    4. q 로 끝내면 평균을 내서 extrinsic/<name>.py 에 저장한다.
    (--touch: 예전 방식. 팔 끝을 pivot 으로 구하고 꼭짓점 네 개를 찍어 보드 위치를 잰다.)

계산
    T_base_board 는 찍은 꼭짓점에서, T_cam_board 는 사진(solvePnP)에서, T_base_link 는 관절 값으로 시뮬 SO-101 의 FK 에서.
        T_link_cam = inv(T_base_link) @ T_base_board @ inv(T_cam_board)
    관절 원시 값 -> 각도는 board_touch.py 와 같이 deg / linear 두 가지 중 hand-eye 가 더 잘 맞는 쪽을 쓰고,
    hand-eye 로 구한 관절 영점 오차를 더한다 (자리마다 board_<placement>.json 에 들어 있다).

키 (창을 클릭한 뒤)
    c  (pivot, 꼭짓점 단계) 지금 팔 끝 위치를 기록    u  마지막 기록 취소    p  pivot 끝내기
    s  지금 보드 자세 (+ 관절 값) 를 표본으로 추가
    h  follower 토크 켜기/끄기 (켜면 지금 자세를 붙잡는다. 손으로 움직일 땐 끈다)
    q  저장하고 종료
"""

import argparse
import json
import math
import sys
import time

import cv2
import mujoco
import numpy as np

from cali_common import HERE, LiveView, load_camera_config, open_camera, write_camera_extrinsic
from charuco_intrinsic_cali import Detector, make_board, parse_squares, usable
from headCam_to_Base import TROLLEY_TOP_Z, average, base_in_trolley, quat_wxyz, transform

sys.path.insert(0, str(HERE.parent))
from board_touch import MAPPINGS, fit_rigid  # noqa: E402
from teleop.constants import JOINT_NAMES, default_port  # noqa: E402

EXT = HERE / "extrinsic"
SCENE = HERE.parents[1] / "models" / "scene" / "scene_empty.xml"
PREFIX = "so101_"
LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
DEFAULT_LINK = {"belly": "shoulder", "wrist": "gripper"}
# 비교할 시뮬 카메라 (CAD 위치)
MODEL_CAMERA = {"wrist": PREFIX + "wrist_cam"}
TOOL = "gripper"
READS = 10
# s 를 누르면 버리는 프레임 수 (카메라 버퍼), 사진 전후 관절 값이 이만큼 넘게 다르면 아직 움직이는 중으로 보고 버린다
FLUSH_FRAMES = 6
SETTLE_TICKS = 4
LABELS = ["origin (yellow)", "2nd corner", "3rd corner", "4th corner"]


# ---------------------------------------------------------------- 로봇

class LinkFK:
    """관절 각도(rad) -> 베이스 기준 링크 자세 4x4."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.qadr = [int(self.model.joint(PREFIX + j).qposadr[0]) for j in JOINT_NAMES]

    def pose(self, body: str) -> np.ndarray:
        b, base = self.model.body(PREFIX + body).id, self.model.body(PREFIX + "base").id
        Rb, pb = self.data.xmat[base].reshape(3, 3), self.data.xpos[base]
        return transform(Rb.T @ self.data.xmat[b].reshape(3, 3), Rb.T @ (self.data.xpos[b] - pb))

    def __call__(self, q: list[float], body: str) -> np.ndarray:
        if body == "base":
            return np.eye(4)
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.pose(body)

    def model_camera(self, camera: str, body: str) -> np.ndarray | None:
        """시뮬 모델에 있는 카메라의 링크 기준 자세 (OpenCV 축). 없으면 None."""
        try:
            cam = self.model.camera(camera).id
        except KeyError:
            return None
        mujoco.mj_forward(self.model, self.data)
        T_base_link = self.pose(body)
        base = self.model.body(PREFIX + "base").id
        Rb, pb = self.data.xmat[base].reshape(3, 3), self.data.xpos[base]
        R = Rb.T @ self.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1, -1, -1])  # MuJoCo 카메라 축 -> OpenCV
        return np.linalg.inv(T_base_link) @ transform(R, Rb.T @ (self.data.cam_xpos[cam] - pb))


class Follower:
    """follower 버스만 연다 (connect() 는 바로 토크를 켜서 팔이 튈 수 있다). 관절 값은 원시 틱으로 읽는다."""

    def __init__(self, port: str, arm_id: str):
        from lerobot.robots.so_follower import SOFollower
        from teleop.follower import SO101FollowerConfig

        self.robot = SOFollower(SO101FollowerConfig(port=port, id=arm_id))
        self.bus = self.robot.bus
        self.bus.connect()
        self.bus.disable_torque()
        self.holding = False
        self.calibration = {j: {"range_min": c.range_min, "range_max": c.range_max} for j, c in self.robot.calibration.items()}
        if not self.calibration:
            print(f"(follower 캘리브레이션 파일이 없다: {self.robot.calibration_fpath}. 관절 매핑은 deg 만 쓴다)")

    def read(self) -> tuple[dict[str, float], float]:
        reads = [self.bus.sync_read("Present_Position", normalize=False, num_retry=2) for _ in range(READS)]
        spread = max(max(r[j] for r in reads) - min(r[j] for r in reads) for j in JOINT_NAMES)
        return {j: float(np.mean([r[j] for r in reads])) for j in JOINT_NAMES}, float(spread)

    def toggle_hold(self) -> None:
        if self.holding:
            self.bus.disable_torque()
        else:
            pos = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
            self.bus.sync_write("Goal_Position", pos, normalize=False)
            self.bus.enable_torque()
        self.holding = not self.holding

    def close(self) -> None:
        if self.holding:
            input("follower 토크를 끈다. 팔을 손으로 받치고 Enter...")
        self.bus.disable_torque()
        self.bus.disconnect(False)


def to_rad(raw: dict, calib: dict, mapping: str, offsets=None) -> np.ndarray:
    q = np.array(MAPPINGS[mapping](raw, calib), float)
    return q if offsets is None else q + np.asarray(offsets)


def se3(x: np.ndarray) -> np.ndarray:
    return transform(cv2.Rodrigues(np.asarray(x[:3], float))[0], x[3:6])


def se3_vec(T: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


# hand-eye 에서 영점을 같이 푸는 관절. shoulder_pan 영점은 보드를 베이스 Z 축으로 돌린 것과, wrist_roll 영점은
# 카메라를 gripper 에서 돌린 것과 구별되지 않아 뺀다 (시뮬에서도 같은 값으로 움직이니 결과는 맞다).
OFFSET_JOINTS = [1, 2, 3]


def attach_points(samples: list[dict], name: str, detector) -> None:
    """예전 표본에 보드 점(obj, img)이 없으면 남겨 둔 원본 사진에서 다시 찾는다."""
    for s in samples:
        if "img" in s or not s.get("image"):
            continue
        bgr = cv2.imread(str(EXT / f"{name}_captures" / s["image"]))
        if bgr is None:
            continue
        corners, ids, _, _ = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if ids is not None and len(ids) > 0:
            obj, img = detector.points(corners, ids)
            s["obj"], s["img"] = obj.reshape(-1, 3).tolist(), img.reshape(-1, 2).tolist()


def solve_handeye(samples: list[dict], calib: dict, fk: LinkFK, board_pts: np.ndarray, fit_offsets: bool,
                  K: np.ndarray | None = None, dist: np.ndarray | None = None) -> dict:
    """wrist 캠 표본 (T_cam_board, 관절) -> T_base_board, T_gripper_cam, 관절 영점 오차.
    T_cam_board = inv(T_gripper_cam) @ inv(FK(q + off)) @ T_base_board 를 맞춘다.
    표본에 보드 점(obj, img)이 있으면 재투영 오차(px)로 맞춘다. 멀리서 작게 찍힌 보드는 기울기가 불확실한데,
    재투영 오차는 그런 표본을 자연히 덜 믿는다. 없으면 보드 꼭짓점 위치 오차(m)로 맞춘다."""
    P = np.vstack([board_pts.T, np.ones(len(board_pts))])
    meas = [np.array(s["T_cam_board"]) for s in samples]
    use_px = K is not None and all("img" in s for s in samples)
    pts = [(np.array(s["obj"], np.float64), np.array(s["img"], np.float64)) for s in samples] if use_px else None
    fits = []
    for mapping in [m for m in MAPPINGS if m == "deg" or calib]:
        q0 = [to_rad(s["q_raw"], calib, mapping) for s in samples]
        T_bg = [fk(q, TOOL) for q in q0]
        # 초기값: OpenCV 의 robot-world-hand-eye (world = 보드)
        Rb, tb, Rc, tc = cv2.calibrateRobotWorldHandEye(
            [T[:3, :3] for T in meas], [T[:3, 3] for T in meas],
            [np.linalg.inv(T)[:3, :3] for T in T_bg], [np.linalg.inv(T)[:3, 3] for T in T_bg])
        T_bb0 = np.linalg.inv(transform(Rb, tb))
        T_gc0 = np.linalg.inv(transform(Rc, tc))

        def residual(x, with_off):
            T_bb, inv_gc = se3(x[:6]), np.linalg.inv(se3(x[6:12]))
            off = np.zeros(6)
            if with_off:
                off[OFFSET_JOINTS] = x[12:]
            out = []
            for i, (q, Tm) in enumerate(zip(q0, meas)):
                pred = inv_gc @ np.linalg.inv(fk(q + off, TOOL)) @ T_bb
                if use_px:
                    proj, _ = cv2.projectPoints(pts[i][0], cv2.Rodrigues(pred[:3, :3])[0], pred[:3, 3], K, dist)
                    out.append((proj.reshape(-1, 2) - pts[i][1]).ravel())
                else:
                    out.append(((pred - Tm)[:3] @ P).ravel())
            return np.concatenate(out)

        def lm(x, with_off):
            lam, r = 1e-3, residual(x, with_off)
            for _ in range(100):
                J = np.empty((len(r), len(x)))
                for i in range(len(x)):
                    d = np.zeros(len(x))
                    d[i] = 1e-6
                    J[:, i] = (residual(x + d, with_off) - r) / 1e-6
                with np.errstate(all="ignore"):  # macOS Accelerate 의 numpy matmul 이 내는 헛경고
                    A, g = J.T @ J, J.T @ r
                dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
                r2 = residual(x + dx, with_off)
                with np.errstate(all="ignore"):
                    better = np.all(np.isfinite(r2)) and r2 @ r2 < r @ r
                if better:
                    x, r, lam = x + dx, r2, lam / 3
                    if np.abs(dx).max() < 1e-9:
                        break
                else:
                    lam *= 5
            return x, r

        x0 = np.concatenate([se3_vec(T_bb0), se3_vec(T_gc0)])
        x, r = lm(x0, False)
        if fit_offsets:
            x, r = lm(np.concatenate([x, np.zeros(len(OFFSET_JOINTS))]), True)
        off = np.zeros(6)
        if fit_offsets:
            off[OFFSET_JOINTS] = x[12:]
        # 표본마다 보드 꼭짓점 위치 오차 (mm, 카메라 좌표) 와 재투영 오차 (px)
        T_bb, inv_gc = se3(x[:6]), np.linalg.inv(se3(x[6:12]))
        per, per_px = [], []
        for i, (q, Tm) in enumerate(zip(q0, meas)):
            pred = inv_gc @ np.linalg.inv(fk(q + off, TOOL)) @ T_bb
            per.append(float(np.sqrt(np.mean(np.sum(((pred - Tm)[:3] @ P) ** 2, axis=0)))) * 1000)
            if use_px:
                proj, _ = cv2.projectPoints(pts[i][0], cv2.Rodrigues(pred[:3, :3])[0], pred[:3, 3], K, dist)
                per_px.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts[i][1]) ** 2, axis=1)))))
        per = np.array(per)
        fits.append({"mapping": mapping, "T_base_board": se3(x[:6]).tolist(), "T_link_cam_handeye": se3(x[6:12]).tolist(),
                     "joint_offsets": off.tolist(), "rms_mm": float(np.sqrt(np.mean(per ** 2))), "per_sample_mm": per.tolist(),
                     "rms_px": float(np.sqrt(np.mean(np.square(per_px)))) if use_px else None, "per_sample_px": per_px,
                     "score": float(np.sqrt(np.mean(np.square(per_px)))) if use_px else float(np.sqrt(np.mean(per ** 2))) / 1000})
    for f in fits:
        offs = " ".join(f"{JOINT_NAMES[j]} {math.degrees(f['joint_offsets'][j]):+.1f}" for j in OFFSET_JOINTS)
        px = f",  재투영 오차 RMS {f['rms_px']:.1f} px" if f["rms_px"] is not None else ""
        print(f"  {f['mapping']:6s}: 보드 꼭짓점 오차 RMS {f['rms_mm']:.1f} mm{px}" + (f",  관절 영점 오차(도) {offs}" if fit_offsets else ""))
    best = min(fits, key=lambda f: f["score"])
    T_bg = [fk(to_rad(s["q_raw"], calib, best["mapping"]), TOOL) for s in samples]
    spread = max(rot_deg(a[:3, :3].T @ b[:3, :3]) for a in T_bg for b in T_bg)
    errs = best["per_sample_px"] or best["per_sample_mm"]
    worst = int(np.argmax(errs))
    print(f"  -> {best['mapping']} 를 쓴다. 표본마다 오차 최대 #{worst + 1} {best['per_sample_mm'][worst]:.1f} mm"
          + (f" / {best['per_sample_px'][worst]:.1f} px" if best["per_sample_px"] else "") + ",  "
          f"손목 방향이 바뀐 폭 최대 {spread:.0f} 도")
    if spread < 60:
        print("  (손목 방향 변화가 작다. 팔을 더 다양하게 돌려 찍으면 정확해진다)")
    print("  RMS 가 5 mm 를 넘으면 오차 큰 표본을 빼거나 (--retouch 로 다시) 더 찍는다.")
    return best


# ---------------------------------------------------------------- 보드 자리 (꼭짓점 찍기)

# 자로 잴 때의 기준: pan 회전축 중심의 상판 위 점. 시뮬 모델에서 베이스 원점 기준 (38.8 mm, 0, 0) 이다.
PAN_AXIS_IN_BASE = np.array([0.0388, 0.0, 0.0])


def measured_board_pose(kind: str, origin: list[float], yaw_deg: float) -> np.ndarray:
    """자로 잰 보드 자세 -> T_base_board.
    origin: 노란 점(원점)의 위치 X Y [Z] (m), pan 회전축 중심에서 상판 위로 잰다. +X 로봇 앞, +Y 로봇 왼쪽, +Z 위.
    yaw: 빨간 X 축이 향하는 방향 (도). 0 = 로봇 앞, 90 = 로봇 왼쪽.
    flat    : 보드를 상판에 눕힘 (인쇄면이 위). 보드 Z 축(인쇄면 안쪽)이 아래를 향한다.
    upright : 보드를 세움 (인쇄된 그대로, 노란 점이 왼쪽 위로 보이게). 보드 Y 축(초록)이 아래를 향한다."""
    yaw = math.radians(yaw_deg)
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    if kind == "flat":
        z = np.array([0.0, 0.0, -1.0])
        y = np.cross(z, x)
    else:
        y = np.array([0.0, 0.0, -1.0])
        z = np.cross(x, y)
    p = np.array(list(origin) + [0.0] * (3 - len(origin))) + PAN_AXIS_IN_BASE
    return transform(np.column_stack([x, y, z]), p)


def extreme_corners(board) -> tuple[list[int], np.ndarray]:
    """안쪽 꼭짓점 중 네 귀퉁이의 id 와 보드 좌표. 첫 번째(id 0)가 원점(노란 점)."""
    sx, sy = board.getChessboardSize()
    n = sx - 1
    ids = [0, n - 1, (sy - 2) * n, (sy - 1) * n - 1]
    return ids, np.asarray(board.getChessboardCorners(), float)[ids]


def tool_poses(raws: list[dict], calib: dict, mapping: str, fk: LinkFK) -> list[np.ndarray]:
    return [fk(to_rad(raw, calib, mapping), TOOL) for raw in raws]


def solve_pivot(raws: list[dict], calib: dict, fk: LinkFK) -> dict:
    """고정점 하나를 여러 자세로 찍은 것 -> 팔 끝 위치 e (gripper 링크 기준).  R_i e + p_i = c  (c 도 모름)."""
    fits = []
    for mapping in [m for m in MAPPINGS if m == "deg" or calib]:
        Ts = tool_poses(raws, calib, mapping, fk)
        A = np.vstack([np.hstack([T[:3, :3], -np.eye(3)]) for T in Ts])
        b = np.concatenate([-T[:3, 3] for T in Ts])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = np.linalg.norm((A @ sol - b).reshape(-1, 3), axis=1) * 1000
        # 가장 작은 특이값이 작으면 어떤 방향으로 자세가 덜 바뀐 것이다 (그 방향의 끝 위치가 불확실)
        sv = np.linalg.svd(A, compute_uv=False)
        fits.append({"mapping": mapping, "tip": sol[:3].tolist(), "rms_mm": float(np.sqrt(np.mean(res ** 2))),
                     "err_mm": res.tolist(), "conditioning": float(sv[-1] / sv[0])})
    for f in fits:
        print(f"  {f['mapping']:6s}: 끝 위치 {' '.join(f'{v * 1000:.1f}' for v in f['tip'])} mm, "
              f"찍기마다 오차 {' / '.join(f'{v:.1f}' for v in f['err_mm'])} mm (RMS {f['rms_mm']:.1f}), 자세 다양성 {f['conditioning']:.2f}")
    best = min(fits, key=lambda f: f["rms_mm"])
    print(f"  -> {best['mapping']} 를 쓴다. RMS 가 3 mm 를 넘거나 자세 다양성이 0.1 보다 작으면 --repivot 으로 다시.")
    return {**best, "raws": raws, "calibration": calib, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def fit_board(raws: list[dict], calib: dict, tip: dict, fk: LinkFK, board_pts: np.ndarray) -> dict:
    """네 꼭짓점 (첫 점 = 원점) -> T_base_board. 나머지 세 점의 순서는 가장 잘 맞는 쪽."""
    import itertools

    e = np.array(tip["tip"])
    pts = np.array([T[:3, :3] @ e + T[:3, 3] for T in tool_poses(raws, calib, tip["mapping"], fk)])
    best = None
    for order in itertools.permutations(range(1, len(pts))):
        idx = [0, *order]
        R, t = fit_rigid(board_pts[idx], pts)
        err = np.linalg.norm(pts - (board_pts[idx] @ R.T + t), axis=1)
        score = float(np.sqrt(np.mean(err ** 2)))
        if best is None or score < best["score"]:
            best = {"score": score, "R": R, "t": t, "err": err}
    print(f"  꼭짓점 오차 {' / '.join(f'{v * 1000:.1f}' for v in best['err'])} mm (RMS {best['score'] * 1000:.1f}),  "
          f"보드 면이 바라보는 방향 (베이스) {np.round(-best['R'][:, 2], 2).tolist()}")
    print("  RMS 가 5 mm 를 넘으면 --retouch 로 다시 찍는다.")
    return {"mapping": tip["mapping"], "tip": tip["tip"], "T_base_board": transform(best["R"], best["t"]).tolist(),
            "score": best["score"], "corner_err_mm": (best["err"] * 1000).tolist()}


# ---------------------------------------------------------------- 계산

def _lm(res, x: np.ndarray, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt (수치 미분)."""
    lam, r = 1e-3, res(x)
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            d = np.zeros(len(x))
            d[i] = 1e-6
            J[:, i] = (res(x + d) - r) / 1e-6
        with np.errstate(all="ignore"):
            A, g = J.T @ J, J.T @ r
        dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        r2 = res(x + dx)
        with np.errstate(all="ignore"):
            better = np.all(np.isfinite(r2)) and r2 @ r2 < r @ r
        if better:
            x, r, lam = x + dx, r2, lam / 3
            if np.abs(dx).max() < 1e-10:
                break
        else:
            lam *= 5
    return x, r


def pan_angle(raw: dict) -> float:
    return (raw["shoulder_pan"] - 2047) * 2 * math.pi / 4096


def solve_pan_only(samples: list[dict], fk: LinkFK, K: np.ndarray, dist: np.ndarray, lens_height: float, cam_yaw_deg: float) -> dict:
    """shoulder_pan 만 돌리며 찍은 표본 -> T_shoulder_cam. 보드 위치는 몰라도 된다.

    한 축으로만 도는 카메라는 (그 축 방향 높이, 그 축 둘레 방향) 두 값이 사진에 나타나지 않는다
    (카메라와 보드를 같이 올리거나 같이 돌려도 사진이 같다). 그래서 이 둘은
        높이: 상판에서 렌즈 중심까지 잰 값 (lens_height)
        방향: pan 이 같을 때 카메라가 팔이 뻗는 방향 (+ cam_yaw_deg) 을 본다
    로 정하고, 나머지 넷(숙인 각도, 옆 기울기, pan 축에서의 거리와 방향 차이)은 사진으로 푼다."""
    th = np.array([pan_angle(s["q_raw"]) for s in samples])
    meas = [np.array(s["T_cam_board"]) for s in samples]
    pts = [(np.array(s["obj"], float), np.array(s["img"], float)) for s in samples]

    def S(t):
        q = np.zeros(6)
        q[0] = t
        return fk(q, "shoulder")

    # 1) 초기값: 보드 좌표에서 본 카메라들은 pan 축을 중심으로 돈다
    C = [np.linalg.inv(T) for T in meas]
    i0 = int(np.argmin(np.abs(th - np.median(th))))
    num, den, A, b = np.zeros(3), 0.0, [], []
    for i in range(len(C)):
        d = th[i] - th[i0]
        if abs(d) < 1e-3:
            continue
        rv = cv2.Rodrigues(C[i][:3, :3] @ C[i0][:3, :3].T)[0].ravel()
        num, den = num + d * rv, den + d * d
    u_raw = num / den
    pan_scale = float(np.linalg.norm(u_raw))  # 카메라가 돈 각도 / 엔코더 각도. 1 이어야 한다
    u = u_raw / pan_scale
    for i in range(len(C)):
        R = cv2.Rodrigues(u * (th[i] - th[i0]))[0]
        A.append(np.eye(3) - R)
        b.append(C[i][:3, 3] - R @ C[i0][:3, 3])
    p_axis = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    # 베이스 축을 보드 좌표로 (pan 축 둘레 방향, 축 방향 위치는 아무거나. 나중에 정한다)
    z_b = -u  # pan 관절 축은 베이스 -z 방향
    x_b = np.cross(np.cross(z_b, [1.0, 0, 0] if abs(z_b[0]) < 0.9 else [0, 1.0, 0]), z_b)
    x_b /= np.linalg.norm(x_b)
    y_b = np.cross(z_b, x_b)
    R_board_base = np.column_stack([x_b, y_b, z_b])
    origin = p_axis - R_board_base @ PAN_AXIS_IN_BASE
    Z = np.linalg.inv(transform(R_board_base, origin))  # T_base_board
    X = average([np.linalg.inv(S(t)) @ Z @ Ci for t, Ci in zip(th, C)])

    # 2) 재투영 오차로 다듬기
    def res(x):
        Zx, iX = se3(x[:6]), np.linalg.inv(se3(x[6:]))
        out = []
        for t, (obj, img) in zip(th, pts):
            pred = iX @ np.linalg.inv(S(t)) @ Zx
            proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(pred[:3, :3])[0], pred[:3, 3], K, dist)
            out.append((proj.reshape(-1, 2) - img).ravel())
        return np.concatenate(out)

    x, r = _lm(res, np.concatenate([se3_vec(Z), se3_vec(X)]))
    Z, X = se3(x[:6]), se3(x[6:])
    per_px = [float(np.sqrt(np.mean(np.sum(rr.reshape(-1, 2) ** 2, axis=1))))
              for rr in np.split(r, np.cumsum([2 * len(p_[1]) for p_ in pts])[:-1])]

    # 3) 사진으로 안 나오는 두 값을 정한다 (shoulder 좌표에서 pan 축 = z 축, 원점을 지난다)
    S0 = S(0.0)
    h = (S0 @ X)[2, 3] - lens_height  # shoulder z 로 h 만큼 옮기면 베이스 높이는 h 만큼 낮아진다
    G = transform(np.eye(3), [0, 0, h])
    X = G @ X
    f = X[:3, 2]  # 카메라가 보는 방향 (shoulder 좌표)
    horiz = float(np.hypot(f[0], f[1]))
    yaw = math.radians(cam_yaw_deg)
    target = S0[:3, :3].T @ np.array([math.cos(yaw), math.sin(yaw), 0.0])  # 팔이 뻗는 방향 (+yaw) 을 shoulder 좌표로
    phi = math.atan2(target[1], target[0]) - math.atan2(f[1], f[0])
    Gr = transform(cv2.Rodrigues(np.array([0, 0, phi]))[0], [0, 0, 0])
    X = Gr @ X
    G = Gr @ G
    Z = S0 @ G @ np.linalg.inv(S0) @ Z

    per = [np.linalg.inv(S(t)) @ Z @ Ci for t, Ci in zip(th, C)]
    dpos = [np.linalg.norm(T[:3, 3] - X[:3, 3]) * 1000 for T in per]
    drot = [rot_deg(X[:3, :3].T @ T[:3, :3]) for T in per]
    print(f"\nbelly pan-only (표본 {len(samples)} 개, pan 범위 {math.degrees(np.ptp(th)):.0f} 도):")
    print(f"  재투영 오차 RMS {np.sqrt(np.mean(np.square(per_px))):.2f} px (최대 #{int(np.argmax(per_px)) + 1} {max(per_px):.2f} px)")
    print(f"  카메라가 돈 각도 / 엔코더 각도 = {pan_scale:.3f}  (1 에 가까워야 한다)")
    print(f"  표본별 퍼짐 RMS {np.sqrt(np.mean(np.square(dpos))):.1f} mm, {np.sqrt(np.mean(np.square(drot))):.2f} 도")
    if math.degrees(np.ptp(th)) < 30:
        print("  (pan 범위가 좁다. 보드가 보이는 한 넓게 돌릴수록 pan 축 위치가 정확해진다)")
    if horiz < 0.3:
        print("  (카메라가 거의 수직으로 내려다봐서 '정면' 방향을 정하기 애매하다. 결과의 방향을 눈으로 확인할 것)")
    tilt = math.degrees(math.asin(np.clip(-(S0 @ X)[2, 2], -1, 1)))
    print(f"  카메라가 아래로 숙인 각도 {tilt:.1f} 도,  pan 축에서 렌즈까지 수평 거리 {np.hypot(*(S0 @ X)[:2, 3] - PAN_AXIS_IN_BASE[:2]) * 1000:.0f} mm")
    R, t = X[:3, :3], X[:3, 3]
    return {
        "link": "shoulder", "T_link_cam": X.tolist(), "position": t.tolist(), "quat_wxyz": quat_wxyz(R),
        "mujoco_body": '<body name="shoulder"> (so101.xml)', "mujoco_pos": t.tolist(),
        "mujoco_xyaxes": np.concatenate([R[:, 0], -R[:, 1]]).tolist(), "mujoco_fovy": None,
        "num_samples": len(samples), "placements": sorted({s["placement"] for s in samples}),
        "spread_mm": float(np.sqrt(np.mean(np.square(dpos)))), "spread_deg": float(np.sqrt(np.mean(np.square(drot)))),
        "reprojection_px": float(np.sqrt(np.mean(np.square(per_px)))), "T_base_board": Z.tolist(), "pan_scale": pan_scale,
    }


def solve_eye_to_hand(samples: list[dict], calib: dict, fk: LinkFK, K: np.ndarray, dist: np.ndarray, bootstrap: int = 20) -> dict:
    """보드를 gripper 에 달고 고정 카메라로 찍은 표본 -> T_base_cam, T_gripper_board.
    T_cam_board = inv(T_base_cam) @ FK(q) @ T_gripper_board 를 재투영 오차로 맞춘다."""
    meas = [np.array(s["T_cam_board"]) for s in samples]
    pts = [(np.array(s["obj"], float), np.array(s["img"], float)) for s in samples]
    fits = []
    for mapping in [m for m in MAPPINGS if m == "deg" or calib]:
        F = [fk(to_rad(s["q_raw"], calib, mapping), TOOL) for s in samples]
        # 초기값: 식을 뒤집으면 (T_board_cam = inv(B) inv(F) Y) 눈-손 꼴이 되어 OpenCV robot-world-hand-eye 로 푼다
        inv_meas = [np.linalg.inv(T) for T in meas]
        Rb, tb, Rc, tc = cv2.calibrateRobotWorldHandEye(
            [T[:3, :3] for T in inv_meas], [T[:3, 3] for T in inv_meas],
            [np.linalg.inv(T)[:3, :3] for T in F], [np.linalg.inv(T)[:3, 3] for T in F])
        Y0 = np.linalg.inv(transform(Rb, tb))  # T_base_cam
        B0 = np.linalg.inv(transform(Rc, tc))  # T_gripper_board

        def res(x):
            iY, B = np.linalg.inv(se3(x[:6])), se3(x[6:])
            out = []
            for Fi, (obj, img) in zip(F, pts):
                pred = iY @ Fi @ B
                proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(pred[:3, :3])[0], pred[:3, 3], K, dist)
                out.append((proj.reshape(-1, 2) - img).ravel())
            return np.concatenate(out)

        x, r = _lm(res, np.concatenate([se3_vec(Y0), se3_vec(B0)]))
        Y, B = se3(x[:6]), se3(x[6:])
        per_px = [float(np.sqrt(np.mean(np.sum(rr.reshape(-1, 2) ** 2, axis=1))))
                  for rr in np.split(r, np.cumsum([2 * len(p_[1]) for p_ in pts])[:-1])]
        # 표본마다 카메라 위치가 얼마나 다르게 나오는지 (그 표본만으로 본 T_base_cam)
        per_T = [Fi @ B @ np.linalg.inv(Tm) for Fi, Tm in zip(F, meas)]
        dpos = [np.linalg.norm(T[:3, 3] - Y[:3, 3]) * 1000 for T in per_T]
        drot = [rot_deg(Y[:3, :3].T @ T[:3, :3]) for T in per_T]
        fits.append({"mapping": mapping, "T_base_cam": Y, "T_gripper_board": B, "per_px": per_px,
                     "rms_px": float(np.sqrt(np.mean(np.square(per_px)))), "dpos": dpos, "drot": drot})
    for f in fits:
        print(f"  {f['mapping']:6s}: 재투영 오차 RMS {f['rms_px']:.1f} px")
    best = min(fits, key=lambda f: f["rms_px"])
    worst = int(np.argmax(best["per_px"]))
    print(f"  -> {best['mapping']} 를 쓴다. 표본마다 오차 최대 #{worst + 1} {best['per_px'][worst]:.1f} px,  "
          f"그리퍼에서 보드 원점까지 {np.linalg.norm(best['T_gripper_board'][:3, 3]) * 1000:.0f} mm")
    # 최종값이 얼마나 믿을 만한지: 표본을 무작위로 다시 뽑아 여러 번 풀어 본 흔들림 (부트스트랩)
    if bootstrap and len(samples) >= 8:
        import contextlib
        import io
        rng = np.random.default_rng(0)
        Y = best["T_base_cam"]
        dp, dr = [], []
        calib_b = calib if best["mapping"] == "linear" else {}
        for _ in range(bootstrap):
            pick = [samples[i] for i in rng.integers(0, len(samples), len(samples))]
            with contextlib.redirect_stdout(io.StringIO()):
                Tb = solve_eye_to_hand(pick, calib_b, fk, K, dist, bootstrap=0)["T_base_cam"]
            dp.append(np.linalg.norm(Tb[:3, 3] - Y[:3, 3]) * 1000)
            dr.append(rot_deg(Y[:3, :3].T @ Tb[:3, :3]))
        best["boot_mm"], best["boot_deg"] = float(np.sqrt(np.mean(np.square(dp)))), float(np.sqrt(np.mean(np.square(dr))))
        print(f"  추정 오차 (부트스트랩 {bootstrap} 번): 위치 약 {best['boot_mm']:.0f} mm, 각도 약 {best['boot_deg']:.1f} 도"
              "   <- 이게 목표(예: 10 mm, 1 도) 안으로 들어오면 그만 찍어도 된다")
    return best


def joint_check(pair: list[dict]) -> str | None:
    """직전 표본과 비교해서 관절 하나(또는 lift/elbow/flex 만)만 움직였으면, 엔코더 각도와 카메라가 돈 각도를 비교한다.
    단단한 팔이면 둘이 같아야 한다. 차이가 크면 그 관절(또는 마운트)이 엔코더 밖에서 휘거나 노는 것이다."""
    if len(pair) != 2 or pair[0]["q_raw"] is None:
        return None
    a, b = pair
    dq = {j: (b["q_raw"][j] - a["q_raw"][j]) * 360 / 4096 for j in JOINT_NAMES[:5]}
    moved = [j for j, v in dq.items() if abs(v) > 1.0]
    if not moved:
        return None
    if len(moved) == 1:
        label, enc = moved[0], abs(dq[moved[0]])
    elif set(moved) <= {"shoulder_lift", "elbow_flex", "wrist_flex"}:
        label, enc = "+".join(moved), abs(sum(dq[j] for j in moved))  # 축이 평행이라 합만큼 돈다
    else:
        return None
    cam = rot_deg(np.array(a["T_cam_board"])[:3, :3] @ np.array(b["T_cam_board"])[:3, :3].T)
    return f"[joint check] {label}: encoder {enc:.1f} deg vs camera {cam:.1f} deg (diff {cam - enc:+.1f})"


def rot_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def compute(name: str, store: dict, fk: LinkFK, mapping_override: str | None, fovy: float | None) -> dict:
    link = store["link"]
    boards, poses, errs, owners = {}, [], [], []
    for s in store["samples"]:
        p = s["placement"]
        if p not in boards:
            boards[p] = json.loads((EXT / f"board_{p}.json").read_text(encoding="utf-8"))
        bf = boards[p]
        mapping = mapping_override or bf["mapping"]
        T_base_link = np.eye(4) if link == "base" else fk(
            to_rad(s["q_raw"], store["calibration"], mapping, bf.get("joint_offsets")), link)
        T = np.linalg.inv(T_base_link) @ np.array(bf["T_base_board"]) @ np.linalg.inv(np.array(s["T_cam_board"]))
        poses.append(T)
        errs.append(s["reproj_px"])
        owners.append(p)
    T_mean = average(poses)
    dpos = [np.linalg.norm(T[:3, 3] - T_mean[:3, 3]) * 1000 for T in poses]
    drot = [rot_deg(T_mean[:3, :3].T @ T[:3, :3]) for T in poses]

    print(f"\n표본 {len(poses)} 개 ({', '.join(sorted(boards))}), 보드 재투영 오차 평균 {np.mean(errs):.2f} px")
    for p in sorted(boards):
        sel = [i for i, o in enumerate(owners) if o == p]
        Tp = average([poses[i] for i in sel])
        print(f"  {p}: {len(sel)} 개, 평균과의 차이 {np.linalg.norm(Tp[:3, 3] - T_mean[:3, 3]) * 1000:.1f} mm "
              f"/ {rot_deg(T_mean[:3, :3].T @ Tp[:3, :3]):.2f} 도,  자리 안 퍼짐 "
              f"{np.sqrt(np.mean(np.square([dpos[i] for i in sel]))):.1f} mm")
    worst = int(np.argmax(dpos))
    print(f"  가장 벗어난 표본: #{worst + 1} ({owners[worst]}) {dpos[worst]:.1f} mm / {drot[worst]:.2f} 도")

    R, t = T_mean[:3, :3], T_mean[:3, 3]
    if link == "base":
        # 고정 캠은 씬(트롤리) 좌표로 바꿔서 worldbody 에 넣는다
        Ts = transform(np.eye(3), base_in_trolley() + [0, 0, TROLLEY_TOP_Z]) @ T_mean
        body, mj_pos, mj_R = "worldbody (scene_trolley.xml)", Ts[:3, 3], Ts[:3, :3]
    else:
        body, mj_pos, mj_R = f"<body name=\"{link}\"> (so101.xml)", t, R
    return {
        "link": link,
        "T_link_cam": T_mean.tolist(),
        "position": t.tolist(),
        "quat_wxyz": quat_wxyz(R),
        "mujoco_body": body,
        "mujoco_pos": mj_pos.tolist(),
        "mujoco_xyaxes": np.concatenate([mj_R[:, 0], -mj_R[:, 1]]).tolist(),
        "mujoco_fovy": fovy,
        "num_samples": len(poses),
        "placements": sorted(boards),
        "spread_mm": float(np.sqrt(np.mean(np.square(dpos)))),
        "spread_deg": float(np.sqrt(np.mean(np.square(drot)))),
        "reprojection_px": float(np.mean(errs)),
    }


# ---------------------------------------------------------------- 메인

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="카메라 이름 (<name>_cam/config.py 의 intrinsic 을 쓴다)")
    ap.add_argument("--link", choices=LINKS, help="카메라가 붙은 링크 (기본: belly=shoulder, wrist=gripper, 그 밖=base)")
    ap.add_argument("--placement", default="p1", help="보드 자리 이름 (기본 p1). 자리를 옮기면 이름을 바꾼다")
    ap.add_argument("--retouch", action="store_true", help="이 자리의 보드 위치를 다시 잰다 (wrist hand-eye 또는 --touch)")
    ap.add_argument("--touch", action="store_true", help="보드 위치를 팔 끝으로 꼭짓점을 찍어 잰다 (예전 방식, pivot 필요)")
    ap.add_argument("--measured", choices=["flat", "upright"], help="보드 위치를 자로 재서 준다 (--origin, --yaw 와 같이)")
    ap.add_argument("--origin", type=float, nargs="+", metavar="M", help="--measured: 노란 점 위치 X Y [Z] m (pan 축 중심, 상판 기준)")
    ap.add_argument("--yaw", type=float, default=0.0, help="--measured: 빨간 X 축 방향, 도 (0 = 로봇 앞, 90 = 로봇 왼쪽)")
    ap.add_argument("--measure-board", action="store_true",
                    help="이미 구한 고정 카메라(top 등)로 놓아 둔 보드를 보고 --placement 자리의 보드 위치를 저장한다 (팔 불필요)")
    ap.add_argument("--board-on-gripper", action="store_true",
                    help="고정 카메라(top, left): 보드를 그리퍼에 달고 팔을 움직이며 찍어 베이스 기준 카메라 위치를 푼다")
    ap.add_argument("--lens-height", type=float, metavar="M",
                    help="belly pan-only: 상판에서 렌즈 중심까지 높이 m. 주면 보드 위치 없이 pan 만 돌려 찍어 푼다")
    ap.add_argument("--cam-yaw", type=float, default=0.0,
                    help="belly pan-only: 카메라가 팔 뻗는 방향에서 돌아간 각도, 도 (+ = 로봇 왼쪽). 기본 0 = 정면")
    ap.add_argument("--offsets", action="store_true",
                    help="hand-eye 에서 관절 영점 오차도 같이 푼다 (표본이 아주 다양하지 않으면 불안정해서 기본은 끈다)")
    ap.add_argument("--reset", action="store_true", help="이 카메라의 이전 표본을 지우고 시작")
    ap.add_argument("--replay", action="store_true", help="찍지 않고 쌓인 표본으로 계산만 다시")
    ap.add_argument("--mapping", choices=list(MAPPINGS), help="관절 매핑을 강제 (기본: 자리마다 꼭짓점이 잘 맞은 쪽)")
    ap.add_argument("--repivot", action="store_true", help="팔 끝 위치 캘리(pivot)를 다시 한다")
    ap.add_argument("--squares", default="5x7")
    ap.add_argument("--square", type=float, help="칸 한 변 m (intrinsic 때와 같은 보드)")
    ap.add_argument("--marker", type=float, help="마커 한 변 m (기본 --square x 0.6)")
    ap.add_argument("--dict", default="DICT_6X6_250")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--min-corners", type=int, default=8)
    ap.add_argument("--camera", help="장치 경로 또는 번호 (기본: config.py 의 port)")
    ap.add_argument("--arm-port", default=default_port("so101_follower"), help="follower 포트")
    ap.add_argument("--arm-id", default="follower", help="follower 캘리브레이션 이름 (lerobot 의 --robot.id)")
    args = ap.parse_args()

    name = args.name
    link = args.link or DEFAULT_LINK.get(name, "base")
    EXT.mkdir(exist_ok=True)
    store_path = EXT / f"{name}_samples.json"
    store = None if args.reset or not store_path.exists() else json.loads(store_path.read_text(encoding="utf-8"))
    if store is not None and store["link"] != link:
        raise SystemExit(f"{store_path.name} 의 표본은 link={store['link']} 로 찍었다. --reset 으로 지우거나 --link 를 맞춘다.")
    cam_cfg = load_camera_config(name)
    fk = LinkFK()

    if not args.replay:
        if args.square is None:
            raise SystemExit("--square 를 주세요 (인쇄한 보드의 칸 한 변, m).")
        marker = args.marker or round(args.square * 0.6, 6)
        squares = parse_squares(args.squares)
        board = make_board(squares, args.square, marker, args.dict, args.legacy)
        board_info = {"squares_x": squares[0], "squares_y": squares[1], "square": args.square,
                      "marker": marker, "dictionary": args.dict, "legacy": args.legacy}
        if args.measure_board:  # 표본 파일은 건드리지 않는다
            measure_board(args, name, board, board_info, cam_cfg)
            return
        store = store or {"link": link, "board": board_info, "calibration": {}, "samples": []}
        if store["board"] != board_info:
            raise SystemExit(f"이전 표본은 다른 보드 설정으로 찍었다: {store['board']}. --reset 으로 지운다.")
        capture(args, name, link, board, board_info, store, store_path, cam_cfg, fk)

    if not store or not store["samples"]:
        raise SystemExit("표본이 없습니다.")
    if args.board_on_gripper:
        if link != "base":
            raise SystemExit("--board-on-gripper 는 고정 카메라(link=base) 용입니다.")
        mine = [s_ for s_ in store["samples"] if s_["placement"] == args.placement and s_.get("q_raw")]
        board = make_board((store["board"]["squares_x"], store["board"]["squares_y"]), store["board"]["square"],
                           store["board"]["marker"], store["board"]["dictionary"], store["board"]["legacy"])
        attach_points(mine, name, Detector(board))
        mine = [s_ for s_ in mine if "img" in s_]
        if len(mine) < 8:
            raise SystemExit(f"표본이 {len(mine)} 개뿐입니다 (최소 8, 15~20 추천). 같은 명령으로 이어서 찍을 수 있습니다.")
        print(f"\n{name}: 보드를 그리퍼에 달고 찍은 표본 {len(mine)} 개")
        fit = solve_eye_to_hand(mine, store["calibration"], fk, np.array(cam_cfg.camera_matrix), np.array(cam_cfg.dist_coeffs))
        T = fit["T_base_cam"]
        Ts = transform(np.eye(3), base_in_trolley() + [0, 0, TROLLEY_TOP_Z]) @ T
        result = {
            "link": "base", "T_link_cam": T.tolist(), "position": T[:3, 3].tolist(), "quat_wxyz": quat_wxyz(T[:3, :3]),
            "mujoco_body": "worldbody (scene_trolley.xml)", "mujoco_pos": Ts[:3, 3].tolist(),
            "mujoco_xyaxes": np.concatenate([Ts[:3, 0], -Ts[:3, 1]]).tolist(),
            "mujoco_fovy": cam_cfg.fovy_deg if cam_cfg.calibrated else None, "num_samples": len(mine),
            "placements": [args.placement], "spread_mm": float(np.sqrt(np.mean(np.square(fit["dpos"])))),
            "spread_deg": float(np.sqrt(np.mean(np.square(fit["drot"])))), "reprojection_px": fit["rms_px"],
        }
        out = write_camera_extrinsic(name, result)
        fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
        fwd = T[:3, 2]
        print(f"\n베이스 기준 카메라 위치 (m): {fmt(T[:3, 3])}")
        print(f"보는 방향: yaw {math.degrees(math.atan2(fwd[1], fwd[0])):.1f} 도, 아래로 {math.degrees(math.asin(np.clip(-fwd[2], -1, 1))):.1f} 도")
        print(f"표본별 퍼짐 RMS: {result['spread_mm']:.1f} mm, {result['spread_deg']:.2f} 도")
        fovy = f' fovy="{result["mujoco_fovy"]:.2f}"' if result["mujoco_fovy"] else ""
        print("MuJoCo 에 넣을 카메라 (worldbody 안):")
        print(f'  <camera name="{name}" pos="{fmt(result["mujoco_pos"])}" xyaxes="{fmt(result["mujoco_xyaxes"])}"{fovy} '
              f'resolution="{cam_cfg.width} {cam_cfg.height}"/>')
        print("저장:", out)
        return
    if args.lens_height is not None:
        if link != "shoulder":
            raise SystemExit("--lens-height (pan-only) 는 shoulder 링크 카메라(belly) 용입니다.")
        mine = [s_ for s_ in store["samples"] if s_["placement"] == args.placement and s_.get("q_raw")]
        board = make_board((store["board"]["squares_x"], store["board"]["squares_y"]), store["board"]["square"],
                           store["board"]["marker"], store["board"]["dictionary"], store["board"]["legacy"])
        attach_points(mine, name, Detector(board))
        mine = [s_ for s_ in mine if "img" in s_]
        if len({round(pan_angle(s_["q_raw"]), 2) for s_ in mine}) < 3:
            raise SystemExit(f"pan 각도가 다른 표본이 3 개 이상 필요합니다 (지금 {len(mine)} 개). 이어서 찍으려면 같은 명령을 다시.")
        result = solve_pan_only(mine, fk, np.array(cam_cfg.camera_matrix), np.array(cam_cfg.dist_coeffs),
                                args.lens_height, args.cam_yaw)
        result["mujoco_fovy"] = cam_cfg.fovy_deg if cam_cfg.calibrated else None
        out = write_camera_extrinsic(name, result)
        fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
        print(f"\nshoulder 기준 카메라 위치 (m): {fmt(result['position'])}")
        fovy = f' fovy="{result["mujoco_fovy"]:.2f}"' if result["mujoco_fovy"] else ""
        print('MuJoCo 에 넣을 카메라 (<body name="shoulder"> (so101.xml) 안):')
        print(f'  <camera name="{name}" pos="{fmt(result["mujoco_pos"])}" xyaxes="{fmt(result["mujoco_xyaxes"])}"{fovy} '
              f'resolution="{cam_cfg.width} {cam_cfg.height}"/>')
        print("저장:", out)
        return
    if args.replay and link == TOOL:
        for pl in sorted({s_["placement"] for s_ in store["samples"]}):
            bpath = EXT / f"board_{pl}.json"
            old = json.loads(bpath.read_text(encoding="utf-8")) if bpath.exists() else None
            if old is not None and old.get("method") != "handeye":
                continue
            mine = [s_ for s_ in store["samples"] if s_["placement"] == pl]
            print(f"\n보드 자리 {pl} (wrist hand-eye 다시 풀기, 표본 {len(mine)} 개):")
            board = make_board((store["board"]["squares_x"], store["board"]["squares_y"]), store["board"]["square"],
                               store["board"]["marker"], store["board"]["dictionary"], store["board"]["legacy"])
            corner_ids, corner_pts = extreme_corners(board)
            attach_points(mine, name, Detector(board))
            fit = solve_handeye(mine, store["calibration"], fk, corner_pts, args.offsets and len(mine) >= 12,
                                np.array(cam_cfg.camera_matrix), np.array(cam_cfg.dist_coeffs))
            bpath.write_text(json.dumps({"board": store["board"], "method": "handeye", "calibration": store["calibration"],
                                         "corner_ids": corner_ids, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **fit},
                                        indent=1), encoding="utf-8")
    result = compute(name, store, fk, args.mapping, cam_cfg.fovy_deg if cam_cfg.calibrated else None)
    out = write_camera_extrinsic(name, result)

    fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
    print(f"\n{link} 기준 카메라 위치 (m): {fmt(result['position'])}")
    print(f"퍼짐 RMS: {result['spread_mm']:.1f} mm, {result['spread_deg']:.2f} 도")
    if name in MODEL_CAMERA and link == "gripper":
        T_cad = fk.model_camera(MODEL_CAMERA[name], link)
        if T_cad is not None:
            T = np.array(result["T_link_cam"])
            print(f"시뮬 모델 {MODEL_CAMERA[name]} (CAD) 와의 차이: {np.linalg.norm(T[:3, 3] - T_cad[:3, 3]) * 1000:.1f} mm, "
                  f"{rot_deg(T_cad[:3, :3].T @ T[:3, :3]):.1f} 도   (CAD 위치 {fmt(T_cad[:3, 3])})")
    print(f"MuJoCo 에 넣을 카메라 ({result['mujoco_body']} 안):")
    fovy = f' fovy="{result["mujoco_fovy"]:.2f}"' if result["mujoco_fovy"] else ""
    print(f'  <camera name="{name}" pos="{fmt(result["mujoco_pos"])}" xyaxes="{fmt(result["mujoco_xyaxes"])}"{fovy} '
          f'resolution="{cam_cfg.width} {cam_cfg.height}"/>')
    print("저장:", out)


def load_extrinsic(name: str) -> np.ndarray:
    """extrinsic/<name>.py 의 T_link_cam."""
    path = EXT / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"{name} 의 extrinsic 이 없습니다 ({path}). 먼저 구하세요.")
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    cfg = ns[f"{name.capitalize()}CamExtrinsicConfig"]()
    if cfg.link != "base":
        raise SystemExit(f"{name} 는 link={cfg.link} 라서 고정 카메라가 아닙니다.")
    return np.array(cfg.T_link_cam)


def measure_board(args, name, board, board_info, cam_cfg) -> None:
    """이미 구한 고정 카메라로 놓아 둔 보드를 보고, 베이스 기준 보드 위치를 board_<placement>.json 에 저장한다."""
    T_base_cam = load_extrinsic(name)
    K, dist = np.array(cam_cfg.camera_matrix), np.array(cam_cfg.dist_coeffs)
    detector = Detector(board)
    corner_ids, corner_pts = extreme_corners(board)
    cap = open_camera(args.camera or cam_cfg.port, cam_cfg.width, cam_cfg.height)
    view = LiveView(f"measure board ({name}, {args.placement})", cam_cfg.width, cam_cfg.height)
    poses, errs, message = [], [], "board must not move. s: add (3-5), q: save"
    try:
        done = False
        while view.open and not done:
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit("카메라에서 화면을 읽지 못했습니다.")
            corners, ids, _, _ = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            shown, found, why = bgr.copy(), None, "no board"
            if ids is not None and len(ids) > 0:
                obj, img = detector.points(corners, ids)
                why = usable(corners, ids, obj, args.min_corners)
                cv2.aruco.drawDetectedCornersCharuco(shown, corners, ids, (0, 0, 255))
                if why is None:
                    ok_pnp, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
                    if ok_pnp:
                        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
                        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                        found = (transform(cv2.Rodrigues(rvec)[0], tvec),
                                 float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=-1)))), len(ids))
                        cv2.drawFrameAxes(shown, K, dist, rvec, tvec, 2 * board_info["square"], 3)
                        o, _ = cv2.projectPoints(corner_pts[:1], rvec, tvec, K, dist)
                        cv2.circle(shown, tuple(int(v) for v in o.ravel()), 8, (0, 255, 255), -1)
            state = why or f"{found[2]} corners, reproj {found[1]:.2f}px"
            for key in view.show(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                                 f"[{args.placement} measure board with {name}] samples {len(poses)}  [{state}]\n{message}"):
                if key == "s":
                    if found is None:
                        message = f"(not added: {why})"
                    else:
                        poses.append(found[0])
                        errs.append(found[1])
                        message = f"(added #{len(poses)})"
                elif key == "q":
                    done = True
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    if not poses:
        raise SystemExit("표본이 없습니다.")
    T_cam_board = average(poses)
    spread = max(np.linalg.norm(T[:3, 3] - T_cam_board[:3, 3]) for T in poses) * 1000
    T_base_board = T_base_cam @ T_cam_board
    placement = {"board": board_info, "method": f"camera:{name}", "mapping": args.mapping or "deg",
                 "T_base_board": T_base_board.tolist(), "corner_ids": corner_ids, "num_samples": len(poses),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = EXT / f"board_{args.placement}.json"
    path.write_text(json.dumps(placement, indent=1), encoding="utf-8")
    print(f"\n보드 자리 {args.placement}: {name} 카메라로 잼 (표본 {len(poses)} 개, 흔들림 최대 {spread:.1f} mm, "
          f"재투영 {np.mean(errs):.2f} px)")
    print(f"  베이스 기준 노란 점 위치 (m): {np.round(T_base_board[:3, 3], 4).tolist()}")
    print(f"  ({name} 의 extrinsic 오차가 그대로 들어간다)")
    print("저장:", path)


def capture(args, name, link, board, board_info, store, store_path, cam_cfg, fk) -> None:
    if not cam_cfg.calibrated:
        raise SystemExit(f"{name} 카메라의 intrinsic 이 없습니다. 먼저 charuco_intrinsic_cali.py {name}.")
    K, dist = np.array(cam_cfg.camera_matrix), np.array(cam_cfg.dist_coeffs)
    width, height = cam_cfg.width, cam_cfg.height
    detector = Detector(board)
    corner_ids, corner_pts = extreme_corners(board)

    board_path = EXT / f"board_{args.placement}.json"
    placement = None if args.retouch or not board_path.exists() else json.loads(board_path.read_text(encoding="utf-8"))
    if placement is not None and placement["board"] != board_info:
        raise SystemExit(f"{board_path.name} 는 다른 보드 설정으로 찍었다. --retouch 로 다시 찍는다.")
    # wrist 는 그 자리를 정하는 카메라라서, hand-eye 로 잰 자리면 찍을 때마다 모든 표본으로 다시 푼다
    if args.measured and (placement is None or args.retouch):
        if not args.origin or len(args.origin) not in (2, 3):
            raise SystemExit("--measured 에는 --origin X Y [Z] 가 필요합니다.")
        placement = {"board": board_info, "method": "measured", "kind": args.measured, "origin": args.origin, "yaw": args.yaw,
                     "mapping": args.mapping or "deg", "T_base_board": measured_board_pose(args.measured, args.origin, args.yaw).tolist(),
                     "corner_ids": corner_ids, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        board_path.write_text(json.dumps(placement, indent=1), encoding="utf-8")
        print(f"보드 자리 {args.placement}: 자로 잰 값으로 저장 ({args.measured}, origin {args.origin}, yaw {args.yaw})")
    panonly = args.lens_height is not None or args.board_on_gripper
    handeye = panonly or (not args.touch and (placement is None or (link == TOOL and placement.get("method") == "handeye")))
    if handeye:
        placement = None
    if handeye and link != TOOL and not panonly:
        raise SystemExit(f"보드 자리 {args.placement} 가 아직 측정되지 않았습니다. 먼저 wrist 캠으로 이 자리를 잽니다:\n"
                         f"  extrinsic_cali.py wrist --square {board_info['square']} --placement {args.placement}\n"
                         "(또는 --touch 로 팔 끝으로 꼭짓점을 찍어 잰다)")
    if args.retouch:
        # 이 자리의 예전 표본은 다시 잴 때 섞이지 않게 뺀다
        store["samples"] = [s_ for s_ in store["samples"] if s_["placement"] != args.placement]
    tip_path = EXT / "tip.json"
    tip = json.loads(tip_path.read_text(encoding="utf-8")) if tip_path.exists() and not args.repivot else None
    if placement is not None or handeye:
        tip = tip or {}
    need_arm = placement is None or link != "base" or args.board_on_gripper

    arm = Follower(args.arm_port, args.arm_id) if need_arm else None
    if arm is not None:
        if store["calibration"] and arm.calibration and store["calibration"] != arm.calibration:
            raise SystemExit("follower 캘리브레이션이 이전 표본 때와 다르다. --reset 으로 지우고 다시 찍는다.")
        store["calibration"] = store["calibration"] or arm.calibration
        if tip and tip.get("calibration") is not None and tip["calibration"] != arm.calibration:
            raise SystemExit("follower 캘리브레이션이 pivot 때와 다르다. --repivot 으로 다시 한다.")
    cap = open_camera(args.camera or cam_cfg.port, width, height)
    view = LiveView(f"extrinsic_cali ({name}, link={link}, {args.placement})", width, height)
    touches: list[dict] = []
    if tip is None:
        message = "PIVOT: keep the jaw tip on one fixed point, change arm pose a lot, c each time (6+), p to finish"
    elif args.board_on_gripper:
        message = "BOARD ON GRIPPER: move the arm so the camera sees the board in many poses. h, wait 1s, s (15-20). q to solve"
    elif panonly:
        message = "PAN-ONLY: move ONLY shoulder_pan (wide range), h, wait 1s, s. 5-10 angles, q to solve"
    elif handeye:
        message = "HAND-EYE: vary the arm pose a lot (pan, wrist flex, wrist roll). h: hold, s: add, q: solve (12+)"
    elif placement is None:
        message = "touch the YELLOW corner with the fixed jaw tip, press c"
    else:
        message = ""
    added = 0

    def analyze(bgr):
        """한 장 -> (보드 자세 등 또는 None, 못 쓰는 이유, 그려 넣은 화면)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        corners, ids, _, _ = detector.detect(rgb)
        shown = bgr.copy()
        found, why = None, "no board"
        if ids is not None and len(ids) > 0:
            obj, img = detector.points(corners, ids)
            why = usable(corners, ids, obj, args.min_corners)
            cv2.aruco.drawDetectedCornersCharuco(shown, corners, ids, (0, 0, 255))
            if why is None:
                ok_pnp, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
                if ok_pnp:
                    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
                    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                    err = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=-1))))
                    found = (transform(cv2.Rodrigues(rvec)[0], tvec), err, len(ids), obj, img)
                    cv2.drawFrameAxes(shown, K, dist, rvec, tvec, 2 * board_info["square"], 3)
                    o, _ = cv2.projectPoints(corner_pts[:1], rvec, tvec, K, dist)
                    cv2.circle(shown, tuple(int(v) for v in o.ravel()), 8, (0, 255, 255), -1)
        return found, why, cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)

    def fresh_frame():
        """카메라가 쌓아 둔 예전 프레임을 버리고 지금 찍힌 것을 읽는다 (macOS 는 몇 장씩 밀려 있다)."""
        for _ in range(FLUSH_FRAMES):
            cap.grab()
        ok_, frame = cap.read()
        if not ok_:
            raise SystemExit("카메라에서 화면을 읽지 못했습니다.")
        return frame

    def save_store():
        store_path.write_text(json.dumps(store, indent=1), encoding="utf-8")

    try:
        done = False
        while view.open and not done:
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit("카메라에서 화면을 읽지 못했습니다.")
            found, why, shown = analyze(bgr)

            hold = ("  h: torque " + ("ON" if arm.holding else "off")) if arm else ""
            if tip is None:
                status = f"[pivot] recorded {len(touches)}  c: record  u: undo  p: finish (6+){hold}\n{message}"
            elif handeye:
                n_he = sum(1 for s_ in store["samples"] if s_["placement"] == args.placement)
                state = why or f"{found[2]} corners, reproj {found[1]:.2f}px"
                status = f"[{args.placement} {'board-on-gripper' if args.board_on_gripper else 'pan-only' if panonly else 'hand-eye'}] samples {n_he}  [{state}]  s: add  u: undo  q: solve & quit{hold}\n{message}"
            elif placement is None:
                step = f"touch {LABELS[len(touches)]}  c: record  u: undo"
                status = f"[{args.placement}] {step}{hold}\n{message}"
            else:
                state = why or f"{found[2]} corners, reproj {found[1]:.2f}px"
                status = (f"[{args.placement}] samples {added} (total {len(store['samples'])})  [{state}]  "
                          f"s: add  u: undo  q: save & quit{hold}\n{message}")
            for key in view.show(shown, status):
                if key == "h" and arm:
                    arm.toggle_hold()
                elif key == "q":
                    done = True
                elif (handeye or placement is not None) and key == "u":
                    # 이번 실행에서 찍은 것만 지운다 (예전 실행의 표본은 건드리지 않는다)
                    if added > 0 and store["samples"] and store["samples"][-1]["placement"] == args.placement:
                        gone = store["samples"].pop()
                        save_store()
                        added -= 1
                        message = f"(removed last sample{' ' + gone['image'] if gone.get('image') else ''})"
                    else:
                        message = "(nothing to remove in this run)"
                elif placement is None and not handeye and key == "u" and touches:
                    touches.pop()
                    message = "(undone)"
                elif placement is None and not handeye and key in ("c", "p"):
                    if arm.holding:
                        message = "(torque is ON: press h to release first)"
                        continue
                    if key == "c":
                        raw, spread = arm.read()
                        touches.append(raw)
                        message = f"(recorded {len(touches)}" + (f", arm moved {spread:.0f} ticks while reading" if spread > 10 else "") + ")"
                    if tip is None and key == "p":
                        if len(touches) < 6:
                            message = f"(need 6+ poses, have {len(touches)})"
                            continue
                        print("\n팔 끝 위치 (pivot):")
                        tip = solve_pivot(touches, arm.calibration, fk)
                        tip_path.write_text(json.dumps(tip, indent=1), encoding="utf-8")
                        touches = []
                        message = f"tip saved (RMS {tip['rms_mm']:.1f} mm). now touch the YELLOW corner, press c"
                    elif tip is not None and len(touches) == 4:
                        print(f"\n보드 자리 {args.placement}:")
                        fit = fit_board(touches, arm.calibration, tip, fk, corner_pts)
                        placement = {"board": board_info, "touches": touches, "calibration": arm.calibration,
                                     "corner_ids": corner_ids, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **fit}
                        board_path.write_text(json.dumps(placement, indent=1), encoding="utf-8")
                        message = f"board placement saved (corner RMS {fit['score'] * 1000:.1f} mm). move the arm away, capture with s"
                        if link == "base":
                            arm.close()
                            arm = None
                elif (placement is not None or handeye) and key == "s":
                    # 관절 읽기 -> 새 사진 -> 관절 다시 읽기. 사진과 관절 값이 같은 순간이 되게 하고, 그사이 움직였으면 버린다
                    q_raw = q_before = None
                    if link != "base" or args.board_on_gripper:
                        q_before, spread = arm.read()
                        if spread > 10:
                            message = f"(not added: arm moving {spread:.0f} ticks. hold still or press h)"
                            continue
                    bgr = fresh_frame()
                    found, why, _ = analyze(bgr)
                    if found is None:
                        message = f"(not added: {why})"
                        continue
                    if link != "base" or args.board_on_gripper:
                        q_after, spread = arm.read()
                        moved = max(abs(q_after[j] - q_before[j]) for j in JOINT_NAMES)
                        if spread > 10 or moved > SETTLE_TICKS:
                            message = f"(not added: arm moved {moved:.0f} ticks while capturing. wait until it settles)"
                            continue
                        q_raw = {j: (q_before[j] + q_after[j]) / 2 for j in JOINT_NAMES}
                    shots = EXT / f"{name}_captures"
                    shots.mkdir(exist_ok=True)
                    shot = f"{args.placement}_{time.strftime('%H%M%S')}_{len(store['samples']) + 1:02d}.png"
                    cv2.imwrite(str(shots / shot), bgr)  # 나중에 다시 계산/확인할 수 있게 원본을 남긴다
                    store["samples"].append({"placement": args.placement, "T_cam_board": found[0].tolist(),
                                             "q_raw": q_raw, "reproj_px": found[1], "corners": found[2], "image": shot,
                                             "obj": found[3].reshape(-1, 3).tolist(), "img": found[4].reshape(-1, 2).tolist()})
                    save_store()
                    added += 1
                    message = f"(added #{len(store['samples'])})"
                    if q_raw is not None:
                        check = joint_check(store["samples"][-2:] if len(store["samples"]) > 1 else [])
                        if check:
                            print(check)
                            message += "  " + check
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if arm is not None:
            arm.close()
    if panonly:
        return
    if handeye:
        mine = [s_ for s_ in store["samples"] if s_["placement"] == args.placement]
        if len(mine) < 8:
            raise SystemExit(f"hand-eye 표본이 {len(mine)} 개뿐입니다 (최소 8, 15~20 추천). 같은 명령으로 이어서 찍을 수 있습니다.")
        print(f"\n보드 자리 {args.placement} (wrist hand-eye, 표본 {len(mine)} 개):")
        attach_points(mine, name, detector)
        fit = solve_handeye(mine, store["calibration"], fk, corner_pts, args.offsets and len(mine) >= 12, K, dist)
        placement = {"board": board_info, "method": "handeye", "calibration": store["calibration"],
                     "corner_ids": corner_ids, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **fit}
        board_path.write_text(json.dumps(placement, indent=1), encoding="utf-8")
        print("저장:", board_path)
        print(f"  (이 자리로 이미 계산한 다른 카메라는 --replay 로 다시 계산한다: extrinsic_cali.py <name> --replay)")
    if placement is None:
        raise SystemExit("꼭짓점 4 개를 다 찍지 못해서 보드 자리를 저장하지 않았습니다.")


if __name__ == "__main__":
    main()
