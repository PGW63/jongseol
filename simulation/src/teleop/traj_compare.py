"""정해진 끝점 궤적을 리더암 텔레옵과 키보드 IK 로 따라가 보고, 둘의 오차(RMSE)를 비교한다.

    # 1) 빈 월드에서 궤적 따라가기. 궤적마다 기록하고 다음 궤적으로 넘어간다
    .venv/bin/python src/teleop/traj_compare.py run --mode teleop
    .venv/bin/python src/teleop/traj_compare.py run --mode keyboard
    .venv/bin/python src/teleop/traj_compare.py run --mode keyboard --traj circle_xy square_xy --repeat 3
    .venv/bin/python src/teleop/traj_compare.py run --mode keyboard --pace timed --slow 2   # 시간 맞춰 쫓기

    # 2) 지금까지 모은 기록을 한 장의 그래프로
    .venv/bin/python src/teleop/traj_compare.py plot --save results/traj_compare.png

    # 궤적 목록
    .venv/bin/python src/teleop/traj_compare.py list

check_connection_with_leader.py 처럼 MuJoCo 뷰어 대신 오프스크린 렌더를 그래프 창 안에 그린다.
키 입력도 이 창이 받는다 (창을 한 번 클릭해 포커스를 줄 것).

진행
    ready     : 회색 선이 기준 궤적, 초록 공이 시작점. 끝점(gripperframe)을 시작점에 1 초 대고 있거나
                space 를 누르면 시작. 키보드 모드는 팔이 시작점에 놓인 채로 시작한다.
    countdown : 3 초. 그 사이 시작점에 머문다.
    record    : --pace free (기본) 자기 속도로 경로를 따라 그린다. 따라온 구간은 초록으로 칠해지고,
                빨간 공은 조금 앞의 길잡이. 끝까지 가면 (또는 enter) 끝.
                --pace timed  빨간 공이 궤적을 일정한 속도로 움직이고 그걸 쫓는다. --slow 로 늦출 수 있다.
    끝나면 results/traj_compare/<궤적>__<모드>__<시각>.npz 로 저장하고 다음 궤적으로.

키
    space 시작 · enter 여기서 끝내고 저장 · r 지금 궤적 다시 · n 건너뛰기 · esc 끝내기 · [ ] 카메라 회전
    키보드 모드: w/s 앞뒤(x) · a/d 좌우(y) · q/e 위아래(z)  (누르고 있는 동안 움직인다)

평가
    path RMSE     : 끝점에서 기준 궤적(선)까지의 최단 거리 (mm). 모양만 맞으면 작다. 두 방식 공통.
    time to finish: free 에서 끝까지 가는 데 걸린 시간 (s).
    tracking RMSE : timed 에서 같은 시각의 기준점과 끝점 사이 거리 (mm). 타이밍까지 맞아야 작다.
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from teleop.constants import JOINT_NAMES, MUJOCO_JOINT_RANGE, default_port

ROOT = Path(__file__).resolve().parents[2]
EMPTY_SCENE = ROOT / "models" / "scene" / "scene_empty.xml"
RESULTS = ROOT / "results" / "traj_compare"

PREFIX = "so101_"
EE_SITE = PREFIX + "gripperframe"
# IK 로 움직이는 관절. wrist_roll 과 gripper 는 끝점 위치에 거의 영향이 없어 고정한다
IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
# 키보드 모드의 기본 자세 (rad, JOINT_NAMES 순). IK 의 남는 자유도를 이 쪽으로 당긴다
REST_Q = np.array([0.0, -1.0, 1.0, 0.8, 0.0, 0.3])

MODES = ("teleop", "keyboard")
MODE_COLOR = {"teleop": "#2a78d6", "keyboard": "#eb6834"}
MODE_LABEL = {"teleop": "leader teleop", "keyboard": "keyboard IK"}
REF_COLOR = "#555555"

COUNTDOWN_S = 3.0
START_TOL_M = 0.015  # 끝점이 시작점에서 이 거리 안에
START_HOLD_S = 1.0  # 이만큼 머무르면 자동 시작


# ---------------------------------------------------------------- 궤적


@dataclass
class Traj:
    """끝점이 따라갈 기준 궤적. 모양(s: 0~1 -> xyz)과 한 바퀴 시간."""

    name: str
    shape: callable
    duration: float
    plane: str  # 2D 로 그릴 때 쓸 평면 ("xy" 위에서 / "xz" 옆에서)

    def __post_init__(self):
        # 모양을 촘촘히 찍고 호 길이로 다시 나눠 기준점이 일정한 속도로 움직이게 한다
        pts = np.array([self.shape(s) for s in np.linspace(0.0, 1.0, 2000)])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(arc[-1])
        u = np.linspace(0.0, self.length, 1000)
        self.points = np.stack([np.interp(u, arc, pts[:, k]) for k in range(3)], axis=1)

    def at(self, t: float, duration: float | None = None) -> np.ndarray:
        """기록 시작 후 t 초에 기준점이 있어야 할 위치 (duration 동안 한 바퀴)."""
        u = np.clip(t / (duration or self.duration), 0.0, 1.0) * (len(self.points) - 1)
        i = min(int(u), len(self.points) - 2)
        return self.points[i] + (u - i) * (self.points[i + 1] - self.points[i])

    @property
    def start(self) -> np.ndarray:
        return self.points[0]


def _circle(center, r, plane):
    a, b = {"xy": (0, 1), "xz": (0, 2)}[plane]

    def f(s):
        p = np.array(center, dtype=float)
        p[a] += r * np.cos(2 * np.pi * s)
        p[b] += r * np.sin(2 * np.pi * s)
        return p

    return f


def _polyline(corners):
    corners = np.asarray(corners, dtype=float)

    def f(s):
        u = s * (len(corners) - 1)
        i = min(int(u), len(corners) - 2)
        return corners[i] + (u - i) * (corners[i + 1] - corners[i])

    return f


def _figure8(center, a):
    def f(s):
        th = 2 * np.pi * s
        return np.array(center) + np.array([a * np.sin(2 * th) / 2, a * np.sin(th), 0.0])

    return f


# 로봇 베이스가 원점, +x 가 앞. gripperframe 이 편하게 닿는 곳은 대략 (0.24, 0, 0.12) 근처
TRAJS = {
    t.name: t
    for t in [
        Traj("line_xy", _polyline([[0.18, -0.08, 0.10], [0.30, 0.08, 0.10], [0.18, -0.08, 0.10]]), 6.0, "xy"),
        Traj("circle_xy", _circle([0.24, 0.0, 0.10], 0.06, "xy"), 8.0, "xy"),
        Traj("circle_xz", _circle([0.24, 0.0, 0.15], 0.05, "xz"), 8.0, "xz"),
        Traj("square_xy", _polyline([[0.19, -0.05, 0.10], [0.29, -0.05, 0.10], [0.29, 0.05, 0.10],
                                     [0.19, 0.05, 0.10], [0.19, -0.05, 0.10]]), 8.0, "xy"),
        Traj("figure8_xy", _figure8([0.24, 0.0, 0.10], 0.08), 10.0, "xy"),
    ]
}


def path_distance(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """각 점에서 기준 궤적(촘촘한 점들)까지의 최단 거리."""
    return np.min(np.linalg.norm(points[:, None, :] - ref[None, :, :], axis=2), axis=1)


def metrics(traj: Traj, ref: np.ndarray, ee: np.ndarray, t: np.ndarray, progress: np.ndarray) -> dict[str, float]:
    """tracking / path RMSE (mm), 걸린 시간 (s), 진행률 (%)."""
    track = np.linalg.norm(ee - ref, axis=1)
    path = path_distance(ee, traj.points)
    return {
        "tracking_rmse_mm": float(np.sqrt(np.mean(track**2)) * 1e3),
        "path_rmse_mm": float(np.sqrt(np.mean(path**2)) * 1e3),
        "path_max_mm": float(path.max() * 1e3),
        "time_s": float(t[-1]),
        "completion_pct": float(progress[-1] * 100),
    }


class Progress:
    """끝점이 기준 궤적을 어디까지 따라왔는지 (0~1). 앞쪽 일부 구간에서만 가장 가까운 점을 찾아,
    시작점과 끝점이 같은 닫힌 궤적이나 8자처럼 겹치는 궤적에서도 진행이 건너뛰지 않게 한다."""

    WINDOW = 0.08  # 한 번에 앞으로 찾아보는 범위 (궤적 길이 비율)

    def __init__(self, traj: Traj):
        self.points = traj.points
        self.idx = 0
        self.win = max(2, int(self.WINDOW * len(self.points)))

    def update(self, ee: np.ndarray) -> float:
        seg = self.points[self.idx:self.idx + self.win]
        self.idx += int(np.argmin(np.linalg.norm(seg - ee, axis=1)))
        return self.fraction

    @property
    def fraction(self) -> float:
        return self.idx / (len(self.points) - 1)

    @property
    def point(self) -> np.ndarray:
        return self.points[self.idx]

    def ahead(self, frac: float) -> np.ndarray:
        return self.points[min(self.idx + int(frac * len(self.points)), len(self.points) - 1)]


# ---------------------------------------------------------------- 입력 (리더암 / 키보드 IK)


class PositionIK:
    """끝점 위치만 맞추는 감쇠 최소제곱 IK. 남는 자유도는 기본 자세 쪽으로 당긴다."""

    def __init__(self, model: mujoco.MjModel, damping: float = 0.02, rest_gain: float = 0.05):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(EE_SITE).id
        self.idx = [JOINT_NAMES.index(j) for j in IK_JOINTS]
        self.qadr = [int(model.joint(PREFIX + j).qposadr[0]) for j in JOINT_NAMES]
        self.dofs = [int(model.joint(PREFIX + j).dofadr[0]) for j in IK_JOINTS]
        self.lo = np.array([MUJOCO_JOINT_RANGE[j][0] for j in IK_JOINTS])
        self.hi = np.array([MUJOCO_JOINT_RANGE[j][1] for j in IK_JOINTS])
        self.damping = damping
        self.rest_gain = rest_gain
        self.jac = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy()

    def solve(self, q0: np.ndarray, target: np.ndarray, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """q0(관절 6개, rad) 에서 출발해 target 에 끝점을 맞춘다 -> (관절, 실제로 닿은 끝점)."""
        q = q0.copy()
        for _ in range(iters):
            pos = self.fk(q)
            err = target - pos
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self.jac, None, self.site)
            J = self.jac[:, self.dofs]
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.damping**2 * np.eye(3))
            null = np.eye(len(self.idx)) - J_pinv @ J
            qa = q[self.idx]
            q[self.idx] = np.clip(qa + J_pinv @ err + null @ (self.rest_gain * (REST_Q[self.idx] - qa)), self.lo, self.hi)
        return q, self.fk(q)


class KeyboardIK:
    """w/s a/d q/e 로 3D 커서를 움직이고, 팔은 IK 로 커서를 쫓는다."""

    KEYS = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1), "q": (2, +1), "e": (2, -1)}

    def __init__(self, model: mujoco.MjModel, speed: float):
        self.ik = PositionIK(model)
        self.speed = speed
        self.held: set[str] = set()
        self.q = REST_Q.copy()
        self.cursor = self.ik.fk(self.q)

    def place(self, target: np.ndarray) -> np.ndarray:
        """팔을 target 에 바로 놓는다 (궤적 시작점에 맞출 때)."""
        self.q, self.cursor = self.ik.solve(REST_Q, target, iters=200)
        return self.q

    def press(self, key: str) -> None:
        if key in self.KEYS:
            self.held.add(key)

    def release(self, key: str) -> None:
        self.held.discard(key)

    def step(self, dt: float) -> np.ndarray:
        """눌린 키만큼 커서를 옮기고 관절 목표(rad)를 돌려준다."""
        vel = np.zeros(3)
        for k in self.held:
            axis, sign = self.KEYS[k]
            vel[axis] += sign
        if vel.any():
            self.cursor = self.cursor + self.speed * dt * vel / np.linalg.norm(vel)
            self.q, reached = self.ik.solve(self.q, self.cursor)
            # 닿지 않는 곳이면 커서를 닿는 곳에 붙여 둔다 (커서만 멀리 달아나지 않게)
            if np.linalg.norm(reached - self.cursor) > 0.005:
                self.cursor = reached
        return self.q


class LeaderInput:
    """리더암 정규화값을 MuJoCo 관절 목표(rad)로 읽는다."""

    def __init__(self, port: str, calib_id: str):
        # lerobot 은 리더 모드에서만 필요하다
        from teleop.leader_mj import SO101LeaderMj, SO101LeaderMjConfig

        cfg = SO101LeaderMjConfig(xml_path=EMPTY_SCENE, port=port, id=calib_id)
        self.teleop = SO101LeaderMj(cfg)
        self.teleop.connect(calibrate=False)
        if not self.teleop.is_calibrated:
            print("경고: 모터에 저장된 캘리브레이션이 파일과 다릅니다. lerobot-calibrate 를 다시 실행하세요.")
        print(f"리더암 연결됨: {cfg.port}")

    def read(self) -> np.ndarray:
        action = self.teleop.get_action()
        return np.array([action[f"{j}.pos"] for j in JOINT_NAMES])

    def close(self) -> None:
        if self.teleop.is_connected:
            self.teleop.disconnect()


# ---------------------------------------------------------------- 시뮬레이션 + 화면


def add_sphere(scn: mujoco.MjvScene, pos, radius: float, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(pos, dtype=float), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def add_polyline(scn: mujoco.MjvScene, pts: np.ndarray, width: float, rgba) -> None:
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1


def hex_rgba(color: str, alpha: float = 1.0) -> list[float]:
    return [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [alpha]


class Runner:
    """빈 월드에서 궤적을 하나씩 보여 주고 따라간 결과를 기록한다."""

    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.pace = args.pace
        self.trajs = [TRAJS[n] for n in args.traj for _ in range(args.repeat)]
        self.model = mujoco.MjModel.from_xml_path(str(EMPTY_SCENE))
        self.data = mujoco.MjData(self.model)
        self.act_ids = [self.model.actuator(PREFIX + j).id for j in JOINT_NAMES]
        self.qadr = [int(self.model.joint(PREFIX + j).qposadr[0]) for j in JOINT_NAMES]
        self.site = self.model.site(EE_SITE).id
        self.dt = 1.0 / args.fps
        self.steps_per_tick = max(1, round(self.dt / self.model.opt.timestep))

        self.keyboard = KeyboardIK(self.model, args.speed) if self.mode == "keyboard" else None
        self.leader = LeaderInput(args.port, args.calib_id) if self.mode == "teleop" else None
        self.quit = False
        self.azimuth = -45.0
        self._build_figure()

    # ---- 상태
    def reset_traj(self, i: int) -> None:
        self.i = i
        self.traj = self.trajs[i]
        self.phase = "ready"
        self.phase_t = 0.0
        self.near_t = 0.0
        self.duration = self.traj.duration * self.args.slow
        self.progress = Progress(self.traj)
        self.log_t, self.log_ref, self.log_ee, self.log_q, self.log_p, self.log_d = [], [], [], [], [], []
        if self.keyboard is not None:
            q = self.keyboard.place(self.traj.start)
        else:
            q = self.leader.read()
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.model, self.data)
        self._setup_plots()

    def ee(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    def on_key(self, event, pressed: bool) -> None:
        key = (event.key or "").lower()
        if self.keyboard is not None:
            (self.keyboard.press if pressed else self.keyboard.release)(key)
        if not pressed:
            return
        if key == " " and self.phase == "ready":
            self.phase, self.phase_t = "countdown", 0.0
        elif key == "enter" and self.phase == "record" and self.log_t:
            self.save()  # 끝까지 못 갔어도 여기까지로 저장
            self.next_traj()
        elif key == "r":
            self.reset_traj(self.i)
        elif key == "n":
            self.next_traj()
        elif key == "escape":
            self.quit = True
        elif key in ("[", "]"):
            self.azimuth += 15 if key == "]" else -15

    def next_traj(self) -> None:
        if self.i + 1 >= len(self.trajs):
            self.quit = True
        else:
            self.reset_traj(self.i + 1)

    def tick(self) -> None:
        target = self.keyboard.step(self.dt) if self.keyboard is not None else self.leader.read()
        self.data.ctrl[self.act_ids] = target
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        ee = self.ee()
        self.phase_t += self.dt

        if self.phase == "ready":
            near = np.linalg.norm(ee - self.traj.start) < START_TOL_M
            self.near_t = self.near_t + self.dt if near else 0.0
            if self.near_t >= START_HOLD_S:
                self.phase, self.phase_t = "countdown", 0.0
        elif self.phase == "countdown" and self.phase_t >= COUNTDOWN_S:
            self.phase, self.phase_t = "record", 0.0
        elif self.phase == "record":
            t = self.phase_t - self.dt
            p = self.progress.update(ee)
            self.log_t.append(t)
            # free: 기준점 = 지금까지 따라온 궤적 위의 점 / timed: 시간에 맞춰 움직이는 점
            self.log_ref.append(self.progress.point if self.pace == "free" else self.traj.at(t, self.duration))
            self.log_ee.append(ee)
            self.log_q.append(self.data.qpos[self.qadr].copy())
            self.log_p.append(p)
            self.log_d.append(path_distance(ee[None], self.traj.points)[0])
            if self.pace == "free":  # 거의 다 왔고 끝점 근처면 끝
                done = p >= 0.97 and np.linalg.norm(ee - self.traj.points[-1]) < START_TOL_M
            else:
                done = t >= self.duration
            if done:
                self.save()
                self.next_traj()

    def ref_now(self) -> np.ndarray:
        if self.phase != "record":
            return self.traj.start
        if self.pace == "free":
            return self.progress.ahead(0.04)  # 조금 앞의 점을 길잡이로
        return self.traj.at(self.phase_t, self.duration)

    def save(self) -> None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        t, ref, ee, prog = (np.array(x) for x in (self.log_t, self.log_ref, self.log_ee, self.log_p))
        m = metrics(self.traj, ref, ee, t, prog)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = RESULTS / f"{self.traj.name}__{self.mode}__{stamp}.npz"
        np.savez(path, traj=self.traj.name, mode=self.mode, pace=self.pace, fps=self.args.fps,
                 t=t, ref=ref, ee=ee, q=np.array(self.log_q), progress=prog)
        print(f"[{self.traj.name:<11} {self.mode:<8} {self.pace}] path {m['path_rmse_mm']:5.1f} mm "
              f"(max {m['path_max_mm']:5.1f}) | tracking {m['tracking_rmse_mm']:5.1f} mm | "
              f"{m['time_s']:5.1f} s | {m['completion_pct']:3.0f}%  -> {path.name}")

    # ---- 화면
    def _build_figure(self) -> None:
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []  # matplotlib 기본 단축키(s 저장, q 종료 등)와 겹치지 않게
        self.fig = plt.figure(figsize=(14, 7.5))
        self.fig.canvas.manager.set_window_title("traj_compare")
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1.3, 1])
        self.ax_img = self.fig.add_subplot(gs[:, 0])
        self.ax_img.axis("off")
        self.ax_2d = self.fig.add_subplot(gs[0, 1])
        self.ax_err = self.fig.add_subplot(gs[1, 1])
        self.img = self.ax_img.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.txt = self.ax_img.set_title("", loc="left", fontsize=11, family="monospace")
        self.fig.canvas.mpl_connect("key_press_event", lambda e: self.on_key(e, True))
        self.fig.canvas.mpl_connect("key_release_event", lambda e: self.on_key(e, False))
        self.renderer = mujoco.Renderer(self.model, 480, 640, max_geom=4000)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.22, 0.0, 0.08]
        self.cam.distance = 0.75
        self.cam.elevation = -35

    def _setup_plots(self) -> None:
        a, b = self.traj.plane
        ia, ib = "xyz".index(a), "xyz".index(b)
        self.axes2d = (ia, ib)
        ax = self.ax_2d
        ax.clear()
        pts = self.traj.points * 1e3
        ax.plot(pts[:, ia], pts[:, ib], color=REF_COLOR, lw=1.2, ls="--", label="reference")
        (self.l_ref_pt,) = ax.plot([], [], "o", color="#e34948", ms=8, label="target")
        (self.l_ee,) = ax.plot([], [], color=MODE_COLOR[self.mode], lw=2, label=MODE_LABEL[self.mode])
        ax.set_xlabel(f"{a} (mm)")
        ax.set_ylabel(f"{b} (mm)")
        ax.set_aspect("equal", adjustable="box")
        pad = 30
        ax.set_xlim(pts[:, ia].min() - pad, pts[:, ia].max() + pad)
        ax.set_ylim(pts[:, ib].min() - pad, pts[:, ib].max() + pad)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{self.traj.name}  ({self.i + 1}/{len(self.trajs)})", fontsize=10)

        ax = self.ax_err
        ax.clear()
        (self.l_err,) = ax.plot([], [], color=MODE_COLOR[self.mode], lw=1.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 40)
        ax.set_xlabel("progress along path (%)")
        ax.set_ylabel("distance to path (mm)")
        ax.grid(alpha=0.3)
        self.err_title = ax.set_title("", fontsize=10, loc="left")

    def draw(self) -> None:
        ee = self.ee()
        ref_now = self.ref_now()

        self.cam.azimuth = self.azimuth
        self.renderer.update_scene(self.data, self.cam)
        scn = self.renderer.scene
        add_polyline(scn, self.traj.points[::10], 0.002, [0.35, 0.35, 0.35, 0.8])
        if self.phase == "record" and self.progress.idx >= 10:
            # 이미 지나온 구간은 초록으로
            add_polyline(scn, self.traj.points[:self.progress.idx + 1:10], 0.0028, [0.1, 0.7, 0.3, 0.9])
        add_sphere(scn, self.traj.start, 0.008, [0.1, 0.7, 0.3, 0.6])
        if self.phase == "record":
            add_sphere(scn, ref_now, 0.010, [0.9, 0.25, 0.25, 0.9])
        if len(self.log_ee) > 1:
            add_polyline(scn, np.array(self.log_ee[::2]), 0.0025, hex_rgba(MODE_COLOR[self.mode], 0.9))
        if self.keyboard is not None:
            add_sphere(scn, self.keyboard.cursor, 0.006, [0.95, 0.8, 0.1, 0.8])
        self.img.set_data(self.renderer.render())

        status = {
            "ready": f"READY  hold tip on green ball {START_HOLD_S:.0f}s or press space "
                     f"({np.linalg.norm(ee - self.traj.start) * 1e3:5.1f} mm away)",
            "countdown": f"COUNTDOWN  {COUNTDOWN_S - self.phase_t:3.1f}",
            "record": (f"RECORD  {self.progress.fraction * 100:3.0f}%  {self.phase_t:4.1f} s  (enter: finish here)"
                       if self.pace == "free" else f"RECORD  {self.phase_t:4.1f} / {self.duration:.0f} s"),
        }[self.phase]
        self.txt.set_text(f"[{MODE_LABEL[self.mode]}] {status}")

        ia, ib = self.axes2d
        self.l_ref_pt.set_data([ref_now[ia] * 1e3], [ref_now[ib] * 1e3])
        if self.log_ee:
            e = np.array(self.log_ee) * 1e3
            dist = np.array(self.log_d) * 1e3
            self.l_ee.set_data(e[:, ia], e[:, ib])
            self.l_err.set_data(np.array(self.log_p) * 100, dist)
            self.ax_err.set_ylim(0, max(40, dist.max() * 1.1))
            self.err_title.set_text(f"path RMSE {np.sqrt(np.mean(dist**2)):.1f} mm")
        else:
            self.l_ee.set_data([], [])
            self.l_err.set_data([], [])
            self.err_title.set_text("")
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        self.reset_traj(0)
        self.fig.tight_layout()
        plt.show(block=False)
        print("창을 클릭해 포커스를 주고 시작하세요. (space 시작 · r 다시 · n 건너뛰기 · esc 끝)")
        deadline = time.perf_counter()
        try:
            while plt.fignum_exists(self.fig.number) and not self.quit:
                self.tick()
                if self.quit:
                    break
                self.draw()
                # 남은 시간 동안 키 입력을 처리하며 기다린다. 늦으면 그냥 다음 틱으로
                deadline += self.dt
                rest = deadline - time.perf_counter()
                if rest > 0:
                    self.fig.canvas.start_event_loop(rest)
                else:
                    self.fig.canvas.flush_events()
                    deadline = time.perf_counter()
        except KeyboardInterrupt:
            pass
        finally:
            if self.leader is not None:
                self.leader.close()
            self.renderer.close()


# ---------------------------------------------------------------- 결과 모아 그리기


def load_runs(folder: Path, pace: str) -> list[dict]:
    """pace 가 같은 기록만 읽는다 (free 와 timed 는 점수의 뜻이 달라 섞지 않는다)."""
    runs = []
    for f in sorted(folder.glob("*.npz")):
        z = np.load(f)
        name = str(z["traj"])
        run_pace = str(z["pace"]) if "pace" in z.files else "timed"  # pace 가 없는 예전 기록은 timed
        if name not in TRAJS or run_pace != pace:
            continue
        traj = TRAJS[name]
        run = {"file": f.name, "traj": name, "mode": str(z["mode"]), "t": z["t"], "ref": z["ref"], "ee": z["ee"]}
        run["progress"] = z["progress"] if "progress" in z.files else np.clip(z["t"] / traj.duration, 0, 1)
        run["dist"] = path_distance(run["ee"], traj.points)
        run.update(metrics(traj, run["ref"], run["ee"], run["t"], run["progress"]))
        runs.append(run)
    return runs


def plot_summary(runs: list[dict], pace: str, save: str | None) -> None:
    names = [n for n in TRAJS if any(r["traj"] == n for r in runs)]
    modes = [m for m in MODES if any(r["mode"] == m for r in runs)]

    cols = [("path_rmse_mm", "path mm"), ("tracking_rmse_mm", "tracking mm"), ("time_s", "time s"), ("completion_pct", "done %")]
    print(f"\n{'trajectory':<12}{'mode':<10}{'n':>3}" + "".join(f"{h:>14}" for _, h in cols))
    for n in names:
        for m in modes:
            sel = [r for r in runs if r["traj"] == n and r["mode"] == m]
            if sel:
                vals = "".join(f"{np.mean([r[k] for r in sel]):8.1f} ±{np.std([r[k] for r in sel]):4.1f}" for k, _ in cols)
                print(f"{n:<12}{m:<10}{len(sel):>3}{vals}")

    ncol = len(names)
    fig = plt.figure(figsize=(max(10, 3.4 * ncol), 11))
    fig.canvas.manager.set_window_title("traj_compare summary")
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.1, 1.2, 0.8])
    top = gs[0, :].subgridspec(1, 2, wspace=0.15)

    # 1) 궤적별 RMSE 막대 (여러 번 했으면 평균 ± 표준편차)
    x = np.arange(len(names))
    w = 0.8 / len(modes)
    second = ("time_s", "s", "time to finish") if pace == "free" else \
        ("tracking_rmse_mm", "mm", "tracking RMSE  (same-time distance)")
    for k, (key, unit, title) in enumerate([("path_rmse_mm", "mm", "path RMSE  (distance to reference path)"), second]):
        ax = fig.add_subplot(top[0, k])
        for j, m in enumerate(modes):
            vals = [[r[key] for r in runs if r["traj"] == n and r["mode"] == m] for n in names]
            mean = [np.mean(v) if v else np.nan for v in vals]
            std = [np.std(v) if len(v) > 1 else 0.0 for v in vals]
            pos = x - 0.4 + w * (j + 0.5)
            ax.bar(pos, mean, w * 0.92, yerr=std, color=MODE_COLOR[m], label=MODE_LABEL[m],
                   error_kw={"ecolor": "#333333", "lw": 1, "capsize": 3})
            for p, v, c in zip(pos, mean, vals):
                if c:
                    ax.text(p, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_xticks(x, names, fontsize=9)
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # 범례 자리
        if k == 0:
            ax.legend(fontsize=9, frameon=False, loc="upper left", ncols=len(modes))

    # 2) 궤적별 경로 겹쳐 그리기  3) 진행률에 따른 경로 오차
    for c, n in enumerate(names):
        traj = TRAJS[n]
        ia, ib = ("xyz".index(traj.plane[0]), "xyz".index(traj.plane[1]))
        ax = fig.add_subplot(gs[1, c])
        ax.plot(traj.points[:, ia] * 1e3, traj.points[:, ib] * 1e3, color=REF_COLOR, lw=1.5, ls="--", label="reference")
        ax_e = fig.add_subplot(gs[2, c])
        for m in modes:
            sel = [r for r in runs if r["traj"] == n and r["mode"] == m]
            for i, r in enumerate(sel):
                lab = f"{MODE_LABEL[m]} (n={len(sel)})" if i == 0 else None
                ax.plot(r["ee"][:, ia] * 1e3, r["ee"][:, ib] * 1e3, color=MODE_COLOR[m], lw=1.2, alpha=0.7, label=lab)
                ax_e.plot(r["progress"] * 100, r["dist"] * 1e3, color=MODE_COLOR[m], lw=1, alpha=0.7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(n, fontsize=10)
        ax.set_xlabel(f"{traj.plane[0]} (mm)", fontsize=8)
        ax.set_ylabel(f"{traj.plane[1]} (mm)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=7, loc="best")
        ax_e.set_xlim(0, 100)
        ax_e.set_ylim(bottom=0)
        ax_e.set_xlabel("progress along path (%)", fontsize=8)
        if c == 0:
            ax_e.set_ylabel("distance to path (mm)", fontsize=8)
        ax_e.tick_params(labelsize=8)
        ax_e.grid(alpha=0.3)

    fig.suptitle(f"SO-101 end-effector tracking: leader teleop vs keyboard IK  ({len(runs)} runs, pace={pace})", fontsize=12)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=120)
        print("저장:", save)
    else:
        plt.show()


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="궤적을 따라가며 기록")
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--traj", nargs="+", choices=list(TRAJS), default=list(TRAJS), help="할 궤적 (기본: 전부)")
    r.add_argument("--repeat", type=int, default=1, help="궤적마다 반복 횟수")
    r.add_argument("--fps", type=int, default=20, help="제어/기록 주기 Hz (리더암이 안정적인 20 Hz 가 기본)")
    r.add_argument("--pace", choices=("free", "timed"), default="free",
                   help="free: 자기 속도로 경로를 끝까지 그리면 끝 / timed: 시간에 맞춰 움직이는 점을 쫓는다")
    r.add_argument("--slow", type=float, default=1.0, help="timed 에서 궤적 시간을 이 배수만큼 늘린다")
    r.add_argument("--speed", type=float, default=0.06, help="키보드 커서 속도 (m/s)")
    r.add_argument("--port", default=default_port("so101_onearm"), help="리더암 포트")
    r.add_argument("--calib-id", default="leader",
                   help="캘리브레이션 파일 이름 (~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json)")

    p = sub.add_parser("plot", help="기록을 모아 한 장의 그래프로")
    p.add_argument("--dir", type=Path, default=RESULTS, help="기록 폴더")
    p.add_argument("--pace", choices=("free", "timed"), default="free", help="어느 방식의 기록을 그릴지")
    p.add_argument("--save", metavar="PNG", help="창 대신 이미지로 저장")

    sub.add_parser("list", help="궤적 목록")
    args = ap.parse_args()

    if args.cmd == "list":
        for t in TRAJS.values():
            print(f"{t.name:<12} length {t.length * 1e3:5.0f} mm  {t.duration:4.1f} s  "
                  f"({t.length / t.duration * 1e3:4.0f} mm/s)  plane {t.plane}")
    elif args.cmd == "run":
        Runner(args).run()
        print("끝. 그래프: .venv/bin/python src/teleop/traj_compare.py plot")
    else:
        runs = load_runs(args.dir, args.pace)
        if not runs:
            raise SystemExit(f"pace={args.pace} 기록이 없습니다: {args.dir}")
        plot_summary(runs, args.pace, args.save)


if __name__ == "__main__":
    main()
