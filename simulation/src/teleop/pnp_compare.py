"""빈 월드에서 큐브 pick & place 를 리더암 텔레옵과 키보드 IK 로 각각 해 보고 비교한다.

    # 1) 시행 (seed 가 같으면 큐브/목표 배치가 같다 -> 두 모드를 같은 배치로 비교)
    .venv/bin/python src/teleop/pnp_compare.py run --mode teleop
    .venv/bin/python src/teleop/pnp_compare.py run --mode keyboard --trials 10 --seed 0

    # 2) 모은 기록을 한 장의 그래프로
    .venv/bin/python src/teleop/pnp_compare.py plot --save results/pnp_compare.png

시행 하나
    빨간 큐브와 초록 원(목표)이 무작위로 놓인다. 2 초 카운트다운 뒤 타이머 시작.
    큐브를 초록 원 안에 내려놓고 그리퍼를 떼서 0.5 초 지나면 성공 -> 다음 시행.
    --timeout 초 안에 못 하거나 enter 를 누르면 실패로 기록.
    결과는 results/pnp_compare/<모드>__seed<배치번호>__<시각>.npz

키 (창을 한 번 클릭해 포커스를 줄 것)
    enter 포기(실패로 기록) · r 지금 시행 다시 · esc 끝내기 · [ ] 카메라 회전
    키보드 모드: w/s 앞뒤(x) · a/d 좌우(y) · q/e 위아래(z) · z/x 손목 회전 · space 그리퍼 열기/닫기
                그리퍼는 항상 아래를 보도록 IK 가 자세를 잡는다.

비교 지표
    success rate, time to place (성공한 시행만), 끝점 이동 거리, RMS jerk (움직임이 얼마나 거친지),
    lifts (큐브를 들어 올린 횟수. 1 이면 한 번에 성공)
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from teleop.constants import JOINT_NAMES, MUJOCO_JOINT_RANGE, default_port
from teleop.traj_compare import (KeyboardIK, LeaderInput, MODE_COLOR, MODE_LABEL, MODES, PREFIX, REST_Q,
                                 add_polyline, add_sphere, hex_rgba)

ROOT = Path(__file__).resolve().parents[2]
PNP_SCENE = ROOT / "models" / "scene" / "scene_pnp.xml"
RESULTS = ROOT / "results" / "pnp_compare"

CUBE_HALF = 0.0125
TARGET_R = 0.03  # 큐브 중심이 목표 중심에서 이 거리 안이면 들어간 것 (scene_pnp.xml 의 target 반지름)
LIFT_Z = 0.025  # 큐브 중심이 이보다 높으면 들려 있는 것
SUCCESS_HOLD_S = 0.5
COUNTDOWN_S = 2.0

# 큐브와 목표를 놓는 범위 (로봇 앞, 그리퍼를 세워서 닿는 곳)
SPAWN_X = (0.16, 0.28)
SPAWN_Y = (-0.12, 0.12)
MIN_GAP = 0.10  # 큐브와 목표 사이 최소 거리

GRIP_OPEN = 1.2
GRIP_CLOSED = -0.1  # SO101LeaderMjConfig.gripper_closed_safe 와 같게
HOME = np.array([0.20, 0.0, 0.10])


def layout(seed: int) -> tuple[np.ndarray, float, np.ndarray]:
    """배치 번호 -> (큐브 xy, 큐브 yaw, 목표 xy). 모드와 상관없이 같은 번호면 같은 배치."""
    rng = np.random.default_rng(seed)
    while True:
        cube = rng.uniform([SPAWN_X[0], SPAWN_Y[0]], [SPAWN_X[1], SPAWN_Y[1]])
        target = rng.uniform([SPAWN_X[0], SPAWN_Y[0]], [SPAWN_X[1], SPAWN_Y[1]])
        if np.linalg.norm(cube - target) >= MIN_GAP:
            return cube, float(rng.uniform(-0.6, 0.6)), target


# ---------------------------------------------------------------- 키보드 IK (그리퍼 아래 방향 고정)


class DownIK:
    """끝점 위치 + "그리퍼가 아래를 본다" 를 같이 맞추는 감쇠 최소제곱 IK.

    pan/lift/elbow/wrist_flex 4 개로 위치 3 + 기울기 1 을 맞추고, wrist_roll 은 키로 따로 정한다.
    그리퍼 바디의 +z 가 손목 쪽을 향하므로 (끝이 -z) 그 축을 월드 +z 에 맞춘다.
    """

    def __init__(self, model: mujoco.MjModel, rot_weight: float = 0.05, damping: float = 0.01):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(PREFIX + "gripperframe").id
        self.body = model.body(PREFIX + "gripper").id
        self.qadr = [int(model.joint(PREFIX + j).qposadr[0]) for j in JOINT_NAMES]
        self.dofs = [int(model.joint(PREFIX + j).dofadr[0]) for j in JOINT_NAMES[:4]]
        self.lo = np.array([MUJOCO_JOINT_RANGE[j][0] for j in JOINT_NAMES[:4]])
        self.hi = np.array([MUJOCO_JOINT_RANGE[j][1] for j in JOINT_NAMES[:4]])
        self.w = rot_weight
        self.damping = damping
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def solve(self, q0: np.ndarray, target: np.ndarray, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """q0(관절 6개) 에서 출발 -> (관절, 실제로 닿은 끝점). q0 의 wrist_roll/gripper 는 그대로 둔다."""
        q = q0.copy()
        for _ in range(iters):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            ep = target - self.data.site_xpos[self.site]
            er = np.cross(self.data.xmat[self.body].reshape(3, 3)[:, 2], [0.0, 0.0, 1.0])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jacp, None, self.site)
            mujoco.mj_jacBody(self.model, self.data, None, self.jacr, self.body)
            J = np.vstack([self.jacp[:, self.dofs], self.w * self.jacr[:, self.dofs]])
            e = np.concatenate([ep, self.w * er])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), e)
            q[:4] = np.clip(q[:4] + dq, self.lo, self.hi)
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return q, self.data.site_xpos[self.site].copy()


class KeyboardPnP:
    """w/s a/d q/e 로 끝점, z/x 로 손목 회전, space 로 그리퍼."""

    MOVE = KeyboardIK.KEYS
    ROLL = {"z": +1, "x": -1}
    ROLL_LO, ROLL_HI = MUJOCO_JOINT_RANGE["wrist_roll"]

    def __init__(self, model: mujoco.MjModel, speed: float, roll_speed: float = 1.5):
        self.ik = DownIK(model)
        self.speed = speed
        self.roll_speed = roll_speed
        self.held: set[str] = set()
        self.released_at: dict[str, float] = {}
        self.reset()

    def reset(self) -> np.ndarray:
        q0 = REST_Q.copy()
        q0[4], q0[5] = 0.0, GRIP_OPEN
        self.q, self.cursor = self.ik.solve(q0, HOME, iters=200)
        self.grip_open = True
        return self.q

    def press(self, key: str) -> None:
        if key == " ":
            # 키를 누르고 있으면 OS 가 release/press 를 반복해 보낸다. 방금 뗀 게 아닐 때만 토글
            if time.perf_counter() - self.released_at.get(key, 0.0) > 0.03:
                self.grip_open = not self.grip_open
        self.held.add(key)

    def release(self, key: str) -> None:
        self.held.discard(key)
        self.released_at[key] = time.perf_counter()

    def step(self, dt: float) -> np.ndarray:
        vel = np.zeros(3)
        for k in self.held & self.MOVE.keys():
            axis, sign = self.MOVE[k]
            vel[axis] += sign
        if vel.any():
            self.cursor = self.cursor + self.speed * dt * vel / np.linalg.norm(vel)
            self.cursor[2] = max(self.cursor[2], 0.008)  # 바닥을 파고들지 않게
            self.q, reached = self.ik.solve(self.q, self.cursor)
            if np.linalg.norm(reached - self.cursor) > 0.005:
                self.cursor = reached
        roll = sum(self.ROLL[k] for k in self.held & self.ROLL.keys())
        self.q[4] = np.clip(self.q[4] + roll * self.roll_speed * dt, self.ROLL_LO, self.ROLL_HI)
        self.q[5] = GRIP_OPEN if self.grip_open else GRIP_CLOSED
        return self.q


# ---------------------------------------------------------------- 시행


class Runner:
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.model = mujoco.MjModel.from_xml_path(str(PNP_SCENE))
        self.data = mujoco.MjData(self.model)
        self.act_ids = [self.model.actuator(PREFIX + j).id for j in JOINT_NAMES]
        self.qadr = [int(self.model.joint(PREFIX + j).qposadr[0]) for j in JOINT_NAMES]
        self.site = self.model.site(PREFIX + "gripperframe").id
        self.cube_body = self.model.body("cube").id
        self.cube_qadr = int(self.model.joint("cube").qposadr[0])
        self.cube_dof = int(self.model.joint("cube").dofadr[0])
        self.target_mocap = int(self.model.body("target").mocapid[0])
        self.cube_geom = self.model.geom("cube").id
        self.grip_bodies = {self.model.body(PREFIX + b).id for b in ("gripper", "moving_jaw_so101_v1")}
        self.dt = 1.0 / args.fps
        self.steps_per_tick = max(1, round(self.dt / self.model.opt.timestep))

        self.keyboard = KeyboardPnP(self.model, args.speed) if self.mode == "keyboard" else None
        self.leader = LeaderInput(args.port, args.calib_id) if self.mode == "teleop" else None
        self.quit = False
        self.azimuth = -45.0
        self.results: list[dict] = []
        self._build_figure()

    # ---- 상태
    def reset_trial(self, i: int) -> None:
        self.i = i
        self.seed = self.args.seed + i
        cube_xy, yaw, target_xy = layout(self.seed)
        self.target = target_xy
        mujoco.mj_resetData(self.model, self.data)
        q = self.keyboard.reset() if self.keyboard is not None else self.leader.read()
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.act_ids] = q
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = [cube_xy[0], cube_xy[1], CUBE_HALF]
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.mocap_pos[self.target_mocap] = [target_xy[0], target_xy[1], 0.0005]
        mujoco.mj_forward(self.model, self.data)
        self.phase, self.phase_t, self.hold_t = "countdown", 0.0, 0.0
        self.log = {k: [] for k in ("t", "ee", "cube", "q", "ctrl")}

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def gripper_touching_cube(self) -> bool:
        for c in self.data.contact[:self.data.ncon]:
            if self.cube_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.cube_geom else c.geom1
                if self.model.geom_bodyid[other] in self.grip_bodies:
                    return True
        return False

    def placed(self) -> bool:
        """큐브가 목표 안 바닥에 놓여 있고 그리퍼가 떨어져 있다."""
        c = self.cube_pos()
        still = np.linalg.norm(self.data.qvel[self.cube_dof:self.cube_dof + 3]) < 0.02
        return (np.linalg.norm(c[:2] - self.target) < TARGET_R and c[2] < CUBE_HALF + 0.004
                and still and not self.gripper_touching_cube())

    def on_key(self, event, pressed: bool) -> None:
        key = (event.key or "").lower()
        if self.keyboard is not None:
            (self.keyboard.press if pressed else self.keyboard.release)(key)
        if not pressed:
            return
        if key == "enter" and self.phase == "run":
            self.finish(success=False)
        elif key == "r":
            self.reset_trial(self.i)
        elif key == "escape":
            self.quit = True
        elif key in ("[", "]"):
            self.azimuth += 15 if key == "]" else -15

    def tick(self) -> None:
        target = self.keyboard.step(self.dt) if self.keyboard is not None else self.leader.read()
        self.data.ctrl[self.act_ids] = target
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        self.phase_t += self.dt

        if self.phase == "countdown":
            if self.phase_t >= COUNTDOWN_S:
                self.phase, self.phase_t = "run", 0.0
        elif self.phase == "run":
            self.log["t"].append(self.phase_t)
            self.log["ee"].append(self.data.site_xpos[self.site].copy())
            self.log["cube"].append(self.cube_pos())
            self.log["q"].append(self.data.qpos[self.qadr].copy())
            self.log["ctrl"].append(np.array(target))
            self.hold_t = self.hold_t + self.dt if self.placed() else 0.0
            if self.hold_t >= SUCCESS_HOLD_S:
                self.finish(success=True)
            elif self.phase_t >= self.args.timeout:
                self.finish(success=False)

    def finish(self, success: bool) -> None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        rec = {k: np.array(v) for k, v in self.log.items()}
        rec.update(mode=self.mode, seed=self.seed, success=success, fps=self.args.fps,
                   time_s=float(rec["t"][-1] - (self.hold_t if success else 0.0)) if len(rec["t"]) else 0.0)
        m = trial_metrics(rec)
        self.results.append(m)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = RESULTS / f"{self.mode}__seed{self.seed}__{stamp}.npz"
        np.savez(path, **rec)
        print(f"[{self.mode:<8} seed {self.seed:3d}] {'성공' if success else '실패'} | {m['time_s']:5.1f} s | "
              f"path {m['path_m'] * 100:5.1f} cm | jerk {m['jerk_rms']:6.1f} m/s^3 | lifts {m['lifts']}  -> {path.name}")
        if self.i + 1 >= self.args.trials:
            self.quit = True
        else:
            self.reset_trial(self.i + 1)

    # ---- 화면
    def _build_figure(self) -> None:
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []
        self.fig = plt.figure(figsize=(14, 7.5))
        self.fig.canvas.manager.set_window_title("pnp_compare")
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1.3, 1])
        self.ax_img = self.fig.add_subplot(gs[:, 0])
        self.ax_img.axis("off")
        self.img = self.ax_img.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.txt = self.ax_img.set_title("", loc="left", fontsize=11, family="monospace")

        ax = self.ax_map = self.fig.add_subplot(gs[0, 1])
        ax.set_xlim(SPAWN_X[0] - 0.06, SPAWN_X[1] + 0.06)
        ax.set_ylim(SPAWN_Y[0] - 0.05, SPAWN_Y[1] + 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.3)
        ax.set_title("top view", fontsize=10)
        self.c_target = plt.Circle((0, 0), TARGET_R, color="#1baf7a", alpha=0.35)
        ax.add_patch(self.c_target)
        (self.l_ee,) = ax.plot([], [], color=MODE_COLOR[self.mode], lw=1.5, label=MODE_LABEL[self.mode])
        (self.l_cube,) = ax.plot([], [], "s", color="#e34948", ms=8, label="cube")
        ax.legend(loc="upper right", fontsize=8)

        self.ax_txt = self.fig.add_subplot(gs[1, 1])
        self.ax_txt.axis("off")
        self.stats = self.ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10)

        self.fig.canvas.mpl_connect("key_press_event", lambda e: self.on_key(e, True))
        self.fig.canvas.mpl_connect("key_release_event", lambda e: self.on_key(e, False))
        self.renderer = mujoco.Renderer(self.model, 480, 640, max_geom=2000)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.22, 0.0, 0.03]
        self.cam.distance = 0.7
        self.cam.elevation = -40

    def draw(self) -> None:
        self.cam.azimuth = self.azimuth
        self.renderer.update_scene(self.data, self.cam)
        scn = self.renderer.scene
        if len(self.log["ee"]) > 1:
            add_polyline(scn, np.array(self.log["ee"][::3]), 0.002, hex_rgba(MODE_COLOR[self.mode], 0.7))
        if self.keyboard is not None:
            add_sphere(scn, self.keyboard.cursor, 0.005, [0.95, 0.8, 0.1, 0.8])
        self.img.set_data(self.renderer.render())

        head = f"[{MODE_LABEL[self.mode]}] trial {self.i + 1}/{self.args.trials} (seed {self.seed})  "
        if self.phase == "countdown":
            head += f"get ready {COUNTDOWN_S - self.phase_t:3.1f}"
        else:
            head += f"{self.phase_t:5.1f} / {self.args.timeout:.0f} s"
            if self.hold_t > 0:
                head += "  placed!"
        if self.keyboard is not None:
            head += f"  grip {'OPEN' if self.keyboard.grip_open else 'CLOSED'}"
        self.txt.set_text(head)

        self.c_target.center = tuple(self.target)
        c = self.cube_pos()
        self.l_cube.set_data([c[0]], [c[1]])
        if self.log["ee"]:
            e = np.array(self.log["ee"])
            self.l_ee.set_data(e[:, 0], e[:, 1])
        else:
            self.l_ee.set_data([], [])

        lines = ["so far", ""]
        if self.results:
            ok = [r for r in self.results if r["success"]]
            lines.append(f"success   {len(ok)}/{len(self.results)}")
            if ok:
                lines.append(f"time      {np.mean([r['time_s'] for r in ok]):5.1f} s (mean, success)")
        lines += ["", "enter: give up  r: redo  esc: quit"]
        if self.keyboard is not None:
            lines.append("w/s a/d q/e move  z/x roll  space grip")
        self.stats.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        self.reset_trial(0)
        self.fig.tight_layout()
        plt.show(block=False)
        print("창을 클릭해 포커스를 주고 시작하세요. (enter 포기 · r 다시 · esc 끝)")
        deadline = time.perf_counter()
        try:
            while plt.fignum_exists(self.fig.number) and not self.quit:
                self.tick()
                if self.quit:
                    break
                self.draw()
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


# ---------------------------------------------------------------- 지표 / 결과 그래프


def trial_metrics(rec: dict) -> dict:
    ee, cube = rec["ee"], rec["cube"]
    dt = 1.0 / float(rec["fps"])
    m = {"mode": str(rec["mode"]), "seed": int(rec["seed"]), "success": bool(rec["success"]),
         "time_s": float(rec["time_s"]), "path_m": 0.0, "jerk_rms": 0.0, "lifts": 0}
    if len(ee) > 4:
        m["path_m"] = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
        jerk = np.diff(ee, n=3, axis=0) / dt**3
        m["jerk_rms"] = float(np.sqrt(np.mean(np.sum(jerk**2, axis=1))))
    if len(cube):
        up = cube[:, 2] > LIFT_Z
        m["lifts"] = int(up[0]) + int(np.sum(up[1:] & ~up[:-1]))
    return m


def load_trials(folder: Path) -> list[dict]:
    out = []
    for f in sorted(folder.glob("*.npz")):
        z = np.load(f)
        out.append(trial_metrics({k: z[k] for k in z.files}))
    return out


def plot_summary(trials: list[dict], save: str | None) -> None:
    modes = [m for m in MODES if any(t["mode"] == m for t in trials)]
    by = {m: [t for t in trials if t["mode"] == m] for m in modes}

    print(f"\n{'mode':<10}{'n':>4}{'success':>10}{'time s':>14}{'path cm':>14}{'jerk':>14}{'lifts':>10}")
    for m in modes:
        ts = by[m]
        ok = [t for t in ts if t["success"]]
        f = lambda k, s=1.0, src=ts: f"{np.mean([t[k] for t in src]) * s:8.1f} ±{np.std([t[k] for t in src]) * s:4.1f}" if src else f"{'-':>14}"
        print(f"{m:<10}{len(ts):>4}{len(ok) / len(ts) * 100:9.0f}%{f('time_s', src=ok)}{f('path_m', 100, ok)}"
              f"{f('jerk_rms', src=ok)}{np.mean([t['lifts'] for t in ts]):10.2f}")

    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title("pnp_compare summary")
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.1])
    panels = [
        ("success rate", "%", lambda ts: [100.0 * np.mean([t["success"] for t in ts])], False),
        ("time to place (success)", "s", lambda ts: [t["time_s"] for t in ts if t["success"]], True),
        ("end-effector path (success)", "cm", lambda ts: [t["path_m"] * 100 for t in ts if t["success"]], True),
        ("RMS jerk (success)", "m/s³", lambda ts: [t["jerk_rms"] for t in ts if t["success"]], True),
        ("lifts per trial", "count", lambda ts: [t["lifts"] for t in ts], True),
    ]
    for k, (title, unit, get, spread) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        for j, m in enumerate(modes):
            v = get(by[m])
            if not v:
                continue
            mean = float(np.mean(v))
            ax.bar(j, mean, 0.7, yerr=np.std(v) if spread and len(v) > 1 else None, color=MODE_COLOR[m],
                   error_kw={"ecolor": "#333333", "lw": 1, "capsize": 3})
            ax.text(j, mean, f"{mean:.1f}", ha="center", va="bottom", fontsize=9, color="#333333")
        ax.set_xticks(range(len(modes)), [MODE_LABEL[m] for m in modes], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 110 if k == 0 else None)

    # 같은 배치끼리 시행별 시간 비교 (실패는 x)
    for k, (key, title, unit, scale) in enumerate([("time_s", "time per trial (same seed = same layout)", "s", 1.0),
                                                  ("path_m", "end-effector path per trial", "cm", 100.0)]):
        ax = fig.add_subplot(gs[1, 0:2] if k == 0 else gs[1, 2:5])
        for m in modes:
            ts = sorted(by[m], key=lambda t: t["seed"])
            seeds = [t["seed"] for t in ts]
            vals = [t[key] * scale for t in ts]
            ax.plot(seeds, vals, color=MODE_COLOR[m], lw=1.2, alpha=0.6)
            for s, v, t in zip(seeds, vals, ts):
                ax.plot(s, v, "o" if t["success"] else "x", color=MODE_COLOR[m], ms=7, mew=2)
            ax.plot([], [], "o-", color=MODE_COLOR[m], label=MODE_LABEL[m])
        ax.plot([], [], "x", color="#555555", mew=2, label="failed")
        ax.set_xlabel("layout seed")
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f"SO-101 cube pick & place: leader teleop vs keyboard IK  ({len(trials)} trials)", fontsize=12)
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

    r = sub.add_parser("run", help="pick & place 시행")
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--trials", type=int, default=10, help="시행 횟수")
    r.add_argument("--seed", type=int, default=0, help="첫 배치 번호 (시행 i 는 seed+i)")
    r.add_argument("--timeout", type=float, default=60.0, help="시행 하나 제한 시간 (s)")
    r.add_argument("--fps", type=int, default=20, help="제어/기록 주기 Hz")
    r.add_argument("--speed", type=float, default=0.08, help="키보드 끝점 속도 (m/s)")
    r.add_argument("--port", default=default_port("so101_onearm"), help="리더암 포트")
    r.add_argument("--calib-id", default="leader", help="리더암 캘리브레이션 파일 이름")

    p = sub.add_parser("plot", help="기록을 모아 한 장의 그래프로")
    p.add_argument("--dir", type=Path, default=RESULTS, help="기록 폴더")
    p.add_argument("--save", metavar="PNG", help="창 대신 이미지로 저장")
    args = ap.parse_args()

    if args.cmd == "run":
        Runner(args).run()
        print("끝. 그래프: .venv/bin/python src/teleop/pnp_compare.py plot")
    else:
        trials = load_trials(args.dir)
        if not trials:
            raise SystemExit(f"기록이 없습니다: {args.dir}")
        plot_summary(trials, args.save)


if __name__ == "__main__":
    main()
