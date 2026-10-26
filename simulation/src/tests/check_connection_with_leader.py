"""리더암과 MuJoCo 의 SO-101 이 제대로 이어졌는지 빈 월드에서 확인한다.

    .venv/bin/python src/tests/check_connection_with_leader.py
    .venv/bin/python src/tests/check_connection_with_leader.py --duration 30 --save check.png

mjpython 이 아니라 일반 python 으로 실행한다. MuJoCo 뷰어 창을 따로 띄우지 않고
오프스크린으로 렌더한 화면을 그래프 창 안에 같이 그리기 때문이다.

창 구성
    위   : 관절 6개. 리더암 정규화값(leader) 과, MuJoCo 로 보낸 뒤 시뮬레이션 관절 각도를
           다시 정규화한 값(sim) 을 겹쳐 그린다.
    오른쪽: 시뮬레이션 화면과 관절별 차이(sim - leader) 의 RMSE / 평균 / 분산 / 최대.
    아래 : 통신 지연 (리더암 한 번 읽는 데 걸린 시간) 과 실제 루프 주기.

차이가 크면 캘리브레이션을 다시 하거나 매핑 코드를 고쳐야 한다.
"""

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from teleop.constants import JOINT_NAMES
from teleop.leader_mj import SO101LeaderMj, SO101LeaderMjConfig
from teleop.mapping import rad_to_normalized

ROOT = Path(__file__).resolve().parents[2]
EMPTY_SCENE = ROOT / "models" / "scene" / "scene_empty.xml"


class Stats:
    """관절별 차이(sim - leader) 의 누적 통계."""

    def __init__(self, n: int):
        self.count = 0
        self.total = np.zeros(n)
        self.total_sq = np.zeros(n)
        self.max_abs = np.zeros(n)

    def add(self, diff: np.ndarray) -> None:
        self.count += 1
        self.total += diff
        self.total_sq += diff**2
        self.max_abs = np.maximum(self.max_abs, np.abs(diff))

    def summary(self) -> dict[str, np.ndarray]:
        n = max(self.count, 1)
        mean = self.total / n
        mean_sq = self.total_sq / n
        return {
            "rmse": np.sqrt(mean_sq),
            "mean": mean,
            "var": np.maximum(mean_sq - mean**2, 0.0),
            "max": self.max_abs,
        }


class Session:
    """리더암을 읽어 시뮬레이션을 움직이고 기록을 쌓는 백그라운드 루프."""

    def __init__(self, cfg: SO101LeaderMjConfig, window_s: float):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.data = mujoco.MjData(self.model)
        self.act_ids = [self.model.actuator(cfg.joint_prefix + j).id for j in JOINT_NAMES]
        self.qpos_adr = [int(self.model.joint(cfg.joint_prefix + j).qposadr[0]) for j in JOINT_NAMES]
        self.steps_per_tick = max(1, round(1.0 / cfg.fps / self.model.opt.timestep))

        self.teleop = SO101LeaderMj(cfg)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error: Exception | None = None

        maxlen = int(window_s * cfg.fps)
        self.t = deque(maxlen=maxlen)
        self.leader = deque(maxlen=maxlen)
        self.sim = deque(maxlen=maxlen)
        self.read_ms = deque(maxlen=maxlen)
        self.period_ms = deque(maxlen=maxlen)
        self.stats = Stats(len(JOINT_NAMES))
        self.read_ms_total = 0.0
        self.read_ms_max = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> tuple[np.ndarray, np.ndarray, float]:
        """리더암을 한 번 읽는다 -> (rad 목표, 정규화값, 걸린 시간 ms)."""
        start = time.perf_counter()
        action = self.teleop.get_action()
        ms = (time.perf_counter() - start) * 1e3
        target = np.array([action[f"{j}.pos"] for j in JOINT_NAMES])
        norm = np.array([self.teleop.last_normalized[j] for j in JOINT_NAMES])
        return target, norm, ms

    def start(self) -> None:
        # calibrate=False: 캘리브레이션이 안 맞아도 백그라운드에서 입력을 기다리며 멈추지 않게
        self.teleop.connect(calibrate=False)
        if not self.teleop.is_calibrated:
            print("경고: 모터에 저장된 캘리브레이션이 파일과 다릅니다. lerobot-calibrate 를 다시 실행하세요.")
        # 시작할 때 시뮬레이션 팔을 리더 자세에 맞춰 놓는다 (첫 프레임에 튀지 않게)
        target, _, _ = self._read()
        self.data.qpos[self.qpos_adr] = target
        self.data.ctrl[self.act_ids] = target
        mujoco.mj_forward(self.model, self.data)
        self.thread.start()

    def _run(self) -> None:
        dt = 1.0 / self.cfg.fps
        t0 = last = deadline = time.perf_counter()
        try:
            while not self.stop.is_set():
                tick = time.perf_counter()
                target, norm, ms = self._read()
                with self.lock:
                    self.data.ctrl[self.act_ids] = target
                    for _ in range(self.steps_per_tick):
                        mujoco.mj_step(self.model, self.data)
                    sim = np.array([
                        rad_to_normalized(j, self.data.qpos[adr], self.cfg.gripper_closed_safe)
                        for j, adr in zip(JOINT_NAMES, self.qpos_adr)
                    ])
                    self.t.append(tick - t0)
                    self.leader.append(norm)
                    self.sim.append(sim)
                    self.read_ms.append(ms)
                    self.period_ms.append((tick - last) * 1e3)
                    self.stats.add(sim - norm)
                    self.read_ms_total += ms
                    self.read_ms_max = max(self.read_ms_max, ms)
                last = tick
                # 절대 시각 기준으로 잰다: sleep 이 조금씩 늦어도 평균 주기는 fps 를 유지한다
                deadline += dt
                rest = deadline - time.perf_counter()
                if rest > 0:
                    time.sleep(rest)
                else:
                    deadline = time.perf_counter()
        except Exception as e:  # 시리얼 오류 등: 메인 루프가 보고 끝내도록 넘긴다
            self.error = e
            self.stop.set()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "t": np.array(self.t),
                "leader": np.array(self.leader),
                "sim": np.array(self.sim),
                "read_ms": np.array(self.read_ms),
                "period_ms": np.array(self.period_ms),
                "stats": self.stats.summary(),
                "count": self.stats.count,
                "read_mean": self.read_ms_total / max(self.stats.count, 1),
                "read_max": self.read_ms_max,
                "qpos": self.data.qpos.copy(),
            }

    def close(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.teleop.is_connected:
            self.teleop.disconnect()


def stats_text(s: dict) -> str:
    st = s["stats"]
    lines = ["sim - leader (normalized units)", "", f"{'joint':<14}{'RMSE':>7}{'mean':>8}{'var':>8}{'max':>7}"]
    for i, j in enumerate(JOINT_NAMES):
        lines.append(f"{j:<14}{st['rmse'][i]:7.2f}{st['mean'][i]:8.2f}{st['var'][i]:8.2f}{st['max'][i]:7.2f}")
    lines.append("")
    lines.append(f"samples: {s['count']}")
    return "\n".join(lines)


def latency_text(s: dict, fps: int) -> str:
    if not len(s["read_ms"]):
        return ""
    hz = 1e3 / np.mean(s["period_ms"][1:]) if len(s["period_ms"]) > 1 else float("nan")
    return (
        f"leader read: now {s['read_ms'][-1]:.1f} ms | mean {s['read_mean']:.1f} ms | max {s['read_max']:.1f} ms"
        f"      loop: {hz:.1f} Hz (target {fps} Hz)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="리더암 포트 (기본: config 의 고정 이름)")
    ap.add_argument("--duration", type=float, help="이 시간(초)이 지나면 끝낸다 (기본: 창을 닫을 때까지)")
    ap.add_argument("--window", type=float, default=10.0, help="그래프에 보이는 최근 구간 (초)")
    ap.add_argument("--save", metavar="PNG", help="끝날 때 창 내용을 이미지로 저장")
    args = ap.parse_args()

    cfg = SO101LeaderMjConfig(xml_path=EMPTY_SCENE)
    if args.port:
        cfg.port = args.port
    session = Session(cfg, args.window)

    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title("check_connection_with_leader")
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.6], width_ratios=[1, 1, 1, 1.25])
    lines = []
    for i, joint in enumerate(JOINT_NAMES):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        (l_leader,) = ax.plot([], [], color="tab:blue", lw=1.8, label="leader")
        (l_sim,) = ax.plot([], [], color="tab:orange", lw=1.2, ls="--", label="sim")
        ax.set_title(joint, fontsize=10)
        ax.set_ylim((-5, 105) if joint == "gripper" else (-105, 105))
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        lines.append((ax, l_leader, l_sim))
    ax_img = fig.add_subplot(gs[0, 3])
    ax_img.axis("off")
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.axis("off")
    txt_stats = ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.5)
    ax_lat = fig.add_subplot(gs[2, :])
    (l_read,) = ax_lat.plot([], [], color="tab:red", lw=1.2, label="leader read")
    (l_period,) = ax_lat.plot([], [], color="tab:gray", lw=1.0, label="loop period")
    ax_lat.set_ylabel("ms")
    ax_lat.set_xlabel("time (s)")
    ax_lat.grid(alpha=0.3)
    ax_lat.legend(loc="upper right", fontsize=8)
    txt_lat = ax_lat.set_title("", fontsize=10, loc="left")
    fig.tight_layout()

    renderer = mujoco.Renderer(session.model, 360, 480)
    render_data = mujoco.MjData(session.model)
    img = ax_img.imshow(np.zeros((360, 480, 3), dtype=np.uint8))

    session.start()
    print(f"연결됨: {cfg.port}  (창을 닫거나 Ctrl+C 로 종료)")
    plt.show(block=False)
    started = time.perf_counter()
    s = session.snapshot()
    try:
        while plt.fignum_exists(fig.number) and not session.stop.is_set():
            if args.duration and time.perf_counter() - started >= args.duration:
                break
            s = session.snapshot()
            if len(s["t"]):
                t = s["t"]
                lo, hi = max(0.0, t[-1] - args.window), max(t[-1], args.window)
                for i, (ax, l_leader, l_sim) in enumerate(lines):
                    l_leader.set_data(t, s["leader"][:, i])
                    l_sim.set_data(t, s["sim"][:, i])
                    ax.set_xlim(lo, hi)
                l_read.set_data(t, s["read_ms"])
                l_period.set_data(t, s["period_ms"])
                ax_lat.set_xlim(lo, hi)
                ax_lat.set_ylim(0, max(1.5e3 / cfg.fps, float(np.max(s["period_ms"])) * 1.1))
                txt_stats.set_text(stats_text(s))
                txt_lat.set_text(latency_text(s, cfg.fps))
                render_data.qpos[:] = s["qpos"]
                mujoco.mj_forward(session.model, render_data)
                renderer.update_scene(render_data, camera="view")
                img.set_data(renderer.render())
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        renderer.close()

    if session.error:
        print(f"오류로 중단: {type(session.error).__name__}: {session.error}")
    print()
    print(stats_text(s))
    print(latency_text(s, cfg.fps))
    if args.save and plt.fignum_exists(fig.number):
        fig.savefig(args.save, dpi=110)
        print("저장:", args.save)


if __name__ == "__main__":
    main()
