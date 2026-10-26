"""리더암으로 트롤리 씬을 조작하면서 LeRobot 형식 데이터셋을 모은다.

    .venv/bin/python src/data_mj.py --task "컵을 약두는곳 옆으로 옮긴다"

mjpython 이 아니라 일반 python 으로 실행한다 (MuJoCo 창 없이 카메라 화면만 그린다).

화면
    위  : 왼쪽 옆에서 바라보는 화면, 서랍을 위에서 내려다보는 화면 (조작자용, 데이터셋에는 넣지 않는다)
    아래: top view (corner 카메라) | wrist view (그리퍼 카메라) | 지금까지 모은 에피소드 목록

키 (창을 클릭한 뒤)
    1  취득 시작. 지금 상태 그대로 새 에피소드를 녹화한다.
    2  취득 정지. 에피소드는 "대기" 상태로 남는다.
    3  환경 초기화. 물체를 models/automatic.json 범위 안에서 흩뜨려 다시 놓는다 (녹화 중에는 무시).
    4  가장 최근 에피소드 삭제 (녹화 중이거나 대기 중인 것).
    q  저장하고 종료.

대기 중인 에피소드는 다음 취득을 시작하거나 종료할 때 디스크에 저장된다.
이미 저장된 에피소드는 4 로 지울 수 없다.

저장되는 값 (프레임마다)
    observation.images.top / wrist : 640x480 RGB
    observation.state : 시뮬레이션 관절 각도를 리더와 같은 정규화 단위로 바꾼 값 (6)
    action            : 리더암 정규화값 (팔 -100~100, 그리퍼 0~100) (6)
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from teleop.constants import JOINT_NAMES
from teleop.leader_mj import LeaderSim, Pacer, SO101LeaderMjConfig

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "models" / "scene" / "scene_trolley.xml"
RANDOMIZE = ROOT / "models" / "automatic.json"

# 데이터셋 이미지 키 -> 씬의 카메라 이름
CAMERAS = {"top": "corner", "wrist": "so101_wrist_cam"}
WIDTH, HEIGHT = 640, 480
# 화면에만 보여주는 카메라 (데이터셋에는 넣지 않는다)
VIEW_CAMERAS = ["left", "drawer_top"]
VIEW_WIDTH, VIEW_HEIGHT = 480, 360


def dataset_features() -> dict:
    names = [f"{j}.pos" for j in JOINT_NAMES]
    features = {
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    for key in CAMERAS:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class Recorder:
    """리더 읽기, 시뮬레이션, 카메라 렌더, 데이터셋 쓰기를 한 스레드에서 돌린다.

    GUI 는 command() 로 명령만 넣고 snapshot() 으로 화면에 그릴 것만 가져간다.
    """

    def __init__(self, cfg: SO101LeaderMjConfig, repo_id: str, root: Path, task: str):
        self.cfg = cfg
        self.task = task
        self.sim = LeaderSim(cfg)
        if (root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4, video_backend="pyav")
            print(f"기존 데이터셋에 이어서 기록: {root} ({self.dataset.meta.total_episodes} 에피소드)")
        else:
            self.dataset = LeRobotDataset.create(
                repo_id, cfg.fps, dataset_features(), root=root, robot_type="so101_mujoco",
                image_writer_threads=4, video_backend="pyav",
            )
            print(f"새 데이터셋: {root}")
        if self.dataset.fps != cfg.fps:
            raise ValueError(f"데이터셋 fps({self.dataset.fps}) 와 설정 fps({cfg.fps}) 가 다릅니다")

        self.existing = self.dataset.meta.total_episodes  # 이번 실행 전에 이미 있던 에피소드 수
        self.episodes: list[dict] = []  # 이번 실행에서 만든 에피소드: {frames, status}
        self.recording = False
        self.message = "1: start   2: stop   3: reset env   4: delete latest   q: save & quit"
        self.hz = 0.0
        self.images = {key: np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for key in CAMERAS}
        self.views = {key: np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8) for key in VIEW_CAMERAS}

        self.commands: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    # ── GUI 쪽에서 부르는 것 ──────────────────────────────────────────

    def start(self) -> None:
        self.sim.connect()
        self.thread.start()

    def command(self, name: str) -> None:
        self.commands.put(name)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "images": dict(self.images),
                "views": dict(self.views),
                "episodes": [dict(e) for e in self.episodes],
                "recording": self.recording,
                "message": self.message,
                "hz": self.hz,
            }

    def close(self) -> None:
        """남은 에피소드를 저장하고 끝낸다."""
        if self.thread.is_alive():
            self.command("quit")
            self.thread.join()
        self.sim.close()

    # ── 작업 스레드 ───────────────────────────────────────────────────

    def _set(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _pending(self) -> dict | None:
        """녹화 중이거나 아직 저장하지 않은 가장 최근 에피소드."""
        if self.episodes and self.episodes[-1]["status"] in ("recording", "pending"):
            return self.episodes[-1]
        return None

    def _commit(self) -> None:
        """대기 중인 에피소드를 디스크에 쓴다 (비디오 인코딩 때문에 몇 초 걸릴 수 있다)."""
        ep = self._pending()
        if ep is None:
            return
        if ep["frames"] == 0:
            self.dataset.clear_episode_buffer()
            with self.lock:
                self.episodes.pop()
            return
        self._set(message=f"saving episode {self.existing + len(self.episodes) - 1} ...")
        self.dataset.save_episode(parallel_encoding=False)
        with self.lock:
            ep["status"] = "saved"

    def _handle(self, name: str) -> bool:
        """명령 하나를 처리한다. 종료해야 하면 False."""
        if name == "start":
            if self.recording:
                return True
            self._commit()
            with self.lock:
                self.episodes.append({"frames": 0, "status": "recording"})
                self.recording = True
                self.message = "recording"
        elif name == "stop":
            if self.recording:
                with self.lock:
                    self.episodes[-1]["status"] = "pending"
                    self.recording = False
                    self.message = "stopped (saved on next start or quit; 4 to delete)"
        elif name == "reset":
            if self.recording:
                self._set(message="recording: press 2 to stop before resetting the environment")
            else:
                self.sim.reset()
                self._set(message="environment reset")
        elif name == "delete":
            if self._pending() is None:
                self._set(message="nothing to delete (saved episodes cannot be deleted here)")
            else:
                self.dataset.clear_episode_buffer()
                with self.lock:
                    self.episodes.pop()
                    self.recording = False
                    self.message = "deleted latest episode"
        elif name == "quit":
            with self.lock:
                if self.recording:
                    self.episodes[-1]["status"] = "pending"
                    self.recording = False
            self._commit()
            return False
        return True

    def _run(self) -> None:
        renderer = view_renderer = None
        try:
            # 렌더러는 쓰는 스레드에서 만들어야 한다 (GL 컨텍스트가 스레드에 묶인다)
            renderer = mujoco.Renderer(self.sim.model, HEIGHT, WIDTH)
            view_renderer = mujoco.Renderer(self.sim.model, VIEW_HEIGHT, VIEW_WIDTH)
            ticks = 0
            pacer = Pacer(self.cfg.fps)
            last = time.perf_counter()
            running = True
            while running:
                while running and not self.commands.empty():
                    running = self._handle(self.commands.get())
                    pacer = Pacer(self.cfg.fps)  # 저장하느라 밀린 시간을 따라잡으려 하지 않게
                if not running:
                    break

                leader, state = self.sim.tick()
                images = {}
                for key, camera in CAMERAS.items():
                    renderer.update_scene(self.sim.data, camera=camera)
                    images[key] = renderer.render().copy()
                # 조작자용 화면은 기록하지 않으니 두 틱에 한 번만 그린다
                ticks += 1
                if ticks % 2 == 0:
                    views = {}
                    for camera in VIEW_CAMERAS:
                        view_renderer.update_scene(self.sim.data, camera=camera)
                        views[camera] = view_renderer.render().copy()
                    with self.lock:
                        self.views = views

                if self.recording:
                    frame = {
                        "action": leader.astype(np.float32),
                        "observation.state": state.astype(np.float32),
                        "task": self.task,
                    }
                    for key, image in images.items():
                        frame[f"observation.images.{key}"] = image
                    self.dataset.add_frame(frame)

                now = time.perf_counter()
                with self.lock:
                    self.images = images
                    if self.recording:
                        self.episodes[-1]["frames"] += 1
                    self.hz = 0.9 * self.hz + 0.1 / max(now - last, 1e-6)
                last = now
                pacer.wait()
        except Exception as e:  # GUI 가 보고 끝낼 수 있게 넘긴다
            self.error = e
        finally:
            try:
                self.dataset.finalize()
            finally:
                for r in (renderer, view_renderer):
                    if r is not None:
                        r.close()
                self.done.set()


def episodes_text(s: dict, existing: int, fps: int) -> str:
    lines = [f"dataset: {existing + sum(e['status'] == 'saved' for e in s['episodes'])} episodes saved", ""]
    if existing:
        lines.append(f"  ep 0-{existing - 1}  (from earlier runs)")
    shown = s["episodes"][-18:]
    offset = existing + len(s["episodes"]) - len(shown)
    for i, ep in enumerate(shown):
        mark = {"recording": "● REC", "pending": "pending", "saved": "saved"}[ep["status"]]
        lines.append(f"  ep {offset + i:<3} {ep['frames']:5d} frames {ep['frames'] / fps:6.1f} s  {mark}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="trolley manipulation", help="에피소드에 붙일 작업 설명")
    ap.add_argument("--repo-id", default="local/so101_trolley", help="데이터셋 이름")
    ap.add_argument("--root", type=Path, help="저장 폴더 (기본: datasets/<repo-id 의 뒷부분>)")
    ap.add_argument("--port", help="리더암 포트 (기본: config 의 고정 이름)")
    ap.add_argument("--no-randomize", action="store_true", help="환경을 초기화할 때 물체를 흩뜨리지 않고 원래 자리에 둔다")
    ap.add_argument("--seed", type=int, help="물체 배치 난수 시드")
    args = ap.parse_args()
    root = args.root or ROOT / "datasets" / args.repo_id.split("/")[-1]

    cfg = SO101LeaderMjConfig(
        xml_path=SCENE,
        randomize_path=None if args.no_randomize else RANDOMIZE,
        randomize_seed=args.seed,
    )
    if args.port:
        cfg.port = args.port
    rec = Recorder(cfg, args.repo_id, root, args.task)

    # 숫자키/q 를 matplotlib 기본 단축키가 가로채지 않게 한다
    for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[key] = []
    # 위: 조작자용 화면 2개 / 아래: top view, wrist view, 에피소드 목록
    fig = plt.figure(figsize=(15, 9.5))
    fig.canvas.manager.set_window_title("data_mj")
    gs = fig.add_gridspec(2, 6)
    shown = {}
    for i, camera in enumerate(VIEW_CAMERAS):
        ax = fig.add_subplot(gs[0, 3 * i:3 * i + 3])
        ax.set_title(camera.replace("_", " "), fontsize=11)
        ax.axis("off")
        shown[camera] = ax.imshow(np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8))
    for i, key in enumerate(CAMERAS):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        ax.set_title(f"{key} view (recorded)", fontsize=11)
        ax.axis("off")
        shown[key] = ax.imshow(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    ax_txt = fig.add_subplot(gs[1, 4:6])
    ax_txt.axis("off")
    txt = ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9)
    status = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    keys = {"1": "start", "2": "stop", "3": "reset", "4": "delete"}
    quit_requested = threading.Event()

    def on_key(event) -> None:
        if event.key in keys:
            rec.command(keys[event.key])
        elif event.key == "q":
            quit_requested.set()

    fig.canvas.mpl_connect("key_press_event", on_key)

    rec.start()
    print(f"연결됨: {cfg.port}  (창에서 1: 시작, 2: 정지, 3: 환경 초기화, 4: 최근 삭제, q: 저장 후 종료)")
    plt.show(block=False)
    try:
        while plt.fignum_exists(fig.number) and not quit_requested.is_set() and not rec.done.is_set():
            s = rec.snapshot()
            for key, image in {**s["images"], **s["views"]}.items():
                shown[key].set_data(image)
            txt.set_text(episodes_text(s, rec.existing, cfg.fps))
            status.set_text(f"{'● REC   ' if s['recording'] else ''}{s['message']}      [{s['hz']:.0f} Hz]")
            status.set_color("tab:red" if s["recording"] else "black")
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print("저장 중...")
        rec.close()

    if rec.error:
        print(f"오류로 중단: {type(rec.error).__name__}: {rec.error}")
    saved = sum(e["status"] == "saved" for e in rec.episodes)
    print(f"이번 실행에서 {saved} 에피소드 저장. 전체 {rec.dataset.meta.total_episodes} 에피소드: {root}")


if __name__ == "__main__":
    main()
