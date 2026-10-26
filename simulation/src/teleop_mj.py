"""리더암으로 트롤리 씬의 SO-101 을 움직인다.

    .venv/bin/mjpython src/teleop_mj.py              # MuJoCo 창 (macOS 는 mjpython 필수)
    .venv/bin/python   src/teleop_mj.py --headless   # 창 없이

종료는 창을 닫거나 Ctrl+C.
"""

import argparse
import sys
import time
from pathlib import Path

from teleop.leader_mj import LeaderSim, Pacer, SO101LeaderMjConfig

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "models" / "scene" / "scene_trolley.xml"
RANDOMIZE = ROOT / "models" / "automatic.json"


def run(sim: LeaderSim, duration: float | None, viewer=None) -> None:
    pacer = Pacer(sim.cfg.fps)
    started = report = time.perf_counter()
    ticks = 0
    while viewer is None or viewer.is_running():
        now = time.perf_counter()
        if duration and now - started >= duration:
            break
        sim.tick()
        ticks += 1
        if viewer is not None:
            viewer.sync()
        elif now - report >= 2.0:
            # 창이 없으니 돌아가고 있다는 것만 주기적으로 알린다
            print(f"  {ticks / (now - started):5.1f} Hz | leader read {sim.read_ms:.1f} ms | sim t={sim.data.time:.1f}s")
            report = now
        pacer.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--headless", action="store_true", help="MuJoCo 창 없이 실행")
    ap.add_argument("--scene", type=Path, default=SCENE, help="씬 XML (기본: 트롤리 씬)")
    ap.add_argument("--port", help="리더암 포트 (기본: config 의 고정 이름)")
    ap.add_argument("--fps", type=int, help="리더를 읽는 주기 Hz (기본: config 값)")
    ap.add_argument("--duration", type=float, help="이 시간(초)이 지나면 끝낸다")
    ap.add_argument("--randomize", action="store_true", help="시작할 때 물체 위치를 models/automatic.json 범위에서 흩뜨린다")
    args = ap.parse_args()

    cfg = SO101LeaderMjConfig(xml_path=args.scene, randomize_path=RANDOMIZE if args.randomize else None)
    if args.port:
        cfg.port = args.port
    if args.fps:
        cfg.fps = args.fps

    sim = LeaderSim(cfg)
    sim.connect()
    print(f"연결됨: {cfg.port} -> {cfg.xml_path.name} ({cfg.fps} Hz, Ctrl+C 로 종료)")
    try:
        if args.headless:
            run(sim, args.duration)
        else:
            import mujoco.viewer

            try:
                viewer_cm = mujoco.viewer.launch_passive(sim.model, sim.data)
            except RuntimeError as e:
                if sys.platform == "darwin" and "mjpython" in str(e):
                    sys.exit("macOS 에서 MuJoCo 창을 띄우려면 mjpython 으로 실행하세요:\n"
                             "  .venv/bin/mjpython src/teleop_mj.py\n"
                             "창 없이 돌리려면 --headless 를 붙이세요.")
                raise
            with viewer_cm as viewer:
                viewer.cam.lookat[:] = [0.35, 0.24, 0.85]
                viewer.cam.distance = 1.4
                viewer.cam.azimuth = 150
                viewer.cam.elevation = -30
                run(sim, args.duration, viewer)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
    print("종료")


if __name__ == "__main__":
    main()
