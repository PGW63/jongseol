"""환경 초기화(물체 랜덤 배치)만 따로 확인한다. 리더암 연결 없이 돈다.

    .venv/bin/python src/tests/randomize_env.py
    .venv/bin/python src/tests/randomize_env.py --seed 0 --duration 5 --save layout.png

data_mj.py 에서 3 을 눌렀을 때와 같은 배치 코드를 쓴다 (models/automatic.json).
로봇은 리더가 없으니 키프레임 자세 그대로 서 있다.

키 (창을 클릭한 뒤)
    3  새 배치를 뽑는다
    0  원래 자리로 되돌린다
    q  종료

mjpython 이 아니라 일반 python 으로 실행한다.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from teleop.randomizer import SceneRandomizer

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "models" / "scene" / "scene_trolley.xml"
RANDOMIZE = ROOT / "models" / "automatic.json"

CAMERAS = ["top", "corner", "left", "drawer_top"]
WIDTH, HEIGHT = 480, 360


def offsets_text(offsets: dict | None, count: int) -> str:
    lines = [f"layout #{count}", "", f"{'object':<12}{'dx cm':>7}{'dy cm':>7}{'rot deg':>9}"]
    if offsets is None:
        lines.append("(original positions)")
    else:
        for name, (dx, dy, theta) in offsets.items():
            lines.append(f"{name:<12}{dx * 100:7.2f}{dy * 100:7.2f}{np.degrees(theta):9.2f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, help="배치 난수 시드")
    ap.add_argument("--duration", type=float, help="이 시간(초)이 지나면 끝낸다 (기본: 창을 닫을 때까지)")
    ap.add_argument("--save", metavar="PNG", help="끝날 때 창 내용을 이미지로 저장")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    randomizer = SceneRandomizer(model, RANDOMIZE, args.seed)
    state = {"offsets": None, "count": 0, "request": "random", "quit": False}

    def reset(randomize: bool) -> None:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if randomize:
            state["offsets"] = randomizer.apply(data)
            if state["offsets"] is None:
                print("경고: 겹치지 않는 배치를 찾지 못해 물체를 원래 자리에 둡니다.")
        else:
            randomizer.restore(data)
            state["offsets"] = None
        mujoco.mj_forward(model, data)
        state["count"] += 1

    # 숫자키/q 를 matplotlib 기본 단축키가 가로채지 않게 한다
    for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[key] = []
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title("randomize_env")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.7])
    shown = {}
    for i, camera in enumerate(CAMERAS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_title(camera.replace("_", " "), fontsize=11)
        ax.axis("off")
        shown[camera] = ax.imshow(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")
    txt = ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10)
    fig.suptitle("3: new layout    0: original positions    q: quit", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def on_key(event) -> None:
        if event.key == "3":
            state["request"] = "random"
        elif event.key == "0":
            state["request"] = "original"
        elif event.key == "q":
            state["quit"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    plt.show(block=False)
    started = last = time.perf_counter()
    try:
        while plt.fignum_exists(fig.number) and not state["quit"]:
            now = time.perf_counter()
            if args.duration and now - started >= args.duration:
                break
            if state["request"]:
                reset(state["request"] == "random")
                state["request"] = None
                last = now
            # 물체가 제자리에 가라앉도록 실제 시간만큼 (밀려도 한 번에 0.1 초까지만) 진행한다
            for _ in range(int(min(now - last, 0.1) / model.opt.timestep)):
                mujoco.mj_step(model, data)
            last = now
            for camera in CAMERAS:
                renderer.update_scene(data, camera=camera)
                shown[camera].set_data(renderer.render())
            txt.set_text(offsets_text(state["offsets"], state["count"]))
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()

    print(offsets_text(state["offsets"], state["count"]))
    if args.save and plt.fignum_exists(fig.number):
        fig.savefig(args.save, dpi=90)
        print("저장:", args.save)


if __name__ == "__main__":
    main()
