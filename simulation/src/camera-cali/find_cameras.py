"""연결된 카메라를 찍어서 보여주고, 어느 것이 top / wrist / belly / left ... 인지 config.py 의 port 에 적는다.

    ../.venv/bin/python src/camera-cali/find_cameras.py                 # 사진 보고 번호 고르기
    ../.venv/bin/python src/camera-cali/find_cameras.py --set top=0 wrist=3 belly=1 left=2
    ../.venv/bin/python src/camera-cali/find_cameras.py --list          # 목록만

macOS 는 카메라 번호가 꽂는 순서/개수에 따라 바뀐다. 그래서 config 에는 번호 대신 uid:<uniqueID> 를 적고,
열 때마다 그 카메라가 지금 몇 번인지 다시 찾는다 (cali_common.resolve_camera).
USB 카메라의 uniqueID 에는 USB 포트 위치가 들어 있다. 같은 모델 두 대(예: Global Shutter Camera)는
꽂은 포트로만 구별되므로, 한 번 정한 뒤에는 각 카메라를 같은 USB 포트에 꽂아 둔다.

ubuntu 는 udev 규칙으로 만든 /dev/so101_cam_<name> 을 그대로 쓰므로 이 스크립트가 필요 없다.
"""

import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cali_common import CAMERAS, HERE, load_camera_config, mac_cameras, set_camera_port


def camera_names() -> list[str]:
    """config 가 있는 카메라 이름 (+ 기본 셋)."""
    found = {p.name[:-4] for p in HERE.glob("*_cam") if (p / "config.py").exists()}
    return sorted(found | set(CAMERAS), key=lambda n: (n not in CAMERAS, n))


def snapshot(index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(15):  # 처음 몇 장은 노출이 안 잡혀 어둡다
            ok, f = cap.read()
            if ok:
                frame = f
        return None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def show_grid(cams: list[dict], shots: dict[int, np.ndarray | None]) -> None:
    n = len(cams)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    fig.canvas.manager.set_window_title("find_cameras: 창을 닫으면 터미널에서 번호를 고른다")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), cams):
        img = shots.get(c["index"])
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "(no image)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"[{c['index']}] {c['name']}\n{c['uid']}", fontsize=9)
    fig.tight_layout()
    plt.show()


def current_ports() -> dict[str, str]:
    out = {}
    for name in camera_names():
        try:
            out[name] = load_camera_config(name).port
        except Exception:  # noqa: BLE001  config 가 깨져 있어도 목록은 보여준다
            out[name] = "?"
    return out


def print_list(cams: list[dict]) -> None:
    ports = current_ports()
    owner = {p[4:]: n for n, p in ports.items() if p.startswith("uid:")}
    print("번호  uniqueID                                이름                        config")
    for c in cams:
        print(f" [{c['index']}]  {c['uid']:38s}  {c['name']:26s}  {owner.get(c['uid'], '')}")
    stale = [n for n, p in ports.items() if not p.startswith("uid:")]
    if stale:
        print(f"\n번호/경로로 적혀 있어 바뀔 수 있는 카메라: " + ", ".join(f"{n}={ports[n]!r}" for n in stale))


def assign(name: str, index: int, cams: list[dict]) -> None:
    cam = next((c for c in cams if c["index"] == index), None)
    if cam is None:
        raise SystemExit(f"{index} 번 카메라가 없습니다.")
    path = set_camera_port(name, f"uid:{cam['uid']}")
    print(f"  {name:6s} <- [{index}] {cam['name']} ({cam['uid']})   {path.relative_to(HERE)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="목록만 보여준다")
    ap.add_argument("--set", nargs="+", metavar="NAME=INDEX", help="사진 없이 바로 지정 (예: top=0 wrist=3)")
    ap.add_argument("--all", action="store_true", help="MacBook/iPhone 카메라도 찍어 본다")
    args = ap.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("ubuntu 는 udev 규칙(/dev/so101_cam_<name>)을 쓴다. 이 스크립트는 macOS 용.")

    cams = mac_cameras()
    print_list(cams)
    if args.list:
        return
    if args.set:
        for item in args.set:
            name, _, idx = item.partition("=")
            assign(name, int(idx), cams)
        return

    builtin = ("MacBook", "iPhone", "iPad", "Continuity", "FaceTime")
    targets = [c for c in cams if args.all or not any(b in c["name"] for b in builtin)]
    print(f"\n카메라 {len(targets)} 대를 한 장씩 찍는 중...")
    shots = {c["index"]: snapshot(c["index"]) for c in targets}
    show_grid(targets, shots)

    print("\n각 카메라의 번호를 적는다 (Enter = 그대로 둔다).")
    for name in camera_names():
        text = input(f"  {name} 번호: ").strip()
        if text:
            assign(name, int(text), cams)
    print("\n결과:")
    print_list(mac_cameras())


if __name__ == "__main__":
    main()
