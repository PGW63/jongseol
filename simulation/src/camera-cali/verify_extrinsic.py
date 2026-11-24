"""구한 extrinsic 들이 서로 맞는지 눈과 숫자로 확인한다.

    ../.venv/bin/python src/camera-cali/verify_extrinsic.py --square 0.0368
    ../.venv/bin/python src/camera-cali/verify_extrinsic.py --square 0.0368 --cams top left
    ../.venv/bin/python src/camera-cali/verify_extrinsic.py --cams top left --no-board     # 팔 뼈대만

무엇을 보나
    1. 보드: 보드를 아무 데나 두면, 보드가 보이는 카메라마다 '베이스 기준 보드 위치'를 따로 계산한다.
       extrinsic 이 맞으면 카메라가 달라도 같은 위치가 나와야 한다. 화면 아래와 터미널에 카메라끼리의 차이 (mm, 도).
    2. 팔 뼈대: follower 관절 값으로 시뮬 SO-101 의 관절 위치를 계산해 각 카메라 화면에 초록 선으로 겹쳐 그린다.
       선이 실제 팔을 따라가면 맞다. (그리퍼는 open gripper 라 모양이 다르니 손목까지만 본다)
    wrist, belly 는 관절 값이 필요하므로 follower 를 연결한다 (토크는 끈다).

키 (창을 클릭한 뒤)
    p  지금 숫자를 터미널에 찍기
    h  follower 토크 켜기/끄기
    q  끝
"""

import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cali_common import HERE, LiveView, load_camera_config, open_camera
from charuco_intrinsic_cali import Detector, make_board, parse_squares
from extrinsic_cali import EXT, Follower, LinkFK, rot_deg, to_rad, transform

sys.path.insert(0, str(HERE.parent))
from teleop.constants import default_port  # noqa: E402

SKELETON = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
TILE = (480, 360)


def load_extrinsic_any(name: str) -> tuple[str, np.ndarray]:
    path = EXT / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"{name} 의 extrinsic 이 없습니다 ({path}).")
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    cfg = ns[f"{name.capitalize()}CamExtrinsicConfig"]()
    return cfg.link, np.array(cfg.T_link_cam)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cams", nargs="+", help="확인할 카메라 (기본: extrinsic/<name>.py 가 있는 것 모두)")
    ap.add_argument("--squares", default="5x7")
    ap.add_argument("--square", type=float, default=0.0368, help="보드 칸 한 변 m (기본 0.0368)")
    ap.add_argument("--marker", type=float)
    ap.add_argument("--dict", default="DICT_6X6_250")
    ap.add_argument("--no-board", action="store_true", help="보드 없이 팔 뼈대만 본다")
    ap.add_argument("--arm-port", default=default_port("so101_follower"))
    ap.add_argument("--arm-id", default="follower")
    args = ap.parse_args()

    names = args.cams or sorted(p.stem for p in EXT.glob("*.py")
                                if p.stem not in ("__init__", "config") and not p.stem.startswith("board"))
    cams = []
    for n in names:
        link, T = load_extrinsic_any(n)
        cfg = load_camera_config(n)
        cams.append({"name": n, "link": link, "T": T, "K": np.array(cfg.camera_matrix), "dist": np.array(cfg.dist_coeffs),
                     "cap": open_camera(cfg.port, cfg.width, cfg.height)})
        print(f"{n}: link={link}")
    arm = Follower(args.arm_port, args.arm_id)  # 뼈대를 그리고 wrist/belly 를 놓으려면 관절 값이 필요하다
    fk = LinkFK()
    board = make_board(parse_squares(args.squares), args.square, args.marker or round(args.square * 0.6, 6), args.dict, False)
    detector = Detector(board)
    cols = 2 if len(cams) > 1 else 1
    rows = (len(cams) + cols - 1) // cols
    view = LiveView("verify_extrinsic", TILE[0] * cols, TILE[1] * rows)
    summary = ""

    try:
        while view.open:
            q = to_rad(arm.bus.sync_read("Present_Position", normalize=False, num_retry=2), {}, "deg") if arm else np.zeros(6)
            links = {b: fk(q, b) for b in SKELETON}
            tiles, boards = [], {}
            for c in cams:
                ok, bgr = c["cap"].read()
                if not ok:
                    bgr = np.zeros((TILE[1], TILE[0], 3), np.uint8)
                T_base_cam = c["T"] if c["link"] == "base" else links[c["link"]] @ c["T"]
                T_cam_base = np.linalg.inv(T_base_cam)
                shown = bgr.copy()
                # 팔 뼈대 (카메라 앞에 있는 점만)
                pts = np.array([links[b][:3, 3] for b in SKELETON])
                pc = (T_cam_base[:3, :3] @ pts.T).T + T_cam_base[:3, 3]
                if (pc[:, 2] > 0.02).all():
                    proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(T_cam_base[:3, :3])[0], T_cam_base[:3, 3], c["K"], c["dist"])
                    proj = proj.reshape(-1, 2).astype(int)
                    cv2.polylines(shown, [proj], False, (0, 255, 0), 3)
                    for pnt in proj:
                        cv2.circle(shown, tuple(pnt), 6, (0, 200, 255), -1)
                # 보드
                if not args.no_board:
                    corners, ids, _, _ = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    if ids is not None and len(ids) >= 6:
                        obj, img = detector.points(corners, ids)
                        ok_pnp, rv, tv = cv2.solvePnP(obj, img, c["K"], c["dist"], flags=cv2.SOLVEPNP_IPPE)
                        if ok_pnp:
                            rv, tv = cv2.solvePnPRefineLM(obj, img, c["K"], c["dist"], rv, tv)
                            boards[c["name"]] = T_base_cam @ transform(cv2.Rodrigues(rv)[0], tv)
                            cv2.drawFrameAxes(shown, c["K"], c["dist"], rv, tv, 2 * args.square, 3)
                cv2.putText(shown, c["name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                tiles.append(cv2.resize(shown, TILE))
            while len(tiles) < rows * cols:
                tiles.append(np.zeros((TILE[1], TILE[0], 3), np.uint8))
            grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])

            # 카메라끼리 보드 위치 차이
            seen = sorted(boards)
            pairs = []
            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    A, B = boards[seen[i]], boards[seen[j]]
                    pairs.append(f"{seen[i]}-{seen[j]} {np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000:.0f}mm "
                                 f"{rot_deg(A[:3, :3].T @ B[:3, :3]):.1f}deg")
            summary = ("board seen by " + ", ".join(seen) + ("   |   " + "   ".join(pairs) if pairs else "")) if seen else "no board"
            hold = ("  h: torque " + ("ON" if arm.holding else "off")) if arm else ""
            for key in view.show(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB), f"{summary}\np: print  q: quit{hold}"):
                if key == "p":
                    print(summary)
                    for n, T in boards.items():
                        print(f"  {n}: board origin in base (m) {np.round(T[:3, 3], 4).tolist()}")
                elif key == "h" and arm:
                    arm.toggle_hold()
                elif key == "q":
                    plt.close(view.fig)
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            c["cap"].release()
        if arm:
            arm.close()


if __name__ == "__main__":
    main()
