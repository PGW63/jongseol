"""체커보드로 카메라 내부 파라미터(intrinsic)를 구한다.

    .venv/bin/python src/camera-cali/intrinsic_cali.py top --board 9x6 --square 0.025
    .venv/bin/python src/camera-cali/intrinsic_cali.py wrist --board 9x6 --square 0.025
    .venv/bin/python src/camera-cali/intrinsic_cali.py top --images src/camera-cali/top_cam/captures

첫 인자로 어느 카메라인지 (top 또는 wrist) 를 준다. 끝나면 결과를 그 카메라의 설정 파일
(top_cam/config.py 또는 wrist_cam/config.py) 에 써 넣는다.

--board 는 칸 수가 아니라 안쪽 꼭짓점 수 (가로x세로). 9x6 은 칸이 10x7 인 보드다.
--square 는 한 칸의 한 변 길이 (m).

키 (창을 클릭한 뒤)
    s  지금 화면을 캡쳐 (체커보드가 잡혀 있을 때만)
    q  캡쳐한 것으로 캘리브레이션하고 저장한 뒤 종료

보드를 화면 가운데, 네 귀퉁이, 가까이, 멀리, 기울여서 15 장쯤 찍는다.
--images 를 주면 카메라 대신 그 폴더의 이미지로 바로 계산한다.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from cali_common import (
    CAMERAS, HERE, LiveView, board_points, find_board, load_camera_config, load_images, open_camera, parse_board,
    write_intrinsic_config,
)

MIN_VIEWS = 5


def calibrate(views: list[np.ndarray], cols: int, rows: int, square: float, size: tuple[int, int]) -> dict:
    """views: 이미지마다의 꼭짓점 (N, 2). size: (가로, 세로)."""
    obj = board_points(cols, rows, square)
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [obj] * len(views), [v.astype(np.float32).reshape(-1, 1, 2) for v in views], size, None, None
    )
    return {
        "image_size": list(size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms_px": float(rms),
        "num_images": len(views),
        "board": {"cols": cols, "rows": rows, "square": square},
        # MuJoCo 카메라에 넣을 수직 화각 (도)
        "fovy_deg": math.degrees(2 * math.atan(size[1] / (2 * K[1, 1]))),
    }


def report(result: dict, out: Path) -> None:
    K = np.array(result["camera_matrix"])
    print(f"\n이미지 {result['num_images']} 장, 재투영 오차 RMS {result['rms_px']:.3f} px")
    print(f"  fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}")
    print(f"  왜곡 계수 {np.round(result['dist_coeffs'], 4).tolist()}")
    print(f"  MuJoCo fovy = {result['fovy_deg']:.2f} 도")
    print("저장:", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", choices=CAMERAS, help="캘리브레이션할 카메라")
    ap.add_argument("--camera", help="장치 경로(udev 이름) 또는 카메라 번호 (기본: 그 카메라 config.py 의 port)")
    ap.add_argument("--board", default="9x6", help="체커보드 안쪽 꼭짓점 수, 가로x세로 (기본 9x6)")
    ap.add_argument("--square", type=float, default=0.025, help="한 칸의 한 변 길이 m (기본 0.025)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--images", help="카메라 대신 쓸 이미지 폴더")
    ap.add_argument("--captures", type=Path, help="캡쳐한 이미지를 남길 폴더 (기본: <name>_cam/captures)")
    args = ap.parse_args()
    cols, rows = parse_board(args.board)
    port = args.camera or load_camera_config(args.name).port
    captures = args.captures or HERE / f"{args.name}_cam" / "captures"

    views, size = [], None
    if args.images:
        for name, rgb in load_images(args.images):
            corners = find_board(rgb, cols, rows)
            print(f"  {name}: {'찾음' if corners is not None else '체커보드 없음'}")
            if corners is not None:
                views.append(corners)
                size = (rgb.shape[1], rgb.shape[0])
    else:
        cap = open_camera(port, args.width, args.height)
        view = LiveView(f"intrinsic_cali ({args.name})", args.width, args.height)
        captures.mkdir(parents=True, exist_ok=True)
        message = ""
        try:
            done = False
            while view.open and not done:
                ok, bgr = cap.read()
                if not ok:
                    raise SystemExit("카메라에서 화면을 읽지 못했습니다.")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                size = (rgb.shape[1], rgb.shape[0])
                corners = find_board(rgb, cols, rows)
                shown = rgb.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(shown, (cols, rows), corners.reshape(-1, 1, 2).astype(np.float32), True)
                state = "board found" if corners is not None else "no board"
                for key in view.show(shown, f"captured {len(views)}   [{state}]   s: capture   q: calibrate & quit   {message}"):
                    if key == "s":
                        if corners is None:
                            message = "(not captured: no board)"
                        else:
                            views.append(corners)
                            cv2.imwrite(str(captures / f"capture_{len(views):02d}.png"), bgr)
                            message = f"(captured #{len(views)})"
                    elif key == "q":
                        done = True
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()

    if len(views) < MIN_VIEWS:
        raise SystemExit(f"체커보드가 잡힌 이미지가 {len(views)} 장뿐입니다. 최소 {MIN_VIEWS} 장이 필요합니다.")
    result = calibrate(views, cols, rows, args.square, size)
    report(result, write_intrinsic_config(args.name, port, size[0], size[1], result))


if __name__ == "__main__":
    main()
