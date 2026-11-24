"""ChArUco 보드로 카메라 내부 파라미터(intrinsic)를 구한다.

    .venv/bin/python src/camera-cali/charuco_intrinsic_cali.py top --square 0.034
    .venv/bin/python src/camera-cali/charuco_intrinsic_cali.py side --camera 1 --square 0.034
    .venv/bin/python src/camera-cali/charuco_intrinsic_cali.py top --images src/camera-cali/top_cam/charuco_captures
    .venv/bin/python src/camera-cali/charuco_intrinsic_cali.py top --square 0.04 --make-board charuco_5x7.png

첫 인자는 카메라 이름. 아무 이름이나 줄 수 있고 (영문 소문자/숫자/_), <이름>_cam/config.py 가 없으면
새로 만든다. 끝나면 결과를 그 config.py 에 써 넣는다. 새 카메라는 --camera 로 장치를 알려 주면
그 값이 config 의 port 로 저장되고, 다음부터는 생략해도 된다.

보드 (https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html)
    기본값은 위 튜토리얼의 charucoboard.png 와 같다: 5x7 칸, DICT_6X6_250, 마커 = 칸의 0.6 배.
    그 이미지를 인쇄했으면 칸 한 변만 자로 재서 --square 로 주면 된다.
    --squares  칸 수 (가로x세로). 체커보드와 달리 꼭짓점이 아니라 '칸' 수다.
    --square   한 칸의 한 변 길이 (m). 인쇄한 뒤 자로 재서 넣는다.
    --marker   칸 안 ArUco 마커의 한 변 길이 (m). 생략하면 --square 의 0.6 배.
    --dict     ArUco 사전 (기본 DICT_6X6_250).
    --legacy   OpenCV 4.6 이전에 만든 보드 (또는 calib.io 의 세로 칸 수가 짝수인 보드) 일 때.
    --make-board 파일.png  위 설정의 보드 이미지를 만들고 끝낸다. 100% 배율로 인쇄할 것.

체커보드와 달리 보드 일부가 가려지거나 화면 밖으로 나가도 보이는 꼭짓점만으로 쓸 수 있어서,
화면 가장자리(왜곡이 큰 곳)까지 보드를 채워 찍기 좋다.

키 (창을 클릭한 뒤)
    s  지금 화면을 캡쳐 (꼭짓점이 --min-corners 개 이상 잡혀 있을 때만)
    q  캡쳐한 것으로 캘리브레이션하고 저장한 뒤 종료

보드를 화면 가운데, 네 귀퉁이, 가장자리, 가까이, 멀리, 기울여서 20 장쯤 찍는다.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from cali_common import HERE, LiveView, check_name, load_camera_config, load_images, open_camera, write_intrinsic_config

MIN_VIEWS = 5


def parse_squares(text: str) -> tuple[int, int]:
    try:
        sx, sy = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--squares 는 '가로x세로' 칸 수로 적습니다 (예: 7x5): {text}") from None
    if sx < 2 or sy < 2:
        raise SystemExit("--squares 는 가로, 세로 모두 2 칸 이상이어야 합니다.")
    return sx, sy


def make_board(squares: tuple[int, int], square: float, marker: float, dict_name: str, legacy: bool) -> cv2.aruco.CharucoBoard:
    if not hasattr(cv2.aruco, dict_name):
        raise SystemExit(f"없는 ArUco 사전입니다: {dict_name} (예: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250)")
    if marker >= square:
        raise SystemExit("--marker 는 --square 보다 작아야 합니다.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard(squares, square, marker, dictionary)
    board.setLegacyPattern(legacy)
    return board


def save_board_image(board: cv2.aruco.CharucoBoard, path: Path, square: float, dpi: int, margin: float) -> None:
    """board 를 dpi 로 인쇄했을 때 한 칸이 square (m) 가 되도록 그린다."""
    px_per_m = dpi / 0.0254
    sx, sy = board.getChessboardSize()
    margin_px = round(margin * px_per_m)
    size = (round(sx * square * px_per_m) + 2 * margin_px, round(sy * square * px_per_m) + 2 * margin_px)
    img = board.generateImage(size, marginSize=margin_px, borderBits=1)
    cv2.imwrite(str(path), img)
    w_mm, h_mm = (v / px_per_m * 1000 for v in size)
    print(f"저장: {path}  ({size[0]}x{size[1]} px, {dpi} dpi 로 인쇄하면 {w_mm:.0f}x{h_mm:.0f} mm)")
    print("100% 배율(맞춤 끄기)로 인쇄한 뒤 칸과 마커 크기를 자로 재서 --square, --marker 로 넣으세요.")


class Detector:
    def __init__(self, board: cv2.aruco.CharucoBoard):
        self.board = board
        self.detector = cv2.aruco.CharucoDetector(board)

    def detect(self, rgb: np.ndarray):
        """(charuco 꼭짓점 (N,1,2), id (N,1), 마커 꼭짓점들, 마커 id). 못 찾으면 꼭짓점과 id 는 None."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return self.detector.detectBoard(gray)

    def points(self, corners: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """잡힌 꼭짓점의 보드 좌표 (N,1,3) 와 이미지 좌표 (N,1,2)."""
        obj, img = self.board.matchImagePoints(corners, ids)
        return obj.astype(np.float32), img.astype(np.float32)


def usable(corners, ids, obj, min_corners: int) -> str | None:
    """캘리에 쓸 수 없으면 그 이유. 꼭짓점이 한 줄로만 늘어서 있으면 calibrateCamera 가 풀리지 않는다."""
    n = 0 if ids is None else len(ids)
    if n < min_corners:
        return f"corners {n} < {min_corners}"
    xy = obj.reshape(-1, 3)[:, :2]
    if np.linalg.matrix_rank(xy - xy.mean(0), tol=1e-6) < 2:
        return "corners are collinear"
    return None


def calibrate(views: list[tuple[np.ndarray, np.ndarray]], size: tuple[int, int], board_info: dict) -> dict:
    """views: 이미지마다 (보드 좌표, 이미지 좌표). size: (가로, 세로)."""
    objs, imgs = zip(*views)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(list(objs), list(imgs), size, None, None)
    per_view = []
    for obj, img, r, t in zip(objs, imgs, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=-1)))))
    return {
        "image_size": list(size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms_px": float(rms),
        "num_images": len(views),
        "per_view_px": per_view,
        "charuco": board_info,
        # MuJoCo 카메라에 넣을 수직 화각 (도)
        "fovy_deg": math.degrees(2 * math.atan(size[1] / (2 * K[1, 1]))),
    }


def report(result: dict, names: list[str], out: Path) -> None:
    K = np.array(result["camera_matrix"])
    print(f"\n이미지 {result['num_images']} 장, 재투영 오차 RMS {result['rms_px']:.3f} px")
    print(f"  fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}")
    print(f"  왜곡 계수 {np.round(result['dist_coeffs'], 4).tolist()}")
    print(f"  MuJoCo fovy = {result['fovy_deg']:.2f} 도")
    worst = sorted(zip(result["per_view_px"], names), reverse=True)[:3]
    print("  오차가 큰 이미지: " + ", ".join(f"{n} {e:.2f}px" for e, n in worst))
    print("저장:", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="카메라 이름. <name>_cam/config.py 에 저장 (없으면 만든다)")
    ap.add_argument("--camera", help="장치 경로(udev 이름) 또는 카메라 번호 (기본: 그 카메라 config.py 의 port)")
    ap.add_argument("--squares", default="5x7", help="보드 칸 수, 가로x세로 (기본 5x7)")
    ap.add_argument("--square", type=float, required=True, help="한 칸의 한 변 길이 m (인쇄한 보드를 자로 잰 값)")
    ap.add_argument("--marker", type=float, help="ArUco 마커 한 변 길이 m (기본: --square 의 0.6 배)")
    ap.add_argument("--dict", default="DICT_6X6_250", help="ArUco 사전 (기본 DICT_6X6_250)")
    ap.add_argument("--legacy", action="store_true", help="OpenCV 4.6 이전 배치의 보드")
    ap.add_argument("--min-corners", type=int, default=8, help="캡쳐에 필요한 최소 꼭짓점 수 (기본 8)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--images", help="카메라 대신 쓸 이미지 폴더")
    ap.add_argument("--captures", type=Path, help="캡쳐한 이미지를 남길 폴더 (기본: <name>_cam/charuco_captures)")
    ap.add_argument("--make-board", type=Path, metavar="PNG", help="보드 이미지를 만들고 끝낸다")
    ap.add_argument("--dpi", type=int, default=300, help="--make-board 인쇄 해상도 (기본 300)")
    ap.add_argument("--margin", type=float, default=0.01, help="--make-board 흰 여백 m (기본 0.01)")
    args = ap.parse_args()
    name = check_name(args.name)
    if args.marker is None:
        args.marker = round(args.square * 0.6, 6)
    squares = parse_squares(args.squares)
    board = make_board(squares, args.square, args.marker, args.dict, args.legacy)
    if args.make_board:
        save_board_image(board, args.make_board, args.square, args.dpi, args.margin)
        return
    board_info = {"squares_x": squares[0], "squares_y": squares[1], "square": args.square,
                  "marker": args.marker, "dictionary": args.dict, "legacy": args.legacy}

    cfg = load_camera_config(name)  # 처음 보는 이름이면 여기서 <name>_cam/config.py 가 만들어진다
    port = args.camera or cfg.port
    captures = args.captures or HERE / f"{name}_cam" / "charuco_captures"
    detector = Detector(board)

    views, names, size = [], [], None
    if args.images:
        for fname, rgb in load_images(args.images):
            corners, ids, _, _ = detector.detect(rgb)
            obj = img = None
            if ids is not None and len(ids) > 0:
                obj, img = detector.points(corners, ids)
            why = usable(corners, ids, obj, args.min_corners) if obj is not None else "no board"
            print(f"  {fname}: {why or f'꼭짓점 {len(ids)} 개'}")
            if why is None:
                views.append((obj, img))
                names.append(fname)
                size = (rgb.shape[1], rgb.shape[0])
    else:
        cap = open_camera(port, args.width, args.height)
        view = LiveView(f"charuco_intrinsic_cali ({name})", args.width, args.height)
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
                corners, ids, marker_corners, marker_ids = detector.detect(rgb)
                shown = rgb.copy()
                if marker_ids is not None and len(marker_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(shown, marker_corners, marker_ids)
                obj = img = None
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(shown, corners, ids, (255, 0, 0))
                    obj, img = detector.points(corners, ids)
                why = usable(corners, ids, obj, args.min_corners) if obj is not None else "no board"
                state = why or f"{len(ids)} corners"
                for key in view.show(shown, f"captured {len(views)}   [{state}]   s: capture   q: calibrate & quit   {message}"):
                    if key == "s":
                        if why is not None:
                            message = f"(not captured: {why})"
                        else:
                            views.append((obj, img))
                            fname = f"capture_{len(views):02d}.png"
                            names.append(fname)
                            cv2.imwrite(str(captures / fname), bgr)
                            message = f"(captured #{len(views)})"
                    elif key == "q":
                        done = True
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()

    if len(views) < MIN_VIEWS:
        raise SystemExit(f"보드가 잡힌 이미지가 {len(views)} 장뿐입니다. 최소 {MIN_VIEWS} 장이 필요합니다.")
    result = calibrate(views, size, board_info)
    report(result, names, write_intrinsic_config(name, port, size[0], size[1], result))


if __name__ == "__main__":
    main()
