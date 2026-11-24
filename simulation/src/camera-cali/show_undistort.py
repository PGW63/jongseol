"""캘리브레이션 결과로 왜곡을 편 화면을 원본과 나란히 보여준다.

    .venv/bin/python src/camera-cali/show_undistort.py wrist
    .venv/bin/python src/camera-cali/show_undistort.py wrist --images src/camera-cali/wrist_cam/captures

왼쪽이 원본, 오른쪽이 보정본. 곧은 선(책상 모서리, 체커보드)이 오른쪽에서 곧게 보이면 잘 된 것이다.

키 (창을 클릭한 뒤)
    a  보정본 잘라내기 전환 (검은 테두리 없이 자름 <-> 원본 화각 전부)
    n  다음 이미지 (--images 일 때)
    q  종료
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cali_common import LiveView, find_board, load_camera_config, load_images, open_camera


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="보여줄 카메라 (<name>_cam/config.py)")
    ap.add_argument("--camera", help="장치 경로(udev 이름) 또는 카메라 번호 (기본: 그 카메라 config.py 의 port)")
    ap.add_argument("--images", help="카메라 대신 쓸 이미지 파일 또는 폴더")
    args = ap.parse_args()

    cfg = load_camera_config(args.name)
    if not cfg.calibrated:
        raise SystemExit(f"{args.name} 카메라는 아직 캘리브레이션되지 않았습니다 (intrinsic_cali.py {args.name} 먼저)")
    K = np.array(cfg.camera_matrix)
    dist = np.array(cfg.dist_coeffs)
    size = (cfg.width, cfg.height)
    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}  dist={np.round(dist, 3).tolist()}")
    print(f"재투영 오차 {cfg.rms_px:.2f} px ({cfg.num_images} 장)" + ("  <- 1 보다 커서 다시 찍는 게 좋다" if cfg.rms_px > 1 else ""))

    # alpha 0 = 검은 테두리가 안 보이게 자름, 1 = 원본 화각을 다 담음
    maps = {}
    for alpha in (0.0, 1.0):
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
        maps[alpha] = cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_16SC2)
    alpha = 0.0

    images = load_images(args.images) if args.images else None
    cap = None if images else open_camera(args.camera or cfg.port, *size)
    index = 0
    view = LiveView(f"{args.name} undistort", size[0] * 2, size[1])

    while view.open:
        if images:
            label, rgb = images[index]
        else:
            ok, bgr = cap.read()
            if not ok:
                continue
            label, rgb = "live", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1::-1] != size:
            rgb = cv2.resize(rgb, size)

        fixed = cv2.remap(rgb, *maps[alpha], cv2.INTER_LINEAR)
        for img in (rgb, fixed):
            corners = find_board(img, cfg.board_cols, cfg.board_rows) if cfg.board_cols else None
            if corners is not None:
                cv2.drawChessboardCorners(img, (cfg.board_cols, cfg.board_rows), corners.reshape(-1, 1, 2), True)

        crop = "crop" if alpha == 0 else "Full view"
        status = f"{label}   Left : Original | Right : undistort ({crop})   a: shift  n: next  q: quit"
        for key in view.show(np.hstack([rgb, fixed]), status):
            if key == "a":
                alpha = 1.0 - alpha
            elif key == "n" and images:
                index = (index + 1) % len(images)
            elif key == "q":
                plt.close(view.fig)

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
