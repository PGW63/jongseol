"""헤드캠이 로봇 베이스에서 어디에, 어느 방향으로 달려 있는지(extrinsic) 구한다.

    .venv/bin/python src/camera-cali/headCam_to_Base.py --board 9x6 --square 0.025 --origin 0.30 0.20 --yaw 0

방법
    체커보드를 트롤리 상판에 평평하게 놓고, 보드가 놓인 자리를 자로 재서 알려준다.
    카메라가 본 보드의 자세와 합치면 카메라의 자세가 나온다.

    1. 먼저 intrinsic_cali.py top 으로 top 카메라의 내부 파라미터를 구해 둔다 (top_cam/config.py).
    2. 이 스크립트를 띄우면 화면의 보드 위에 축이 그려진다. 빨강 = 보드 X, 초록 = 보드 Y.
       축이 시작하는 꼭짓점(노란 점)이 보드 원점이다.
    3. 그 원점 꼭짓점의 위치를 재서 --origin X Y 로 준다 (m).
       기본은 트롤리 원점 꼭짓점 기준이고 (+X 손잡이 쪽, +Y 로봇 왼쪽, 상판 윗면이 Z=0),
       --frame base 를 주면 로봇 베이스 기준으로 읽는다.
    4. 빨간 X 축이 향하는 방향을 --yaw 로 준다 (도). 0 = +X, 90 = +Y, 180 = -X, -90 = -Y.
       보드를 트롤리 변과 나란히 놓으면 이 넷 중 하나다.
    원점이 재기 불편한 쪽에 잡혔으면 --flip 으로 반대쪽 꼭짓점으로 옮길 수 있다.

키 (창을 클릭한 뒤)
    s  지금 자세를 표본으로 추가 (여러 번 누르면 평균을 낸다)
    q  저장하고 종료

결과는 extrinsic/config.py 에 써 넣는다. 베이스 기준 카메라 자세와, 씬 XML 의 <camera> 에 그대로 넣을
pos / xyaxes / fovy 가 같이 들어간다.
--image 를 주면 카메라 대신 그 이미지(또는 폴더)로 바로 계산한다.
"""

import argparse
import math

import cv2
import mujoco
import numpy as np

from cali_common import (
    HERE, LiveView, board_points, find_board, load_camera_config, load_images, open_camera, parse_board,
    write_extrinsic_config,
)

ROOT = HERE.parents[1]
SCENE = ROOT / "models" / "scene" / "scene_trolley.xml"
# 씬에서 트롤리 상판 윗면의 높이 (m). 트롤리 좌표의 Z=0 이 여기다.
TROLLEY_TOP_Z = 0.73


def base_in_trolley() -> np.ndarray:
    """트롤리 좌표에서 본 로봇 베이스 원점. 씬 XML 에 놓인 자리를 그대로 읽는다 (베이스는 +X 를 본다)."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data.body("so101_base").xpos - [0, 0, TROLLEY_TOP_Z]


def transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def board_pose_in_camera(corners, cols, rows, square, K, dist):
    """-> (T_cam_board, rvec, tvec, 재투영 오차 px)."""
    obj = board_points(cols, rows, square)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32).reshape(-1, 1, 2), K, dist)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    return transform(cv2.Rodrigues(rvec)[0], tvec), rvec, tvec, err


def board_pose_in_frame(T_cam_board: np.ndarray, origin: np.ndarray, yaw_deg: float) -> np.ndarray:
    """잰 값으로 만든 보드 자세. 보드는 수평면에 놓여 있고 카메라는 위에서 내려다본다고 본다."""
    yaw = math.radians(yaw_deg)
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    # 보드 Z 가 카메라에서 멀어지는 쪽이면 (카메라가 위에 있으니) 아래를 향한다
    z = np.array([0.0, 0.0, -1.0 if T_cam_board[2, 2] > 0 else 1.0])
    return transform(np.column_stack([x, np.cross(z, x), z]), origin)


def average(poses: list[np.ndarray]) -> np.ndarray:
    """자세 여러 개의 평균. 회전은 행렬 평균을 가장 가까운 회전으로 되돌린다."""
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in poses], axis=0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return transform(R, np.mean([T[:3, 3] for T in poses], axis=0))


def quat_wxyz(R: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", help="장치 경로(udev 이름) 또는 카메라 번호 (기본: top_cam/config.py 의 port)")
    ap.add_argument("--board", default="9x6", help="체커보드 안쪽 꼭짓점 수, 가로x세로 (기본 9x6)")
    ap.add_argument("--square", type=float, default=0.025, help="한 칸의 한 변 길이 m (기본 0.025)")
    ap.add_argument("--origin", type=float, nargs="+", required=True, metavar="M",
                    help="보드 원점 꼭짓점의 위치 X Y [Z] (m). Z 를 빼면 0 (상판 위)")
    ap.add_argument("--yaw", type=float, default=0.0, help="보드 X 축(빨강)의 방향, 도 (0 = +X, 90 = +Y)")
    ap.add_argument("--frame", choices=["trolley", "base"], default="trolley", help="--origin, --yaw 를 잰 기준")
    ap.add_argument("--flip", action="store_true", help="보드 원점을 반대쪽 꼭짓점으로 옮긴다 (축이 180 도 돈다)")
    ap.add_argument("--image", help="카메라 대신 쓸 이미지 파일 또는 폴더")
    args = ap.parse_args()
    cols, rows = parse_board(args.board)
    if len(args.origin) not in (2, 3):
        raise SystemExit("--origin 은 X Y 또는 X Y Z 로 줍니다.")
    origin = np.array(list(args.origin) + [0.0] * (3 - len(args.origin)))

    cam = load_camera_config("top")
    if not cam.calibrated:
        raise SystemExit("top 카메라의 내부 파라미터가 없습니다. 먼저 intrinsic_cali.py top 을 실행하세요.")
    K, dist = np.array(cam.camera_matrix), np.array(cam.dist_coeffs)
    width, height = cam.width, cam.height

    def solve(rgb):
        """이미지 한 장 -> (재 기준에서의 카메라 자세, 그려 넣은 화면, 재투영 오차). 보드가 없으면 None."""
        corners = find_board(rgb, cols, rows)
        if corners is None:
            return None
        if args.flip:
            corners = corners[::-1].copy()
        found = board_pose_in_camera(corners, cols, rows, args.square, K, dist)
        if found is None:
            return None
        T_cam_board, rvec, tvec, err = found
        shown = rgb.copy()
        cv2.drawFrameAxes(shown, K, dist, rvec, tvec, 3 * args.square, 3)
        cv2.circle(shown, tuple(int(v) for v in corners[0]), 7, (255, 255, 0), -1)
        T_frame_cam = board_pose_in_frame(T_cam_board, origin, args.yaw) @ np.linalg.inv(T_cam_board)
        return T_frame_cam, shown, err

    samples, errors = [], []
    if args.image:
        for name, rgb in load_images(args.image):
            if rgb.shape[1::-1] != (width, height):
                raise SystemExit(f"{name}: 크기 {rgb.shape[1]}x{rgb.shape[0]} 가 intrinsic 의 {width}x{height} 와 다릅니다.")
            found = solve(rgb)
            print(f"  {name}: {'체커보드 없음' if found is None else f'재투영 오차 {found[2]:.3f} px'}")
            if found is not None:
                samples.append(found[0])
                errors.append(found[2])
    else:
        cap = open_camera(args.camera or cam.port, width, height)
        view = LiveView("headCam_to_Base", width, height)
        message = ""
        try:
            done = False
            while view.open and not done:
                ok, bgr = cap.read()
                if not ok:
                    raise SystemExit("카메라에서 화면을 읽지 못했습니다.")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                found = solve(rgb)
                state = "no board" if found is None else f"board found, reprojection {found[2]:.2f} px"
                keys = view.show(rgb if found is None else found[1],
                                 f"samples {len(samples)}   [{state}]   s: add sample   q: save & quit   {message}")
                for key in keys:
                    if key == "s":
                        if found is None:
                            message = "(not added: no board)"
                        else:
                            samples.append(found[0])
                            errors.append(found[2])
                            message = f"(added #{len(samples)})"
                    elif key == "q":
                        done = True
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()

    if not samples:
        raise SystemExit("표본이 없습니다. 체커보드가 잡힌 상태에서 s 를 눌러야 합니다.")
    T_frame_cam = average(samples)

    # 잰 기준 -> 트롤리 기준, 베이스 기준 둘 다 만든다 (베이스는 트롤리와 방향이 같고 위치만 다르다)
    T_trolley_base = transform(np.eye(3), base_in_trolley())
    if args.frame == "trolley":
        T_trolley_cam = T_frame_cam
        T_base_cam = np.linalg.inv(T_trolley_base) @ T_trolley_cam
    else:
        T_base_cam = T_frame_cam
        T_trolley_cam = T_trolley_base @ T_base_cam

    # MuJoCo 카메라 축은 x 오른쪽, y 위, 보는 방향이 -z. OpenCV 는 x 오른쪽, y 아래, 보는 방향이 +z.
    R_scene = T_trolley_cam[:3, :3]
    scene_pos = T_trolley_cam[:3, 3] + [0, 0, TROLLEY_TOP_Z]
    forward = R_scene[:, 2]
    result = {
        "T_base_cam": T_base_cam.tolist(),
        "position": T_base_cam[:3, 3].tolist(),
        "quat_wxyz": quat_wxyz(T_base_cam[:3, :3]),
        "trolley_position": T_trolley_cam[:3, 3].tolist(),
        "trolley_quat_wxyz": quat_wxyz(R_scene),
        "mujoco_pos": scene_pos.tolist(),
        "mujoco_xyaxes": np.concatenate([R_scene[:, 0], -R_scene[:, 1]]).tolist(),
        "mujoco_fovy": cam.fovy_deg,
        "num_samples": len(samples),
        "reprojection_px": float(np.mean(errors)),
    }
    out = write_extrinsic_config(result)

    fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
    print(f"\n표본 {len(samples)} 개, 재투영 오차 평균 {np.mean(errors):.3f} px")
    print(f"베이스 기준 카메라 위치 (m): {fmt(T_base_cam[:3, 3])}")
    print(f"트롤리 기준 카메라 위치 (m): {fmt(T_trolley_cam[:3, 3])}   (상판 윗면이 Z=0)")
    print(f"보는 방향: yaw {math.degrees(math.atan2(forward[1], forward[0])):.1f} 도, 아래로 {math.degrees(math.asin(-forward[2])):.1f} 도")
    print("씬 XML 에 넣을 카메라:")
    print(f'  <camera name="corner" pos="{fmt(result["mujoco_pos"])}" xyaxes="{fmt(result["mujoco_xyaxes"])}" '
          f'fovy="{result["mujoco_fovy"]:.2f}" resolution="{width} {height}"/>')
    print("저장:", out)


if __name__ == "__main__":
    main()
