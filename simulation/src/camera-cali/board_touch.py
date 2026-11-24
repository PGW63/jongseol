"""로봇 자리에 둔 리더암 끝으로 체커보드 꼭짓점을 찍어서, 로봇 베이스 기준 보드 위치를 구한다 (자 대신 로봇으로 잰다).

    .venv/bin/python src/camera-cali/board_touch.py --board 9x7 --square 0.0205
    .venv/bin/python src/camera-cali/board_touch.py --board 9x7 --square 0.0205 --replay

방법
    1. 보드를 트롤리 상판에 놓고, headCam_to_Base.py 화면에서 노란 점(원점), 빨간 선(X), 초록 선(Y)이
       실제 보드의 어느 꼭짓점과 어느 방향인지 봐 둔다.
    2. 리더암(토크가 꺼져 있다) 끝을 보드의 안쪽 꼭짓점 네 귀퉁이에 차례로 대고 Enter 를 누른다.
       첫 점은 반드시 노란 점. 나머지 세 점은 순서가 바뀌어도 알아서 맞춘다.
       꼭짓점마다 두 번 찍는다: 한 번 찍고, 손목(wrist_roll)을 크게 (90 도 이상) 돌려 같은 꼭짓점을 한 번 더.
    3. 끝나면 headCam_to_Base.py 에 그대로 넣을 --frame base --origin ... --yaw ... 를 출력한다.

계산
    리더 관절 값을 시뮬레이션 SO-101 (models/so101) 에 넣어 gripper 링크의 자세를 구한다.
    리더는 그리퍼 대신 손잡이라 실제로 닿는 끝이 어디인지 모른다. 그래서 끝 위치(gripper 링크 기준)를 모르는 값으로 두고,
    "같은 꼭짓점을 찍은 두 자세는 끝이 같은 점" + "네 꼭짓점은 상판 높이(베이스 Z=0) 평면 위" 를 최소제곱으로 푼다.
    그다음 네 꼭짓점에 보드 모양을 맞춘다. 남는 오차(찍기마다, 꼭짓점마다)가 이 측정의 정확도다.

관절 값 -> 각도는 두 가지로 계산해 비교하고, 오차가 작은 쪽을 쓴다.
    deg    : 모터 원시 값 그대로 (4096 = 360 도). 0 도 = 캘리브레이션 때 잡은 가운데 자세 (2047)
    linear : teleop/mapping.py 와 같은 정규화값 선형 매핑 (캘리 범위 양 끝 = 시뮬 관절 범위 양 끝)
찍은 원시 값은 extrinsic/touch_raw.json 에 남기고, --replay 로 다시 찍지 않고 계산만 다시 할 수 있다.
"""

import argparse
import itertools
import json
import math
import sys

import mujoco
import numpy as np

from cali_common import HERE, parse_board

# teleop 패키지 (src/teleop) 를 쓴다
sys.path.insert(0, str(HERE.parent))
from lerobot.teleoperators.so_leader import SOLeader  # noqa: E402

from teleop.constants import JOINT_NAMES  # noqa: E402
from teleop.leader import SO101LeaderConfig  # noqa: E402
from teleop.mapping import normalized_to_rad  # noqa: E402

SCENE = HERE.parents[1] / "models" / "scene" / "scene_empty.xml"
RAW_FILE = HERE / "extrinsic" / "touch_raw.json"
PREFIX = "so101_"
TOOL_BODY = PREFIX + "gripper"
# 한 점에서 읽어 평균 낼 횟수
READS = 20
# 캘리브레이션 가운데 자세의 원시 값, 한 바퀴 틱 수 (STS3215)
MID_TICK, TICKS = 2047, 4096
# teleop/leader_mj/config.py 의 gripper_closed_safe 와 같게 (그리퍼 각도는 끝 위치에 안 쓰인다)
GRIPPER_CLOSED = -0.1
LABELS = ["원점 (노란 점)", "두 번째 귀퉁이", "세 번째 귀퉁이", "네 번째 귀퉁이"]


def to_rad_deg(raw: dict[str, float], calib: dict) -> list[float]:
    return [(raw[j] - MID_TICK) * 2 * math.pi / TICKS for j in JOINT_NAMES]


def to_rad_linear(raw: dict[str, float], calib: dict) -> list[float]:
    out = []
    for j in JOINT_NAMES:
        lo, hi = calib[j]["range_min"], calib[j]["range_max"]
        norm = (raw[j] - lo) / (hi - lo) * (100 if j == "gripper" else 200) - (0 if j == "gripper" else 100)
        out.append(normalized_to_rad(j, norm, GRIPPER_CLOSED))
    return out


MAPPINGS = {"deg": to_rad_deg, "linear": to_rad_linear}


class ToolFK:
    """관절 각도(rad) -> 시뮬레이션 모델 gripper 링크의 자세 (R, p), 베이스 기준."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.qadr = [int(self.model.joint(PREFIX + j).qposadr[0]) for j in JOINT_NAMES]
        self.tool = self.model.body(TOOL_BODY).id
        self.base = self.model.body(PREFIX + "base").id

    def __call__(self, q: list[float]) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        Rb = self.data.xmat[self.base].reshape(3, 3)
        R = Rb.T @ self.data.xmat[self.tool].reshape(3, 3)
        return R, Rb.T @ (self.data.xpos[self.tool] - self.data.xpos[self.base])


def read_raw(arm: SOLeader) -> dict[str, float]:
    """원시 Present_Position 을 READS 번 읽어 평균."""
    reads = [arm.bus.sync_read("Present_Position", normalize=False, num_retry=2) for _ in range(READS)]
    spread = max(max(r[j] for r in reads) - min(r[j] for r in reads) for j in JOINT_NAMES)
    if spread > 10:
        print(f"  (읽는 동안 관절이 {spread} 틱 흔들렸다. 팔을 가만히 잡고 다시 찍는 게 좋다)")
    return {j: float(np.mean([r[j] for r in reads])) for j in JOINT_NAMES}


def record(args) -> dict:
    arm = SOLeader(SO101LeaderConfig(port=args.port, id=args.id))
    arm.connect()
    touches = []
    try:
        calib = {j: {"range_min": c.range_min, "range_max": c.range_max} for j, c in arm.calibration.items()}
        for label in LABELS:
            input(f"끝을 {label} 꼭짓점에 대고 Enter...")
            first = read_raw(arm)
            input("  손목을 크게 (90 도 이상) 돌려 같은 꼭짓점에 다시 대고 Enter...")
            second = read_raw(arm)
            turn = abs(second["wrist_roll"] - first["wrist_roll"]) * 360 / TICKS
            if turn < 60:
                print(f"  (손목이 {turn:.0f} 도만 돌았다. 많이 돌릴수록 끝 위치를 잘 구한다)")
            touches.append([first, second])
    finally:
        arm.disconnect()
    saved = {"board": args.board, "square": args.square, "calibration": calib, "touches": touches}
    RAW_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print("원시 값 저장:", RAW_FILE)
    return saved


def board_corners(cols: int, rows: int, square: float) -> np.ndarray:
    """보드 좌표 (m): 원점 = 노란 점, X 끝 (빨간 선), Y 끝 (초록 선), 대각선."""
    w, h = (cols - 1) * square, (rows - 1) * square
    return np.array([[0, 0, 0], [w, 0, 0], [0, h, 0], [w, h, 0]], float)


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dst ≈ R src + t 인 회전 R, 이동 t (Kabsch)."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - sc).T @ (dst - dc))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, dc - R @ sc


def evaluate(saved: dict, to_rad, fk: ToolFK, cols: int, rows: int, square: float) -> dict:
    calib = saved["calibration"]
    # 모르는 값: 끝 위치 e (gripper 링크 기준 3), 꼭짓점 k 의 위치 (x, y) (z = 0)
    # 찍기마다 R e + p = (x_k, y_k, 0)  ->  R e - [x_k, y_k, 0] = -p
    n = len(saved["touches"])
    A, b, owner = [], [], []
    for k, pair in enumerate(saved["touches"]):
        for raw in pair:
            R, p = fk(to_rad(raw, calib))
            M = np.zeros((3, 3 + 2 * n))
            M[:, :3] = R
            M[0, 3 + 2 * k] = M[1, 4 + 2 * k] = -1
            A.append(M)
            b.append(-p)
            owner.append(k)
    A, b = np.vstack(A), np.concatenate(b)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    touch_err = np.linalg.norm((A @ sol - b).reshape(-1, 3), axis=1)
    corners = np.column_stack([sol[3:].reshape(n, 2), np.zeros(n)])

    # 첫 점은 노란 점으로 두고, 나머지 세 점이 보드의 어느 귀퉁이인지는 가장 잘 맞는 순서로 고른다
    board = board_corners(cols, rows, square)
    best = None
    for order in itertools.permutations(range(1, n)):
        pts = corners[[0, *order]]
        R, t = fit_rigid(board, pts)
        err = np.linalg.norm(pts - (board @ R.T + t), axis=1)
        if best is None or np.sqrt(np.mean(err ** 2)) < best["score"]:
            best = {"order": order, "corner_err": err, "R": R, "t": t, "score": float(np.sqrt(np.mean(err ** 2)))}
    best.update(tip=sol[:3], touch_err=touch_err, owner=owner,
                yaw=math.degrees(math.atan2(best["R"][1, 0], best["R"][0, 0])))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=SO101LeaderConfig.port, help="리더 포트 (기본 ~/dev/so101_leader)")
    ap.add_argument("--id", default="leader", help="리더 캘리브레이션 파일 이름 (lerobot-calibrate 의 --teleop.id)")
    ap.add_argument("--board", default="9x6", help="체커보드 안쪽 꼭짓점 수, 가로x세로 (headCam_to_Base.py 와 같게)")
    ap.add_argument("--square", type=float, default=0.025, help="한 칸의 한 변 길이 m (headCam_to_Base.py 와 같게)")
    ap.add_argument("--flip", action="store_true", help="headCam_to_Base.py 에서 --flip 을 쓸 거면 똑같이 준다 (출력 명령에 붙는다)")
    ap.add_argument("--replay", action="store_true", help=f"다시 찍지 않고 {RAW_FILE.name} 로 계산만 다시 한다")
    args = ap.parse_args()
    cols, rows = parse_board(args.board)

    saved = json.loads(RAW_FILE.read_text(encoding="utf-8")) if args.replay else record(args)
    if isinstance(saved["touches"], dict):  # 예전 형식 (원점, X 끝, Y 끝, 대각선 이름으로 저장)
        saved["touches"] = list(saved["touches"].values())
    fk = ToolFK()
    results = {name: evaluate(saved, fn, fk, cols, rows, args.square) for name, fn in MAPPINGS.items()}

    names = ["X 끝", "Y 끝", "대각선"]
    print("\n| 매핑 | 꼭짓점 오차 (원점 / 2 / 3 / 4번째) | 평균 | 같은 점 두 번 찍기 오차 | 구한 끝 위치 (gripper 링크 기준) |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        by_touch = [r["corner_err"][[0, *r["order"]].index(k)] for k in range(len(r["order"]) + 1)]
        corner = " / ".join(f"{v * 1000:.0f}" for v in by_touch)
        touch = " / ".join(f"{v * 1000:.0f}" for v in r["touch_err"])
        print(f"| {name} | {corner} mm | {r['score'] * 1000:.1f} mm | {touch} mm "
              f"| {' '.join(f'{v * 1000:.0f}' for v in r['tip'])} mm |")
    best = min(results, key=lambda n: results[n]["score"])
    r = results[best]
    order = [names[i - 1] for i in sorted(range(1, 4), key=lambda i: r["order"].index(i))]
    print(f"\n오차가 작은 쪽: {best}.  찍은 순서로 보면 2, 3, 4번째 = {', '.join(order)}")
    print("  꼭짓점 오차 평균이 5 mm 안쪽이면 쓸 만하다. 한 점만 크면 그 점을 다시 찍는다.")
    print("  두 번 찍기 오차가 크면 (10 mm 넘게) 리더 관절 각도 자체가 부정확한 것이다 (캘리 가운데 자세, 기어 유격).")
    o = r["t"]
    print("\nheadCam_to_Base.py 에 넣을 명령:")
    print(f"  .venv/bin/python src/camera-cali/headCam_to_Base.py --board {args.board} --square {args.square} "
          f"--frame base --origin {o[0]:.4f} {o[1]:.4f} 0 --yaw {r['yaw']:.1f}" + (" --flip" if args.flip else ""))


if __name__ == "__main__":
    main()
