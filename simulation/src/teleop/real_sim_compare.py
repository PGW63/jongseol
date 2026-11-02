"""실물 텔레옵 데이터의 액션을 시뮬레이션 SO-101 에 그대로 넣고, 관절 움직임이 실물과 얼마나 다른지 RMSE 로 잰다.

    .venv/bin/python src/teleop/real_sim_compare.py
    .venv/bin/python src/teleop/real_sim_compare.py --data ../IDM/data/cache/teleop_1015/arrays.npz --episode 81

입력은 IDM prepare_data.py 가 만든 arrays.npz (action, state, episode_index). 단위는 LeRobot 정규화값
(팔 -100~100, 그리퍼 0~100) 이고, 시뮬레이션 관절 각도도 mapping.py 로 같은 단위로 바꿔서 비교한다.

진행
    1) raw       : 액션을 그대로 시뮬레이션에 넣는다. 첫 자세는 실물 state[0].
                   frame t 에서 시뮬레이션 관절을 기록하고 action[t] 를 넣어 1/fps 초 진행 (LeRobot 녹화 순서와 같다).
    2) 캘리브레이션 보정 : 관절마다 median(실물 state - 시뮬레이션) 을 영점 차이로 보고, 액션에 더해 다시 재생한다.
                   리더/팔로워 영점이 다르면 실물은 액션에서 일정하게 어긋난 자세로 가는데, 시뮬레이션은 리더값을 그대로 쓰기 때문.
    빈 월드(scene_empty) 에서 돌리므로 물체 접촉은 없다. 그리퍼는 실물이 물체를 쥐면 덜 닫혀서 오차가 커진다.
    그래서 실물 그리퍼가 raw 시뮬레이션보다 GRASP_THRESH 이상 더 열린 상태가 GRASP_MIN_S 초 넘게 이어지면 "쥔 구간" 으로 보고,
    (그보다 짧은 건 빠르게 닫을 때 실물 서보가 2~3 frame 늦게 따라와서 생기는 것이라 쥔 구간이 아니다)
    그리퍼 영점은 쥔 구간을 뺀 frame 으로 잡고, 쥔 구간을 뺀 그리퍼 RMSE 도 따로 낸다.

출력: results/real_sim_compare/ 에 real_vs_sim.png(관절별 한 에피소드의 sim/real 궤적, 전체 에피소드 real - sim 분포), rmse.json
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from teleop.constants import JOINT_NAMES, MUJOCO_JOINT_RANGE
from teleop.mapping import normalized_to_rad, rad_to_normalized

ROOT = Path(__file__).resolve().parents[2]
EMPTY_SCENE = ROOT / "models" / "scene" / "scene_empty.xml"
DEFAULT_DATA = ROOT.parent / "IDM" / "data" / "cache" / "teleop_1015" / "arrays.npz"
RESULTS = ROOT / "results" / "real_sim_compare"

PREFIX = "so101_"
FPS = 30
# teleop/leader_mj/config.py 의 gripper_closed_safe 와 같게
GRIPPER_CLOSED = -0.1
# 실물 그리퍼가 시뮬레이션보다 이만큼(정규화 단위) 더 열려 있으면 물체에 막혀 있다고 본다
GRASP_THRESH = 3.0
# 이 시간(초)보다 짧게 열려 있는 건 닫을 때의 지연이라 쥔 구간에서 뺀다
GRASP_MIN_S = 0.5
GRIP = JOINT_NAMES.index("gripper")
NORM_RANGE = [(0.0, 100.0) if j == "gripper" else (-100.0, 100.0) for j in JOINT_NAMES]

SIM_COLOR = "#2a78d6"
REAL_COLOR = "#eb6834"
ERR_COLOR = "#52514e"
GRID_COLOR = "#e4e2dc"
INK2 = "#52514e"


def deg_per_unit(j: str) -> float:
    """정규화값 1 이 시뮬레이션에서 몇 도인지 (mapping.py 의 선형 매핑 기준)."""
    lo, hi = MUJOCO_JOINT_RANGE[j]
    if j == "gripper":
        return np.degrees(hi - GRIPPER_CLOSED) / 100.0
    return np.degrees(hi - lo) / 200.0


# ---------------------------------------------------------------- 재생

_sim = None


def _init_sim():
    global _sim
    model = mujoco.MjModel.from_xml_path(str(EMPTY_SCENE))
    data = mujoco.MjData(model)
    act = [model.actuator(PREFIX + j).id for j in JOINT_NAMES]
    qadr = [int(model.joint(PREFIX + j).qposadr[0]) for j in JOINT_NAMES]
    vadr = [int(model.joint(PREFIX + j).dofadr[0]) for j in JOINT_NAMES]
    steps = max(1, round(1.0 / FPS / model.opt.timestep))
    _sim = (model, data, act, qadr, vadr, steps)


def _to_rad(norm: np.ndarray) -> np.ndarray:
    return np.array([normalized_to_rad(j, v, GRIPPER_CLOSED) for j, v in zip(JOINT_NAMES, norm)])


def replay(args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """액션 [T, 6] 을 재생해 각 frame 의 시뮬레이션 관절값 [T, 6] (정규화 단위) 을 돌려준다."""
    actions, state0 = args
    model, data, act, qadr, vadr, steps = _sim
    mujoco.mj_resetData(model, data)
    data.qpos[qadr] = _to_rad(state0)
    data.qvel[vadr] = 0.0
    data.ctrl[act] = _to_rad(actions[0])
    mujoco.mj_forward(model, data)
    out = np.empty_like(actions)
    for t, a in enumerate(actions):
        out[t] = [rad_to_normalized(j, data.qpos[adr], GRIPPER_CLOSED) for j, adr in zip(JOINT_NAMES, qadr)]
        data.ctrl[act] = _to_rad(a)
        for _ in range(steps):
            mujoco.mj_step(model, data)
    return out


def replay_all(pool: Pool, actions: list[np.ndarray], states: list[np.ndarray], offset: np.ndarray) -> list[np.ndarray]:
    lo = np.array([r[0] for r in NORM_RANGE])
    hi = np.array([r[1] for r in NORM_RANGE])
    jobs = [(np.clip(a + offset, lo, hi), s[0]) for a, s in zip(actions, states)]
    return pool.map(replay, jobs)


def grasp_mask(real_grip: np.ndarray, sim_grip: np.ndarray) -> np.ndarray:
    """실물 그리퍼가 물체에 막혀 시뮬레이션보다 덜 닫힌 frame (GRASP_MIN_S 초 넘게 이어진 구간만)."""
    m = real_grip - sim_grip > GRASP_THRESH
    edges = np.diff(np.r_[0, m.astype(int), 0])
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if b - a < GRASP_MIN_S * FPS:
            m[a:b] = False
    return m


def rmse(err: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(err**2, axis=0))


# ---------------------------------------------------------------- 그래프


def _style(ax):
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c7bf")
    ax.tick_params(colors=INK2, labelsize=9)


def plot_error(err: np.ndarray, grasp: np.ndarray, ep: int, real_ep: np.ndarray, sim_ep: np.ndarray,
               grasp_ep: np.ndarray, r: np.ndarray, path: Path) -> None:
    """관절마다 한 에피소드의 sim / real 궤적, 오른쪽에 전체 에피소드의 real - sim 분포.

    sim 은 영점 보정 후. 그리퍼는 쥔 구간을 회색으로 칠하고 RMSE / 분포에서 뺀다.
    """
    t = np.arange(len(real_ep)) / FPS
    err_ep = real_ep - sim_ep
    n = len(JOINT_NAMES)
    fig = plt.figure(figsize=(15, 2.2 * n))
    gs = fig.add_gridspec(n, 2, width_ratios=[3.2, 1], hspace=0.35, wspace=0.08,
                          left=0.06, right=0.99, top=0.95, bottom=0.06)
    bins = np.linspace(-15, 15, 61)
    first = hist_x = None
    for i, j in enumerate(JOINT_NAMES):
        keep = ~grasp if i == GRIP else np.ones(len(err), bool)
        keep_ep = ~grasp_ep if i == GRIP else np.ones(len(err_ep), bool)
        e = err[keep, i]

        ax = fig.add_subplot(gs[i, 0], sharex=first)
        first = first or ax
        if i == GRIP:
            ax.fill_between(t, 0, 1, where=grasp_ep, color=GRID_COLOR, transform=ax.get_xaxis_transform(),
                            linewidth=0, label="grasp (excluded)")
        ax.plot(t, sim_ep[:, i], color=SIM_COLOR, linewidth=1.5, label="sim")
        ax.plot(t, real_ep[:, i], color=REAL_COLOR, linewidth=1.5, label="real")
        ax.set_ylabel(j, fontsize=9)
        ax.set_title(f"ep {ep}  RMSE {rmse(err_ep[keep_ep, i]):.2f}", fontsize=9, loc="right", color=INK2)
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        _style(ax)
        if i in (0, GRIP):
            ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)

        hx = fig.add_subplot(gs[i, 1], sharex=hist_x)
        hist_x = hist_x or hx
        hx.hist(np.clip(e, bins[0], bins[-1]), bins=bins, color=ERR_COLOR, edgecolor="white", linewidth=0.5)
        hx.axvline(0, color="#8a8880", linewidth=0.8)
        hx.set_yticks([])
        hx.set_title(f"all eps real − sim  RMSE {r[i]:.2f} ({r[i] * deg_per_unit(j):.1f}°)  mean {e.mean():+.2f}",
                     fontsize=9, loc="right", color=INK2)
        if i == n - 1:
            hx.set_xlabel("real − sim (normalized units)")
        else:
            hx.tick_params(labelbottom=False)
        _style(hx)
    ax.set_xlabel("time (s)")
    arm = np.delete(r, GRIP)
    fig.suptitle(f"Real vs sim joints (sim replays real actions, zero offset corrected)   "
                 f"mean RMSE {np.mean(r):.2f}  (arm {arm.mean():.2f}, gripper w/o grasp {r[GRIP]:.2f})",
                 x=0.01, y=0.995, ha="left", fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="IDM 캐시의 arrays.npz")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--episode", type=int, default=18, help="그래프 왼쪽에 시간축으로 그릴 에피소드")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data)
    ep_idx = d["episode_index"]
    eps = np.unique(ep_idx)
    actions = [d["action"][ep_idx == e].astype(np.float64) for e in eps]
    states = [d["state"][ep_idx == e].astype(np.float64) for e in eps]
    print(f"{len(eps)} episodes, {len(ep_idx)} frames")

    with Pool(args.workers, initializer=_init_sim) as pool:
        sim_raw = replay_all(pool, actions, states, np.zeros(len(JOINT_NAMES)))
        err_raw = np.concatenate([s - r for s, r in zip(states, sim_raw)])
        grasp = np.concatenate([grasp_mask(s[:, GRIP], r[:, GRIP]) for s, r in zip(states, sim_raw)])
        offset = np.median(err_raw, axis=0)
        offset[GRIP] = np.median(err_raw[~grasp, GRIP])
        print("offset:", np.round(offset, 2))
        sim_cor = replay_all(pool, actions, states, offset)
    err_cor = np.concatenate([s - r for s, r in zip(states, sim_cor)])

    r_raw, r_cor = rmse(err_raw), rmse(err_cor)
    g_raw, g_cor = rmse(err_raw[~grasp, GRIP]), rmse(err_cor[~grasp, GRIP])
    # 팔 5개 + 쥔 구간을 뺀 그리퍼
    m_raw = float(np.mean(np.r_[np.delete(r_raw, GRIP), g_raw]))
    m_cor = float(np.mean(np.r_[np.delete(r_cor, GRIP), g_cor]))
    starts = np.cumsum([0] + [len(s) for s in states])
    grasp_ep = [grasp[a:b] for a, b in zip(starts[:-1], starts[1:])]
    dpu = np.array([deg_per_unit(j) for j in JOINT_NAMES])
    res = {
        "data": str(args.data),
        "unit": "normalized (arm -100~100, gripper 0~100)",
        "joints": JOINT_NAMES,
        "deg_per_unit": dpu.tolist(),
        "offset": offset.tolist(),
        "rmse_raw": r_raw.tolist(),
        "rmse_corrected": r_cor.tolist(),
        "rmse_raw_deg": (r_raw * dpu).tolist(),
        "rmse_corrected_deg": (r_cor * dpu).tolist(),
        "rmse_mean_raw": float(r_raw.mean()),
        "rmse_mean_corrected": float(r_cor.mean()),
        "grasp_thresh": GRASP_THRESH,
        "grasp_min_s": GRASP_MIN_S,
        "grasp_frame_ratio": float(grasp.mean()),
        "gripper_rmse_no_grasp_raw": float(g_raw),
        "gripper_rmse_no_grasp_corrected": float(g_cor),
        "gripper_rmse_no_grasp_corrected_deg": float(g_cor * dpu[GRIP]),
        "rmse_mean_no_grasp_raw": m_raw,
        "rmse_mean_no_grasp_corrected": m_cor,
        "per_episode": {
            int(e): {
                "rmse_raw": rmse(s - a).tolist(),
                "rmse_corrected": rmse(s - b).tolist(),
                "mean_raw": float(rmse(s - a).mean()),
                "mean_corrected": float(rmse(s - b).mean()),
            }
            for e, s, a, b in zip(eps, states, sim_raw, sim_cor)
        },
    }
    (args.out / "rmse.json").write_text(json.dumps(res, indent=2))

    print("\n| joint | offset | RMSE raw | RMSE corrected | corrected (deg) |\n|---|---|---|---|---|")
    for i, j in enumerate(JOINT_NAMES):
        print(f"| {j} | {offset[i]:+.2f} | {r_raw[i]:.2f} | {r_cor[i]:.2f} | {r_cor[i] * dpu[i]:.2f} |")
    print(f"| gripper (쥔 구간 제외, {100 * (1 - grasp.mean()):.0f}% frame) | {offset[GRIP]:+.2f} | {g_raw:.2f} | {g_cor:.2f} | {g_cor * dpu[GRIP]:.2f} |")
    print(f"| mean | | {r_raw.mean():.2f} | {r_cor.mean():.2f} | |")
    print(f"| mean (그리퍼 쥔 구간 제외) | | {m_raw:.2f} | {m_cor:.2f} | |")

    k = int(np.searchsorted(eps, args.episode))
    if k >= len(eps) or eps[k] != args.episode:
        raise SystemExit(f"에피소드 {args.episode} 가 데이터에 없습니다")
    r_plot = r_cor.copy()
    r_plot[GRIP] = g_cor
    plot_error(err_cor, grasp, args.episode, states[k], sim_cor[k], grasp_ep[k], r_plot, args.out / "real_vs_sim.png")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
