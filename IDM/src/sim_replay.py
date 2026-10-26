"""물리 검증: IDM 이 라벨링한 액션을 SimStudio MuJoCo 장면(simple_pick)에서 실제로 실행해
GT 액션 실행 결과와 엔드이펙터(wrist_site) 궤적을 비교한다.

sim 실험 재현용. 장면은 so101-simstudio 저장소의 SO101/scenes/simple_pick/scene.xml 이다.
MuJoCo 가 있는 simulation/.venv 로 실행 (evaluate.py 가 만든 pred_test.npz 가 먼저 있어야 한다):
  ../simulation/.venv/bin/python src/sim_replay.py --run runs/sim_top_resnet --scene <simstudio>/SO101/scenes/simple_pick/scene.xml
"""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PHYSICS_DT, RECORD_FPS = 1 / 360, 20  # configs/so101_mujoco_pick_leader.yaml


class Arm:
    def __init__(self, scene):
        self.m = mj.MjModel.from_xml_path(str(scene))
        self.m.opt.timestep = PHYSICS_DT
        self.d = mj.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
        self.act = [mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS]
        self.site = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_SITE, "wrist_site")
        self.n_sub = int((1 / RECORD_FPS) / PHYSICS_DT)

    def rollout(self, q0, actions):
        mj.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = q0
        self.d.ctrl[self.act] = q0
        mj.mj_forward(self.m, self.d)
        ee, q = [], []
        for a in actions:
            self.d.ctrl[self.act] = a
            for _ in range(self.n_sub):
                mj.mj_step(self.m, self.d)
            ee.append(self.d.site_xpos[self.site].copy())
            q.append(self.d.qpos[self.qadr].copy())
        return np.array(ee), np.array(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="test")
    ap.add_argument("--scene", required=True, help="simple_pick scene.xml 경로")
    args = ap.parse_args()
    run = ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    raw = ROOT / json.loads((ROOT / cfg["data"] / "meta.json").read_text())["source"]
    df = pd.concat([pd.read_parquet(p) for p in sorted((raw / "data").rglob("*.parquet"))]).sort_values("index")
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)  # [N, 15], ee = [12:15]

    preds = np.load(run / f"pred_{args.tag}.npz")
    arm = Arm(args.scene)
    per_ep, rows = {}, []
    for e in cfg["test_eps"]:
        s = int(preds[f"ep{e}_start"][0])
        gt, pr = preds[f"ep{e}_gt"].astype(np.float64), preds[f"ep{e}_pred"].astype(np.float64)
        q0 = state[s, :6]
        ee_gt, _ = arm.rollout(q0, gt)
        ee_pr, _ = arm.rollout(q0, pr)
        ee_rec = state[s + 1 : s + 1 + len(gt), 12:15]
        err = np.linalg.norm(ee_pr - ee_gt, axis=1) * 100
        sanity = np.linalg.norm(ee_gt - ee_rec, axis=1) * 100
        per_ep[e] = {"ee_err_cm_mean": err.mean(), "ee_err_cm_p95": np.percentile(err, 95),
                     "ee_err_cm_final": err[-1], "gt_replay_vs_recorded_cm": sanity.mean()}
        rows.append(err)
        np.savez(run / f"replay_ep{e}.npz", ee_gt=ee_gt, ee_pred=ee_pr, ee_rec=ee_rec)
    allerr = np.concatenate(rows)
    res = {"per_episode": per_ep, "ee_err_cm_mean": allerr.mean(), "ee_err_cm_median": np.median(allerr),
           "ee_err_cm_p95": np.percentile(allerr, 95),
           "gt_replay_vs_recorded_cm": float(np.mean([v["gt_replay_vs_recorded_cm"] for v in per_ep.values()]))}
    res = json.loads(json.dumps(res, default=float))
    (run / f"replay_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_episode"}, indent=1))


if __name__ == "__main__":
    main()
