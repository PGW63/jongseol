"""학습된 IDM 으로 held-out 에피소드 전체를 라벨링(DreamGen 의 dump_idm_actions 방식:
겹치지 않는 H-step 윈도우)하고 GT 액션과 비교한다.

출력: runs/<run>/eval_<tag>.json, traj_<tag>.png, pred_<tag>.npz
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import ActionNormalizer, Cache
from models import build_model

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_model(run):
    cfg = json.loads((run / "config.json").read_text())
    kw = {}
    if cfg["model"].startswith("dreamgen"):
        # 가중치는 아래에서 덮어쓰므로 pretrained 로딩은 생략
        kw = dict(single_embodiment=cfg.get("single_embodiment", False))
        name = "dreamgen_scratch"
    else:
        name = cfg["model"]
    model = build_model(name, horizon=cfg["horizon"], **kw)
    sd = torch.load(run / "model_bf16.pt", map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    if cfg["model"].startswith("dreamgen"):
        model = model.to(torch.bfloat16)  # 추론은 bf16 가중치로 (학습 중 GPU 공유 가능하도록)
    return model.cuda().eval(), cfg


@torch.no_grad()
def relabel_episode(model, cache, s, t, H, normalizer, bs=16):
    starts = list(range(s, t - H, H))
    preds = []
    for i in range(0, len(starts), bs):
        idx = starts[i : i + bs]
        fr = torch.from_numpy(np.stack([np.stack([cache.frames[j], cache.frames[j + H]]) for j in idx])).cuda()
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(model.predict(fr).cpu().numpy())
    pred = normalizer.denorm(np.concatenate(preds)).reshape(-1, len(JOINTS))
    return pred, len(starts) * H


def metrics(pred, gt, rng):
    err = pred - gt
    rmse = np.sqrt((err**2).mean(0))
    ss_res = (err**2).sum(0)
    ss_tot = ((gt - gt.mean(0)) ** 2).sum(0)
    # GT 가 거의 상수인 관절(OOD 녹화의 wrist_roll 등)은 R² 정의가 무의미 -> NaN
    r2 = np.where(gt.std(0) > 1e-3 * rng, 1 - ss_res / np.maximum(ss_tot, 1e-9), np.nan)
    return {
        "rmse": rmse.tolist(),
        "rmse_mean": float(rmse.mean()),
        "nrmse_pct": (100 * rmse / rng).tolist(),
        "nrmse_pct_mean": float((100 * rmse / rng).mean()),
        "r2": [None if np.isnan(v) else float(v) for v in r2],
        "r2_mean_arm": float(np.nanmean(r2[:5])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=None, help="기본: 학습 데이터의 test 에피소드")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--plot-episode", type=int, default=None)
    args = ap.parse_args()

    run = Path(args.run)
    model, cfg = load_model(run)
    H = cfg["horizon"]
    normalizer = ActionNormalizer(**cfg["normalizer"])
    rng = normalizer.hi - normalizer.lo
    cache = Cache(args.data or cfg["data"])
    episodes = cfg["test_eps"] if args.data is None else cache.episodes

    train_cache = Cache(cfg["data"])
    train_mean = train_cache.action[np.isin(train_cache.ep, cfg["train_eps"])].mean(0)

    all_pred, all_gt, all_interp, saved = [], [], [], {}
    torch.cuda.reset_peak_memory_stats()
    t0, n_win = time.time(), 0
    for e in episodes:
        s, t = cache.episode_range(e)
        pred, n = relabel_episode(model, cache, s, t, H, normalizer)
        gt = cache.action[s : s + n]
        # 특권(privileged) 참고 baseline: 두 프레임 시점의 GT 관절 상태를 선형 보간
        st = cache.state
        interp = np.concatenate([
            st[j] + (st[j + H] - st[j]) * (np.arange(H)[:, None] / H) for j in range(s, s + n, H)])
        all_pred.append(pred), all_gt.append(gt), all_interp.append(interp)
        saved[f"ep{e}_pred"], saved[f"ep{e}_gt"], saved[f"ep{e}_start"] = pred, gt, np.array([s])
        n_win += n // H
    elapsed = time.time() - t0
    pred, gt, interp = map(np.concatenate, (all_pred, all_gt, all_interp))

    to_deg = 180 / np.pi if np.abs(gt).max() < 4 else 1.0  # sim 은 rad -> deg 로 보고
    res = {
        "run": str(run), "data": args.data or cfg["data"], "episodes": episodes, "frames": len(gt),
        "unit": "deg" if to_deg != 1 else "raw(-100~100)",
        "idm": metrics(pred * to_deg, gt * to_deg, rng * to_deg),
        "baseline_mean_action": metrics(np.broadcast_to(train_mean, gt.shape) * to_deg, gt * to_deg, rng * to_deg),
        "baseline_state_interp_oracle": metrics(interp * to_deg, gt * to_deg, rng * to_deg),
        "infer_windows_per_s": n_win / elapsed,
        "infer_peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (run / f"eval_{args.tag}.json").write_text(json.dumps(res, indent=2))
    np.savez(run / f"pred_{args.tag}.npz", **saved)

    e = args.plot_episode if args.plot_episode is not None else episodes[0]
    p, g = saved[f"ep{e}_pred"] * to_deg, saved[f"ep{e}_gt"] * to_deg
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
    tt = np.arange(len(g)) / cache.meta["fps"]
    for j, ax in enumerate(axs.flat):
        ax.plot(tt, g[:, j], color="#222", lw=1.6, label="GT action")
        ax.plot(tt, p[:, j], color="#d9480f", lw=1.2, label="IDM pseudo-label")
        for b in range(0, len(g), H):
            ax.axvline(b / cache.meta["fps"], color="#ccc", lw=0.4, zorder=0)
        ax.set_title(f"{JOINTS[j]}  (RMSE {np.sqrt(((p[:, j]-g[:, j])**2).mean()):.2f} {res['unit']})", fontsize=10)
        ax.grid(alpha=0.25)
        if j >= 3:
            ax.set_xlabel("time [s]")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{run.name} | {args.tag} episode {e} | mean RMSE {res['idm']['rmse_mean']:.2f} {res['unit']}, "
                 f"R²(arm) {res['idm']['r2_mean_arm']:.3f}")
    fig.tight_layout()
    fig.savefig(run / f"traj_{args.tag}.png", dpi=110)
    print(json.dumps({k: res[k] for k in ["frames", "unit", "infer_windows_per_s"]}),
          "\nIDM", json.dumps(res["idm"], default=lambda x: None),
          "\nmean", round(res["baseline_mean_action"]["rmse_mean"], 3),
          "| interp-oracle", round(res["baseline_state_interp_oracle"]["rmse_mean"], 3))


if __name__ == "__main__":
    main()
