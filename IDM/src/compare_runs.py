"""같은 test 에피소드로 평가한 run 여러 개를 비교한다 (evaluate.py 결과 pred_<tag>.npz, eval_<tag>.json 필요).

출력 (--out 폴더, 파일 이름 앞에 --name):
  <name>_ep<e>.png   에피소드별 관절 6개 궤적: GT vs 각 run 의 IDM 라벨 (제목에 관절별 RMSE)
  <name>_rmse.png    test 에피소드 전체 관절별 RMSE 막대 비교
  <name>.json        관절별 RMSE 표 (run x joint), 에피소드별 RMSE
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import JOINTS

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # 범주형 순서 고정 (run 순서대로)


def rmse(p, g):
    return np.sqrt(((p - g) ** 2).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None, help="범례 이름 (기본: run 폴더 이름)")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--episodes", nargs="+", type=int, default=None, help="궤적을 그릴 에피소드 (기본: test 에피소드 앞 3개)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="compare")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    labels = args.labels or [r.name for r in runs]
    preds = [np.load(r / f"pred_{args.tag}.npz") for r in runs]
    evals = [json.loads((r / f"eval_{args.tag}.json").read_text()) for r in runs]
    episodes = evals[0]["episodes"]
    assert all(e["episodes"] == episodes for e in evals), "run 마다 test 에피소드가 다르다"
    fps = json.loads((Path(json.loads((runs[0] / "config.json").read_text())["data"]) / "meta.json").read_text())["fps"]
    unit = evals[0]["unit"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 에피소드별 / 전체 RMSE
    per_ep = {lab: {} for lab in labels}
    for lab, pr in zip(labels, preds):
        for e in episodes:
            per_ep[lab][e] = rmse(pr[f"ep{e}_pred"], pr[f"ep{e}_gt"]).tolist()
    total = {lab: ev["idm"]["rmse"] for lab, ev in zip(labels, evals)}
    summary = {"unit": unit, "episodes": episodes, "joints": JOINTS,
               "rmse": total, "rmse_mean": {lab: float(np.mean(v)) for lab, v in total.items()},
               "r2_mean_arm": {lab: ev["idm"]["r2_mean_arm"] for lab, ev in zip(labels, evals)},
               "rmse_per_episode": per_ep}
    (out / f"{args.name}.json").write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False})

    # 1) 에피소드별 궤적
    for e in args.episodes or episodes[:3]:
        g = preds[0][f"ep{e}_gt"]
        tt = np.arange(len(g)) / fps
        fig, axs = plt.subplots(2, 3, figsize=(16, 7), sharex=True, facecolor=SURFACE)
        for j, ax in enumerate(axs.flat):
            ax.set_facecolor(SURFACE)
            ax.plot(tt, g[:, j], color=INK, lw=2, label="GT action")
            parts = []
            for k, (lab, pr) in enumerate(zip(labels, preds)):
                p = pr[f"ep{e}_pred"]
                ax.plot(tt, p[:, j], color=SERIES[k], lw=1.2, alpha=0.9, label=f"IDM ({lab})")
                parts.append(f"{lab} {np.sqrt(((p[:, j] - g[:, j]) ** 2).mean()):.2f}")
            ax.set_title(f"{JOINTS[j]}\nRMSE  " + "  |  ".join(parts), fontsize=10)
            ax.grid(alpha=0.2)
            if j >= 3:
                ax.set_xlabel("time [s]")
        axs[0, 0].legend(fontsize=8, frameon=False)
        means = "  |  ".join(f"{lab} {np.mean(per_ep[lab][e]):.2f}" for lab in labels)
        fig.suptitle(f"test episode {e}  ·  mean RMSE ({unit}): {means}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"{args.name}_ep{e}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)

    # 2) 관절별 RMSE 막대
    names = JOINTS + ["mean"]
    x = np.arange(len(names))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, lab in enumerate(labels):
        v = total[lab] + [float(np.mean(total[lab]))]
        bars = ax.bar(x + (k - (len(labels) - 1) / 2) * w, v, w * 0.92, color=SERIES[k], label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(x, names)
    ax.set_ylabel(f"RMSE ({unit})")
    ax.set_title(f"IDM RMSE on {len(episodes)} held-out episodes (lower is better)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_rmse.png", dpi=120, facecolor=SURFACE)

    print("| joint | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for j, n in enumerate(names):
        print(f"| {n} | " + " | ".join(f"{(total[lab] + [np.mean(total[lab])])[j]:.2f}" for lab in labels) + " |")


if __name__ == "__main__":
    main()
