"""학습 로그(loss.csv)로 Loss 그래프를 그린다.
- results/loss_<run>.png : 모델별 train loss(EMA) + val loss (+ val RMSE 가 기록된 run 은 오른쪽 패널)
- results/val_rmse_<prefix>.png : 같은 데이터셋 모델 간 val RMSE 비교 (공통 단위)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# entity -> 고정 색 (dataviz 기본 팔레트 slot 1~3, 검증 통과)
COLOR = {"resnet": "#2a78d6", "dreamgen_pretrained": "#eb6834", "dreamgen_scratch": "#1baf7a"}
LABEL = {"resnet": "ResNet-IDM", "dreamgen_pretrained": "DreamGen IDM (IDM_so100 init)",
         "dreamgen_scratch": "DreamGen IDM (SigLIP2 init only)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
VAL = "val_rmse_mean"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load(run):
    df = pd.read_csv(run / "loss.csv")
    cfg = json.loads((run / "config.json").read_text())
    return df, cfg


def has_val_rmse(df):
    return VAL in df and df[VAL].notna().any()


def plot_run(run, out):
    df, cfg = load(run)
    c = COLOR[cfg["model"]]
    unit = "deg" if "sim" in cfg["data"] else "raw"
    to = 180 / np.pi if unit == "deg" else 1
    with_rmse = has_val_rmse(df)
    fig, axs = plt.subplots(1, 2 if with_rmse else 1, figsize=(11 if with_rmse else 6.5, 3.8), squeeze=False)
    for ax in axs.flat:
        style(ax)
    a1 = axs[0, 0]
    ema = df.train_loss.ewm(alpha=0.1).mean()
    a1.plot(df.step, df.train_loss, color=c, alpha=0.25, lw=1)
    a1.plot(df.step, ema, color=c, lw=2, label="train (EMA)")
    v = df.dropna(subset=["val_loss"])
    a1.plot(v.step, v.val_loss, color=INK, lw=2, marker="o", ms=5, label="val (held-out eps)")
    a1.set_yscale("log")
    loss_name = "L1 (normalized action)" if cfg["model"] == "resnet" else "flow-matching velocity MSE"
    a1.set_title(f"Loss — {loss_name}", fontsize=10, color=INK, loc="left")
    a1.set_xlabel("step", color=MUTED)
    a1.legend(fontsize=8, frameon=False)
    if with_rmse:
        a2 = axs[0, 1]
        a2.plot(v.step, v[VAL] * to, color=c, lw=2, marker="o", ms=6)
        a2.annotate(f"{v[VAL].iloc[-1]*to:.2f} {unit}", (v.step.iloc[-1], v[VAL].iloc[-1] * to),
                    textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=9, color=INK)
        a2.set_title(f"Val action RMSE (mean of 6 joints, {unit})", fontsize=10, color=INK, loc="left")
        a2.set_xlabel("step", color=MUTED)
        a2.set_ylim(bottom=0)
    fig.suptitle(f"{run.name}  ·  {LABEL[cfg['model']]}  ·  bs{cfg['bs']}  ·  peak {df.peak_mem_gb.max():.1f} GB"
                 f"  ·  {df.sec_per_step.median():.2f} s/step", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"loss_{run.name}.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def plot_compare(runs, out, name):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    style(ax)
    for run in runs:
        df, cfg = load(run)
        to = 180 / np.pi if "sim" in cfg["data"] else 1
        v = df.dropna(subset=[VAL])
        samples = v.step * cfg["bs"] / 1000
        ax.plot(samples, v[VAL] * to, color=COLOR[cfg["model"]], lw=2, marker="o", ms=6, label=LABEL[cfg["model"]])
        ax.annotate(f"{v[VAL].iloc[-1]*to:.2f}", (samples.iloc[-1], v[VAL].iloc[-1] * to),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=9, color=INK)
    unit = "deg" if "sim" in name else "raw units"
    ax.set_xlabel("training samples seen (×1000)", color=MUTED)
    ax.set_ylabel(f"val RMSE ({unit})", color=MUTED)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Held-out action RMSE vs. training samples — {name}", fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"val_rmse_{name}.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    runs = sorted(p for p in (ROOT / "runs").iterdir() if (p / "loss.csv").exists())
    for r in runs:
        plot_run(r, out)
    for prefix in ["sim", "real"]:
        rs = [r for r in runs if r.name.startswith(prefix) and has_val_rmse(load(r)[0])]
        if rs:
            plot_compare(rs, out, prefix)
    print("plotted", [r.name for r in runs])


if __name__ == "__main__":
    main()
