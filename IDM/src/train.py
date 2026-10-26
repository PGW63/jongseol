"""IDM 학습 공통 루프 (bf16 autocast, fp32 master weight — 원본 HF Trainer 설정과 동일)."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ActionNormalizer, Cache, IDMWindowDataset, split_episodes
from models import build_model


def make_optimizer(params, name, lr, wd):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.95, 0.999), weight_decay=wd)
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.95, 0.999), weight_decay=wd)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, normalizer, max_batches):
    model.eval()
    errs, losses = [], []
    g = torch.Generator(device="cuda").manual_seed(0)
    for i, (frames, act) in enumerate(loader):
        if i >= max_batches:
            break
        frames, act = frames.cuda(non_blocking=True), act.cuda(non_blocking=True)
        torch.manual_seed(i)  # flow-matching 샘플 노이즈 고정
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model.loss(frames, act).item())
            pred = model.predict(frames)
        errs.append(normalizer.denorm(pred.cpu().numpy()) - normalizer.denorm(act.cpu().numpy()))
    model.train()
    e = np.concatenate(errs)  # [N, H, D]
    return float(np.mean(losses)), np.sqrt((e**2).mean(axis=(0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="prepare_data.py 로 만든 캐시 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--optim", default="adamw")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--single-embodiment", action="store_true")
    ap.add_argument("--bf16-weights", action="store_true", help="fp32 master weight 대신 bf16 파라미터로 학습")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = Cache(args.data)
    train_eps, test_eps = split_episodes(cache, args.n_test)
    train_idx = np.isin(cache.ep, train_eps)
    normalizer = ActionNormalizer.fit(cache.action[train_idx])
    train_ds = IDMWindowDataset(cache, train_eps, args.horizon, normalizer)
    val_ds = IDMWindowDataset(cache, test_eps, args.horizon, normalizer, stride=args.horizon)
    print(f"train windows={len(train_ds)} ({len(train_eps)} eps) | val windows={len(val_ds)} eps={test_eps}")

    kw = {}
    if args.model.startswith("dreamgen"):
        kw = dict(freeze_vision=args.freeze_vision, grad_ckpt=args.grad_ckpt,
                  single_embodiment=args.single_embodiment)
    model = build_model(args.model, horizon=args.horizon, **kw)
    model = (model.to(torch.bfloat16) if args.bf16_weights else model).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params trainable={n_train/1e6:.1f}M / total={n_total/1e6:.1f}M")
    opt = make_optimizer(params, args.optim, args.lr, args.wd)
    warm = int(args.steps * args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(args.steps - warm, 1))))

    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2)

    cfg = vars(args) | {"train_eps": train_eps, "test_eps": test_eps, "normalizer": normalizer.to_dict(),
                        "params_trainable_M": n_train / 1e6, "params_total_M": n_total / 1e6,
                        "gpu": torch.cuda.get_device_name(0)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    log_f = open(out / "loss.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "train_loss", "val_loss", "val_rmse_mean", "lr", "sec_per_step", "peak_mem_gb"])
    torch.cuda.reset_peak_memory_stats()
    model.train()
    step, t0, run_loss, it = 0, time.time(), [], iter(loader)
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                frames, act = next(it)
            except StopIteration:
                it = iter(loader)
                frames, act = next(it)
            frames, act = frames.cuda(non_blocking=True), act.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(frames, act) / args.grad_accum
            loss.backward()
            run_loss.append(loss.item() * args.grad_accum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 10 == 0 or step == args.steps:
            dt = (time.time() - t0) / 10
            t0 = time.time()
            row = [step, np.mean(run_loss), "", "", sched.get_last_lr()[0], round(dt, 3),
                   round(torch.cuda.max_memory_allocated() / 2**30, 2)]
            if step % args.eval_every == 0 or step == args.steps:
                vl, rmse = evaluate(model, val_loader, normalizer, args.eval_batches)
                row[2], row[3] = vl, float(rmse.mean())
                print(f"  [eval] step {step} val_loss={vl:.4f} val_rmse/joint={np.round(rmse, 4)}")
                t0 = time.time()
            log.writerow(row)
            log_f.flush()
            print(f"step {step} loss={row[1]:.4f} lr={row[4]:.2e} {dt:.2f}s/it mem={row[6]}GB", flush=True)
            run_loss = []
    if args.save:
        sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
        torch.save(sd, out / "model_bf16.pt")
    print("done")


if __name__ == "__main__":
    main()
