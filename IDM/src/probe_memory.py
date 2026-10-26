"""8GB GPU 에서 어떤 설정으로 (풀)트레이닝이 가능한지 측정한다.
각 설정을 별도 프로세스에서 합성 배치로 8 step 돌려 peak 메모리 / step 시간 / OOM 여부를 기록."""
import json
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    # name, model, bs, flags
    ("A. 원본 레시피 (fp32+AdamW, bs16)", "dreamgen_pretrained", 16, {}),
    ("B. 원본 레시피 bs4", "dreamgen_pretrained", 4, {}),
    ("C. 원본 레시피 bs1", "dreamgen_pretrained", 1, {}),
    ("D. +grad ckpt bs4", "dreamgen_pretrained", 4, {"grad_ckpt": True}),
    ("E. +grad ckpt +single-emb bs4", "dreamgen_pretrained", 4, {"grad_ckpt": True, "single_embodiment": True}),
    ("F. +grad ckpt +single-emb +8bit AdamW bs8", "dreamgen_pretrained", 8,
     {"grad_ckpt": True, "single_embodiment": True, "optim": "adamw8bit"}),
    ("G. +grad ckpt +single-emb +8bit AdamW bs16", "dreamgen_pretrained", 16,
     {"grad_ckpt": True, "single_embodiment": True, "optim": "adamw8bit"}),
    ("G2. +grad ckpt +single-emb +8bit AdamW bs32", "dreamgen_pretrained", 32,
     {"grad_ckpt": True, "single_embodiment": True, "optim": "adamw8bit"}),
    ("G3. +grad ckpt +8bit AdamW (single-emb 없음) bs16", "dreamgen_pretrained", 16,
     {"grad_ckpt": True, "optim": "adamw8bit"}),
    ("H. single-emb +8bit AdamW (ckpt 없음) bs8", "dreamgen_pretrained", 8,
     {"single_embodiment": True, "optim": "adamw8bit"}),
    ("I. 비전 동결(부분학습) bs16", "dreamgen_pretrained", 16, {"freeze_vision": True}),
    ("J. ResNet IDM bs64", "resnet", 64, {}),
]

CHILD = r"""
import json, sys, time, torch
sys.path.insert(0, "{src}")
from models import build_model
from train import make_optimizer
cfg = json.loads(sys.argv[1])
fl = cfg["flags"]
kw = {{}}
if cfg["model"].startswith("dreamgen"):
    kw = dict(grad_ckpt=fl.get("grad_ckpt", False), single_embodiment=fl.get("single_embodiment", False),
              freeze_vision=fl.get("freeze_vision", False))
res = {{}}
try:
    m = build_model(cfg["model"], **kw).cuda().train()
    params = [p for p in m.parameters() if p.requires_grad]
    res["trainable_M"] = sum(p.numel() for p in params) / 1e6
    opt = make_optimizer(params, fl.get("optim", "adamw"), 1e-4, 1e-5)
    bs = cfg["bs"]
    fr = torch.randint(0, 255, (bs, 2, 224, 224, 3), dtype=torch.uint8, device="cuda")
    a = torch.rand(bs, 16, 6, device="cuda") * 2 - 1
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(8):
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m.loss(fr, a)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t)
    res.update(ok=True, peak_gb=torch.cuda.max_memory_allocated() / 2**30,
               reserved_gb=torch.cuda.max_memory_reserved() / 2**30, sec_per_step=sum(times[3:]) / 5)
except torch.cuda.OutOfMemoryError as e:
    res.update(ok=False, error="CUDA OOM", peak_gb=torch.cuda.max_memory_allocated() / 2**30)
print("RESULT" + json.dumps(res))
"""


def main():
    src = Path(__file__).resolve().parent
    out = src.parent / "results/memory_probe.json"
    out.parent.mkdir(exist_ok=True)
    results = []
    for name, model, bs, flags in CONFIGS:
        t = time.time()
        p = subprocess.run([sys.executable, "-c", CHILD.format(src=src),
                            json.dumps({"model": model, "bs": bs, "flags": flags})],
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        r = json.loads(line[0][6:]) if line else {"ok": False, "error": p.stderr.strip().splitlines()[-1][:200]}
        r.update(name=name, model=model, bs=bs, flags=flags)
        results.append(r)
        print(f"{name:48s} -> {r}  ({time.time()-t:.0f}s)", flush=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
