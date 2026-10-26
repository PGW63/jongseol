#!/usr/bin/env bash
# IDM 실행 환경을 만든다: IDM/.venv 를 만들고 패키지를 설치한다.
#
#   ./install.sh             ResNet-IDM 만 (기본)
#   ./install.sh --dreamgen  DreamGen IDM 까지: 추가 패키지 + GR00T-Dreams 코드 + 체크포인트(약 3GB)
#
# simulation/.venv(MuJoCo, LeRobot) 와 torch 버전이 달라서 IDM 은 별도 venv 를 쓴다.
# 다른 CUDA 휠을 쓰려면 TORCH_INDEX 를 바꾼다 (기본 cu121: NVIDIA 드라이버 535 이상).
set -euo pipefail
cd "$(dirname "$0")"

log() { printf '%s\n' "$*" >&2; }
die() { log "오류: $*"; exit 1; }

DREAMGEN=0
for a in "$@"; do
  case "$a" in
    --dreamgen) DREAMGEN=1 ;;
    *) die "알 수 없는 인자: $a (사용법: ./install.sh [--dreamgen])" ;;
  esac
done

VENV=.venv
PY="$VENV/bin/python"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
GR00T_REPO=https://github.com/NVIDIA/GR00T-Dreams
GR00T_COMMIT=ec3881d44545016871997f8e17dd15f1d792e91d

command -v uv >/dev/null 2>&1 || die "uv 가 필요합니다: https://docs.astral.sh/uv/"

log "== 기본 환경 (ResNet-IDM)"
[ -x "$PY" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$PY" torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX"
uv pip install --python "$PY" -r requirements.txt

if [ "$DREAMGEN" = 1 ]; then
  log "== DreamGen IDM"
  uv pip install --python "$PY" -r requirements-dreamgen.txt

  if [ ! -d third_party/GR00T-Dreams ]; then
    mkdir -p third_party
    git clone "$GR00T_REPO" third_party/GR00T-Dreams
    git -C third_party/GR00T-Dreams checkout --quiet "$GR00T_COMMIT"
  fi

  # IDM_so100: SO100 용 사전학습 IDM. siglip2-large-vision: 같은 구조를 만들 때 쓰는 SigLIP2 (텍스트 타워 제외)
  "$PY" - <<'PY'
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

ckpt = Path("checkpoints")
if not (ckpt / "IDM_so100/model.safetensors").exists():
    snapshot_download("seonghyeonye/IDM_so100", local_dir=ckpt / "IDM_so100")

out = ckpt / "siglip2-large-vision"
if not (out / "model.safetensors").exists():
    out.mkdir(parents=True, exist_ok=True)
    repo = "google/siglip2-large-patch16-256"
    for f in ["config.json", "preprocessor_config.json"]:
        hf_hub_download(repo, f, local_dir=out)
    full = hf_hub_download(repo, "model.safetensors", local_dir=out / "_full")
    sd = {k: v for k, v in load_file(full).items() if not k.startswith("text_model.")}
    save_file(sd, str(out / "model.safetensors"))
    shutil.rmtree(out / "_full")
print("checkpoints ready")
PY
fi

log ""
log "설치 확인:"
"$PY" - <<'PY' >&2
import torch

print(f"  torch {torch.__version__}, CUDA 사용 가능: {torch.cuda.is_available()}")
PY
