#!/usr/bin/env bash
# REPORT.md 실험 재현: 공개 데이터셋의 top 계열 카메라로 ResNet-IDM / DreamGen IDM 학습 + 평가.
#   ./experiments/run_experiments.sh
# 선택 환경변수
#   SMOKE_ROOT=<leader-smoke-20261005 경로>        로컬 녹화 OOD 평가까지 실행
#   SCENE=<simstudio>/SO101/scenes/simple_pick/scene.xml  MuJoCo 리플레이 검증까지 실행 (simulation/.venv 의 mujoco 사용)
# DreamGen run 은 ./install.sh --dreamgen 으로 설치된 경우에만 실행한다.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DG="--bs 16 --optim adamw8bit --grad-ckpt --single-embodiment --lr 1e-4 --steps 3000 --eval-every 500 --save"
RN="--bs 64 --lr 3e-4 --wd 1e-4 --steps 6000 --eval-every 500 --save"

# 1) 데이터 다운로드 (top 계열 카메라 영상만) + 캐시
$PY - <<'EOF'
from huggingface_hub import snapshot_download
for repo, local, cam in [("alexhegit/so101-simstudio-lab01-pnp", "data/raw/so101_sim_pnp", "camera_top"),
                         ("lerobot/svla_so101_pickplace", "data/raw/svla_so101_pickplace", "up")]:
    snapshot_download(repo, repo_type="dataset", local_dir=local,
                      allow_patterns=["meta/*", "data/*", f"videos/observation.images.{cam}/*"])
EOF
[ -d data/cache/sim_top ] || $PY src/prepare_data.py --root data/raw/so101_sim_pnp --camera observation.images.camera_top --out data/cache/sim_top
[ -d data/cache/real_up ] || $PY src/prepare_data.py --root data/raw/svla_so101_pickplace --camera observation.images.up --out data/cache/real_up
if [ -n "${SMOKE_ROOT:-}" ] && [ ! -d data/cache/ours_smoke_top ]; then
  $PY src/prepare_data.py --root "$SMOKE_ROOT" --camera observation.images.camera_top --out data/cache/ours_smoke_top
fi

# 2) 학습 + held-out 평가 (+ 로컬 녹화 OOD, MuJoCo 리플레이)
run() {
  name=$1; shift; mkdir -p runs/$name; echo "=== $name ($(date))"
  $PY src/train.py --out runs/$name "$@" > runs/$name/train.log 2>&1; echo "    train exit=$?"
  $PY src/evaluate.py --run runs/$name > runs/$name/eval.log 2>&1; echo "    eval exit=$?"
  case $name in sim_*)
    [ -d data/cache/ours_smoke_top ] && $PY src/evaluate.py --run runs/$name --data data/cache/ours_smoke_top --tag ours_smoke > /dev/null 2>&1
    [ -n "${SCENE:-}" ] && ../simulation/.venv/bin/python src/sim_replay.py --run runs/$name --scene "$SCENE" > runs/$name/replay.log 2>&1
  esac
}
run sim_top_resnet --model resnet --data data/cache/sim_top $RN
run real_up_resnet --model resnet --data data/cache/real_up $RN
if [ -d third_party/GR00T-Dreams ]; then
  run sim_top_dreamgen_pretrained --model dreamgen_pretrained --data data/cache/sim_top $DG
  run real_up_dreamgen_pretrained --model dreamgen_pretrained --data data/cache/real_up $DG
fi
$PY src/plot_results.py
echo "ALL DONE $(date)"
