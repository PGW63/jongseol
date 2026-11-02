#!/usr/bin/env bash
# 우리 실물 텔레옵 데이터(Gom-sy/teleop_1015)로 카메라 구성 비교: top only vs top + wrist (ResNet-IDM).
#   ./experiments/run_teleop_1015.sh
# 선택 환경변수
#   DATASET=<경로>  데이터셋 위치 (기본: ../datasets/teleop_1015, 없으면 HF 에서 받는다. gated 라 hf auth login 필요)
#   STEPS=<n>       run 마다 학습 step 수 (기본 15000, batch 64 기준 약 3.5 epoch)
# 두 run 은 같은 캐시, 같은 test 에피소드(seed 0 으로 10개), 같은 하이퍼파라미터를 쓴다.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATASET=${DATASET:-../datasets/teleop_1015}
STEPS=${STEPS:-15000}
CACHE=data/cache/teleop_1015
RN="--model resnet --data $CACHE --n-test 10 --bs 64 --lr 3e-4 --wd 1e-4 --steps $STEPS --eval-every 1000 --save"

# 1) 데이터 + 캐시 (카메라 2개를 한 캐시에, 카메라당 프레임 수 x 150KB)
if [ ! -f "$DATASET/meta/info.json" ]; then
  $PY -c "from huggingface_hub import snapshot_download as d; d('Gom-sy/teleop_1015', repo_type='dataset', local_dir='$DATASET')"
fi
[ -f $CACHE/meta.json ] || $PY src/prepare_data.py --root "$DATASET" \
  --camera observation.images.top observation.images.wrist --out $CACHE

# 2) 학습 + held-out 평가
run() {
  name=$1; shift; mkdir -p runs/$name; echo "=== $name ($(date))"
  $PY src/train.py --out runs/$name "$@" > runs/$name/train.log 2>&1; echo "    train exit=$?"
  $PY src/evaluate.py --run runs/$name > runs/$name/eval.log 2>&1; echo "    eval exit=$?"
}
run teleop_top_resnet $RN --cameras top
run teleop_topwrist_resnet $RN --cameras top wrist

# 3) 비교 그래프 -> results/teleop_cam_*.png, results/teleop_cam.json
$PY src/compare_runs.py --runs runs/teleop_top_resnet runs/teleop_topwrist_resnet \
  --labels "top only" "top + wrist" --name teleop_cam
echo "ALL DONE $(date)"
