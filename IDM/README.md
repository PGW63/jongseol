# IDM (Inverse Dynamics Model)

영상의 두 프레임 (o<sub>t</sub>, o<sub>t+16</sub>)을 보고 그 사이 16스텝의 액션(6-DoF 관절 + 그리퍼)을 예측하는 모델입니다.
비디오 생성 모델이 만든 영상에는 액션이 없으므로, IDM으로 **가짜 액션 라벨(pseudo-label)** 을 붙여 정책 학습 데이터로 씁니다 ([DreamGen](https://arxiv.org/abs/2505.12705) 방식).

## 결론

| 항목 | 결정 | 근거 |
|---|---|---|
| 모델 | **ResNet-IDM** (전체 파라미터 학습) | 모든 실험에서 DreamGen IDM보다 정확하고(real RMSE 2.50 vs 3.44) 학습이 약 7배 빠름. GPU 2.1GB |
| 카메라 | **top 뷰 하나** | 우리 `care_world` 의 top 헤드캠은 팔 전체가 보임. wrist 뷰는 팔이 안 보여서 단독으로는 관절각을 알 수 없음 |
| 학습 데이터 | **실물 데이터만** | IDM은 라벨을 붙일 영상과 같은 종류(카메라, 장면)의 영상으로 학습해야 함. 카메라 배치가 다르면 크게 실패함(OOD RMSE 35°) |

자세한 실험 내용, 수치, FAQ는 [REPORT.md](REPORT.md)를 보세요.

## 설치

`simulation/.venv`(MuJoCo, LeRobot)와 torch 버전이 달라서 `IDM/.venv` 를 따로 만듭니다. [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
cd IDM
./install.sh              # ResNet-IDM (기본)
./install.sh --dreamgen   # DreamGen IDM 비교까지 (GR00T-Dreams 코드 + 체크포인트 약 3GB 추가)
```

- 기본 torch는 `2.5.1+cu121` 입니다 (NVIDIA 드라이버 535 이상). 다른 CUDA 휠은 `TORCH_INDEX=<휠 인덱스 URL> ./install.sh`.
- `--dreamgen` 은 `third_party/GR00T-Dreams`(커밋 고정)와 `checkpoints/`(IDM_so100, SigLIP2 비전 가중치)를 받습니다. 둘 다 git에는 올라가지 않습니다.

## 사용법 (우리 데이터)

LeRobot 포맷 데이터셋(영상 + `action`) 기준입니다. `simulation/src/data_mj.py` 로 녹화하면 저장소 최상위 `datasets/<이름>/` 에 저장되고, 카메라 키는 `observation.images.top` 입니다.

```bash
cd IDM
# 1) 캐시 만들기: top 영상 -> 224px 프레임 + 액션
.venv/bin/python src/prepare_data.py --root ../datasets/<데이터셋> --camera observation.images.top --out data/cache/ours_top

# 2) 학습 (에피소드 10개는 held-out test 로 자동 분리)
.venv/bin/python src/train.py --model resnet --data data/cache/ours_top --out runs/ours_top_resnet \
  --n-test 10 --bs 64 --lr 3e-4 --wd 1e-4 --steps 6000 --eval-every 500 --save

# 3) held-out 에피소드 평가 -> runs/ours_top_resnet/eval_test.json, traj_test.png
.venv/bin/python src/evaluate.py --run runs/ours_top_resnet
```

### 지켜야 할 것

- **카메라와 fps를 생성 모델과 맞춥니다.** IDM은 학습한 카메라 시점에서만 동작합니다. 생성 영상도 같은 top 시점, 같은 fps여야 합니다(30fps면 16프레임 = 0.53초).
- **캘리브레이션을 바꾸지 않습니다.** 녹화 중간에 리더/팔로워를 다시 캘리브레이션하면 같은 자세의 라벨 값이 달라집니다. 정책을 돌릴 로봇도 같은 캘리브레이션이어야 합니다.
- **관절 범위를 골고루 녹화합니다.** IDM은 학습 때 본 관절 범위 밖의 값을 예측하지 못합니다.
- **디스크.** 캐시는 프레임당 약 150KB입니다. 2분 30초 × 100개(약 45만 프레임)면 약 68GB가 필요하므로, 이 규모에서는 mp4를 학습 중에 바로 디코딩하도록 `prepare_data.py`/`dataset.py` 를 바꿔야 합니다.
- **학습량.** 45만 프레임이면 1 epoch가 batch 64 기준 약 7k step(약 15분)입니다. `--steps` 를 데이터 양에 맞게 늘리세요.

### 라벨 없는 영상 추론 (pseudo-label)

학습한 run으로 액션이 없는 영상(생성 영상 등)에 라벨을 붙입니다. 입력은 mp4 폴더(파일 하나 = 에피소드 하나)나 LeRobot v3 데이터셋입니다. LeRobot 데이터셋은 `action` 컬럼이 없어도 됩니다.

```bash
# mp4 폴더
.venv/bin/python src/infer.py --run runs/ours_top_resnet --input <mp4 폴더> --out runs/ours_top_resnet/labels

# LeRobot 데이터셋
.venv/bin/python src/infer.py --run runs/ours_top_resnet --input <데이터셋> --camera observation.images.top --out <출력 폴더>
```

- 출력은 에피소드마다 `<이름>.npy`(float32 `[프레임 수, 6]`, 학습 데이터와 같은 단위)와 `meta.json` 입니다.
- 16프레임 단위로 예측하고, 16으로 나누어떨어지지 않는 끝부분은 마지막 프레임에서 끝나는 윈도우를 하나 더 돌려 채웁니다. 마지막 프레임은 다음 프레임이 없으므로 직전 액션을 복사합니다.
- 영상 fps가 학습 데이터와 다르면 경고를 출력합니다. 프레임이 17개 미만인 영상은 건너뜁니다.

## 폴더 구조

```
IDM/
├── README.md                  # 이 문서 (사용법)
├── REPORT.md                  # 실험 보고서 (모델/데이터 선정, 결과, 검증, FAQ)
├── install.sh                 # 환경 설치 (--dreamgen 으로 DreamGen 추가)
├── requirements.txt           # 기본 패키지
├── requirements-dreamgen.txt  # DreamGen 추가 패키지
├── src/                       # 코드 (파일별 설명: src/info.md)
├── experiments/
│   └── run_experiments.sh     # REPORT.md 실험 재현 (공개 데이터 다운로드 -> 학습 -> 평가)
└── results/                   # REPORT.md 에 쓰인 결과. 그림(loss_*.png, traj_<run>_<tag>.png)은 여기 바로 아래
    └── runs/<run>/            # run별 config.json, loss.csv, eval_*.json, replay_test.json

# install.sh / 실행 시 생기는 폴더 (git 제외)
.venv/  data/  runs/  third_party/  checkpoints/
```
