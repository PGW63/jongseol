# 종합설계

SO-101 로봇팔로 트롤리 위 약 전달 작업(약 파우치 확인, 서랍에서 약 꺼내기, 컵에 붓고 젓기, 벨 누르기, 트레이·컵 건네기)을 학습시키는 프로젝트입니다.

MuJoCo 시뮬레이션에서 리더암 텔레옵으로 시연 데이터를 모으고, IDM(Inverse Dynamics Model)으로 액션이 없는 영상(비디오 생성 모델 결과 등)에 액션 라벨을 붙여 정책 학습 데이터를 늘립니다.

## 폴더 구조

```
jongseol/
├── README.md       # 이 문서 (전체 소개)
├── simulation/     # MuJoCo 씬, 리더암 텔레옵, 카메라 캘리브레이션, 데이터 취득
├── IDM/            # Inverse Dynamics Model 학습 · 평가 · 추론
└── datasets/       # LeRobot 포맷 데이터셋 (git 제외). simulation 이 저장하고 IDM 이 읽는다
```

| 폴더 | 내용 | 가상환경 | 문서 |
|---|---|---|---|
| `simulation/` | 트롤리 씬과 SO-101 모델, 환경 랜덤화, 리더암 연결, 카메라 캘리브레이션, `data_mj.py` 데이터 취득 | `simulation/.venv` (MuJoCo, LeRobot) | [simulation/README.md](simulation/README.md) |
| `IDM/` | 두 프레임으로 16스텝 액션을 예측하는 IDM. 학습, held-out 평가, 라벨 없는 영상 추론 | `IDM/.venv` (torch) | [IDM/README.md](IDM/README.md), [IDM/REPORT.md](IDM/REPORT.md) |
| `datasets/` | 녹화한 데이터셋 또는 외부에서 받은 데이터셋 | | |

`simulation` 과 `IDM` 은 torch 버전이 달라서 가상환경을 따로 씁니다. 각 폴더의 명령은 그 폴더 안에서 실행합니다.

## 설치

```bash
cd simulation && ./scripts/auto_install.sh   # simulation/.venv
cd ../IDM && ./install.sh                     # IDM/.venv
```

## 전체 흐름

1. **데이터 취득** (`simulation/`): 리더암으로 시뮬레이션 속 SO-101을 조작해 녹화합니다. 결과는 `datasets/<이름>/` 에 LeRobot 포맷으로 저장됩니다.
   ```bash
   cd simulation
   .venv/bin/python src/data_mj.py --task "작업 설명"
   ```
   외부 데이터셋을 쓸 때도 `datasets/` 아래에 두면 됩니다 (LeRobot v3, 6차원 `action`/`observation.state`).

2. **IDM 학습** (`IDM/`): top 카메라 영상과 액션으로 학습합니다.
   ```bash
   cd IDM
   .venv/bin/python src/prepare_data.py --root ../datasets/<데이터셋> --camera observation.images.top --out data/cache/ours_top
   .venv/bin/python src/train.py --model resnet --data data/cache/ours_top --out runs/ours_top_resnet \
     --n-test 10 --bs 64 --lr 3e-4 --wd 1e-4 --steps 6000 --eval-every 500 --save
   .venv/bin/python src/evaluate.py --run runs/ours_top_resnet
   ```

3. **라벨 없는 영상 추론** (`IDM/`): mp4 폴더나 LeRobot 데이터셋에 pseudo-label 을 붙입니다.
   ```bash
   .venv/bin/python src/infer.py --run runs/ours_top_resnet --input <mp4 폴더 또는 ../datasets/<데이터셋>> \
     [--camera observation.images.top] --out <출력 폴더>
   ```

자세한 옵션과 주의사항(카메라·fps 맞추기, 캘리브레이션 유지 등)은 각 폴더의 README를 보세요.

## 참고

1. [so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio)
2. [robosuite](https://github.com/ARISE-Initiative/robosuite)
3. [DreamGen](https://arxiv.org/abs/2505.12705)
