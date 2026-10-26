# 현재 폴더에 들어갈 것.
IDM(Inverse Dynamics Model) 코드. 실행은 전부 `IDM/` 폴더에서 `IDM/.venv` 로 한다.

## 기본 흐름 (항상 쓰는 파일)

### `prepare_data.py`
- LeRobot 데이터셋(mp4 + parquet)에서 카메라 하나를 골라 학습용 캐시를 만든다.
- 영상을 중앙 정사각 크롭 후 224×224로 줄여 `frames.npy`에, 액션/상태/에피소드 번호를 `arrays.npz`에 저장한다.
- 우리 `simulation/src/data_mj.py` 데이터셋은 `--camera observation.images.top`.
- 주의: 프레임을 압축 없이 저장해서 프레임당 약 150KB. 45만 프레임이면 약 68GB.

### `dataset.py`
- 캐시를 (o<sub>t</sub>, o<sub>t+16</sub>) → 액션 a<sub>t:t+16</sub> 윈도우로 잘라 준다. 학습은 1프레임 간격으로 가능한 모든 윈도우를 쓴다.
- 관절별 min-max 정규화([-1, 1])는 학습 데이터 기준으로 계산한다.
- train/test는 에피소드 단위로 나눈다 (seed 0 셔플, `--n-test` 개를 test로).

### `models.py`
- `ResNetIDM`: 주력 모델. ImageNet ResNet18로 두 프레임 특징을 뽑고 Transformer 인코더/디코더로 16개 액션을 회귀한다 (18.7M, L1 loss).
- `DreamGenIDM`: NVIDIA GR00T-Dreams의 IDM 코드를 그대로 불러와 쓰는 비교용 모델 (SigLIP2 + flow-matching DiT, 612M).
  `./install.sh --dreamgen` 으로 설치했을 때만 쓸 수 있다.
- 두 모델 모두 학습 시 같은 증강(crop 0.95 → resize → color jitter)을 쓴다.

### `train.py`
- 공통 학습 루프 (bf16 autocast, cosine LR, warmup 5%).
- 결과는 `runs/<이름>/` 에 저장: `config.json`(설정, train/test 에피소드, 정규화 범위), `loss.csv`(step별 loss, val RMSE), `model_bf16.pt`(`--save` 시).
- DreamGen을 8GB GPU에서 풀트레이닝하려면 `--optim adamw8bit --grad-ckpt --single-embodiment`.

### `evaluate.py`
- 학습된 IDM으로 test 에피소드 전체를 라벨링한다 (겹치지 않는 16프레임 윈도우, DreamGen `dump_idm_actions` 와 같은 방식).
- 정규화를 풀어서 원래 단위로 GT 액션과 비교: 관절별 RMSE, nRMSE(학습 범위 대비 %), R².
- 기준선 2개도 같이 계산: 평균 액션(하한), 상태 보간 오라클(GT 관절 상태 사용, 참고값).
- `--data <다른 캐시>` 를 주면 그 데이터의 모든 에피소드를 평가한다 (OOD 테스트).
- 출력: `runs/<이름>/eval_<tag>.json`, `traj_<tag>.png`(GT vs 라벨 궤적), `pred_<tag>.npz`.

### `infer.py`
- 액션 라벨이 없는 영상(mp4 폴더 또는 LeRobot v3 데이터셋)에 학습된 IDM으로 pseudo-label을 붙인다. `evaluate.py` 의 `relabel_episode` 를 그대로 쓴다.
- 끝부분은 마지막 프레임에서 끝나는 윈도우로 채우고, 마지막 프레임은 직전 액션을 복사해서 프레임 수와 같은 길이로 맞춘다.
- 출력: `<out>/<에피소드>.npy` (`[프레임 수, 6]`), `<out>/meta.json`.

## 실험 재현용 파일

### `plot_results.py`
- `runs/*/loss.csv` 로 `results/loss_<run>.png`(train/val loss), `results/val_rmse_<sim|real>.png`(모델 간 val RMSE 비교)를 그린다.

### `probe_memory.py`
- 8GB GPU에서 DreamGen IDM을 어떤 설정으로 학습할 수 있는지(OOM 여부, peak 메모리, step 시간) 측정한다.
- 결과: `results/memory_probe.json`. 다른 GPU에서 DreamGen을 돌릴 때만 필요.

### `sim_replay.py`
- IDM 라벨을 SimStudio `simple_pick` MuJoCo 장면에서 실행하고, GT 액션을 실행했을 때와 엔드이펙터 위치 차이(cm)를 비교한다.
- sim 데이터셋(alexhegit/so101-simstudio-lab01-pnp) 실험 재현 전용. `simulation/.venv`(MuJoCo)로 실행하고 `--scene` 으로 장면 경로를 준다.
