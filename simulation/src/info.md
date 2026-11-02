# 현재 폴더에 들어갈 것.

## `teleop_mj.py`
- 리더암과 무조코 실제 환경에 연결
- 인자로 --headless를 주면 gui없이 실행

## `data_mj.py`
- lerobot format으로 top view, wrist view, robot state의 데이터 취득
- gui를 통해 top view와 wrist view를 보여준다.
- 화면을 세로로 2분할을 한다.
- 밑의 화면을 3분할하고, 1-2는 view를 보여준다.
- 밑의 화면 3에는 지금까지 dataset을 보여준다.
- 위의 화면을 2분할 한다.
- 위의 화면에는 로봇의 앞쪽 뷰 두개를 추가하여 로봇을 바라본다.
- '1'을 누를 시 데이터 취득 시작
- '2'를 누를 시 데이터 취득 정지
- '3'을 누를 시 환경 초기화.
- '4'를 누를 시 가장 최근 데이터 삭제
- 환경 초기화시, models/automatic.json 을 읽고 이거에 맞게 환경을 랜덤하게 초기화한다.
## `image_to_3d/`
- 사진 한 장으로 물체 mesh 만들기. 설치와 사용법은 [IMAGE_TO_3D.md](../IMAGE_TO_3D.md)
- `run_triposg.py`: TripoSG 를 8GB GPU 에서 돌리는 실행 스크립트 (→ GLB)
- `glb_to_mj.py`: GLB 를 씬 규약(OBJ, cm, Z-up, 바닥 중심)으로 변환

## `teleop/traj_compare.py`
- 빈 월드에 SO-101 만 띄우고, 정해진 끝점 궤적(line / circle / square / figure8)을 화면에 그린다
- `run --mode teleop`: 리더암으로, `run --mode keyboard`: 키보드(w/s a/d q/e) + IK 로 궤적을 따라간다
- 궤적마다 기록을 `results/traj_compare/*.npz` 로 저장 (기준점, 끝점, 관절)
- `plot`: 모은 기록으로 궤적별 tracking / path RMSE 막대그래프 + 경로 겹쳐 그리기 + 시간별 오차를 한 장에

## `teleop/pnp_compare.py`
- 빈 월드에 큐브와 목표 구역을 놓은 씬(`models/scene/scene_pnp.xml`)에서 pick & place
- `run --mode teleop | keyboard`: 같은 seed 면 큐브/목표 배치가 같아 두 모드를 같은 조건에서 비교
- 키보드 모드는 그리퍼가 항상 아래를 보게 하는 IK + 손목 회전(z/x) + 그리퍼(space)
- 시행마다 `results/pnp_compare/*.npz` 로 저장, `plot` 으로 성공률 / 시간 / 이동 거리 / jerk / 들어 올린 횟수 비교

## `teleop/real_sim_compare.py`
- 실물 텔레옵 데이터(IDM 캐시 `arrays.npz`)의 액션을 빈 월드 SO-101 에 그대로 재생하고, 실물 관절 state 와 RMSE 비교 (정규화 단위 + 도)
- 관절마다 median(실물 - 시뮬) 을 영점(제로 캘리브레이션) 차이로 보고 액션에 더해 다시 재생 → 보정 전/후 RMSE
- 그리퍼는 실물이 물체를 쥔 구간(실물이 3 이상 더 열린 상태가 0.5 초 넘게)을 빼고 따로 RMSE
- `results/real_sim_compare/` 에 `real_vs_sim.png`(관절별 sim(파랑)/real(주황) 궤적 + 전체 real - sim 분포 한 장), `rmse.json`
