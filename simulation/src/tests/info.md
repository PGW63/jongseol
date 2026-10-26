# 현재 폴더에 들어갈 것들

## python 파일들
### `run_scene.py`
- models들이 잘 올라가졌는지 무조코를 켜서 확인할 수 있는 코드
- Fusion을 통해 모델링한 환경
- meshAI를 통해 생성한 물체들
- robot
- camera view

### `check_connection_with_leader.py`
- 리더암이랑 무조코랑 빈 월드에 띄워서 잘 연결됐는지 확인하는 코드
- 리더암의 정규화값과 시뮬레이션 내의 로봇암의 움직임이 얼마나 다른지 시각화
- 구성은 위에 subplot 6개로, 각 configuration마다 그래프로 리더암의 정규화 configurations와 무조코로 변환된 뒤의 다시 정규화를 거쳐 현재 값을 표시
- 각 관절별로 차이를 기록하고 있다가 최종적으로 각 관절마다의 RMSE, 평균값, 분산, 최대차이 를 plot들 옆에 실시간으로 시각화
- 이 외에도 통신지연을 측정하여 subplot들 밑에 표시
- 만약 여기서 차이가 발생하면, 캘리브레이션을 다시 하거나 코드 수정이 필요

### `randomize_env.py`
- data_mj에서 로봇 및 텔레옵없이 구현.

### `replay_real_traj.py`
- 실제 lerobot 형식으로 취한 trajectory를 직접 실행
- 실제 trajectory는 root폴더에, real_traj이라는 이름으로 저장
- 리더암을 실제 traj으로 교체하여 check_connection_with_leader와 동일한 기능을 제공
- 빈 월드를 기본으로 하되, **argument --env** 를 통해, 실제 우리 scene에서도 평가 가능
