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