# 현재 폴더에 들어가야 할 것

## `intrinsic_cali.py`
- 카메라의 udev 포트를 통해 연결
- 체커보드를 통해 캘리브레이션을 진행한다.
- 사용자가 's' 키를 누를 시 화면 캡쳐
- argument로 체커보드 사이즈를 입력받을 수 있게 한다.
- 인자로 top, wrist, belly 중 하나를 받아서 캘리가 끝나면 config.py에 저장해준다.


## `charuco_intrinsic_cali.py`
- intrinsic_cali.py 와 같지만 체커보드 대신 ChArUco 보드를 쓴다 (보드가 일부 가려지거나 화면 밖으로 나가도 된다).
- 첫 인자로 아무 카메라 이름이나 받고, `<name>_cam/config.py` 가 없으면 만들어서 결과를 저장한다.
- 기본 보드는 OpenCV 튜토리얼의 charucoboard.png (5x7 칸, DICT_6X6_250, 마커 = 칸 x 0.6). 인쇄한 칸 길이만 `--square` 로 준다.
- `--make-board board.png` 로 인쇄용 보드 이미지를 만든다.


## `headCam_to_Base.py`
- 헤드캠의 로봇 베이스로부터의 extrinsic을 구한다.
- 캘리가 끝나면 extrinsic의 config에 저장해준다.



## `extrinsic_cali.py`
- ChArUco 보드로 카메라 extrinsic 을 구한다. 첫 인자로 카메라 이름 (intrinsic 이 먼저 있어야 한다).
- 기준 링크: top, left = base (고정), belly = shoulder (shoulder_pan 과 같이 돈다), wrist = gripper. 다른 이름은 `--link`.
- 보드 자리(`--placement`)는 wrist 캠으로 먼저 잰다 (hand-eye): 팔 자세를 바꿔가며 15~20 장 찍으면 베이스 기준 보드 위치, gripper 기준 wrist 캠, 관절 영점 오차를 같이 푼다 (`extrinsic/board_<자리>.json`). 보드는 눕혀도 세워도 된다.
- 그 뒤 같은 자리로 top, left, belly 를 돌린다. (`--touch` 는 팔 끝으로 꼭짓점을 찍는 예전 방식)
- 카메라마다 `s` 로 표본을 찍는다 (`extrinsic/<name>_samples.json` 에 쌓임). 자리를 바꿔 여러 번 돌리면 같이 평균 낸다.
- 결과는 `extrinsic/<name>.py` 와, MuJoCo 에 넣을 `<camera>` 줄.


## `find_cameras.py` (macOS)
- 연결된 카메라를 한 장씩 찍어 보여주고, 어느 것이 top / wrist / belly / left 인지 골라 각 config 의 port 에 `uid:<uniqueID>` 로 적는다.
- macOS 의 카메라 번호는 꽂는 순서/개수에 따라 바뀌지만, uniqueID (USB 포트 위치 + VID/PID) 는 같은 포트에 꽂는 한 그대로다.
- 모든 스크립트는 `open_camera` 에서 `uid:` 를 지금의 번호로 바꿔 연다.
