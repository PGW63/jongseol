# 현재 폴더에 들어가야 할 것

## `intrinsic_cali.py`
- 카메라의 udev 포트를 통해 연결
- 체커보드를 통해 캘리브레이션을 진행한다.
- 사용자가 's' 키를 누를 시 화면 캡쳐
- argument로 체커보드 사이즈를 입력받을 수 있게 한다.
- 인자로 wrist or top을 받아서 캘리가 끝나면 config.py에 저장해준다.


## `headCam_to_Base.py`
- 헤드캠의 로봇 베이스로부터의 extrinsic을 구한다.
- 캘리가 끝나면 extrinsic의 config에 저장해준다.

