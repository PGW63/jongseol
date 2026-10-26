# 현재 폴더에 들어갈 것.
자동 쉘파일들 모음

## shell 스크립트
기본적으로 실행권한을 부여해주어야 한다.

```chmod 500 <파일명>```

### `auto_install`
- 알아서 필요한 python packages들을 자동 설치해준다.


### `auto_check_udev`
- 무엇이 leader암이고, 무엇이 follower암인지 체크.
- 필요에 따라 사용자에게 USB를 뺏다 꽂으라고 요청.

### `auto_rules_mapping`
- 인자로 OS를 받아서 ubuntu, macOS를 받는다.
- 또한 또다른 인자들로 leader, follower로 받고 udev 포트를 받아서 알아서 매핑을 진행해준다.
- ubuntu의 경우, /dev/so101_leader, /dev/so101_follower
- macOS의 경우, ~/dev/so101_leader, ~/dev/so101_follower

### `auto_rules_mapping_mj`
- auto_rules_mapping과 동일하되, follower를 설정하지 않는다.
- ubuntu의 경우, /dev/so101_leader_mj
- macOS의 경우, ~/dev/so101_leader_mj




