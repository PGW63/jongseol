#!/usr/bin/env bash
# MuJoCo 용 리더암 포트에 고정 이름을 붙인다 (팔로워는 설정하지 않는다).
#
#   ./auto_rules_mapping_mj.sh                       # OS 와 포트를 알아서 찾는다
#   ./auto_rules_mapping_mj.sh macOS                 # 포트만 알아서 찾는다
#   ./auto_rules_mapping_mj.sh ubuntu /dev/ttyACM0   # 둘 다 직접 지정
#
#   ubuntu: /dev/so101_leader_mj   (udev 규칙, sudo 필요)
#   macOS : ~/dev/so101_leader_mj  (심볼릭 링크)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/_common.sh"

NAME=so101_leader_mj

[ "$#" -le 2 ] || die "사용법: $0 [<ubuntu|macOS> [<리더 포트>]]"
if [ "$#" -ge 1 ]; then OS="$(normalize_os "$1")"; else OS="$(detect_os)"; fi

if [ "$#" -ge 2 ]; then
  PORT="$2"
else
  found="$("$HERE/auto_check_udev.sh" leader_mj)"
  PORT="${found#leader_mj=}"
fi

map_port "$OS" "$NAME" "$PORT"
