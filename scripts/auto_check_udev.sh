#!/usr/bin/env bash
# 어느 USB 포트가 어느 암인지 찾는다.
#
#   ./auto_check_udev.sh leader_mj           # 리더만 (MuJoCo 용)
#   ./auto_check_udev.sh leader follower     # 리더 + 팔로워
#
# 결과는 stdout 에 "<역할>=<포트>" 로 한 줄씩 나온다. 안내 문구는 stderr.
# 포트가 하나뿐이고 역할도 하나면 묻지 않고 바로 정한다.
# 그 외에는 역할마다 USB 를 뽑았다 꽂게 해서 사라지는 포트로 구분한다.
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_common.sh"

[ "$#" -ge 1 ] || die "사용법: $0 <역할> [<역할> ...]   (예: leader_mj / leader follower)"

OS="$(detect_os)"
ports="$(list_ports "$OS")"
[ -n "$ports" ] || die "USB 시리얼 포트가 없습니다. 암의 USB 와 전원을 확인하세요."
n_ports="$(printf '%s\n' "$ports" | wc -l | tr -d ' ')"

if [ "$#" -eq 1 ] && [ "$n_ports" -eq 1 ]; then
  log "포트가 하나뿐이라 $1 로 정합니다: $ports"
  echo "$1=$ports"
  exit 0
fi

[ "$n_ports" -ge "$#" ] || die "역할은 $# 개인데 포트는 $n_ports 개뿐입니다: $(echo $ports)"
[ -r /dev/tty ] || die "포트가 여러 개라 USB 를 뽑았다 꽂아야 하는데, 터미널 입력을 받을 수 없습니다."

log "꽂혀 있는 포트:"
printf '%s\n' "$ports" | sed 's/^/  /' >&2

remaining="$ports"
left="$#"
for role in "$@"; do
  n_remaining="$(printf '%s\n' "$remaining" | wc -l | tr -d ' ')"
  if [ "$left" -eq 1 ] && [ "$n_remaining" -eq 1 ]; then
    log "남은 포트가 하나라 $role 로 정합니다: $remaining"
    echo "$role=$remaining"
    break
  fi

  log ""
  log "[$role] 암의 USB 를 뽑고 Enter 를 누르세요."
  read -r _ </dev/tty
  sleep 1
  now="$(list_ports "$OS")"
  found=""
  count=0
  for p in $remaining; do
    if ! printf '%s\n' "$now" | grep -qxF "$p"; then
      found="$p"
      count=$((count + 1))
    fi
  done
  [ "$count" -eq 1 ] || die "사라진 포트가 $count 개입니다. 한 번에 하나만 뽑아야 합니다."
  log "[$role] = $found"
  echo "$role=$found"

  log "[$role] 암의 USB 를 다시 꽂고 Enter 를 누르세요."
  read -r _ </dev/tty
  tries=0
  while [ ! -e "$found" ]; do
    tries=$((tries + 1))
    [ "$tries" -le 10 ] || die "$found 가 다시 나타나지 않습니다."
    sleep 1
  done

  remaining="$(printf '%s\n' "$remaining" | grep -vxF "$found" || true)"
  left=$((left - 1))
done
