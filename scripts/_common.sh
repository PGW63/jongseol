#!/usr/bin/env bash
# auto_* 스크립트들이 같이 쓰는 함수. 직접 실행하지 않고 source 해서 쓴다.

# 메시지는 전부 stderr 로 (stdout 은 다른 스크립트가 읽는 결과 전용)
log() { printf '%s\n' "$*" >&2; }
die() { log "오류: $*"; exit 1; }

# ubuntu | macos
detect_os() {
  case "$(uname -s)" in
    Darwin) echo macos ;;
    Linux) echo ubuntu ;;
    *) die "지원하지 않는 OS: $(uname -s)" ;;
  esac
}

# 인자로 받은 OS 이름을 ubuntu | macos 로 정리
normalize_os() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    ubuntu | linux) echo ubuntu ;;
    macos | mac | darwin | osx) echo macos ;;
    *) die "OS 는 ubuntu 또는 macOS 여야 합니다: $1" ;;
  esac
}

# 지금 꽂혀 있는 USB 시리얼 포트를 한 줄에 하나씩 출력
list_ports() {
  local os="$1" p
  if [ "$os" = macos ]; then
    for p in /dev/cu.usbmodem* /dev/cu.usbserial* /dev/cu.wchusbserial*; do
      [ -e "$p" ] && echo "$p"
    done
  else
    for p in /dev/ttyACM* /dev/ttyUSB*; do
      [ -e "$p" ] && echo "$p"
    done
  fi
  return 0
}

# 고정 이름이 놓일 경로. ubuntu: /dev/<name> (udev), macOS: ~/dev/<name> (심볼릭 링크)
mapped_path() {
  local os="$1" name="$2"
  if [ "$os" = macos ]; then echo "$HOME/dev/$name"; else echo "/dev/$name"; fi
}

# map_port <os> <name> <port>: 포트에 고정 이름을 붙인다
map_port() {
  local os="$1" name="$2" port="$3" target
  [ -e "$port" ] || die "포트가 없습니다: $port"
  target="$(mapped_path "$os" "$name")"

  if [ "$os" = macos ]; then
    # macOS 는 udev 가 없고 /dev 에 만든 링크는 재부팅하면 사라진다.
    # 포트 이름(/dev/cu.usbmodem<시리얼>)이 기기 시리얼로 고정되므로 ~/dev 에 링크만 건다.
    mkdir -p "$HOME/dev"
    ln -sfn "$port" "$target"
  else
    local props vendor product serial rule
    props="$(udevadm info -q property -n "$port")" || die "udevadm 조회 실패: $port"
    vendor="$(printf '%s\n' "$props" | sed -n 's/^ID_VENDOR_ID=//p')"
    product="$(printf '%s\n' "$props" | sed -n 's/^ID_MODEL_ID=//p')"
    serial="$(printf '%s\n' "$props" | sed -n 's/^ID_SERIAL_SHORT=//p')"
    [ -n "$vendor" ] && [ -n "$product" ] || die "USB vendor/product ID 를 읽지 못했습니다: $port"
    rule="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$vendor\", ATTRS{idProduct}==\"$product\""
    if [ -n "$serial" ]; then
      rule="$rule, ATTRS{serial}==\"$serial\""
    else
      log "경고: 이 어댑터는 시리얼 번호가 없어서 같은 종류를 두 개 꽂으면 구분하지 못합니다."
    fi
    rule="$rule, SYMLINK+=\"$name\", MODE=\"0666\""
    log "udev 규칙을 씁니다 (sudo 필요): /etc/udev/rules.d/99-$name.rules"
    printf '%s\n' "$rule" | sudo tee "/etc/udev/rules.d/99-$name.rules" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sleep 1
  fi

  [ -e "$target" ] || die "매핑 실패: $target 이 생기지 않았습니다"
  log "매핑 완료: $target -> $port"
  echo "$target"
}
