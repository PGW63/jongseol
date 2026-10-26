#!/usr/bin/env bash
# 실행 환경을 만든다: 프로젝트 루트에 .venv 를 만들고 pyproject.toml 의 패키지를 설치한다.
#
#   ./scripts/auto_install.sh
#
# uv 가 있으면 uv 로, 없으면 python3 -m venv + pip 로 설치한다.
# 이미 .venv 가 있으면 그대로 두고 빠진 패키지만 채운다.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/_common.sh"

ROOT="$(dirname "$HERE")"
VENV="$ROOT/.venv"
PY_VERSION=3.12

cd "$ROOT"
[ -f pyproject.toml ] || die "pyproject.toml 이 없습니다: $ROOT"

if command -v uv >/dev/null 2>&1; then
  log "uv 로 설치합니다 ($(uv --version))"
  [ -x "$VENV/bin/python" ] || uv venv --python "$PY_VERSION" "$VENV"
  uv pip install --python "$VENV/bin/python" -e .
else
  log "uv 가 없어서 python3 -m venv + pip 로 설치합니다"
  if [ ! -x "$VENV/bin/python" ]; then
    command -v python3 >/dev/null 2>&1 || die "python3 가 없습니다. Python $PY_VERSION 이상을 설치하세요."
    python3 -c "import sys; sys.exit(sys.version_info < (3, 12))" ||
      die "Python $PY_VERSION 이상이 필요합니다 (지금: $(python3 --version 2>&1))"
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -e .
fi

log ""
log "설치 확인:"
"$VENV/bin/python" - <<'PY' >&2
import importlib.metadata as md

import lerobot  # noqa: F401
import mujoco  # noqa: F401
import teleop  # noqa: F401

for name in ("mujoco", "lerobot", "feetech-servo-sdk", "matplotlib", "imageio"):
    print(f"  {name}=={md.version(name)}")
PY

log ""
log "완료. 실행은 .venv 의 파이썬으로:"
if [ "$(detect_os)" = macos ]; then
  log "  .venv/bin/mjpython src/tests/run_scene.py   (macOS 뷰어는 mjpython 필요)"
else
  log "  .venv/bin/python src/tests/run_scene.py"
fi
