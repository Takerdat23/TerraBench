#!/usr/bin/env bash
set -euo pipefail

AQUACROP_HOME="${AQUACROP_HOME:-${TERRABENCH_AQUACROP_HOME:-$HOME/simulators/aquacrop-7.1-x86_64-linux}}"
PROJECT_PATH="${1:-${AQUACROP_PROJECT_PATH:-}}"
RUN_DIR="${TERRABENCH_OFFICIAL_RUN_DIR:-${CLIMATE_AGENT_OFFICIAL_RUN_DIR:-}}"

if [[ ! -x "$AQUACROP_HOME/aquacrop" ]]; then
  echo "AquaCrop executable was not found or is not executable: $AQUACROP_HOME/aquacrop" >&2
  exit 2
fi

mkdir -p "$AQUACROP_HOME/LIST" "$AQUACROP_HOME/OUTP"

if [[ -n "$PROJECT_PATH" ]]; then
  if [[ ! -f "$PROJECT_PATH" ]]; then
    echo "AquaCrop project file was not found: $PROJECT_PATH" >&2
    exit 2
  fi
  case "$PROJECT_PATH" in
    *.PRO|*.PRM|*.pro|*.prm) ;;
    *)
      echo "AquaCrop project should normally be a .PRO or .PRM file: $PROJECT_PATH" >&2
      exit 2
      ;;
  esac
  if [[ "$(realpath "$(dirname "$PROJECT_PATH")")" != "$(realpath "$AQUACROP_HOME/LIST")" ]]; then
    cp "$PROJECT_PATH" "$AQUACROP_HOME/LIST/$(basename "$PROJECT_PATH")"
  fi
fi

shopt -s nullglob
projects=(
  "$AQUACROP_HOME"/LIST/*.PRO
  "$AQUACROP_HOME"/LIST/*.PRM
  "$AQUACROP_HOME"/LIST/*.pro
  "$AQUACROP_HOME"/LIST/*.prm
)

if (( ${#projects[@]} == 0 )); then
  echo "No AquaCrop .PRO or .PRM project files found in $AQUACROP_HOME/LIST." >&2
  echo "Create a project with the official AquaCrop GUI or copy an existing project file into LIST." >&2
  exit 2
fi

rm -f "$AQUACROP_HOME/LIST/ListProjectsTemp.txt"

(
  cd "$AQUACROP_HOME"
  ./aquacrop
)

if [[ -n "$RUN_DIR" ]]; then
  mkdir -p "$RUN_DIR/aquacrop_OUTP" "$RUN_DIR/aquacrop_LIST"
  cp -a "$AQUACROP_HOME/OUTP/." "$RUN_DIR/aquacrop_OUTP/" 2>/dev/null || true
  cp -a "$AQUACROP_HOME/LIST/." "$RUN_DIR/aquacrop_LIST/" 2>/dev/null || true
fi
