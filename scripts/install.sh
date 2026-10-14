#!/bin/zsh
# Install GaussQueue as a per-user launchd agent.
#
#   scripts/install.sh --inbox <dir> [--outbox <dir>] [--status <dir>] [--work <dir>]
#                      [--no-brush] [--no-start]
#
# - Downloads a pinned Brush release (checksum-verified) into $GAUSSQUEUE_HOME/bin
# - If Node.js is available, installs a pinned SplatTransform (for .sog/.spz output)
#   into $GAUSSQUEUE_HOME/tools (a private npm prefix, not a global install)
# - Writes $GAUSSQUEUE_HOME/config.json from config.example.json if there isn't one
#   (existing configs are never overwritten; the --dir flags only apply to new ones)
# - Renders the launchd plist and (re)starts the agent
#
# Re-running is safe; it's also how you update after a `git pull`.
set -euo pipefail

BRUSH_VERSION="v0.3.0"
BRUSH_ASSET="brush-app-aarch64-apple-darwin.tar.xz"
BRUSH_SHA256="65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c"
SPLAT_TRANSFORM_VERSION="3.10.1"   # @playcanvas/splat-transform, for .sog/.spz output
LABEL="com.gaussqueue.agent"

REPO="${0:A:h:h}"
APP_HOME="${GAUSSQUEUE_HOME:-$HOME/Library/Application Support/GaussQueue}"
PYTHON="${PYTHON:-$(command -v python3 || true)}"
PLIST="${GAUSSQUEUE_PLIST:-$HOME/Library/LaunchAgents/$LABEL.plist}"

inbox="" outbox="" status_dir="" work="" want_brush=1 start=1
while (( $# )); do
  case "$1" in
    --inbox)  inbox="$2"; shift 2 ;;
    --outbox) outbox="$2"; shift 2 ;;
    --status) status_dir="$2"; shift 2 ;;
    --work)   work="$2"; shift 2 ;;
    --no-brush) want_brush=0; shift ;;
    --no-start) start=0; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$(uname)" == Darwin ]] || { echo "GaussQueue only runs on macOS." >&2; exit 1; }
[[ -n "$PYTHON" ]] || { echo "python3 not found (install Xcode Command Line Tools or set PYTHON=)." >&2; exit 1; }
"$PYTHON" -I -c 'import sys; sys.exit(sys.version_info < (3, 9))' \
  || { echo "Python 3.9+ required ($PYTHON is $("$PYTHON" --version))." >&2; exit 1; }

mkdir -p "$APP_HOME/bin"

# --- Brush -------------------------------------------------------------------
if (( want_brush )) && [[ ! -x "$APP_HOME/bin/brush_app" ]]; then
  [[ "$(uname -m)" == arm64 ]] || { echo "The pinned Brush build is Apple Silicon only; build Brush yourself and set \"brush\" in config.json, then rerun with --no-brush." >&2; exit 1; }
  echo "Downloading Brush $BRUSH_VERSION…"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL -o "$tmp/brush.tar.xz" \
    "https://github.com/ArthurBrussee/brush/releases/download/$BRUSH_VERSION/$BRUSH_ASSET"
  got="$(shasum -a 256 "$tmp/brush.tar.xz" | cut -d' ' -f1)"
  [[ "$got" == "$BRUSH_SHA256" ]] || { echo "Brush checksum mismatch ($got); aborting." >&2; exit 1; }
  tar -xf "$tmp/brush.tar.xz" -C "$tmp"
  install -m 755 "$tmp"/brush-app-*/brush_app "$APP_HOME/bin/brush_app"
  echo "Installed Brush to $APP_HOME/bin/brush_app"
fi

# --- SplatTransform (optional compression) ----------------------------------
ST_DIR="$APP_HOME/tools/splat-transform"
if command -v npm >/dev/null; then
  have="$("$ST_DIR/node_modules/.bin/splat-transform" --version 2>/dev/null | awk '{print $2}' || true)"
  if [[ "$have" != "v$SPLAT_TRANSFORM_VERSION" ]]; then
    echo "Installing SplatTransform $SPLAT_TRANSFORM_VERSION…"
    mkdir -p "$ST_DIR"
    npm install --no-fund --no-audit --silent --prefix "$ST_DIR" \
      "@playcanvas/splat-transform@$SPLAT_TRANSFORM_VERSION"
  fi
else
  echo "Node.js not found: skipping SplatTransform; results will be .ply only (brew install node, then rerun)."
fi

# --- config ------------------------------------------------------------------
if [[ ! -f "$APP_HOME/config.json" ]]; then
  [[ -n "$inbox" ]] || { echo "First install needs --inbox <dir> (where captures will arrive)." >&2; exit 2; }
  INBOX="$inbox" OUTBOX="$outbox" STATUS_DIR="$status_dir" WORK="$work" \
  "$PYTHON" -I - "$REPO/config.example.json" "$APP_HOME/config.json" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
for key, env in (("inbox_dir", "INBOX"), ("outbox_dir", "OUTBOX"),
                 ("status_dir", "STATUS_DIR"), ("work_dir", "WORK")):
    if os.environ.get(env):
        cfg[key] = os.environ[env]
if not os.environ.get("OUTBOX"):   # keep results next to the inbox by default
    cfg["outbox_dir"] = os.path.join(os.path.dirname(os.environ["INBOX"].rstrip("/")), "outbox")
if not os.environ.get("STATUS_DIR"):
    cfg["status_dir"] = os.path.join(os.path.dirname(os.environ["INBOX"].rstrip("/")), "status")
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
open(sys.argv[2], "a").write("\n")
PY
  echo "Wrote $APP_HOME/config.json"
else
  echo "Keeping existing $APP_HOME/config.json"
fi

GAUSSQUEUE_HOME="$APP_HOME" "$PYTHON" -I "$REPO/gaussqueue.py" --check || {
  echo "Fix the problems above (edit $APP_HOME/config.json) and rerun." >&2; exit 1; }

# --- launchd -----------------------------------------------------------------
mkdir -p "$HOME/Library/LaunchAgents"
GQ_PYTHON="$PYTHON" GQ_SCRIPT="$REPO/gaussqueue.py" GQ_APP_HOME="$APP_HOME" GQ_LOG="$APP_HOME/gaussqueue.log" \
"$PYTHON" -I - "$REPO/launchd/$LABEL.plist.template" "$PLIST" <<'PY'
import os, sys
from xml.sax.saxutils import escape
t = open(sys.argv[1]).read()
for k in ("PYTHON", "SCRIPT", "APP_HOME", "LOG"):
    t = t.replace("{{" + k + "}}", escape(os.environ["GQ_" + k]))
open(sys.argv[2], "w").write(t)
PY
plutil -lint -s "$PLIST"

if (( start )); then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  port="$("$PYTHON" -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("dashboard_port", 8765))' "$APP_HOME/config.json")"
  echo "GaussQueue is running. Dashboard: http://127.0.0.1:$port   Log: $APP_HOME/gaussqueue.log"
else
  echo "Installed $PLIST (not started; run: launchctl bootstrap gui/\$(id -u) \"$PLIST\")"
fi
