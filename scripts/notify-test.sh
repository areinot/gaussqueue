#!/bin/zsh
# Send a test iMessage to config.json's "imessage_to", optionally with an image:
#   scripts/notify-test.sh [path/to/image.png]
set -euo pipefail
REPO="${0:A:h:h}"
exec "${PYTHON:-python3}" -I "$REPO/gaussqueue.py" --notify-test "$@"
