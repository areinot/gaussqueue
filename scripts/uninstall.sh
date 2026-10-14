#!/bin/zsh
# Stop GaussQueue and remove its launchd agent. Your config, history, inbox,
# outbox and work folders are left untouched.
set -euo pipefail
LABEL="com.gaussqueue.agent"
PLIST="${GAUSSQUEUE_PLIST:-$HOME/Library/LaunchAgents/$LABEL.plist}"
APP_HOME="${GAUSSQUEUE_HOME:-$HOME/Library/Application Support/GaussQueue}"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && echo "Stopped $LABEL" || echo "$LABEL was not running"
rm -f "$PLIST"
echo "Removed $PLIST"
echo "Data kept in: $APP_HOME (delete it yourself if you want a clean slate)"
