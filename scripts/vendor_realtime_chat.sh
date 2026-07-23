#!/usr/bin/env bash
# Helper to vendor agents/RealtimeVoiceChat into the parent repository as regular files.
# WARNING: This will remove the nested .git directory inside agents/RealtimeVoiceChat.
# Run from repository root: bash scripts/vendor_realtime_chat.sh

set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$BASE_DIR/agents/RealtimeVoiceChat"

if [ ! -d "$TARGET_DIR" ]; then
  echo "ERROR: $TARGET_DIR not found"
  exit 1
fi

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Found nested .git. Removing it to convert submodule into normal folder..."
  rm -rf "$TARGET_DIR/.git"
  echo "Removed $TARGET_DIR/.git"
else
  echo "No nested .git found. Folder already vendored or not a git repo."
fi

# Add and commit instructions for user
cat <<'EOF'
Next steps (run these yourself):

# Stage the vendored files
git add agents/RealtimeVoiceChat

# Commit the vendored directory
git commit -m "Vendor RealtimeVoiceChat as regular directory"

# Push to remote
git push

If you previously added the path as a submodule entry and it still appears as a gitlink, run:

# Remove potential stale submodule entry from the index
git rm --cached agents/RealtimeVoiceChat || true
git add agents/RealtimeVoiceChat
git commit -m "Replace submodule with vendored RealtimeVoiceChat contents"
git push
EOF

echo "Done. Please follow the printed next steps to finalize the vendor operation."
