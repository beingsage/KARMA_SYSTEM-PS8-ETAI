#!/bin/bash
set -e # Good practice: exit script if any command fails.

# This script runs as root  # Correct assumption based on Dockerfile changes.
echo "Entrypoint: Ensuring cache directory ownership for user appuser (1001)..." # Informative log.

# Ensure the target directories exist. Runs as root, so has permission.
mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/torch

# Chown command runs as root and should succeed now.
# Uses UID:GID which is robust. Targets the parent .cache dir recursively.
chown -R 1001:1001 /home/appuser/.cache

echo "Entrypoint: Switching to user appuser (1001) and executing command: $@" # Informative log.

# Drop privileges using gosu before executing the CMD passed as arguments ($@).
# 'gosu appuser' finds the user named 'appuser' (which is UID 1001).
# 'exec' replaces the script process with the application process.

# if Ollama is configured, wait for the service to come up so the app
# doesn't immediately error on startup. we poll the /api/ps endpoint.
if [ -n "${OLLAMA_BASE_URL:-}" ]; then
	echo "Entrypoint: waiting for Ollama at $OLLAMA_BASE_URL ..."
	until curl --silent --fail "$OLLAMA_BASE_URL/api/ps" >/dev/null 2>&1; do
		echo -n '.'
		sleep 2
	done
	echo "\nEntrypoint: Ollama is responsive. Continuing startup."
fi

exec gosu appuser "$@"