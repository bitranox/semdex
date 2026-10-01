#!/usr/bin/env bash
# ==============================================================================
# semdex restart hook - restart an ollama server running in a docker container
# ==============================================================================
#
# semdex runs this script (via [health].restart_command) when it judges an
# ollama-backed embedding/summary endpoint to be DOWN: the reactive self-heal
# runs it after a hard provider failure, and the opt-in [health] check loop runs
# it when a probe fails. semdex expects a zero exit to mean "restart issued".
#
# Environment contract:
#   * The script inherits semdex's environment. Set OLLAMA_CONTAINER to name the
#     container to restart (default: "ollama"). Set DOCKER_HOST to target a remote
#     or non-default docker daemon (standard docker CLI variable).
#   * semdex bounds the run with [health].restart_timeout; keep this quick.
#
# Make it executable before pointing [health].restart_command at it:
#   git update-index --chmod=+x scripts/restart-hooks/ollama-docker.sh
# ==============================================================================
set -euo pipefail

container="${OLLAMA_CONTAINER:-ollama}"

echo "semdex restart hook: restarting docker container ${container}" >&2
docker restart "${container}"
