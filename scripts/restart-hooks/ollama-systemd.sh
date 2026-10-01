#!/usr/bin/env bash
# ==============================================================================
# semdex restart hook - restart an ollama server managed by systemd
# ==============================================================================
#
# semdex runs this script (via [health].restart_command) when it judges an
# ollama-backed embedding/summary endpoint to be DOWN: the reactive self-heal
# runs it after a hard provider failure, and the opt-in [health] check loop runs
# it when a probe fails. semdex expects a zero exit to mean "restart issued".
#
# Environment contract:
#   * The script inherits semdex's environment. Set OLLAMA_UNIT to name a
#     non-default systemd unit (default: "ollama"). To restart ollama on a REMOTE
#     host, set OLLAMA_SSH_HOST to a host semdex can reach with a passwordless key
#     (the restart then runs over ssh).
#   * semdex bounds the run with [health].restart_timeout; keep this quick.
#
# Make it executable before pointing [health].restart_command at it:
#   git update-index --chmod=+x scripts/restart-hooks/ollama-systemd.sh
# ==============================================================================
set -euo pipefail

unit="${OLLAMA_UNIT:-ollama}"

if [[ -n "${OLLAMA_SSH_HOST:-}" ]]; then
    echo "semdex restart hook: restarting ${unit} on ${OLLAMA_SSH_HOST}" >&2
    # The unit name is expanded here (client side) on purpose - it is an operator
    # config value, so remote-side re-expansion is neither needed nor wanted.
    # shellcheck disable=SC2029
    ssh "${OLLAMA_SSH_HOST}" systemctl restart "${unit}"
else
    echo "semdex restart hook: restarting local unit ${unit}" >&2
    systemctl restart "${unit}"
fi
