#!/usr/bin/env bash
# Run a long experiment while preserving its live terminal output in a file.
#
# Usage:
#   scripts/run_logged_job.sh path/to/terminal.log command [arguments...]
#
# The training scripts render batch progress with carriage returns. `tee` keeps
# that behaviour in the launching terminal and stores the same stream for a
# later `tail -F` session.
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 LOG_FILE COMMAND [ARGUMENT ...]" >&2
  exit 2
fi

log_file=$1
shift
mkdir -p "$(dirname "$log_file")"
printf '[launch] %q ' "$@" | tee -a "$log_file"
printf '\n' | tee -a "$log_file"

"$@" 2>&1 | tee -a "$log_file"
