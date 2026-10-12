#!/bin/bash
# Count random sub-queries on all QLever endpoints of the slice (labels for training the estimator
# on sub-queries, src/datastructures/subquery_labels.py). Run it on a QLever node (it has the
# QLever access token in data/qlever/qlever_yago/qlever_yago.env, needed for counts above the
# server's 60 s limit), e.g. in tmux:
#
#   cd ~/rl-based-join-optimization && git pull
#   QLEVER_HOSTS="10.2.32.224,10.2.32.214,..." bash dockerfiles/virtual_wall/run_subquery_counts.sh
#
# Needs ~/subquery_tasks.jsonl.gz (made locally by `subquery_labels sample`, copied with scp).
# Without QLEVER_HOSTS the endpoints come from the online config's execution section.
# One thread per endpoint pulls from a shared queue: slow counts occupy one endpoint each, the
# fast ones keep flowing. Results are appended to ~/subquery_results.jsonl as they finish; rerun
# the same command after an interruption and it continues. Copy that file back when done.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/reschauz/rl-based-join-optimization}"
TASKS="${TASKS:-$HOME/subquery_tasks.jsonl.gz}"
RESULTS="${RESULTS:-$HOME/subquery_results.jsonl}"
TIMEOUT="${TIMEOUT:-600}"
cd "$REPO_DIR"

[ -f "$TASKS" ] || { echo "Copy the task file to $TASKS first." >&2; exit 1; }
if [ ! -x .venv/bin/python ]; then
  echo "Creating the Python environment..."
  [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  "$HOME/.local/bin/uv" venv --seed --python 3.10 .venv
  .venv/bin/pip install -r requirements.txt
fi

ENDPOINTS=()
if [ -n "${QLEVER_HOSTS:-}" ]; then
  PORTS="${PORTS_PER_HOST:-15}"; BASE="${BASE_PORT:-7000}"
  list=""
  for host in ${QLEVER_HOSTS//,/ }; do
    for ((p = BASE; p < BASE + PORTS; p++)); do list="${list:+$list,}http://${host#http://}:$p"; done
  done
  ENDPOINTS=(--endpoints "$list")
fi

PYTHONPATH="$REPO_DIR" .venv/bin/python -u -m src.datastructures.subquery_labels count \
  --tasks "$TASKS" --out "$RESULTS" --timeout "$TIMEOUT" "${ENDPOINTS[@]}" 2>&1 | tee -a "$HOME/subquery_counts.log"
echo "Done. Copy $RESULTS back, e.g.: scp <node>:$RESULTS data/generated_queries/mixed_yago_v2/subquery_labels/"
