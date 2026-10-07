#!/bin/bash
# Statistics-based query generation on a QLever node of the slice.
#
#   run_query_generation.sh "[cycle]"            # one or more shapes, hydra list syntax
#   run_query_generation.sh "[path,tree]" 1      # optional seed
#
# The generator works in memory (graph + exact counting) on this node's CPU cores; the node's
# own QLever instances (localhost:7000-7014) only COUNT the few cyclic subsets too large to
# count in memory. Do not run online training while this runs: it shares the QLever cores.
#
# Needs, once per node:
#   - the repository with the generation code (git pull),
#   - the graph cache: ~/generation_graph.tar.zst (made locally with
#       tar -C data/statistics_based_generation -cf - yago/graph.npz yago/statistics.npz | zstd -T0 -3 -o generation_graph.tar.zst
#     and copied here, e.g. scp from the trainer), extracted into data/statistics_based_generation/,
#   - a Python environment: created below like the trainer's (uv + requirements.txt).
# Output: data/generated_queries/statistics_based_yago/<hostname>/statistics_based_<shape>.json
set -euo pipefail

SHAPES="${1:?usage: $0 \"[shape,...]\" [seed]}"
SEED="${2:-0}"
REPO_DIR="${REPO_DIR:-/users/reschauz/rl-based-join-optimization}"
cd "$REPO_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Creating the Python environment..."
  [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  "$HOME/.local/bin/uv" venv --seed --python 3.10 .venv
  .venv/bin/pip install -r requirements.txt
fi

if [ ! -f data/statistics_based_generation/yago/graph.npz ]; then
  [ -f "$HOME/generation_graph.tar.zst" ] || { echo "Copy generation_graph.tar.zst to $HOME first." >&2; exit 1; }
  mkdir -p data/statistics_based_generation
  zstd -dc "$HOME/generation_graph.tar.zst" | tar -xf - -C data/statistics_based_generation
fi

# One worker per physical core, minus one for the parent process (it collects and counts).
PHYSICAL=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
WORKERS=$(( PHYSICAL > 2 ? PHYSICAL - 1 : 1 ))
PORTS=$(docker ps --filter name=qlever_core -q | wc -l)
OUT="data/generated_queries/statistics_based_yago/$(hostname -s)"
mkdir -p "$OUT"
echo "Generating $SHAPES with $WORKERS workers, counting on $PORTS local QLever endpoints -> $OUT"

.venv/bin/python -m src.random_query_generation.statistics_based_generation.generate \
  generation.mode=generate "generation.shapes=$SHAPES" generation.seed="$SEED" generation.workers="$WORKERS" \
  "generation.endpoints.endpoint_hosts=[localhost]" generation.endpoints.ports_per_host="$PORTS" \
  generation.output_directory="$OUT" 2>&1 | grep --line-buffered -v "Counting true cardinalities" | tee "$OUT.log"
