#!/bin/bash
# Pack the git-ignored files the Virtual Wall nodes need into two archives for Google Drive.
#
#   qlever_bundle.tar.zst    the YAGO QLever index, its Qleverfile / env, and the deploy and
#                            pin scripts (~1 GB). Not the 5 GB yago.nt dump: QLever does not
#                            read it at query time.
#   trainer_bundle.tar.zst   what online amortized-DP training reads: the mixed_yago_hll
#                            train/val queries, RDF2Vec vectors, term statistics, the
#                            pretrained oracle / embedder, the trial-85 epinet run, the
#                            held-out split, the label cache and the offline value-net runs.
#
# Paths inside the archives are relative to the repository root, so the setup script
# extracts them straight into the clone. Usage, from anywhere:
#   dockerfiles/virtual_wall/make_data_bundles.sh [output_dir]     (default: ~/data_bundles)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$HOME/data_bundles}"
mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

Q=data/qlever/qlever_yago
O=experiments/experiment_outputs/mixed_yago

QLEVER_FILES=(
  data/qlever/deploy_isolated_qlever_instances.py
  data/qlever/pin_index_in_memory.py
  "$Q"/qleverfile_default
  "$Q"/qlever_yago.env
  "$Q"/yago.settings.json
  "$Q"/yago.meta-data.json
)
QLEVER_FILES+=("$Q"/yago.index.* "$Q"/yago.internal.* "$Q"/yago.vocabulary.*)

TRAINER_FILES=(
  data/generated_queries/mixed_yago_hll/dataset_train
  data/generated_queries/mixed_yago_hll/dataset_val
  data/rdf2vec_embeddings/mixed_yago/model.json
  data/term_occurrences/mixed_yago
  "$O"/pretrained_models/pretrain_experiment_huge_oracle_graph_norm_hll.yaml-31-05-2026-22-46-06
  "$O"/pretrained_models/pretrain_experiment_triple_conv_hll.yaml-27-05-2026-15-37-01
  "$O"/supervised_epinet_training/simulated_cost-02-06-2026-09-47-43
  "$O"/epinet_rerun_trial85/split.json
  "$O"/amortized_dp/labels
)
# Every finished offline run, so init_checkpoint "latest:<layer>" resolves the same way as
# here (tar keeps modification times, which is what "latest" sorts on).
TRAINER_FILES+=("$O"/amortized_dp/run-*)

pack() {
  local name="$1"; shift
  for path in "$@"; do
    [ -e "$path" ] || { echo "Missing: $path" >&2; exit 1; }
  done
  echo "Packing $name ($(du -shc "$@" | tail -1 | cut -f1) uncompressed)..."
  tar -cf - "$@" | zstd -T0 -6 -q -o "$OUT_DIR/$name" -f
  (cd "$OUT_DIR" && sha256sum "$name" > "$name.sha256")
  echo "  -> $OUT_DIR/$name ($(du -sh "$OUT_DIR/$name" | cut -f1))"
}

pack qlever_bundle.tar.zst "${QLEVER_FILES[@]}"
pack trainer_bundle.tar.zst "${TRAINER_FILES[@]}"

echo
echo "Upload both .tar.zst files to Google Drive, set each to 'Anyone with the link',"
echo "and put the share links in QLEVER_BUNDLE_URL / TRAINER_BUNDLE_URL in setup_node.sh."
