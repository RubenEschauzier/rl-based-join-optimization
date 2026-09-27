#!/usr/bin/env bash
# Submit the trial-85 epinet rerun to imec GPULab.
#
# Defaults mirror the job that ran the Optuna sweep (image, cluster, storage, SSH key).
# The image's code is NOT baked in: it is read from the /project_ghent mount, so the
# repository on project storage must contain the rerun code before submitting.
#
# Needs gpulab-cli and GPULAB_CERT. Examples:
#   PROJECT=phdexperimentsruben CLUSTER_ID=5 dockerfiles/gpulab/submit_epinet_rerun.sh
#       -> 10 jobs, one seed each
#   SEEDS="3" ...
#       -> a single job for seed 3
#   SEEDS_PER_JOB=2 ...
#       -> 5 jobs, each training 2 seeds AT THE SAME TIME on its one GPU
#   SEEDS_PER_JOB=2 PACKING=sequential ...
#       -> 5 jobs, each training 2 seeds one after the other
#   DRY_RUN=1 ...
#       -> print the job JSON, submit nothing
set -euo pipefail

: "${PROJECT:?Set PROJECT to your GPULab project name}"
IMAGE="${IMAGE:-rubeneschauzier/epinet-training-sweep:latest}"
: "${CLUSTER_ID:?Set CLUSTER_ID (see gpulab-cli clusters); the sweep ran on 5}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SEEDS_PER_JOB="${SEEDS_PER_JOB:-1}"
# parallel: one process per seed on the same GPU (CPU memory and CPUs scale with it).
# sequential: one process that trains its seeds back to back (same memory as one seed).
PACKING="${PACKING:-parallel}"
SSH_PUB_KEY="${SSH_PUB_KEY:-ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINrM6/FhK42PO98OzVO8kgVCx9ea9yfvmYhpcWmR1ja3}"
# Same scheduling as the sweep: never halted before 7 days, cancelled after 14. One seed
# is estimated at ~8 h on an RTX 3080, so a job should never get near either limit.
MIN_DURATION="${MIN_DURATION:-7 days}"
MAX_DURATION="${MAX_DURATION:-14 days}"
DRY_RUN="${DRY_RUN:-0}"
JOB_ID_LOG="${JOB_ID_LOG:-epinet_rerun_trial85_job_ids.txt}"

read -r -a seed_list <<< "${SEEDS}"
if [[ "${PACKING}" != "parallel" && "${PACKING}" != "sequential" ]]; then
  echo "PACKING must be 'parallel' or 'sequential', got '${PACKING}'" >&2
  exit 1
fi

for (( start = 0; start < ${#seed_list[@]}; start += SEEDS_PER_JOB )); do
  group=("${seed_list[@]:start:SEEDS_PER_JOB}")
  group_name=$(IFS=-; echo "${group[*]}")
  processes=1
  [[ "${PACKING}" == "parallel" ]] && processes=${#group[@]}
  # Peak RSS was measured at 21.6 GB per process (the full datasets are loaded up front);
  # the sweep ran at 25 GB. 28 GB per process leaves headroom for the test split.
  cpu_memory_gb="${CPU_MEMORY_GB:-$(( 28 * processes ))}"
  cpus="${CPUS:-$(( 4 * processes ))}"

  # Exec-form commands keep python as PID 1, so it receives GPULab's SIGUSR1 halt signal
  # and can answer with exit 123 (restartable then re-queues; finished seeds are skipped).
  if (( ${#group[@]} == 1 )); then
    command_json="[\"python\", \"-m\", \"src.supervised_value_estimation.rerun_epinet_seeds\", \"rerun.seed=${group[0]}\"]"
  elif [[ "${PACKING}" == "parallel" ]]; then
    seed_args=$(printf ', "%s"' "${group[@]}")
    command_json="[\"python\", \"-m\", \"src.supervised_value_estimation.launch_seeds_parallel\"${seed_args}]"
  else
    command_json="[\"python\", \"-m\", \"src.supervised_value_estimation.rerun_epinet_seeds\", \"rerun.run_seeds=[$(IFS=,; echo "${group[*]}")]\"]"
  fi

  job_json=$(cat <<EOF
{
  "name": "epinet-rerun-trial85-seeds-${group_name}",
  "description": "Full-data rerun of epinet sweep trial 85 (seeds ${group[*]}, ${PACKING}), with a held-out test split. Needs a lot of RAM: the dataset is held in memory.",
  "request": {
    "docker": {
      "image": "${IMAGE}",
      "command": ${command_json},
      "environment": {"PYTHONUNBUFFERED": "1"},
      "portMappings": [],
      "storage": [
        {
          "hostPath": "/project_ghent",
          "containerPath": "/project_ghent"
        }
      ]
    },
    "resources": {
      "clusterId": ${CLUSTER_ID},
      "gpus": 1,
      "cpus": ${cpus},
      "cpuMemoryGb": ${cpu_memory_gb}
    },
    "scheduling": {
      "interactive": false,
      "restartable": true,
      "minDuration": "${MIN_DURATION}",
      "maxDuration": "${MAX_DURATION}",
      "reservationIds": []
    },
    "extra": {
      "sshPubKeys": ["${SSH_PUB_KEY}"]
    }
  }
}
EOF
)
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${job_json}"
    continue
  fi
  job_id=$(gpulab-cli submit --project "${PROJECT}" <<< "${job_json}")
  echo "seeds ${group[*]}: ${job_id}"
  echo "${group[*]} ${job_id}" >> "${JOB_ID_LOG}"
done
