#!/bin/bash
# Launch a robust 3-4 week P3-Z campaign on int-nano (all QC-pass molecules).
#
# Cluster: batch partition, nano CPU nodes (128c / ~250G), unlimited walltime.
# Strategy:
#   1) Build one shared conformer cache
#   2) Run parallel shard workers (default 8 × 24 CPUs on nano nodes)
#   3) Auto-resubmit incomplete shards every 7 days
#   4) Merge shard checkpoints and build final outputs
#
# Usage:
#   ssh nano-xd
#   cd /home/ws/xd2484/xeco/experiment/P3_Z_dft_descriptors
#   ./submit_p3z_campaign.sh start
#   ./submit_p3z_campaign.sh status
#   ./submit_p3z_campaign.sh finalize

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_p3z_env.sh
source "${SCRIPT_DIR}/slurm_p3z_env.sh"

cmd="${1:-start}"

common_export="ALL,P3Z_ARTIFACTS_DIR=${ARTIFACTS_DIR},P3Z_NUM_SHARDS=${P3Z_NUM_SHARDS},P3Z_COMPUTE_VIBRATIONS=0,P3Z_COMPUTE_VERTICAL_EA=1"

sbatch_shard_resources=(
  --partition="${P3Z_SLURM_PARTITION}"
  --exclude="${P3Z_SLURM_EXCLUDE}"
  --cpus-per-task="${P3Z_CPUS_PER_TASK}"
  --mem="${P3Z_MEM}"
  --time="${P3Z_SHARD_TIME}"
)

case "${cmd}" in
  start)
    mkdir -p "${ARTIFACTS_DIR}"
    echo "Campaign settings:"
    echo "  XECO_ROOT=${XECO_ROOT}"
    echo "  ARTIFACTS_DIR=${ARTIFACTS_DIR}"
    echo "  NUM_SHARDS=${P3Z_NUM_SHARDS}"
    echo "  CPUs/shard=${P3Z_CPUS_PER_TASK}  mem=${P3Z_MEM}  time=${P3Z_SHARD_TIME}"
    echo "  exclude=${P3Z_SLURM_EXCLUDE}"
    echo

    echo "[1/2] Submitting conformer preparation..."
    prep_job="$(sbatch \
      --partition="${P3Z_SLURM_PARTITION}" \
      --exclude="${P3Z_SLURM_EXCLUDE}" \
      --export="${common_export}" \
      "${SCRIPT_DIR}/run_p3z_prepare_conformers.slurm" | awk '{print $4}')"
    echo "      prep job: ${prep_job}"

    echo "[2/2] Submitting shard array (depends on prep)..."
    array_job="$(sbatch \
      --dependency=afterok:"${prep_job}" \
      --export="${common_export},P3Z_AUTO_RESUBMIT_ON_INCOMPLETE=1" \
      --array="0-$((P3Z_NUM_SHARDS - 1))" \
      "${sbatch_shard_resources[@]}" \
      "${SCRIPT_DIR}/run_p3z_dft_array.slurm" | awk '{print $4}')"
    echo "      array job: ${array_job}"
    echo
    echo "Monitor:"
    echo "  squeue -u \"\$USER\" | grep p3z"
    echo "  ./submit_p3z_campaign.sh status"
    ;;
  status)
    p3z_activate_conda
    cd "${XECO_ROOT}"
    p3z_export_campaign_env
    export P3Z_NUM_SHARDS="${P3Z_NUM_SHARDS}"
    python "${SCRIPT_DIR}/p3z_campaign_status.py"
    ;;
  queue)
    echo "SLURM queue (p3z jobs):"
    squeue -u "${USER}" | grep -E "JOBID|p3z" || echo "  (no p3z jobs in queue)"
    ;;
  finalize)
    echo "Submitting merge + finalize job..."
    sbatch \
      --partition="${P3Z_SLURM_PARTITION}" \
      --exclude="${P3Z_SLURM_EXCLUDE}" \
      --export="${common_export}" \
      "${SCRIPT_DIR}/run_p3z_finalize.slurm"
    ;;
  *)
    echo "Usage: $0 {start|status|queue|finalize}"
    exit 1
    ;;
esac
