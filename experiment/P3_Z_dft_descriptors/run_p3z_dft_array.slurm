#!/bin/bash
#SBATCH --job-name=p3z_shard
#SBATCH --mail-user=xd2484@partner.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --partition=batch
#SBATCH --exclude=gpu[01-15],bionano06,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

XECO_ROOT="${XECO_ROOT:-/home/ws/xd2484/xeco}"
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-${XECO_ROOT}/experiment/P3_Z_dft_descriptors}}"
# shellcheck source=slurm_p3z_env.sh
source "${SCRIPT_DIR}/slurm_p3z_env.sh"

NUM_SHARDS="${P3Z_NUM_SHARDS}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${P3Z_SHARD_INDEX:-0}}"

p3z_activate_conda
cd "${XECO_ROOT}"
p3z_set_threading

export P3Z_MAX_MOLECULES=0
export P3Z_SELECTION_MODE=all_qc_pass
export P3Z_PILOT_MODE=none
export P3Z_NUM_SHARDS="${NUM_SHARDS}"
export P3Z_SHARD_INDEX="${SHARD_INDEX}"
export P3Z_SKIP_VALIDATION=1
export P3Z_CHECKPOINT_EVERY=1
export P3Z_DFT_BASIS="${P3Z_DFT_BASIS:-6-31g*}"
export P3Z_MAX_SCF_CYCLES="${P3Z_MAX_SCF_CYCLES:-200}"
export P3Z_SCF_CONV_TOL="${P3Z_SCF_CONV_TOL:-1e-9}"
export P3Z_GEOMOPT_MAX_STEPS="${P3Z_GEOMOPT_MAX_STEPS:-100}"
export P3Z_COMPUTE_VIBRATIONS=0
export P3Z_COMPUTE_VERTICAL_EA=1
export P3Z_AUTO_RESUBMIT_ON_INCOMPLETE="${P3Z_AUTO_RESUBMIT_ON_INCOMPLETE:-1}"

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Partition: ${SLURM_JOB_PARTITION:-batch}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?} | Mem: ${SLURM_MEM_PER_NODE:-?}"
echo "Array task: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Artifacts: ${ARTIFACTS_DIR}"
echo "---"

set +e
python "${SCRIPT_DIR}/p3z_dft_descriptors.py"
pipeline_exit=$?
set -e

python "${SCRIPT_DIR}/p3z_campaign_status.py" --shard "${SHARD_INDEX}"
status_exit=$?

if [[ "${P3Z_AUTO_RESUBMIT_ON_INCOMPLETE}" == "1" && "${status_exit}" -ne 0 ]]; then
  echo "[RESUBMIT] Shard ${SHARD_INDEX} incomplete; queueing continuation (7-day slot)."
  sbatch \
    --chdir="${SCRIPT_DIR}" \
    --dependency=afterany:"${SLURM_JOB_ID}" \
    --partition="${P3Z_SLURM_PARTITION}" \
    --exclude="${P3Z_SLURM_EXCLUDE}" \
    --cpus-per-task="${P3Z_CPUS_PER_TASK}" \
    --mem="${P3Z_MEM}" \
    --time="${P3Z_SHARD_TIME}" \
    --array="${SHARD_INDEX}" \
    --export=ALL,P3Z_ARTIFACTS_DIR="${ARTIFACTS_DIR}",P3Z_NUM_SHARDS="${NUM_SHARDS}",P3Z_AUTO_RESUBMIT_ON_INCOMPLETE=1 \
    "${SCRIPT_DIR}/run_p3z_dft_array.slurm"
else
  echo "[OK] Shard ${SHARD_INDEX} complete or auto-resubmit disabled."
fi

exit "${pipeline_exit}"
