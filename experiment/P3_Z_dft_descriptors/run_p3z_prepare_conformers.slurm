#!/bin/bash
#SBATCH --job-name=p3z_conf
#SBATCH --mail-user=xd2484@partner.kit.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=batch
#SBATCH --exclude=gpu[01-15],bionano06,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

XECO_ROOT="${XECO_ROOT:-/home/ws/xd2484/xeco}"
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-${XECO_ROOT}/experiment/P3_Z_dft_descriptors}}"
# shellcheck source=slurm_p3z_env.sh
source "${SCRIPT_DIR}/slurm_p3z_env.sh"

p3z_activate_conda
cd "${XECO_ROOT}"
p3z_set_threading

export P3Z_MAX_MOLECULES=0
export P3Z_SELECTION_MODE=all_qc_pass
export P3Z_PILOT_MODE=none
export P3Z_NUM_SHARDS=1
export P3Z_PREPARE_CONFORMERS_ONLY=1
export P3Z_SKIP_VALIDATION=1

echo "Host: $(hostname)"
echo "Preparing shared conformer cache (~6100 molecules) in ${ARTIFACTS_DIR}"
python "${SCRIPT_DIR}/p3z_dft_descriptors.py"
echo "Done: $(date -Is)"
