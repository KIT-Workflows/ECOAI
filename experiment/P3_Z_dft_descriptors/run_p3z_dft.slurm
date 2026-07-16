#!/bin/bash
#SBATCH --job-name=p3z_dft
#SBATCH --mail-user=xd2484@partner.kit.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=batch
#SBATCH --exclude=gpu[01-15],bionano06,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
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

# --- run mode ---
# Pilot (16 molecules, production basis, optional vibrations):
#   sbatch --export=ALL,P3Z_ARTIFACTS_DIR=...,P3Z_MAX_MOLECULES=0,P3Z_PILOT_MODE=representative,P3Z_PILOT_SIZE=16,P3Z_COMPUTE_VIBRATIONS=0 run_p3z_dft.slurm
# Single-node full QC-pass (slow; prefer submit_p3z_campaign.sh):
#   sbatch --export=ALL,P3Z_MAX_MOLECULES=0,P3Z_PILOT_MODE=none,P3Z_COMPUTE_VIBRATIONS=0 run_p3z_dft.slurm
export P3Z_MAX_MOLECULES="${P3Z_MAX_MOLECULES:-0}"
export P3Z_SELECTION_MODE="${P3Z_SELECTION_MODE:-all_qc_pass}"
export P3Z_PILOT_MODE="${P3Z_PILOT_MODE:-none}"
export P3Z_PILOT_SIZE="${P3Z_PILOT_SIZE:-0}"

export P3Z_DFT_BASIS="${P3Z_DFT_BASIS:-6-31g*}"
export P3Z_MAX_SCF_CYCLES="${P3Z_MAX_SCF_CYCLES:-200}"
export P3Z_SCF_CONV_TOL="${P3Z_SCF_CONV_TOL:-1e-9}"
export P3Z_GEOMOPT_MAX_STEPS="${P3Z_GEOMOPT_MAX_STEPS:-100}"
export P3Z_VIBRATION_SCF_MAX_CYCLES="${P3Z_VIBRATION_SCF_MAX_CYCLES:-200}"
export P3Z_VIBRATION_SCF_CONV_TOL="${P3Z_VIBRATION_SCF_CONV_TOL:-1e-9}"
export P3Z_ANION_SCF_MAX_CYCLES="${P3Z_ANION_SCF_MAX_CYCLES:-200}"
export P3Z_ANION_SCF_CONV_TOL="${P3Z_ANION_SCF_CONV_TOL:-1e-9}"

export P3Z_COMPUTE_VIBRATIONS="${P3Z_COMPUTE_VIBRATIONS:-0}"
export P3Z_COMPUTE_IR_SPECTRA="${P3Z_COMPUTE_IR_SPECTRA:-0}"
export P3Z_COMPUTE_HD_SHIFTS="${P3Z_COMPUTE_HD_SHIFTS:-0}"
export P3Z_COMPUTE_VERTICAL_EA="${P3Z_COMPUTE_VERTICAL_EA:-1}"
export P3Z_CHECKPOINT_EVERY="${P3Z_CHECKPOINT_EVERY:-1}"

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Partition: ${SLURM_JOB_PARTITION:-batch}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?} | Mem: ${SLURM_MEM_PER_NODE:-?}"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
python --version
echo "xTB: $(command -v xtb || echo 'NOT FOUND')"
xtb --version 2>&1 | head -3 || true
echo "P3Z_MAX_MOLECULES=${P3Z_MAX_MOLECULES}"
echo "P3Z_SELECTION_MODE=${P3Z_SELECTION_MODE}"
echo "P3Z_PILOT_MODE=${P3Z_PILOT_MODE}"
echo "P3Z_PILOT_SIZE=${P3Z_PILOT_SIZE}"
echo "P3Z_DFT_BASIS=${P3Z_DFT_BASIS}"
echo "Stages: vib=${P3Z_COMPUTE_VIBRATIONS} ir=${P3Z_COMPUTE_IR_SPECTRA} hd=${P3Z_COMPUTE_HD_SHIFTS} vea=${P3Z_COMPUTE_VERTICAL_EA}"
echo "Input: ${P3Z_INPUT_PARQUET}"
echo "Output dir: ${P3Z_ARTIFACTS_DIR}"
echo "---"

python "${SCRIPT_DIR}/p3z_dft_descriptors.py"

echo "---"
echo "Done: $(date -Is)"
