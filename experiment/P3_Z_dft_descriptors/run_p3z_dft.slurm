#!/bin/bash
#SBATCH --job-name=p3z_dft
#SBATCH --mail-user=xd2484@partner.kit.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# --- paths (nano-xd / xeco layout) ---
XECO_ROOT="/home/ws/xd2484/xeco"
SCRIPT_DIR="${XECO_ROOT}/experiment/P3_Z_dft_descriptors"
CONDA_ENV="${P3Z_CONDA_ENV:-p3z_dft}"

# --- conda ---
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${XECO_ROOT}"

# --- P3-Z pipeline inputs/outputs ---
export P3Z_SCRIPT_DIR="${SCRIPT_DIR}"
export P3Z_INPUT_PARQUET="${XECO_ROOT}/Datasets/data/curated_molecules_with_splits.parquet"
export P3Z_CLASSICAL_FEATURES="${XECO_ROOT}/Datasets/data/features_combined.parquet"
export MPLCONFIGDIR="${SCRIPT_DIR}/artifacts/_matplotlib"

# Threading: one PySCF worker locally, use all allocated CPUs for BLAS/OpenMP
export OMP_NUM_THREADS="${SLURM_NTASKS:-16}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-16}"

# --- run mode ---
# Smoke test (default): 1 molecule, STO-3G, 3 geom steps (set in script).
# Full production run:
#   sbatch --export=ALL,P3Z_MAX_MOLECULES=0 run_p3z_dft.slurm
export P3Z_MAX_MOLECULES="${P3Z_MAX_MOLECULES:-1}"

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
python --version
echo "xTB: $(command -v xtb || echo 'NOT FOUND')"
xtb --version 2>&1 | head -3 || true
echo "P3Z_MAX_MOLECULES=${P3Z_MAX_MOLECULES}"
echo "Input: ${P3Z_INPUT_PARQUET}"
echo "Output dir: ${SCRIPT_DIR}/artifacts"
echo "---"

python "${SCRIPT_DIR}/p3z_dft_descriptors.py"

echo "---"
echo "Done: $(date -Is)"
