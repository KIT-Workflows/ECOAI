#!/bin/bash
# Shared runtime settings for P3-Z jobs on int-nano (ssh nano-xd).
#
# Cluster notes (2026-07):
#   - Default partition: batch (unlimited walltime; default limit 10:00:00 if --time omitted)
#   - CPU nodes: nano[03-26], 128 cores / ~250 GB RAM (some nodes 512 GB–1 TB)
#   - GPU nodes in batch: gpu[02-15], 32 cores / ~125 GB — not used for PySCF here
#   - DefMemPerCPU: 2000 MB (2 GB per requested CPU)
#   - PreemptMode: REQUEUE on batch (killed jobs may be requeued automatically)
#
# Override any variable before sbatch, e.g.:
#   export P3Z_CPUS_PER_TASK=16
#   sbatch run_p3z_dft_array.slurm

# --- project paths (login node: int-nano; workspace under /home/ws/xd2484) ---
export XECO_ROOT="${XECO_ROOT:-/home/ws/xd2484/xeco}"
export SCRIPT_DIR="${SCRIPT_DIR:-${XECO_ROOT}/experiment/P3_Z_dft_descriptors}"
export CONDA_ENV="${P3Z_CONDA_ENV:-p3z_dft}"
export ARTIFACTS_DIR="${P3Z_ARTIFACTS_DIR:-${SCRIPT_DIR}/artifacts_prod_electronic_v1}"

# --- SLURM resource defaults (CPU-only PySCF; avoid GPU nodes) ---
export P3Z_SLURM_PARTITION="${P3Z_SLURM_PARTITION:-batch}"
export P3Z_SLURM_EXCLUDE="${P3Z_SLURM_EXCLUDE:-gpu[01-15],bionano06,a100}"
export P3Z_CPUS_PER_TASK="${P3Z_CPUS_PER_TASK:-24}"
export P3Z_MEM="${P3Z_MEM:-64G}"
export P3Z_SHARD_TIME="${P3Z_SHARD_TIME:-7-00:00:00}"
export P3Z_PREP_TIME="${P3Z_PREP_TIME:-1-00:00:00}"
export P3Z_FINALIZE_TIME="${P3Z_FINALIZE_TIME:-2:00:00}"
export P3Z_NUM_SHARDS="${P3Z_NUM_SHARDS:-8}"

# --- pipeline inputs ---
export P3Z_SCRIPT_DIR="${SCRIPT_DIR}"
export P3Z_ARTIFACTS_DIR="${ARTIFACTS_DIR}"
export P3Z_INPUT_PARQUET="${P3Z_INPUT_PARQUET:-${XECO_ROOT}/experiment/P2_scaffold_splitting/artifacts/curated_molecules_with_splits.parquet}"
export P3Z_CLASSICAL_FEATURES="${P3Z_CLASSICAL_FEATURES:-${XECO_ROOT}/experiment/P3_feature_engineering/artifacts/features_combined.parquet}"
export MPLCONFIGDIR="${ARTIFACTS_DIR}/_matplotlib"

p3z_activate_conda() {
    # shellcheck source=/dev/null
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
}

p3z_set_threading() {
    # One PySCF worker per task; BLAS/OpenMP uses all CPUs in the allocation.
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${P3Z_CPUS_PER_TASK}}"
    export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
    export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
    export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
}

p3z_export_campaign_env() {
    export P3Z_MAX_MOLECULES=0
    export P3Z_SELECTION_MODE=all_qc_pass
    export P3Z_PILOT_MODE=none
    export P3Z_DFT_BASIS="${P3Z_DFT_BASIS:-6-31g*}"
    export P3Z_COMPUTE_VIBRATIONS=0
    export P3Z_COMPUTE_IR_SPECTRA=0
    export P3Z_COMPUTE_HD_SHIFTS=0
    export P3Z_COMPUTE_VERTICAL_EA=1
    export P3Z_SKIP_VALIDATION=1
}
