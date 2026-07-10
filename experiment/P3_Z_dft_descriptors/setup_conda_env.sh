#!/bin/bash
# One-time setup on nano-xd: create the p3z_dft conda env from environment.yml.
# dftd3 comes from conda-forge (dftd3-python), not pip, to avoid compiling
# against the cluster's old system GCC when building from source.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${P3Z_CONDA_ENV:-p3z_dft}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Update with:"
  echo "  conda env update -n ${ENV_NAME} -f ${SCRIPT_DIR}/environment.yml --prune"
  exit 0
fi

conda env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
conda activate "${ENV_NAME}"

python - <<'PY'
import importlib
import shutil

from dftd3.library import get_api_version

mods = ["numpy", "pandas", "scipy", "pyarrow", "matplotlib", "tqdm", "h5py", "rdkit", "pyscf", "dftd3"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}=={getattr(mod, '__version__', '?')}")
print(f"dftd3_api=={get_api_version()}")
print("xtb:", shutil.which("xtb"))
PY

echo "Setup complete. Activate with: conda activate ${ENV_NAME}"
