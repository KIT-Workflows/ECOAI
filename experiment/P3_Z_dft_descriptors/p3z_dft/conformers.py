from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

def generate_best_conformer(smiles, n_confs=30, max_iters=500, seed=42):
    """
    Generate 3D conformers with ETKDG, MMFF-optimize, and
    return the lowest-energy 3D Mol object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, 0

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.useSmallRingTorsions = True

    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    except Exception:
        conf_ids = []

    if len(conf_ids) == 0:
        params.useRandomCoords = True
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        except Exception:
            pass
        if len(conf_ids) == 0:
            return None, None, 0

    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=0)
    except Exception:
        return mol, 0.0, len(conf_ids)

    best_conf_id = None
    best_energy = float("inf")
    for conf_id, (converged, energy) in enumerate(results):
        if energy < best_energy:
            best_energy = energy
            best_conf_id = conf_id

    if best_conf_id is None:
        return mol, 0.0, len(conf_ids)

    mol_best = Chem.RWMol(mol)
    all_conf_ids = [c.GetId() for c in mol_best.GetConformers()]
    for cid in all_conf_ids:
        if cid != best_conf_id:
            mol_best.RemoveConformer(cid)

    return mol_best, best_energy, len(conf_ids)

