# Repellent/Insecticide Modeling Plan With Staged Quantum Descriptors

## Summary
- Build the project in two stages. `V1` is a chemically curated, probability-calibrated repellency model using standard cheminformatics features. `V2` adds electronic-structure descriptors after `V1` has identified the most informative compounds.
- Use the paper’s strategy as the template: feature-tokenized transformer for tabular molecular descriptors, frozen-backbone fine-tuning on small labeled data, calibrated uncertainty with cross-validation plus conformal prediction, and post-hoc interpretability with SHAP/attention.
- Do **not** implement a 3-class softmax model initially. `non-repellent`, `repellent`, and `insecticidal` are not cleanly exclusive, and the LifeChemicals insecticide file is currently an **unlabeled screening pool**, not supervised insecticide ground truth.
- Treat “effectiveness” as a **calibrated probability of activity** in `V1`. Only treat effectiveness as potency after real numeric labels such as `% repellency`, `% mortality`, `LC50`, or `LD50` are added.

## Implementation Changes
- Create a reproducible data-curation pipeline that reads the three SDFs, strips salts/counterions, canonicalizes structures, keeps one canonical record per InChIKey, preserves source metadata, and flags malformed/placeholder records in the decoy file for exclusion.
- Define the first supervised endpoint as `repellent_active` with `1` from the repellent SDF and `0` from curated non-repellent decoys. Use `non-repellent` as the complement of this probability, not as a separate head.
- Restrict the first production model to the dominant assay scope: `Aedes aegypti` repellency. Keep the other species as metadata for later multi-domain expansion, not mixed into the first supervised target.
- Build `V1` with two model families trained on the same scaffold split: a strong baseline using Morgan fingerprints (`radius=2`, `2048` bits) plus RDKit 2D descriptors, and an `FTTransformer` over curated tabular features. Select the winner by scaffold-split `PR-AUC`, `ROC-AUC`, `MCC`, `Brier score`, and calibration.
- Pretrain the `FTTransformer` backbone on all curated molecules from the three SDFs using self-supervised masked-feature reconstruction. Freeze the backbone and fine-tune only the head on the labeled repellency task, mirroring the paper’s transfer-learning approach.
- Export model outputs in a single prediction table with: `compound_id`, `canonical_smiles`, `scaffold_id`, `p_repellent`, `p_non_repellent=1-p_repellent`, `prediction_interval`, `ood_score`, and `split_id`.
- Add conformal uncertainty exactly as a first-class output. Use scaffold-aware 5-fold cross-validation for out-of-fold predictions, then fit calibration plus conformal intervals on those predictions so each molecule gets both a probability and an uncertainty band.
- Create a quantum-descriptor stage for `V2` that runs **after** `V1` and only on a selected subset. The default subset is `300` labeled molecules chosen by class balance and scaffold diversity (`150` repellent, `150` non-repellent) plus `200` unlabeled LifeChemicals molecules chosen from `100` highest `p_repellent` and `100` highest uncertainty.
- Use `GFN2-xTB` as the broad quantum layer. For each selected molecule, generate conformers with ETKDG, MMFF-optimize, keep the lowest-energy conformer, optimize with xTB, and extract at minimum: `HOMO`, `LUMO`, `gap`, `dipole`, `total energy`, `partial charge mean/std/max`, `polarizability if available`, and simple frontier-orbital-derived reactivity indices.
- Use DFT only as a validation/high-fidelity layer on a smaller subset of `100` molecules drawn from the xTB set with class balance and chemical diversity. Run single-point DFT on the xTB-optimized geometry to benchmark whether xTB descriptors are directionally reliable before expanding DFT usage.
- Train `V2` as an ablation study, not a blind replacement. Compare three feature sets on the same labeled quantum subset: cheminformatics-only, quantum-only, and fused features. Promote the fused model only if it improves scaffold-split discrimination **and** calibration.
- Keep the LifeChemicals library unlabeled in training. Use it for self-supervised pretraining and later screening/ranking only. Do not train an `insecticidal` head until external insecticide assay labels are added from ChEMBL, PubChem, or literature.
- Reserve the later `V3` interface now: once insecticide labels exist, extend the model to a multitask setup with independent outputs `p_repellent` and `p_insecticidal`, each with its own calibration and uncertainty, instead of collapsing them into one multiclass label.

## Public Interfaces and Artifacts
- Standardize on four project artifacts: `curated_molecules.parquet`, `split_manifest.json`, `model_predictions.parquet`, and `quantum_descriptors.parquet`.
- `curated_molecules.parquet` should contain identifiers, canonical structure fields, source dataset, label fields, assay/species metadata, scaffold, and a `qc_status` column for excluded/problematic molecules.
- `quantum_descriptors.parquet` should key by `compound_id` and store conformer provenance, method (`xtb` or `dft`), charge/multiplicity, convergence status, and the extracted electronic descriptors.
- `model_predictions.parquet` should expose only calibrated probabilities and uncertainty outputs that downstream ranking or simulation selection will consume.

## Test Plan
- Verify data curation is deterministic: repeated runs produce identical canonical SMILES, label counts, and exclusion counts.
- Verify no leakage: no Bemis-Murcko scaffold appears in both train and validation folds.
- Benchmark `V1` against random and majority baselines; require materially better `PR-AUC`, `MCC`, and `Brier score`.
- Check probability calibration with reliability plots and expected calibration error; reject any model with good AUC but poor calibration.
- For `V2`, evaluate gain only on the same labeled subset with quantum descriptors; require improvement in at least one ranking metric and one calibration metric before adopting fused features.
- For the DFT benchmark subset, require that xTB-derived HOMO/LUMO gap rankings correlate strongly enough with DFT to justify continued xTB use as the bulk descriptor engine.
- Run interpretability on the final selected model and confirm important features are chemically plausible rather than assay/source artifacts.

## Assumptions and Defaults
- The insecticide dataset obtained from LifeChemical is treated as an **unlabeled candidate library** until real insecticide activity labels are added.
- The first supervised endpoint is `Aedes aegypti` repellency because it is the dominant labeled target and gives the cleanest initial task.
- “Effectiveness” in the first release means calibrated probability of repellency, not potency. Potency modeling is deferred until numeric assay labels exist.
- The decoy SDF contains some malformed or placeholder structures; those will be excluded during curation rather than force-fit into training.
- The default quantum workflow is `xTB first, DFT second`, and quantum descriptors are used to improve a screened subset before any attempt to scale them to the full library.
