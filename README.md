# Variance-Calibrated Cross-Paradigm 2D Modeling for Activity Cliff Prediction

Official reference implementation and reproducible benchmark summary tables for variance-calibrated pure-2D activity cliff prediction across all 30 MoleculeACE benchmark targets.

## Overview

Activity cliffs (pairs of structurally similar compounds with steep potency disparities) pose a fundamental challenge for quantitative structure-activity relationship (QSAR) modeling. This repository implements:

1. **Pure-2D Cross-Paradigm Fusion**: Combines a composite positive-semidefinite multi-kernel SVR (`K_Multi_SVR`: 0.70 ECFP4 Tanimoto + 0.15 MACCS Tanimoto + 0.15 Desc2D RBF) with an antisymmetric pairwise potency-difference network (`Y_A2_2D` / zero-initialized 2D-LFER conditioning `Y_FiLM_2DLFER`) on all 30 MoleculeACE targets (48,714 compounds, 5 random seeds).
2. **Leak-Free, Split-Matched Variance Calibration**: Analytical 3-fold nested out-of-fold Bates-Granger calibration (`OptW` for stratified splits; Murcko `GroupKFold` `GroupOptW` for strict zero-overlap scaffold holdouts) and zero-parameter sample-size shrinkage (`PriorN`), automatically shifting weight toward `K_Multi_SVR` under $n < 1000$ or far out-of-distribution scaffold shifts ($s_{\max} < 0.50$).
3. **Counterfactual Dissection of Ligand-Only Failure Modes**: Controlled 10-target diagnostic ablations isolating why vacuum ETKDG 3D conformers, as-provided SMILES `AIMNet2` charges, reused-validation gating, out-of-fold residual stacking (`KR_*`), and GINE + cross-attention smoothing degrade ligand-only activity cliff resolution.

## Repository Structure

```text
variance-calibrated-cliff-2d/
├── physchem_delta/               # Core featurization, models, anti-leakage tests, evaluation runners
│   ├── data.py                   # MoleculeACE loaders & Murcko scaffold splits
│   ├── featurize.py              # Chiral ECFP4, MACCS, Desc2D, radius-1 2D-LFER & optional 3D/QM features
│   ├── models.py                 # PSD K_Multi_SVR, Y_A2_2D, Y_FiLM_2DLFER, GINE+Attn, Bates-Granger OOF
│   ├── run_jcim_30targets.py     # 30-target main evaluation runner & pure-2D cache equivalence check
│   ├── eval_jcim_step_a.py       # Nested OOF OptW & PriorN benchmark
│   ├── eval_scaffold_robustness.py # 10-target strict zero-overlap Murcko scaffold splits
│   ├── eval_opt13_modular.py     # Gated 3D/QM & GINE+CrossAttn counterfactual ablations
│   ├── eval_path_a_modular.py    # OOF kernel-residual stacking vs y-space controls
│   └── tests_leakage.py          # 5/5 automated anti-leakage & determinism contract tests
└── refine-logs/                  # Verified summary CSVs across 30-target, scaffold, and 10-target suites
```

## Quick Start

```bash
pip install -r requirements.txt
python physchem_delta/tests_leakage.py
```

### Running the Benchmarks

```bash
# 1. Base 30-target composite multi-kernel & pairwise 2D benchmark
python physchem_delta/run_jcim_30targets.py

# 2. Leak-free 3-fold nested OOF Bates-Granger calibration (OptW & PriorN)
python physchem_delta/eval_jcim_step_a.py

# 3. Strict zero-overlap Bemis-Murcko scaffold split evaluation (GroupOptW)
python physchem_delta/eval_scaffold_robustness.py
```

## Citation

If you find this codebase or benchmark results useful in your research, please cite:

```bibtex
@article{variance_calibrated_cliff_2d,
  title={Variance-Calibrated Cross-Paradigm 2D Modeling for Activity Cliff Prediction: When Split-Matched Ensembling Outperforms Ligand-Only 3D, Quantum, and Graph Neural Architectures},
  author={Li, Zhuofan},
  note={Manuscript under review; citation details will be updated upon publication},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE).
