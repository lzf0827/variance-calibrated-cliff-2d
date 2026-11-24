"""Generate Figure 5: RDKit ACS-1996 2D Molecular Case Pairs + Local Descriptor Deltas & Predictions."""

from __future__ import annotations

import io
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jcim_plot_style import PALETTE, REFINELOGS_DIR, ROOT_DIR, apply_jcim_style, panel_tag, save_jcim_fig

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_plan_a import _lfer_atom_matrix
from physchem_delta.featurize import compute_local_pair_deltas, precompute_mol_env_counters
from physchem_delta.models import compute_tanimoto_matrix
from physchem_delta.run_jcim_30targets import extract_or_load_2d_features


def _select_case(target: str, role: str) -> dict[str, object]:
    """Deterministically recover exact verified WIN/FAIL case from data + OptW npz predictions."""
    ds = load_target_dataset(target)
    feats = extract_or_load_2d_features(target, ds.smiles)
    z = np.load(REFINELOGS_DIR / "jcim30_step_a_results" / "per_target" / f"{target}_optw_preds.npz")

    sim = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
    best_pos = np.argmax(sim, axis=1)
    s_max = sim[np.arange(ds.test_idx.size), best_pos]
    nearest_tr = ds.train_idx[best_pos]

    y_te = ds.y[ds.test_idx]
    y_anc = ds.y[nearest_tr]
    dy_true = y_te - y_anc

    pr_rbf = z["K_RBF_SVM_mean"].astype(np.float64)
    pr_mul = z["K_Multi_SVR_mean"].astype(np.float64)
    pr_mlp = z["E_Multi_DirectMLP_OptW_mean"].astype(np.float64)
    pr_film = z["E_Multi_FiLM_OptW_mean"].astype(np.float64)

    err_film = np.abs(y_te - pr_film)
    err_best_ctrl = np.minimum.reduce([np.abs(y_te - pr_rbf), np.abs(y_te - pr_mul), np.abs(y_te - pr_mlp)])
    elig = (s_max >= 0.75) & (np.abs(dy_true) >= 1.2)
    if not np.any(elig):
        elig = (s_max >= 0.70) & (np.abs(dy_true) >= 1.0)

    cand = np.where(elig)[0]
    margin = err_best_ctrl - err_film  # positive = OptW_FiLM beats best ctrl
    pos = int(cand[np.argmax(margin[cand])]) if role == "WIN" else int(cand[np.argmin(margin[cand])])

    lfer_atoms = _lfer_atom_matrix(feats)
    env_c = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)
    d_mat = compute_local_pair_deltas(
        np.array([ds.test_idx[pos]], dtype=np.int64),
        np.array([nearest_tr[pos]], dtype=np.int64),
        feats.atom_ptr,
        feats.atom_env_hash,
        lfer_atoms,
        env_counters=env_c,
    )[0]

    return {
        "target": target,
        "role": role,
        "test_smiles": ds.smiles[int(ds.test_idx[pos])],
        "anchor_smiles": ds.smiles[int(nearest_tr[pos])],
        "s_max": float(s_max[pos]),
        "y_true": float(y_te[pos]),
        "y_anc": float(y_anc[pos]),
        "dy_true": float(dy_true[pos]),
        "d_logp": float(d_mat[1]),
        "d_mr": float(d_mat[2]),
        "d_tpsa": float(d_mat[3]),
        "K_RBF_SVM": float(pr_rbf[pos]),
        "K_Multi_SVR": float(pr_mul[pos]),
        "OptW_MLP": float(pr_mlp[pos]),
        "OptW_FiLM": float(pr_film[pos]),
    }


def _render_aligned_pair_png(smiles_q: str, smiles_a: str, width: int = 640, height: int = 230) -> np.ndarray:
    """Align anchor -> query via MCS and depict side-by-side in ACS 1996 styling highlighting modified atoms/bonds."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_q = Chem.MolFromSmiles(smiles_q)
    mcs = rdFMCS.FindMCS([mol_a, mol_q], ringMatchesRingOnly=True, completeRingsOnly=False, timeout=5)
    core = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None

    rdDepictor.Compute2DCoords(mol_a)
    if core is not None and mol_a.HasSubstructMatch(core) and mol_q.HasSubstructMatch(core):
        rdDepictor.GenerateDepictionMatching2DStructure(mol_q, mol_a, refPatt=core)
        match_a = set(mol_a.GetSubstructMatch(core))
        match_q = set(mol_q.GetSubstructMatch(core))
    else:
        rdDepictor.Compute2DCoords(mol_q)
        match_a = set(range(mol_a.GetNumAtoms()))
        match_q = set(range(mol_q.GetNumAtoms()))

    hl_a = [i for i in range(mol_a.GetNumAtoms()) if i not in match_a]
    hl_q = [i for i in range(mol_q.GetNumAtoms()) if i not in match_q]

    col_a = (0.0, 0.45, 0.70, 0.48)   # Blue highlight on anchor diff
    col_q = (0.84, 0.37, 0.0, 0.48)   # Orange highlight on query diff
    hl_col_a = {i: col_a for i in hl_a}
    hl_col_q = {i: col_q for i in hl_q}

    def _diff_bonds(mol: Chem.Mol, hl_set: set[int]) -> list[int]:
        return [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in hl_set or b.GetEndAtomIdx() in hl_set]

    hb_a = _diff_bonds(mol_a, set(hl_a))
    hb_q = _diff_bonds(mol_q, set(hl_q))
    hbc_a = {b: col_a for b in hb_a}
    hbc_q = {b: col_q for b in hb_q}

    def _draw_one(
        mol: Chem.Mol,
        hl_atoms: list[int],
        hl_cols: dict[int, tuple[float, ...]],
        hl_bonds: list[int],
        hl_bcols: dict[int, tuple[float, ...]],
        legend: str,
    ) -> Image.Image:
        d = rdMolDraw2D.MolDraw2DCairo(width // 2, height)
        opts = d.drawOptions()
        rdMolDraw2D.SetACS1996Mode(opts, rdMolDraw2D.MeanBondLength(mol))
        opts.fixedBondLength = 21.0
        opts.bondLineWidth = 1.4
        opts.padding = 0.04
        opts.clearBackground = True
        opts.legendFontSize = 14
        d.DrawMolecule(
            mol,
            highlightAtoms=hl_atoms,
            highlightAtomColors=hl_cols,
            highlightBonds=hl_bonds,
            highlightBondColors=hl_bcols,
            legend=legend,
        )
        d.FinishDrawing()
        return Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")

    im_a = _draw_one(mol_a, hl_a, hl_col_a, hb_a, hbc_a, "Train Anchor")
    im_q = _draw_one(mol_q, hl_q, hl_col_q, hb_q, hbc_q, "Test Analog")

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(im_a, (0, 0))
    canvas.paste(im_q, (width // 2, 0))
    return np.asarray(canvas)


def main() -> None:
    apply_jcim_style(base_font=7.5)

    specs = [
        ("CHEMBL244_Ki", "WIN", "A", "Protease (CHEMBL244_Ki, WIN)"),
        ("CHEMBL4792_Ki", "WIN", "B", "Kinase (CHEMBL4792_Ki, WIN)"),
        ("CHEMBL3979_EC50", "WIN", "C", "Cellular EC50 (CHEMBL3979_EC50, WIN)"),
        ("CHEMBL4203_Ki", "FAIL", "D", r"Small-$n$ Limit (CHEMBL4203_Ki, $n=731$)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.45), gridspec_kw={"hspace": 0.46, "wspace": 0.12})
    for ax, (tgt, role, tag, hdr) in zip(axes.flat, specs):
        case = _select_case(tgt, role)
        img = _render_aligned_pair_png(case["test_smiles"], case["anchor_smiles"])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#CCCCCC")
            sp.set_linewidth(0.7)

        panel_tag(ax, f"{tag}  {hdr}", x=0.0, y=1.03, fontsize=7.8)
        footer = (
            rf"$s_{{\max}}={case['s_max']:.3f}$  |  $y_{{\mathrm{{anc}}}}={case['y_anc']:.2f} \rightarrow y_{{\mathrm{{true}}}}={case['y_true']:.2f}$ ($\Delta y={case['dy_true']:+.2f}$)" + "\n"
            rf"2D-LFER deltas: $\Delta\mathrm{{LogP}}={case['d_logp']:+.2f}$, $\Delta\mathrm{{MR}}={case['d_mr']:+.1f}$, $\Delta\mathrm{{TPSA}}={case['d_tpsa']:+.1f}$" + "\n"
            rf"Pred: SVM={case['K_RBF_SVM']:.2f} | Multi={case['K_Multi_SVR']:.2f} | OptW-MLP={case['OptW_MLP']:.2f} | OptW-FiLM={case['OptW_FiLM']:.2f}"
        )
        ax.set_xlabel(footer, fontsize=5.7, labelpad=3)

    save_jcim_fig(fig, "fig5_cases", also_tiff=False)


if __name__ == "__main__":
    main()
