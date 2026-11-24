"""Generate Figure 1: JCIM Graphical Workflow Overview (MCS Pair -> Pure-2D Dual Branches -> Split-Matched OOF Calibration)."""

from __future__ import annotations

import io
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_fig5_cases import _select_case
from jcim_plot_style import PALETTE, apply_jcim_style, panel_tag, save_jcim_fig


def _render_single_aligned_mol(
    smiles_a: str,
    smiles_q: str,
    width: int = 380,
    height: int = 185,
) -> tuple[np.ndarray, np.ndarray]:
    """Render MCS-aligned anchor and query molecules separately in ACS-1996 style."""
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
    col_a = (0.0, 0.45, 0.70, 0.48)
    col_q = (0.84, 0.37, 0.0, 0.48)

    def _diff_bonds(mol: Chem.Mol, hl_set: set[int]) -> list[int]:
        return [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in hl_set or b.GetEndAtomIdx() in hl_set]

    def _draw(mol: Chem.Mol, h_atoms: list[int], rgba: tuple[float, ...]) -> np.ndarray:
        h_bonds = _diff_bonds(mol, set(h_atoms))
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        rdMolDraw2D.SetACS1996Mode(opts, rdMolDraw2D.MeanBondLength(mol))
        opts.fixedBondLength = 20.0
        opts.bondLineWidth = 1.4
        opts.padding = 0.05
        opts.clearBackground = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=h_atoms,
            highlightAtomColors={i: rgba for i in h_atoms},
            highlightBonds=h_bonds,
            highlightBondColors={b: rgba for b in h_bonds},
        )
        drawer.FinishDrawing()
        return np.asarray(Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB"))

    return _draw(mol_a, hl_a, col_a), _draw(mol_q, hl_q, col_q)


def _rbox(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str,
    ec: str,
    lw: float = 0.9,
    ls: str = "-",
    rounding: float = 0.015,
    zorder: int = 2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={rounding}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )
    )


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, col: str = "#333333", lw: float = 0.95) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8.5,
            lw=lw,
            color=col,
            zorder=4,
        )
    )


def main() -> None:
    apply_jcim_style(base_font=7.5)

    case = _select_case("CHEMBL244_Ki", "WIN")
    im_anchor, im_query = _render_single_aligned_mol(case["anchor_smiles"], case["test_smiles"])

    fig = plt.figure(figsize=(7.0, 4.75))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.42, 1.0], hspace=0.24)

    # =========================================================================
    # Panel A: Graphical Pure-2D Workflow (Chemistry -> Features -> Dual Tracks)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.axis("off")

    # Clean well-spaced stage headings above boxes
    for xh, lbl in [
        (0.095, "1. MCS Pair"),
        (0.245, "2. 2D Features"),
        (0.578, "3. Global Kernel & Antisymmetric Delta Branches"),
        (0.927, "4. Fusion"),
    ]:
        ax_a.text(xh, 0.985, lbl, ha="center", va="top", fontsize=6.2, fontweight="bold", color=PALETTE["navy"])

    # --- Column 1: Real Factor Xa Activity Cliff Pair (CHEMBL244_Ki) ---
    _rbox(ax_a, 0.008, 0.04, 0.175, 0.87, "#FBFBFD", PALETTE["navy"], lw=0.95)
    ax_a.text(
        0.0955, 0.892,
        r"Factor Xa ($\mathrm{CHEMBL244\_K_i}$)",
        ha="center", va="top", fontsize=5.1, fontweight="bold", color=PALETTE["navy"],
    )

    ax_m1 = ax_a.inset_axes([0.015, 0.56, 0.161, 0.25])
    ax_m1.imshow(im_anchor)
    ax_m1.axis("off")
    ax_a.text(0.0955, 0.555, rf"Anchor $j$ ($y_j = {case['y_anc']:.2f}$)", ha="center", va="top", fontsize=5.1, color=PALETTE["blue"], fontweight="bold")

    ax_m2 = ax_a.inset_axes([0.015, 0.25, 0.161, 0.25])
    ax_m2.imshow(im_query)
    ax_m2.axis("off")
    ax_a.text(0.0955, 0.245, rf"Query $i$ ($y_i = {case['y_true']:.2f}$)", ha="center", va="top", fontsize=5.1, color=PALETTE["orange"], fontweight="bold")

    ax_a.text(
        0.0955, 0.145,
        rf"$s_{{\max}} = {case['s_max']:.3f}$,  $\Delta y = {case['dy_true']:+.2f}$",
        ha="center", va="center", fontsize=5.0, color="#111111",
    )
    ax_a.text(
        0.0955, 0.075,
        r"$\mathbf{p}_{ij}^{\mathrm{2D}}$: $\Delta$LogP, $\Delta$MR, $\Delta$TPSA, $\Delta N$",
        ha="center", va="center", fontsize=4.55, color="#444444",
    )

    # --- Column 2: Visual 2D Binary Fingerprint Barcodes + Desc2D Glyph ---
    _rbox(ax_a, 0.198, 0.04, 0.095, 0.87, "#F4F6F9", PALETTE["darkgray"], lw=0.85)
    ax_a.text(0.2455, 0.885, "Inputs", ha="center", va="top", fontsize=5.6, fontweight="bold", color="#222222")

    rng = np.random.default_rng(244)
    fp1 = (rng.random((2, 36)) > 0.72).astype(float)
    fp2 = (rng.random((2, 24)) > 0.65).astype(float)

    ax_fp1 = ax_a.inset_axes([0.207, 0.66, 0.077, 0.11])
    ax_fp1.imshow(fp1, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1.2)
    ax_fp1.set_xticks([])
    ax_fp1.set_yticks([])
    ax_a.text(0.2455, 0.64, "ECFP4 (2048b)", ha="center", va="top", fontsize=4.8, color="#222222")

    ax_fp2 = ax_a.inset_axes([0.207, 0.45, 0.077, 0.09])
    ax_fp2.imshow(fp2, aspect="auto", cmap="BuPu", interpolation="nearest", vmin=0, vmax=1.2)
    ax_fp2.set_xticks([])
    ax_fp2.set_yticks([])
    ax_a.text(0.2455, 0.43, "MACCS (167b)", ha="center", va="top", fontsize=4.8, color="#222222")

    ax_d2 = ax_a.inset_axes([0.209, 0.22, 0.073, 0.12])
    vals_d = [0.65, -0.45, 0.82, 0.30, -0.60, 0.50]
    ax_d2.bar(range(6), vals_d, color=[PALETTE["teal"] if v > 0 else PALETTE["amber"] for v in vals_d], width=0.72)
    ax_d2.axhline(0, color="#333333", lw=0.5)
    ax_d2.axis("off")
    ax_a.text(0.2455, 0.20, "Desc2D (25d)\n" + r"+ Radius-1 $\mathbf{p}_{ij}^{\mathrm{2D}}$", ha="center", va="top", fontsize=4.65, color="#222222", linespacing=1.22)

    _arrow(ax_a, 0.185, 0.48, 0.196, 0.48, PALETTE["navy"])

    # --- Column 3 Upper Track: Global Multi-Kernel Branch ---
    _rbox(ax_a, 0.314, 0.51, 0.530, 0.40, "#F4FAFD", PALETTE["blue"], lw=0.85, ls=":")
    ax_a.text(0.324, 0.895, r"Global Kernel Branch ($\mathrm{K\_Multi\_SVR}$)", ha="left", va="top", fontsize=5.6, fontweight="bold", color=PALETTE["blue"])

    # PSD Gram Matrix card with miniature heatmap icon
    _rbox(ax_a, 0.324, 0.54, 0.340, 0.28, "#FFFFFF", PALETTE["blue"], lw=0.85)
    gram_ex = np.array([
        [1.00, 0.82, 0.54, 0.31, 0.24, 0.18],
        [0.82, 1.00, 0.61, 0.29, 0.20, 0.15],
        [0.54, 0.61, 1.00, 0.44, 0.33, 0.21],
        [0.31, 0.29, 0.44, 1.00, 0.78, 0.52],
        [0.24, 0.20, 0.33, 0.78, 1.00, 0.69],
        [0.18, 0.15, 0.21, 0.52, 0.69, 1.00],
    ])
    ax_hm = ax_a.inset_axes([0.332, 0.565, 0.068, 0.225])
    ax_hm.imshow(gram_ex, cmap="Blues", vmin=0.1, vmax=1.0)
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])
    for sp in ax_hm.spines.values():
        sp.set_edgecolor(PALETTE["blue"])
        sp.set_linewidth(0.6)

    ax_a.text(0.533, 0.785, "PSD Multi-Kernel Gram Matrix", ha="center", va="top", fontsize=5.35, fontweight="bold", color="#111111")
    ax_a.text(
        0.533, 0.645,
        r"$K_{\mathrm{multi}} = 0.70\,K_{\mathrm{Tan}}^{\mathrm{ECFP4}}$" + "\n"
        r"$+\,0.15\,K_{\mathrm{Tan}}^{\mathrm{MACCS}} + 0.15\,K_{\mathrm{RBF}}^{\mathrm{Desc2D}}$",
        ha="center", va="center", fontsize=5.15, fontweight="normal", color="#111111", linespacing=1.32,
    )

    _rbox(ax_a, 0.686, 0.54, 0.148, 0.28, "#EBF5FB", PALETTE["blue"], lw=0.9)
    ax_a.text(0.760, 0.785, "Precomputed SVR", ha="center", va="top", fontsize=5.35, fontweight="bold", color="#111111")
    ax_a.text(
        0.760, 0.645,
        r"$C=3.0,\ \varepsilon=0.1$" + "\n"
        r"$\Rightarrow\ \hat{y}_K,\ \mathbf{e}_K$",
        ha="center", va="center", fontsize=5.2, fontweight="normal", color="#111111", linespacing=1.32,
    )
    _arrow(ax_a, 0.296, 0.69, 0.320, 0.69, PALETTE["blue"])
    _arrow(ax_a, 0.668, 0.68, 0.683, 0.68, PALETTE["blue"])

    # --- Column 3 Lower Track: Antisymmetric Pairwise Delta Branch ---
    _rbox(ax_a, 0.314, 0.04, 0.530, 0.43, "#FFF9F2", PALETTE["orange"], lw=0.85, ls=":")
    ax_a.text(0.324, 0.455, r"Antisymmetric Pairwise Delta Branch ($\mathrm{Y\_A2\ /\ Y\_FiLM}$)", ha="left", va="top", fontsize=5.6, fontweight="bold", color=PALETTE["orange"])

    # Module (i): Siamese + Zero-Init FiLM (bold title, regular math body)
    _rbox(ax_a, 0.324, 0.07, 0.195, 0.31, "#FFFFFF", PALETTE["orange"], lw=0.85)
    ax_a.text(0.4215, 0.365, "Siamese 2D + Zero-Init FiLM", ha="center", va="top", fontsize=4.85, fontweight="bold", color="#111111")
    ax_a.text(
        0.4215, 0.195,
        r"$\boldsymbol{\gamma} = \mathbf{1} + \tanh(\mathbf{W}_\gamma \mathbf{p}_{ij}^{\mathrm{2D}} + \mathbf{b}_\gamma)$" + "\n"
        r"$\psi = w_{\mathrm{out}}(\boldsymbol{\gamma}\odot\mathbf{z}_{ij}+\boldsymbol{\beta}) + \mathbf{w}_{\mathrm{skip}}^{\top}\mathbf{p}_{ij}^{\mathrm{2D}}$" + "\n"
        r"(init: $\boldsymbol{\gamma}=\mathbf{1},\ \boldsymbol{\beta}=\mathbf{0},\ \mathbf{w}_{\mathrm{skip}}=\mathbf{0}$)",
        ha="center", va="center", fontsize=4.5, fontweight="normal", color="#111111", linespacing=1.32,
    )

    # Module (ii): Exact Antisymmetrization (bold title, regular math body)
    _rbox(ax_a, 0.535, 0.07, 0.125, 0.31, "#FEF5E7", PALETTE["orange"], lw=0.85)
    ax_a.text(0.5975, 0.365, "Antisymmetry", ha="center", va="top", fontsize=4.85, fontweight="bold", color="#111111")
    ax_a.text(
        0.5975, 0.195,
        r"$f_\Delta(i,j) = \frac{1}{2}[\psi(i,j)$" + "\n"
        r"$\quad - \psi(j,i)]$" + "\n"
        r"$f_\Delta(i,i) = 0$",
        ha="center", va="center", fontsize=4.85, fontweight="normal", color="#111111", linespacing=1.30,
    )

    # Module (iii): Top-K Anchor Retrieval (bold title, regular math body)
    _rbox(ax_a, 0.676, 0.07, 0.158, 0.31, "#FFFFFF", PALETTE["orange"], lw=0.9)
    ax_pool = ax_a.inset_axes([0.698, 0.175, 0.114, 0.088])
    alpha_w = [0.32, 0.24, 0.16, 0.11, 0.07, 0.05, 0.03, 0.02]
    ax_pool.bar(range(len(alpha_w)), alpha_w, color=PALETTE["orange"], width=0.72)
    ax_pool.set_xticks([])
    ax_pool.set_yticks([])
    for sp in ("top", "right"):
        ax_pool.spines[sp].set_visible(False)
    ax_a.text(0.755, 0.365, r"Top-$K$ Pool ($k=10, \tau=10$)", ha="center", va="top", fontsize=4.85, fontweight="bold", color="#111111")
    ax_a.text(
        0.755, 0.115,
        r"$\hat{y}_\Delta(i) = \sum_k \alpha_{ik}(y_k + f_\Delta(i,k))$",
        ha="center", va="center", fontsize=4.55, fontweight="normal", color="#111111",
    )

    _arrow(ax_a, 0.296, 0.29, 0.320, 0.24, PALETTE["orange"])
    _arrow(ax_a, 0.522, 0.225, 0.532, 0.225, PALETTE["orange"])
    _arrow(ax_a, 0.663, 0.225, 0.673, 0.225, PALETTE["orange"])

    # --- Column 4: Final Variance-Calibrated Fusion Pill (bold title, regular math body) ---
    _rbox(ax_a, 0.865, 0.23, 0.125, 0.50, PALETTE["navy"], PALETTE["navy"], lw=1.0, rounding=0.022)
    ax_a.text(0.9275, 0.68, "Calibrated\nPrediction", ha="center", va="top", fontsize=5.8, fontweight="bold", color="#FFFFFF", linespacing=1.22)
    ax_a.text(
        0.9275, 0.39,
        r"$\hat{y} = (1 - w^*)\,\hat{y}_K$" + "\n"
        r"$+\,w^*\,\hat{y}_\Delta$" + "\n\n"
        r"($w^* \in [0, 1]$)",
        ha="center", va="center", fontsize=5.4, fontweight="normal", color="#FFFFFF", linespacing=1.28,
    )
    _arrow(ax_a, 0.837, 0.68, 0.861, 0.56, PALETTE["blue"])
    _arrow(ax_a, 0.837, 0.225, 0.861, 0.40, PALETTE["orange"])

    panel_tag(ax_a, "A", x=-0.022, y=0.99, fontsize=10.5)

    # =========================================================================
    # Panel B: Split-Matched Nested OOF Protocol & Self-Limiting Regime Behavior
    # =========================================================================
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], width_ratios=[1.15, 1.0], wspace=0.18)

    # --- Panel B Left: Split-Matched Leak-Free 3-Fold Nested OOF ---
    ax_bl = fig.add_subplot(gs_b[0, 0])
    ax_bl.set_xlim(0.0, 1.0)
    ax_bl.set_ylim(0.0, 1.0)
    ax_bl.axis("off")

    _rbox(ax_bl, 0.01, 0.02, 0.98, 0.95, "#F8FBF9", PALETTE["teal"], lw=0.9)
    ax_bl.text(0.50, 0.942, "Leak-Free 3-Fold Nested OOF Calibration", ha="center", va="top", fontsize=6.4, fontweight="bold", color=PALETTE["navy"])

    # Visual 3-fold CV bar diagram inside train_idx | barrier | test_idx (placed well below title)
    ax_cv = ax_bl.inset_axes([0.05, 0.42, 0.90, 0.38])
    ax_cv.set_xlim(0, 10)
    ax_cv.set_ylim(0, 3.65)
    ax_cv.axis("off")
    for f_idx, y_row in enumerate([2.25, 1.20, 0.15]):
        for seg in range(3):
            col = PALETTE["amber"] if seg == f_idx else "#D6EAF8"
            lbl = "OOF" if seg == f_idx else "fit"
            ax_cv.add_patch(FancyBboxPatch((seg * 2.42 + 0.08, y_row), 2.24, 0.78, boxstyle="round,pad=0.03,rounding_size=0.12", fc=col, ec="#444444", lw=0.55))
            ax_cv.text(seg * 2.42 + 1.20, y_row + 0.39, lbl, ha="center", va="center", fontsize=5.1, fontweight="bold" if seg == f_idx else "normal", color="#111111")
        # Clean Held-Out Test block (no hatch across text)
        ax_cv.add_patch(FancyBboxPatch((7.78, y_row), 2.10, 0.78, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#E8EAED", ec="#666666", lw=0.6))
        ax_cv.text(8.83, y_row + 0.39, r"Held-Out $\mathcal{D}_{\mathrm{te}}$", ha="center", va="center", fontsize=4.9, color="#222222", fontweight="bold")
    ax_cv.plot([7.52, 7.52], [-0.05, 3.35], color=PALETTE["crimson"], lw=1.3, ls="--")
    ax_cv.text(3.75, 3.25, r"Inside $\mathcal{D}_{\mathrm{tr}}$: OptW (Stratified) / GroupOptW (Murcko)", ha="center", va="bottom", fontsize=4.95, fontweight="bold", color=PALETTE["navy"])
    ax_cv.text(8.83, 3.25, "Zero Leakage", ha="center", va="bottom", fontsize=4.95, fontweight="bold", color=PALETTE["crimson"])

    # Clean mathematical equations for Bates-Granger & PriorN
    ax_bl.text(
        0.50, 0.27,
        r"$w^* = \mathrm{clip}\!\left(\frac{\mathbf{e}_K^{\top}(\mathbf{e}_K - \mathbf{e}_\Delta)}{\|\mathbf{e}_K - \mathbf{e}_\Delta\|_2^2},\, 0,\, 1\right)$",
        ha="center", va="center", fontsize=7.0, color="#111111",
    )
    ax_bl.text(
        0.50, 0.10,
        r"Size Prior (zero tuning):  $w_{\mathrm{prior}} = \frac{0.5\,n_{\mathrm{tr}}}{n_{\mathrm{tr}} + 1000}$",
        ha="center", va="center", fontsize=5.8, color="#222222",
    )

    panel_tag(ax_bl, "B", x=-0.045, y=0.98, fontsize=10.5)

    # --- Panel B Right: Regime Shrinkage Mechanism & Track-2 Scope ---
    ax_br = fig.add_subplot(gs_b[0, 1])
    n_grid = np.linspace(350, 3800, 200)
    n_tr_grid = 0.8 * n_grid
    w_prior_curve = 0.5 * n_tr_grid / (n_tr_grid + 1000.0)

    ax_br.axvspan(350, 1000, color="#EBF5FB", alpha=0.80, zorder=0)
    ax_br.axvspan(1000, 3800, color="#FFF5EB", alpha=0.65, zorder=0)
    ax_br.axvline(1000, color="#555555", ls="--", lw=0.85, zorder=1)
    ax_br.axhline(0.50, color=PALETTE["amber"], ls=":", lw=1.0, zorder=1)

    ax_br.plot(n_grid, w_prior_curve, color=PALETTE["teal"], lw=1.8, zorder=3)
    ax_br.text(3200, 0.39, r"$w_{\mathrm{prior}}(n_{\mathrm{tr}})$", fontsize=5.3, color=PALETTE["teal"], fontweight="bold", ha="center")

    # Show scaffold GroupOptW downward shrinkage arrow
    ax_br.annotate(
        "",
        xy=(1750, 0.267),
        xytext=(1750, 0.363),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["vermilion"], lw=1.5),
        zorder=4,
    )
    ax_br.scatter([1750, 1750], [0.363, 0.267], color=[PALETTE["blue"], PALETTE["vermilion"]], s=22, zorder=5, edgecolors="black", linewidths=0.4)
    ax_br.text(1840, 0.368, "Stratified CV (0.363)", fontsize=5.15, color=PALETTE["blue"], va="center")
    ax_br.text(1840, 0.258, r"Murcko GroupKFold (0.267 $\rightarrow$ 0.042)", fontsize=5.15, color=PALETTE["vermilion"], va="center", fontweight="bold")

    ax_br.text(2450, 0.488, r"Uncalibrated 50/50 baseline ($w = 0.50$)", fontsize=5.0, color=PALETTE["amber"], ha="center", va="top", fontweight="bold")
    ax_br.text(620, 0.032, "Small-$n$ /\nFar-OOD\n" + r"($w^* \downarrow$ Kernel)", fontsize=4.55, color=PALETTE["blue"], ha="center", va="bottom", fontweight="bold")
    ax_br.text(2650, 0.055, "Analog-Rich Regime\n" + r"($w^* \uparrow$ Delta Gain)", fontsize=5.0, color=PALETTE["orange"], ha="center", va="bottom", fontweight="bold")

    ax_br.set_xlim(240, 3850)
    ax_br.set_ylim(0.0, 0.65)
    ax_br.set_xlabel(r"Dataset Size $n_{\mathrm{total}}$ (Regime Shrinkage & Track-2 Scope)", fontsize=6.5)
    ax_br.set_ylabel(r"Delta Weight ($w^*$)", fontsize=6.5)
    ax_br.set_title("Self-Limiting Variance Calibration", fontsize=6.7, fontweight="bold", color=PALETTE["navy"], pad=4)
    ax_br.tick_params(labelsize=5.7)

    # Compact header pill cleanly inside axes bounds well above y=0.50 line
    ax_br.text(
        0.515, 0.945,
        "Track-2 Excluded (Fig. 4): Vacuum 3D/AIMNet2, Val-Gate, OOF Residual, GINE+Attn",
        transform=ax_br.transAxes,
        ha="center", va="top",
        fontsize=4.35, color=PALETTE["crimson"],
        bbox=dict(boxstyle="round,pad=0.18", fc="#FFF5F5", ec=PALETTE["crimson"], lw=0.55),
    )

    save_jcim_fig(fig, "fig1_hero", also_tiff=False)


if __name__ == "__main__":
    main()
