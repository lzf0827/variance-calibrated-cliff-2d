"""Generate ACS JCIM TOC Graphic (<= 3.25 in x 1.75 in, <= 3 simple blocks, >= 300 dpi RGB)."""

from __future__ import annotations

import sys
from pathlib import Path
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jcim_plot_style import PALETTE, REFINELOGS_DIR, apply_jcim_style, save_jcim_fig


def main() -> None:
    apply_jcim_style(base_font=7.0)

    step_a = pd.read_csv(REFINELOGS_DIR / "jcim30_step_a_results" / "step_a_summary.csv")
    scaf = pd.read_csv(REFINELOGS_DIR / "scaffold_robustness_results" / "scaffold_summary.csv")
    c_optw = float(step_a[step_a["model"] == "E_Multi_FiLM_OptW"]["cliff_rmse_mean"].mean())
    s_grp = float(scaf[scaf["model"] == "E_Multi_FiLM_GroupOptW"]["rmse_mean"].mean())

    # Exact JCIM TOC size: 3.25 in x 1.75 in -> 975 x 525 px at 300 dpi
    fig = plt.figure(figsize=(3.25, 1.75), dpi=300, facecolor="white")
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    def _card(x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.0) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.015,rounding_size=0.03",
                facecolor=fc, edgecolor=ec, linewidth=lw,
            )
        )

    # Block 1 (Left): Pure-2D Complementary Pair
    _card(0.02, 0.10, 0.29, 0.80, "#EEF6FB", PALETTE["blue"], lw=1.1)
    ax.text(0.165, 0.81, "Pure-2D Branches", ha="center", va="center", fontsize=6.6, fontweight="bold", color=PALETTE["navy"])
    _card(0.045, 0.50, 0.24, 0.20, "#FFFFFF", PALETTE["blue"], lw=0.75)
    ax.text(0.165, 0.60, "Multi-Kernel SVR\nGlobal Kernel", ha="center", va="center", fontsize=5.9, color="#111111")
    ax.text(0.165, 0.43, "+", ha="center", va="center", fontsize=8.0, fontweight="bold", color=PALETTE["orange"])
    _card(0.045, 0.16, 0.24, 0.20, "#FFFFFF", PALETTE["orange"], lw=0.75)
    ax.text(0.165, 0.26, r"Antisym. $\Delta$ Pair" + "\nCliff Sensitive", ha="center", va="center", fontsize=5.9, color="#111111")

    # Arrow 1 -> 2
    ax.add_patch(FancyArrowPatch((0.325, 0.50), (0.375, 0.50), arrowstyle="-|>", mutation_scale=8, lw=1.1, color=PALETTE["navy"]))

    # Block 2 (Center): Split-Matched Nested-OOF Calibrator
    _card(0.38, 0.10, 0.26, 0.80, "#FFF6EB", PALETTE["orange"], lw=1.1)
    ax.text(0.51, 0.81, "Split-Matched OOF", ha="center", va="center", fontsize=6.6, fontweight="bold", color=PALETTE["orange"])
    ax.text(0.51, 0.61, r"$w^* \in [0, 1]$", ha="center", va="center", fontsize=7.6, color="#111111")
    ax.text(0.51, 0.40, r"$n < 1000$ / OOD" + "\nShrink to Kernel", ha="center", va="center", fontsize=5.8, color="#333333")
    ax.text(0.51, 0.19, "OptW | GroupOptW", ha="center", va="center", fontsize=5.8, fontweight="bold", color=PALETTE["navy"])

    # Arrow 2 -> 3
    ax.add_patch(FancyArrowPatch((0.655, 0.50), (0.705, 0.50), arrowstyle="-|>", mutation_scale=8, lw=1.1, color=PALETTE["navy"]))

    # Block 3 (Right): Dual Verdict (Verified Gains vs Isolated Traps)
    _card(0.71, 0.10, 0.27, 0.80, "#F9FAFB", "#444444", lw=1.0)
    _card(0.73, 0.53, 0.23, 0.30, "#E9F7F2", PALETTE["teal"], lw=0.8)
    ax.text(0.845, 0.74, "Validated Gains", ha="center", va="center", fontsize=6.2, fontweight="bold", color=PALETTE["teal"])
    ax.text(0.845, 0.60, f"30-Tgt Cliff: {c_optw:.3f}\nScaffold OOD: {s_grp:.3f}", ha="center", va="center", fontsize=5.8, color="#111111")

    _card(0.73, 0.17, 0.23, 0.30, "#FDEDEC", PALETTE["crimson"], lw=0.8)
    ax.text(0.845, 0.38, "Controlled Traps", ha="center", va="center", fontsize=6.2, fontweight="bold", color=PALETTE["crimson"])
    ax.text(0.845, 0.24, "3D/QM | Val-Gate\nOOF KR | GraphAttn", ha="center", va="center", fontsize=5.6, color="#111111")

    save_jcim_fig(fig, "toc_graphic", also_tiff=True, tight_box=None)


if __name__ == "__main__":
    main()
