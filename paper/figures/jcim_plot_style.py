"""Shared ACS JCIM publication figure styling, Okabe-Ito palette, and export helpers."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FIG_OUT_DIR = Path(__file__).resolve().parent
REFINELOGS_DIR = ROOT_DIR / "refine-logs"

# Colorblind-safe Okabe-Ito + crisp neutral palette (RGB hex)
PALETTE: dict[str, str] = {
    "orange": "#D55E00",      # Primary hero / E_Multi_FiLM_OptW
    "vermilion": "#990000",   # Scaffold-matched GroupOptW
    "amber": "#E69F00",       # Fixed 50/50 E_Multi_FiLM
    "teal": "#009E73",        # PriorN / Y_A2 / Positive
    "blue": "#0072B2",        # K_Multi_SVR
    "skyblue": "#56B4E9",     # K_Tan_SVR / CatBoost light
    "purple": "#CC79A7",      # DirectMLP OptW / E_Multi_DirectMLP_OptW
    "indigo": "#7A5195",      # CatBoost OptW / DirectMLP GroupOptW
    "darkgray": "#555555",    # K_RBF_SVM
    "lightgray": "#BDBDBD",   # DirectMLP / Reference
    "yellow": "#F0E442",      # Standalone Y_FiLM / A2
    "crimson": "#C0392B",     # Negative failure highlight
    "navy": "#1F3A5F",        # Schematic framing
}


def apply_jcim_style(base_font: float = 7.5) -> None:
    """Configure vector-clean, ACS-compliant sans-serif Arial typography."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_font,
        "axes.labelsize": base_font + 0.5,
        "axes.titlesize": base_font + 0.5,
        "xtick.labelsize": base_font - 0.5,
        "ytick.labelsize": base_font - 0.5,
        "legend.fontsize": base_font - 0.7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel_tag(ax: plt.Axes, tag: str, x: float = -0.14, y: float = 1.05, fontsize: float = 10.0) -> None:
    """Place bold sans-serif uppercase panel letter outside top-left without inner plot titles."""
    ax.text(
        x, y, tag,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def save_jcim_fig(
    fig: plt.Figure,
    stem: str,
    also_tiff: bool = False,
    tight_box: str | None = "tight",
) -> list[Path]:
    """Save PDF (vector), 300-dpi RGB PNG, and optionally 300-dpi RGB TIFF."""
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_OUT_DIR / f"{stem}.pdf"
    png_path = FIG_OUT_DIR / f"{stem}.png"

    save_kw: dict[str, object] = {"dpi": 300, "facecolor": "white"}
    if tight_box is not None:
        save_kw["bbox_inches"] = tight_box
        save_kw["pad_inches"] = 0.08
        fig.savefig(pdf_path, **save_kw)
        fig.savefig(png_path, **save_kw)
    else:
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(pdf_path, bbox_inches=fig.bbox_inches, **save_kw)
            fig.savefig(png_path, bbox_inches=fig.bbox_inches, **save_kw)
    written = [pdf_path, png_path]

    if also_tiff:
        tif_path = FIG_OUT_DIR / f"{stem}.tif"
        img = Image.open(png_path).convert("RGB")
        img.save(tif_path, format="TIFF", dpi=(300, 300), compression="tiff_lzw")
        written.append(tif_path)

    plt.close(fig)
    print(f"Saved: {', '.join(p.name for p in written)}", flush=True)
    return written
