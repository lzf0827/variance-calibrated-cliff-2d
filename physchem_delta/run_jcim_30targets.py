"""JCIM Main Benchmark: Pure-2D E_Multi_FiLM across all 30 MoleculeACE targets x 5 seeds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_opt13_modular import _fit_kernel_bases, _train_yspace_modular
from physchem_delta.eval_plan_a import (
    SEEDS,
    _fit_rbf_svr,
    _lfer_atom_matrix,
    _train_delta_backbone,
)
from physchem_delta.featurize import (
    TargetFeatures,
    _atom_env_hashes,
    _compute_desc2d,
    _maccs_array,
    _morgan_bit_array,
    _sanitize_array,
    get_feature_cache_dir,
)
from physchem_delta.metrics import evaluate_predictions
from physchem_delta.models import (
    _train_direct_mlp_with_inner_split,
    build_tier_matrices,
    make_inner_split,
    run_tabular_baseline,
)
from physchem_delta.run_30_targets import ALL_30_TARGETS

JCIM_MODELS: tuple[str, ...] = (
    "B1_CatBoost",
    "B1b_LightGBM",
    "A1_Direct_MLP",
    "K_RBF_SVM",
    "K_Tan_SVR",
    "K_Multi_SVR",
    "Y_A2_2D",
    "Y_FiLM_2DLFER",
    "E_Multi_CatBoost",
    "E_Multi_DirectMLP",
    "E_SVM_A2Delta",
    "E_Multi_A2Delta",
    "E_Multi_FiLM",
)

COMPARATOR_ORDER: tuple[str, ...] = (
    "K_RBF_SVM",
    "K_Tan_SVR",
    "K_Multi_SVR",
    "B1_CatBoost",
    "B1b_LightGBM",
    "A1_Direct_MLP",
    "E_Multi_DirectMLP",
    "E_Multi_CatBoost",
    "E_SVM_A2Delta",
    "E_Multi_A2Delta",
)


def _out_root() -> Path:
    d = ROOT_DIR / "refine-logs" / "jcim30_results"
    (d / "per_target").mkdir(parents=True, exist_ok=True)
    return d


def _fig_dir() -> Path:
    d = ROOT_DIR / "refine-logs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_2d_dir() -> Path:
    d = Path(__file__).resolve().parent / "feature_cache_2d"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_or_load_2d_features(target: str, smiles_list: list[str]) -> TargetFeatures:
    """Fast pure-2D + RDKit atomic Crippen/TPSA extractor cached separately from 3D cache."""
    cdir = _cache_2d_dir()
    npz_path = cdir / f"{target}_2d.npz"
    meta_path = cdir / f"{target}_2d_meta.json"
    n = len(smiles_list)

    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n_mols") == n and meta.get("electronic_backend") == "rdkit_2d_lfer":
            data = np.load(npz_path)
            total_atoms = int(data["atom_ptr"][-1])
            return TargetFeatures(
                target=target,
                electronic_backend="rdkit_2d_lfer",
                ecfp_nochiral=data["ecfp_nochiral"],
                ecfp_chiral=data["ecfp_chiral"],
                maccs=data["maccs"],
                desc2d=data["desc2d"],
                feat_3d=np.zeros((n, 16), dtype=np.float32),
                feat_elec=np.zeros((n, 63), dtype=np.float32),
                atom_ptr=data["atom_ptr"],
                atom_env_hash=data["atom_env_hash"],
                atom_3d_props=np.zeros((total_atoms, 3), dtype=np.float32),
                atom_elec_props=data["atom_elec_props"],
                meta=meta,
            )

    t0 = time.time()
    ecfp_nc_list: list[np.ndarray] = []
    ecfp_c_list: list[np.ndarray] = []
    maccs_list: list[np.ndarray] = []
    desc2d_list: list[np.ndarray] = []
    atom_ptr = [0]
    atom_env_chunks: list[np.ndarray] = []
    atom_elec_chunks: list[np.ndarray] = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {idx} ({target}): {smi}")

        ecfp_nc_list.append(_morgan_bit_array(mol, use_chirality=False))
        ecfp_c_list.append(_morgan_bit_array(mol, use_chirality=True))
        maccs_list.append(_maccs_array(mol))
        desc2d_list.append(_compute_desc2d(mol))
        atom_env_chunks.append(_atom_env_hashes(mol))

        n_heavy = mol.GetNumAtoms()
        try:
            crip = np.asarray(rdMolDescriptors._CalcCrippenContribs(mol), dtype=np.float32)
            crip_logp = crip[:, 0]
            crip_mr = crip[:, 1]
        except Exception:
            crip_logp = np.zeros((n_heavy,), dtype=np.float32)
            crip_mr = np.zeros((n_heavy,), dtype=np.float32)

        try:
            tpsa_c = np.asarray(rdMolDescriptors._CalcTPSAContribs(mol), dtype=np.float32)
        except Exception:
            tpsa_c = np.zeros((n_heavy,), dtype=np.float32)

        zeros = np.zeros((n_heavy,), dtype=np.float32)
        # Stored with 6 cols so _lfer_atom_matrix extracts [crip_logp, crip_mr, tpsa_c] identically at cols 2,3,4
        atom_elec = _sanitize_array(np.stack([zeros, zeros, crip_logp, crip_mr, tpsa_c, zeros], axis=1))
        atom_elec_chunks.append(atom_elec)
        atom_ptr.append(atom_ptr[-1] + n_heavy)

    ecfp_nochiral = np.stack(ecfp_nc_list, axis=0)
    ecfp_chiral = np.stack(ecfp_c_list, axis=0)
    maccs = np.stack(maccs_list, axis=0)
    desc2d = np.stack(desc2d_list, axis=0)
    atom_ptr_arr = np.asarray(atom_ptr, dtype=np.int64)
    atom_env_hash = np.concatenate(atom_env_chunks, axis=0)
    atom_elec_props = np.concatenate(atom_elec_chunks, axis=0)
    total_atoms = int(atom_ptr_arr[-1])

    elapsed = round(time.time() - t0, 2)
    meta = {
        "target": target,
        "n_mols": n,
        "electronic_backend": "rdkit_2d_lfer",
        "elapsed_sec": elapsed,
    }
    np.savez_compressed(
        npz_path,
        ecfp_nochiral=ecfp_nochiral,
        ecfp_chiral=ecfp_chiral,
        maccs=maccs,
        desc2d=desc2d,
        atom_ptr=atom_ptr_arr,
        atom_env_hash=atom_env_hash,
        atom_elec_props=atom_elec_props,
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TargetFeatures(
        target=target,
        electronic_backend="rdkit_2d_lfer",
        ecfp_nochiral=ecfp_nochiral,
        ecfp_chiral=ecfp_chiral,
        maccs=maccs,
        desc2d=desc2d,
        feat_3d=np.zeros((n, 16), dtype=np.float32),
        feat_elec=np.zeros((n, 63), dtype=np.float32),
        atom_ptr=atom_ptr_arr,
        atom_env_hash=atom_env_hash,
        atom_3d_props=np.zeros((total_atoms, 3), dtype=np.float32),
        atom_elec_props=atom_elec_props,
        meta=meta,
    )


def _verify_2d_equivalence() -> None:
    """Assert pure-2D extractor matches existing 3D+AIMNet cache identically on CHEMBL2147_Ki."""
    ref_npz = get_feature_cache_dir() / "CHEMBL2147_Ki_features.npz"
    if not ref_npz.exists():
        raise FileNotFoundError(f"Missing reference cache for equivalence check: {ref_npz}")
    ds = load_target_dataset("CHEMBL2147_Ki")
    feats_2d = extract_or_load_2d_features("CHEMBL2147_Ki", ds.smiles)
    ref = np.load(ref_npz)
    lfer_2d = _lfer_atom_matrix(feats_2d)[:, [1, 2, 3]]
    lfer_ref = ref["atom_elec_props"][:, [4, 5, 6]] if ref["atom_elec_props"].shape[1] >= 8 else ref["atom_elec_props"][:, [2, 3, 4]]

    diffs = {
        "ecfp_chiral": float(np.max(np.abs(feats_2d.ecfp_chiral - ref["ecfp_chiral"]))),
        "maccs": float(np.max(np.abs(feats_2d.maccs - ref["maccs"]))),
        "desc2d": float(np.max(np.abs(feats_2d.desc2d - ref["desc2d"]))),
        "atom_ptr": float(np.max(np.abs(feats_2d.atom_ptr - ref["atom_ptr"]))),
        "atom_env_hash": float(np.max(np.abs(feats_2d.atom_env_hash - ref["atom_env_hash"]))),
        "lfer_2d_atoms": float(np.max(np.abs(lfer_2d - lfer_ref))),
    }
    max_diff = max(diffs.values())
    if max_diff > 1e-6:
        raise RuntimeError(f"2D vs 3D reference equivalence failed: {diffs}")
    print(f"[PASS] 2D vs 3D cache max_abs_diff == {max_diff:.1f} ({diffs})", flush=True)


def _bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(diff[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cross_target_stats(df_wide: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    diff = (df_wide[col_a] - df_wide[col_b]).to_numpy(dtype=np.float64)
    wins = int(np.sum(diff < -1e-9))
    ties = int(np.sum(np.abs(diff) <= 1e-9))
    losses = int(np.sum(diff > 1e-9))
    ci_lo, ci_hi = _bootstrap_ci(diff, n_boot=5000, seed=42)
    if diff.size >= 2 and not np.all(np.abs(diff) < 1e-12):
        try:
            _, pval = stats.wilcoxon(df_wide[col_a], df_wide[col_b], alternative="two-sided")
            p = float(pval)
        except Exception:
            p = float("nan")
    else:
        p = 1.0
    return {
        "n": int(diff.size),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_value": p,
    }


def _evaluate_target(target: str, seeds: tuple[int, ...], force: bool, save_cache: bool) -> dict[str, Any]:
    out_dir = _out_root() / "per_target"
    json_path = out_dir / f"{target}.json"
    npz_path = out_dir / f"{target}_preds.npz"

    if save_cache and len(seeds) == len(SEEDS) and json_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if all(m in cached.get("metrics", {}) for m in JCIM_MODELS):
            print(f"[CACHE] {target}: loaded JCIM-30 checkpoint", flush=True)
            return cached

    t0 = time.time()
    ds = load_target_dataset(target)
    feats = extract_or_load_2d_features(target, ds.smiles)
    lfer_atoms = _lfer_atom_matrix(feats)
    dummy_graphs = [{"x_cont": np.zeros((1, 3), dtype=np.float32)}] * len(ds.smiles)

    y_te = ds.y[ds.test_idx]
    c_te = ds.cliff_mol[ds.test_idx]
    cmask = c_te == 1

    chemistry = {
        "target": target,
        "assay_type": "EC50" if target.endswith("_EC50") else "Ki",
        "n_total": int(len(ds.smiles)),
        "n_train": int(ds.train_idx.size),
        "n_test": int(ds.test_idx.size),
        "n_test_cliff": int(np.sum(cmask)),
        "cliff_frac": float(np.mean(cmask)),
        "mean_mw": float(np.mean(feats.desc2d[:, 0])),
        "mean_logp": float(np.mean(feats.desc2d[:, 1])),
        "mean_tpsa": float(np.mean(feats.desc2d[:, 2])),
        "mean_rot_bonds": float(np.mean(feats.desc2d[:, 5])),
    }

    # Deterministic kernel bases fit strictly on train_idx
    pred_rbf_svm = _fit_rbf_svr(feats.ecfp_chiral, ds.y, ds.train_idx, ds.test_idx)
    pred_tan_svr, pred_multi_svr = _fit_kernel_bases(feats, ds.y, ds.train_idx, ds.test_idx)

    seed_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in JCIM_MODELS}
    seed_preds: dict[str, list[np.ndarray]] = {m: [] for m in JCIM_MODELS}

    for seed in seeds:
        inner_tr_rel, inner_va_rel = make_inner_split(ds.y[ds.train_idx], seed=seed, val_frac=0.15)
        inner_tr_idx = ds.train_idx[inner_tr_rel]
        inner_va_idx = ds.train_idx[inner_va_rel]

        y_cat = run_tabular_baseline("B1_ECFP_Chiral_CatBoost", feats, ds.y, ds.train_idx, ds.test_idx, seed)["y_pred_test"]
        y_lgb = run_tabular_baseline("B1b_ECFP_Chiral_LGBM", feats, ds.y, ds.train_idx, ds.test_idx, seed)["y_pred_test"]
        bundle_2d_inner = build_tier_matrices(feats, tier="2d", fit_idx=inner_tr_idx)
        y_mlp = _train_direct_mlp_with_inner_split(
            bundle_2d_inner.x_mol, ds.y, ds.train_idx, inner_tr_rel, inner_va_rel, ds.test_idx, seed
        )["y_pred_test"]

        y_a2 = _train_delta_backbone(feats, lfer_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="2d")["y_pred_test"]
        y_film = _train_yspace_modular(
            "Y_FiLM_2DLFER", feats, dummy_graphs, lfer_atoms, ds.y,
            ds.train_idx, inner_tr_idx, inner_va_idx, ds.test_idx, seed,
        )["y_pred_test"]

        preds_map = {
            "B1_CatBoost": y_cat,
            "B1b_LightGBM": y_lgb,
            "A1_Direct_MLP": y_mlp,
            "K_RBF_SVM": pred_rbf_svm,
            "K_Tan_SVR": pred_tan_svr,
            "K_Multi_SVR": pred_multi_svr,
            "Y_A2_2D": y_a2,
            "Y_FiLM_2DLFER": y_film,
            "E_Multi_CatBoost": 0.5 * pred_multi_svr + 0.5 * y_cat,
            "E_Multi_DirectMLP": 0.5 * pred_multi_svr + 0.5 * y_mlp,
            "E_SVM_A2Delta": 0.5 * pred_rbf_svm + 0.5 * y_a2,
            "E_Multi_A2Delta": 0.5 * pred_multi_svr + 0.5 * y_a2,
            "E_Multi_FiLM": 0.5 * pred_multi_svr + 0.5 * y_film,
        }
        for m_name, yp in preds_map.items():
            if not np.all(np.isfinite(yp)):
                raise RuntimeError(f"Non-finite prediction in {target} {m_name} seed={seed}")
            seed_metrics[m_name].append(evaluate_predictions(y_te, yp, c_te))
            seed_preds[m_name].append(yp)

    metrics_summary: dict[str, dict[str, float]] = {}
    npz_payload: dict[str, np.ndarray] = {"y_te": y_te, "c_te": c_te}
    for m_name in JCIM_MODELS:
        rmse_arr = np.asarray([r["rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        cliff_arr = np.asarray([r["cliff_rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        ncliff_arr = np.asarray([r["non_cliff_rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        sp_arr = np.asarray([r["spearman"] for r in seed_metrics[m_name]], dtype=np.float64)
        mae_arr = np.asarray([r["mae"] for r in seed_metrics[m_name]], dtype=np.float64)
        cmae_arr = np.asarray([r["cliff_mae"] for r in seed_metrics[m_name]], dtype=np.float64)
        r2_arr = np.asarray([r["r2"] for r in seed_metrics[m_name]], dtype=np.float64)
        metrics_summary[m_name] = {
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),
            "cliff_rmse_mean": float(np.mean(cliff_arr)),
            "cliff_rmse_std": float(np.std(cliff_arr)),
            "non_cliff_rmse_mean": float(np.mean(ncliff_arr)),
            "mae_mean": float(np.mean(mae_arr)),
            "cliff_mae_mean": float(np.mean(cmae_arr)),
            "r2_mean": float(np.mean(r2_arr)),
            "spearman_mean": float(np.mean(sp_arr)),
        }
        stacked = np.stack(seed_preds[m_name], axis=0)
        npz_payload[f"{m_name}_seeds"] = stacked
        npz_payload[f"{m_name}_mean"] = np.mean(stacked, axis=0)

    em_c = metrics_summary["E_Multi_FiLM"]["cliff_rmse_mean"]
    em_c_std = metrics_summary["E_Multi_FiLM"]["cliff_rmse_std"]
    rbf_c = metrics_summary["K_RBF_SVM"]["cliff_rmse_mean"]
    mul_c = metrics_summary["K_Multi_SVR"]["cliff_rmse_mean"]
    mlp_e_c = metrics_summary["E_Multi_DirectMLP"]["cliff_rmse_mean"]

    flags = {
        "LOSE_VS_RBF_SVM": bool(em_c > rbf_c),
        "LOSE_VS_MULTI_SVR": bool(em_c > mul_c),
        "LOSE_VS_MULTI_DIRECTMLP": bool(em_c > mlp_e_c),
    }

    elapsed = round(time.time() - t0, 2)
    result = {
        "target": target,
        "elapsed_sec": elapsed,
        "chemistry": chemistry,
        "metrics": metrics_summary,
        "flags": flags,
    }

    if save_cache and len(seeds) == len(SEEDS):
        np.savez_compressed(npz_path, **npz_payload)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    active = [k for k, v in flags.items() if v]
    alert = f" [ALERT: {', '.join(active)}]" if active else " [OK]"
    print(
        f"[DONE {elapsed:.1f}s] {target} (n={chemistry['n_total']}, cliff={chemistry['n_test_cliff']}): "
        f"RBF={rbf_c:.4f} | MultiSVR={mul_c:.4f} | Cat={metrics_summary['B1_CatBoost']['cliff_rmse_mean']:.4f} | "
        f"E_MLP={mlp_e_c:.4f} | E_SVM_A2={metrics_summary['E_SVM_A2Delta']['cliff_rmse_mean']:.4f} | "
        f"E_Multi_FiLM={em_c:.4f}±{em_c_std:.4f}{alert}",
        flush=True,
    )
    return result


def _plot_jcim_3panel(df_long: pd.DataFrame, df_wide: pd.DataFrame, fig_path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8))

    # Panel A: 30-Target Mean Cliff RMSE & Overall RMSE
    key_models = [
        ("A1_Direct_MLP", "Direct MLP", "#BBBBBB"),
        ("B1_CatBoost", "CatBoost", "#A6CEE3"),
        ("B1b_LightGBM", "LightGBM", "#6BAED6"),
        ("K_Tan_SVR", "Tanimoto-SVR", "#56B4E9"),
        ("K_RBF_SVM", "RBF-SVM", "#777777"),
        ("K_Multi_SVR", "MultiKernel-SVR", "#0072B2"),
        ("Y_A2_2D", "Y_A2_Delta", "#F0E442"),
        ("Y_FiLM_2DLFER", "Y_FiLM_Delta", "#E69F00"),
        ("E_Multi_DirectMLP", "Multi+DirectMLP", "#CC79A7"),
        ("E_Multi_CatBoost", "Multi+CatBoost", "#9999CC"),
        ("E_SVM_A2Delta", "SVM+A2Delta", "#009E73"),
        ("E_Multi_FiLM", "E_Multi_FiLM", "#D55E00"),
    ]
    agg = df_long.groupby("model")[["cliff_rmse_mean", "rmse_mean"]].mean()
    x = np.arange(len(key_models))
    w = 0.38
    cvals = [float(agg.loc[m, "cliff_rmse_mean"]) for m, _, _ in key_models]
    ovals = [float(agg.loc[m, "rmse_mean"]) for m, _, _ in key_models]
    cols = [c for _, _, c in key_models]
    lbls = [l for _, l, _ in key_models]

    ax1.bar(x - w / 2, cvals, width=w, color=cols, edgecolor="black", linewidth=0.5, label="Cliff RMSE")
    ax1.bar(x + w / 2, ovals, width=w, color=cols, alpha=0.45, edgecolor="black", linewidth=0.5, hatch="//", label="Overall RMSE")
    for xi, cv in zip(x, cvals):
        ax1.text(xi - w / 2, cv + 0.002, f"{cv:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=32, ha="right", fontsize=7.5)
    ax1.set_ylim(0.55, max(cvals) + 0.04)
    ax1.set_ylabel("30-Target Mean RMSE (Lower = Better)")
    ax1.set_title("A. 30-Target Mean Performance", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel B: Sorted 30-Target Waterfall vs K_RBF_SVM & E_Multi_DirectMLP
    d_svm = (df_wide["E_Multi_FiLM_cliff"] - df_wide["K_RBF_SVM_cliff"]).to_numpy()
    d_mlp = (df_wide["E_Multi_FiLM_cliff"] - df_wide["E_Multi_DirectMLP_cliff"]).to_numpy()
    order = np.argsort(d_svm)
    tgts_s = df_wide["target"].to_numpy()[order]
    yp = np.arange(len(tgts_s))
    bh = 0.36
    ax2.barh(yp - bh / 2, d_svm[order], height=bh, color="#D55E00", label="E_Multi_FiLM - K_RBF_SVM")
    ax2.barh(yp + bh / 2, d_mlp[order], height=bh, color="#009E73", alpha=0.85, label="E_Multi_FiLM - E_Multi_DirectMLP")
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax2.set_yticks(yp)
    ax2.set_yticklabels([t.replace("CHEMBL", "") for t in tgts_s], fontsize=6.5)
    ax2.set_xlabel("Delta Cliff RMSE (< 0 = E_Multi_FiLM Better)")
    ax2.set_title("B. Per-Target Gains Across 30 Targets", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")

    # Panel C: Scatter E_Multi_FiLM vs K_RBF_SVM Cliff RMSE colored by Ki vs EC50
    for assay, col, mk in [("Ki", "#0072B2", "o"), ("EC50", "#D55E00", "^")]:
        sub = df_wide[df_wide["assay_type"] == assay]
        ax3.scatter(sub["K_RBF_SVM_cliff"], sub["E_Multi_FiLM_cliff"], color=col, marker=mk, s=55, edgecolor="black", linewidth=0.6, label=f"{assay} (N={len(sub)})")
    lo = float(min(df_wide["K_RBF_SVM_cliff"].min(), df_wide["E_Multi_FiLM_cliff"].min())) - 0.03
    hi = float(max(df_wide["K_RBF_SVM_cliff"].max(), df_wide["E_Multi_FiLM_cliff"].max())) + 0.03
    ax3.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x (Parity)")
    ax3.set_xlim(lo, hi)
    ax3.set_ylim(lo, hi)
    ax3.set_xlabel("K_RBF_SVM Cliff RMSE")
    ax3.set_ylabel("E_Multi_FiLM Cliff RMSE")
    ax3.set_title("C. E_Multi_FiLM vs K_RBF_SVM (Below Line = Win)", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def _generate_final_reports(results: list[dict[str, Any]], total_elapsed: float) -> None:
    out_dir = _out_root()
    fig_dir = _fig_dir()

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for r in results:
        tgt = r["target"]
        ch = r["chemistry"]
        wrow: dict[str, Any] = {"target": tgt, **ch, **r["flags"]}
        for m_name in JCIM_MODELS:
            mm = r["metrics"][m_name]
            long_rows.append({"target": tgt, "assay_type": ch["assay_type"], "n_total": ch["n_total"], "model": m_name, **mm})
            wrow[f"{m_name}_cliff"] = mm["cliff_rmse_mean"]
            wrow[f"{m_name}_cliff_std"] = mm["cliff_rmse_std"]
            wrow[f"{m_name}_rmse"] = mm["rmse_mean"]
            wrow[f"{m_name}_noncliff"] = mm["non_cliff_rmse_mean"]
            wrow[f"{m_name}_spearman"] = mm["spearman_mean"]
        wide_rows.append(wrow)

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)
    long_csv = out_dir / "jcim30_summary_long.csv"
    wide_csv = out_dir / "jcim30_summary_wide.csv"
    df_long.to_csv(long_csv, index=False)
    df_wide.to_csv(wide_csv, index=False)

    fig_path = fig_dir / "jcim_30target_main.png"
    _plot_jcim_3panel(df_long, df_wide, fig_path)

    agg = df_long.groupby("model")[["rmse_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "mae_mean", "cliff_mae_mean", "r2_mean", "spearman_mean"]].mean()
    agg_sorted = agg.sort_values("cliff_rmse_mean")

    md_path = ROOT_DIR / "refine-logs" / "JCIM_30TARGET_MAIN_RESULTS.md"
    lines: list[str] = [
        "# JCIM 30-Target MoleculeACE Benchmark Results (Pure-2D `E_Multi_FiLM`)",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Targets**: {len(results)} | **Seeds per Target**: {len(SEEDS)} | **Elapsed**: {total_elapsed:.1f}s",
        "",
        "## 1. 30-Target Overall Leaderboard (Mean Across 30 Targets x 5 Seeds, Sorted by Cliff RMSE)",
        "",
        "| Model | Overall RMSE | Cliff RMSE | Non-Cliff RMSE | MAE | Cliff MAE | R2 | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, row in agg_sorted.iterrows():
        lines.append(
            f"| `{m_name}` | {row['rmse_mean']:.4f} | {row['cliff_rmse_mean']:.4f} | "
            f"{row['non_cliff_rmse_mean']:.4f} | {row['mae_mean']:.4f} | {row['cliff_mae_mean']:.4f} | "
            f"{row['r2_mean']:.4f} | {row['spearman_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Cross-Target Statistical Significance (`E_Multi_FiLM` vs Every Baseline/Control, N=30)",
        "",
        "### A. Activity Cliff RMSE Comparisons (Two-Sided Wilcoxon + 5000-Bootstrap 95% CI of Mean Diff)",
        "",
        "| Comparison (`E_Multi_FiLM` vs Ref) | Win / Tie / Loss | Mean Diff | Median Diff | 95% Bootstrap CI | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for ref in COMPARATOR_ORDER:
        st = _cross_target_stats(df_wide, "E_Multi_FiLM_cliff", f"{ref}_cliff")
        lines.append(
            f"| vs `{ref}` | {st['wins']} / {st['ties']} / {st['losses']} | "
            f"{st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] | `{st['p_value']:.4e}` |"
        )

    lines.extend([
        "",
        "### B. Overall RMSE Comparisons (N=30)",
        "",
        "| Comparison (`E_Multi_FiLM` vs Ref) | Win / Tie / Loss | Mean Diff | Median Diff | 95% Bootstrap CI | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for ref in COMPARATOR_ORDER:
        st = _cross_target_stats(df_wide, "E_Multi_FiLM_rmse", f"{ref}_rmse")
        lines.append(
            f"| vs `{ref}` | {st['wins']} / {st['ties']} / {st['losses']} | "
            f"{st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] | `{st['p_value']:.4e}` |"
        )

    lines.extend([
        "",
        "## 3. Subgroup Breakdowns (Assay Type & Dataset Size)",
        "",
        "### By Assay Type (`Ki` vs `EC50`)",
        "",
        "| Assay | N Targets | K_RBF_SVM Cliff | K_Multi_SVR Cliff | E_Multi_DirectMLP Cliff | E_Multi_FiLM Cliff | Mean (FiLM - RBF) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for assay in ("Ki", "EC50"):
        sub = df_wide[df_wide["assay_type"] == assay]
        if len(sub) > 0:
            diff_m = (sub["E_Multi_FiLM_cliff"] - sub["K_RBF_SVM_cliff"]).mean()
            lines.append(
                f"| `{assay}` | {len(sub)} | {sub['K_RBF_SVM_cliff'].mean():.4f} | "
                f"{sub['K_Multi_SVR_cliff'].mean():.4f} | {sub['E_Multi_DirectMLP_cliff'].mean():.4f} | "
                f"{sub['E_Multi_FiLM_cliff'].mean():.4f} | {diff_m:+.4f} |"
            )

    lines.extend([
        "",
        "### By Dataset Size (`n_total`)",
        "",
        "| Size Bucket | N Targets | K_RBF_SVM Cliff | K_Multi_SVR Cliff | E_Multi_DirectMLP Cliff | E_Multi_FiLM Cliff | Mean (FiLM - RBF) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for lbl, mask in [
        ("n < 1000", df_wide["n_total"] < 1000),
        ("1000 <= n <= 2000", (df_wide["n_total"] >= 1000) & (df_wide["n_total"] <= 2000)),
        ("n > 2000", df_wide["n_total"] > 2000),
    ]:
        sub = df_wide[mask]
        if len(sub) > 0:
            diff_m = (sub["E_Multi_FiLM_cliff"] - sub["K_RBF_SVM_cliff"]).mean()
            lines.append(
                f"| `{lbl}` | {len(sub)} | {sub['K_RBF_SVM_cliff'].mean():.4f} | "
                f"{sub['K_Multi_SVR_cliff'].mean():.4f} | {sub['E_Multi_DirectMLP_cliff'].mean():.4f} | "
                f"{sub['E_Multi_FiLM_cliff'].mean():.4f} | {diff_m:+.4f} |"
            )

    neg_df = df_wide[df_wide["LOSE_VS_RBF_SVM"] | df_wide["LOSE_VS_MULTI_SVR"] | df_wide["LOSE_VS_MULTI_DIRECTMLP"]]
    lines.extend([
        "",
        "## 4. Negative-Result & Loss Breakdown (`E_Multi_FiLM` Lost to SVM / MultiSVR / MultiDirectMLP)",
        "",
    ])
    if len(neg_df) == 0:
        lines.append("Zero negative alerts across all 30 targets.")
    else:
        lines.extend([
            f"Triggered on **{len(neg_df)}/{len(df_wide)}** targets:",
            "",
            "| Target | Assay | n (Test/Cliff) | MW | LogP | TPSA | RotB | RBF_SVM | Multi_SVR | E_Multi_MLP | E_Multi_FiLM | Delta vs RBF | Delta vs MultiSVR |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for _, rw in neg_df.iterrows():
            lines.append(
                f"| `{rw['target']}` | {rw['assay_type']} | {rw['n_total']} ({rw['n_test']}/{rw['n_test_cliff']}) | "
                f"{rw['mean_mw']:.1f} | {rw['mean_logp']:.2f} | {rw['mean_tpsa']:.1f} | {rw['mean_rot_bonds']:.1f} | "
                f"{rw['K_RBF_SVM_cliff']:.4f} | {rw['K_Multi_SVR_cliff']:.4f} | {rw['E_Multi_DirectMLP_cliff']:.4f} | "
                f"{rw['E_Multi_FiLM_cliff']:.4f} | {rw['E_Multi_FiLM_cliff'] - rw['K_RBF_SVM_cliff']:+.4f} | "
                f"{rw['E_Multi_FiLM_cliff'] - rw['K_Multi_SVR_cliff']:+.4f} |"
            )

    lines.extend([
        "",
        "## 5. Complete 30-Target Cliff RMSE Table",
        "",
        "| Target | Assay | n | B1_CatBoost | A1_MLP | K_RBF_SVM | K_Multi_SVR | E_Multi_MLP | E_SVM_A2 | E_Multi_A2 | E_Multi_FiLM (Mean±Std) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, rw in df_wide.iterrows():
        lines.append(
            f"| `{rw['target']}` | {rw['assay_type']} | {rw['n_total']} | "
            f"{rw['B1_CatBoost_cliff']:.4f} | {rw['A1_Direct_MLP_cliff']:.4f} | "
            f"{rw['K_RBF_SVM_cliff']:.4f} | {rw['K_Multi_SVR_cliff']:.4f} | "
            f"{rw['E_Multi_DirectMLP_cliff']:.4f} | {rw['E_SVM_A2Delta_cliff']:.4f} | "
            f"{rw['E_Multi_A2Delta_cliff']:.4f} | {rw['E_Multi_FiLM_cliff']:.4f}±{rw['E_Multi_FiLM_cliff_std']:.4f} |"
        )

    lines.extend([
        "",
        f"Long CSV: `{long_csv.as_posix()}`",
        f"Wide CSV: `{wide_csv.as_posix()}`",
        f"Figure: `{fig_path.as_posix()}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generated JCIM 30-target report: {md_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 30-target JCIM benchmark for pure-2D E_Multi_FiLM.")
    parser.add_argument("--targets", type=str, default="", help="Optional comma-separated subset of targets.")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seeds (subset of 42..46).")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()

    _verify_2d_equivalence()

    target_list = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else list(ALL_30_TARGETS)
    seeds_tuple = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip()) if args.seeds.strip() else SEEDS
    save_cache = not args.no_save

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting JCIM 30-target pure-2D run "
        f"({len(target_list)} targets, seeds={seeds_tuple}, save_cache={save_cache})",
        flush=True,
    )
    t0 = time.time()
    results = [_evaluate_target(tgt, seeds=seeds_tuple, force=args.force, save_cache=save_cache) for tgt in target_list]
    elapsed = time.time() - t0

    if save_cache and len(results) == len(ALL_30_TARGETS) and len(seeds_tuple) == len(SEEDS):
        _generate_final_reports(results, total_elapsed=elapsed)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Partial/smoke run completed in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
