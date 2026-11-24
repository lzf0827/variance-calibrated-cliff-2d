"""External Robustness Check: Strict Bemis-Murcko Scaffold Split across 10 Representative MoleculeACE Targets."""

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
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_jcim_step_a import _bates_granger_weight, _stratified_3fold_indices
from physchem_delta.eval_opt13_modular import _fit_kernel_bases, _train_yspace_modular
from physchem_delta.eval_plan_a import _fit_rbf_svr, _lfer_atom_matrix
from physchem_delta.metrics import compute_mae, compute_rmse, evaluate_predictions
from physchem_delta.models import (
    _train_direct_mlp_with_inner_split,
    build_tier_matrices,
    compute_tanimoto_matrix,
    make_inner_split,
    run_tabular_baseline,
)
from physchem_delta.run_30_targets import WAVE_1_TARGETS
from physchem_delta.run_jcim_30targets import _cross_target_stats, extract_or_load_2d_features

SCAFFOLD_SEEDS: tuple[int, ...] = (42, 43)

SCAFFOLD_MODELS: tuple[str, ...] = (
    "K_RBF_SVM",
    "K_Multi_SVR",
    "B1_CatBoost",
    "A1_Direct_MLP",
    "Y_FiLM_2DLFER",
    "E_Multi_FiLM",
    "E_Multi_FiLM_PriorN",
    "E_Multi_DirectMLP_OptW",
    "E_Multi_FiLM_OptW",
    "E_Multi_DirectMLP_GroupOptW",
    "E_Multi_FiLM_GroupOptW",
)


def _out_root() -> Path:
    d = ROOT_DIR / "refine-logs" / "scaffold_robustness_results"
    (d / "per_target").mkdir(parents=True, exist_ok=True)
    return d


def _fig_dir() -> Path:
    d = ROOT_DIR / "refine-logs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compute_murcko_scaffolds(smiles_list: list[str]) -> list[str]:
    scaffolds: list[str] = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {idx}: {smi}")
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaf if scaf else "__ACYCLIC__")
    return scaffolds


def _make_scaffold_split(scaffolds: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Deterministic seed-shuffled Bemis-Murcko scaffold 80/20 split with zero scaffold overlap."""
    n_total = len(scaffolds)
    groups: dict[str, list[int]] = {}
    for idx, scaf in enumerate(scaffolds):
        groups.setdefault(scaf, []).append(idx)

    keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))

    n_target_test = int(round(0.20 * n_total))
    n_min_test = int(round(0.15 * n_total))
    n_max_test = int(round(0.28 * n_total))

    test_list: list[int] = []
    train_list: list[int] = []
    skipped_keys: list[str] = []

    for pos in perm.tolist():
        key = keys[pos]
        members = groups[key]
        if len(test_list) < n_target_test and (len(test_list) + len(members) <= n_max_test):
            test_list.extend(members)
        elif len(test_list) < n_target_test:
            skipped_keys.append(key)
        else:
            train_list.extend(members)

    # Top up from skipped groups (smallest first) only if test set is still underweight
    if len(test_list) < n_min_test and skipped_keys:
        skipped_keys.sort(key=lambda k: len(groups[k]))
        for key in skipped_keys:
            members = groups[key]
            if len(test_list) < n_min_test and (len(test_list) + len(members) <= int(round(0.35 * n_total)) or len(test_list) == 0):
                test_list.extend(members)
            else:
                train_list.extend(members)
    else:
        for key in skipped_keys:
            train_list.extend(groups[key])

    train_idx = np.sort(np.asarray(train_list, dtype=np.int64))
    test_idx = np.sort(np.asarray(test_list, dtype=np.int64))

    # Strict anti-leakage assertions
    assert train_idx.size + test_idx.size == n_total, "Scaffold split lost/duplicated molecules"
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, "Train/test molecule overlap"
    tr_scafs = {scaffolds[i] for i in train_idx.tolist()}
    te_scafs = {scaffolds[i] for i in test_idx.tolist()}
    assert len(tr_scafs & te_scafs) == 0, "Train/test Murcko scaffold overlap detected!"
    return train_idx, test_idx, len(tr_scafs), len(te_scafs)


def _evaluate_split(
    target: str,
    ds: Any,
    feats: Any,
    lfer_atoms: np.ndarray,
    dummy_graphs: list[dict[str, np.ndarray]],
    scaffolds: list[str],
    split_seed: int,
) -> dict[str, Any]:
    train_idx, test_idx, n_tr_scaf, n_te_scaf = _make_scaffold_split(scaffolds, split_seed)
    n_tr = int(train_idx.size)
    n_te = int(test_idx.size)
    y_tr = ds.y[train_idx]
    y_te = ds.y[test_idx]
    c_te = ds.cliff_mol[test_idx]

    sim_te_tr = compute_tanimoto_matrix(feats.ecfp_chiral[test_idx], feats.ecfp_chiral[train_idx])
    s_max = np.max(sim_te_tr, axis=1).astype(np.float64)

    # 1. Full-train models on this scaffold split
    pred_rbf = _fit_rbf_svr(feats.ecfp_chiral, ds.y, train_idx, test_idx)
    _, pred_multi = _fit_kernel_bases(feats, ds.y, train_idx, test_idx)
    pred_cat = run_tabular_baseline("B1_ECFP_Chiral_CatBoost", feats, ds.y, train_idx, test_idx, seed=split_seed)["y_pred_test"]

    in_tr_rel, in_va_rel = make_inner_split(y_tr, seed=split_seed, val_frac=0.15)
    in_tr_idx = train_idx[in_tr_rel]
    in_va_idx = train_idx[in_va_rel]
    bundle_2d = build_tier_matrices(feats, tier="2d", fit_idx=in_tr_idx)

    pred_mlp = _train_direct_mlp_with_inner_split(
        bundle_2d.x_mol, ds.y, train_idx, in_tr_rel, in_va_rel, test_idx, seed=split_seed
    )["y_pred_test"]
    pred_film = _train_yspace_modular(
        "Y_FiLM_2DLFER", feats, dummy_graphs, lfer_atoms, ds.y,
        train_idx, in_tr_idx, in_va_idx, test_idx, seed=split_seed,
    )["y_pred_test"]

    # 2. Exact Step-A 3-fold StratifiedKFold nested OOF inside train_idx ONLY
    oof_k_st = np.zeros((n_tr,), dtype=np.float64)
    oof_mlp_st = np.zeros((n_tr,), dtype=np.float64)
    oof_film_st = np.zeros((n_tr,), dtype=np.float64)

    for tr_rel_f, va_rel_f in _stratified_3fold_indices(y_tr, seed=split_seed):
        tr_f = train_idx[tr_rel_f]
        va_f = train_idx[va_rel_f]
        assert len(set(tr_f.tolist()) & set(test_idx.tolist())) == 0
        assert len(set(va_f.tolist()) & set(test_idx.tolist())) == 0

        _, oof_k_st[va_rel_f] = _fit_kernel_bases(feats, ds.y, tr_f, va_f)

        f_in_tr_rel, f_in_va_rel = make_inner_split(ds.y[tr_f], seed=split_seed, val_frac=0.15)
        f_in_tr_idx = tr_f[f_in_tr_rel]
        f_in_va_idx = tr_f[f_in_va_rel]
        b_f = build_tier_matrices(feats, tier="2d", fit_idx=f_in_tr_idx)

        oof_mlp_st[va_rel_f] = _train_direct_mlp_with_inner_split(
            b_f.x_mol, ds.y, tr_f, f_in_tr_rel, f_in_va_rel, va_f, seed=split_seed
        )["y_pred_test"]
        oof_film_st[va_rel_f] = _train_yspace_modular(
            "Y_FiLM_2DLFER", feats, dummy_graphs, lfer_atoms, ds.y,
            tr_f, f_in_tr_idx, f_in_va_idx, va_f, seed=split_seed,
        )["y_pred_test"]

    w_strat = {
        "mlp": _bates_granger_weight(y_tr, oof_k_st, oof_mlp_st),
        "film": _bates_granger_weight(y_tr, oof_k_st, oof_film_st),
    }

    # 3. Scaffold-aware 3-fold GroupKFold nested OOF inside train_idx ONLY
    assert n_tr_scaf >= 3, f"{target} has < 3 training scaffolds"
    tr_groups = np.asarray([scaffolds[i] for i in train_idx.tolist()])
    gkf = GroupKFold(n_splits=3)
    oof_k_gk = np.zeros((n_tr,), dtype=np.float64)
    oof_mlp_gk = np.zeros((n_tr,), dtype=np.float64)
    oof_film_gk = np.zeros((n_tr,), dtype=np.float64)

    for tr_rel_f, va_rel_f in gkf.split(np.zeros((n_tr, 1)), y_tr, groups=tr_groups):
        tr_f = train_idx[tr_rel_f]
        va_f = train_idx[va_rel_f]
        assert len({scaffolds[i] for i in tr_f} & {scaffolds[i] for i in va_f}) == 0, "Inner GroupKFold scaffold leak!"

        _, oof_k_gk[va_rel_f] = _fit_kernel_bases(feats, ds.y, tr_f, va_f)

        f_in_tr_rel, f_in_va_rel = make_inner_split(ds.y[tr_f], seed=split_seed, val_frac=0.15)
        f_in_tr_idx = tr_f[f_in_tr_rel]
        f_in_va_idx = tr_f[f_in_va_rel]
        b_f = build_tier_matrices(feats, tier="2d", fit_idx=f_in_tr_idx)

        oof_mlp_gk[va_rel_f] = _train_direct_mlp_with_inner_split(
            b_f.x_mol, ds.y, tr_f, f_in_tr_rel, f_in_va_rel, va_f, seed=split_seed
        )["y_pred_test"]
        oof_film_gk[va_rel_f] = _train_yspace_modular(
            "Y_FiLM_2DLFER", feats, dummy_graphs, lfer_atoms, ds.y,
            tr_f, f_in_tr_idx, f_in_va_idx, va_f, seed=split_seed,
        )["y_pred_test"]

    w_group = {
        "mlp": _bates_granger_weight(y_tr, oof_k_gk, oof_mlp_gk),
        "film": _bates_granger_weight(y_tr, oof_k_gk, oof_film_gk),
    }
    w_prior = float(0.5 * n_tr / (n_tr + 1000.0))

    preds_map: dict[str, np.ndarray] = {
        "K_RBF_SVM": pred_rbf,
        "K_Multi_SVR": pred_multi,
        "B1_CatBoost": pred_cat,
        "A1_Direct_MLP": pred_mlp,
        "Y_FiLM_2DLFER": pred_film,
        "E_Multi_FiLM": 0.5 * pred_multi + 0.5 * pred_film,
        "E_Multi_FiLM_PriorN": (1.0 - w_prior) * pred_multi + w_prior * pred_film,
        "E_Multi_DirectMLP_OptW": (1.0 - w_strat["mlp"]) * pred_multi + w_strat["mlp"] * pred_mlp,
        "E_Multi_FiLM_OptW": (1.0 - w_strat["film"]) * pred_multi + w_strat["film"] * pred_film,
        "E_Multi_DirectMLP_GroupOptW": (1.0 - w_group["mlp"]) * pred_multi + w_group["mlp"] * pred_mlp,
        "E_Multi_FiLM_GroupOptW": (1.0 - w_group["film"]) * pred_multi + w_group["film"] * pred_film,
    }

    metrics: dict[str, dict[str, float]] = {}
    for m_name, yp in preds_map.items():
        if not np.all(np.isfinite(yp)):
            raise RuntimeError(f"Non-finite prediction in {target} split={split_seed} {m_name}")
        metrics[m_name] = evaluate_predictions(y_te, yp, c_te)

    return {
        "split_seed": int(split_seed),
        "n_train": n_tr,
        "n_test": n_te,
        "n_train_scaffolds": int(n_tr_scaf),
        "n_test_scaffolds": int(n_te_scaf),
        "n_test_cliff": int(np.sum(c_te == 1)),
        "mean_s_max": float(np.mean(s_max)),
        "frac_smax_lt_05": float(np.mean(s_max < 0.50)),
        "frac_smax_ge_07": float(np.mean(s_max >= 0.70)),
        "w_strat": w_strat,
        "w_group": w_group,
        "w_prior": w_prior,
        "y_te": y_te,
        "c_te": c_te,
        "s_max": s_max,
        "preds": preds_map,
        "metrics": metrics,
    }


def _evaluate_target_scaffold(
    target: str,
    seeds: tuple[int, ...],
    force: bool,
    save_cache: bool,
) -> dict[str, Any]:
    out_dir = _out_root() / "per_target"
    json_path = out_dir / f"{target}_scaffold.json"
    npz_path = out_dir / f"{target}_scaffold_preds.npz"

    if save_cache and seeds == SCAFFOLD_SEEDS and json_path.exists() and npz_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if all(m in cached.get("metrics", {}) for m in SCAFFOLD_MODELS):
            print(f"[CACHE] {target}: loaded Scaffold Split checkpoint", flush=True)
            return cached

    t0 = time.time()
    ds = load_target_dataset(target)
    feats = extract_or_load_2d_features(target, ds.smiles)
    lfer_atoms = _lfer_atom_matrix(feats)
    dummy_graphs = [{"x_cont": np.zeros((1, 3), dtype=np.float32)}] * len(ds.smiles)
    scaffolds = _compute_murcko_scaffolds(ds.smiles)

    # Record original MoleculeACE stratified split mean_s_max for shift quantification
    sim_orig = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
    orig_mean_smax = float(np.mean(np.max(sim_orig, axis=1)))

    split_records: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_c: list[np.ndarray] = []
    pooled_smax: list[np.ndarray] = []
    pooled_preds: dict[str, list[np.ndarray]] = {m: [] for m in SCAFFOLD_MODELS}

    for sp_seed in seeds:
        sp = _evaluate_split(target, ds, feats, lfer_atoms, dummy_graphs, scaffolds, sp_seed)
        split_records.append(
            {k: v for k, v in sp.items() if k not in ("y_te", "c_te", "s_max", "preds")}
        )
        pooled_y.append(sp["y_te"])
        pooled_c.append(sp["c_te"])
        pooled_smax.append(sp["s_max"])
        for m in SCAFFOLD_MODELS:
            pooled_preds[m].append(sp["preds"][m])

    metrics_agg: dict[str, dict[str, float]] = {}
    for m in SCAFFOLD_MODELS:
        rmses = np.asarray([sr["metrics"][m]["rmse"] for sr in split_records], dtype=np.float64)
        maes = np.asarray([sr["metrics"][m]["mae"] for sr in split_records], dtype=np.float64)
        cliffs = np.asarray([sr["metrics"][m]["cliff_rmse"] for sr in split_records], dtype=np.float64)
        ncliffs = np.asarray([sr["metrics"][m]["non_cliff_rmse"] for sr in split_records], dtype=np.float64)
        sps = np.asarray([sr["metrics"][m]["spearman"] for sr in split_records], dtype=np.float64)
        metrics_agg[m] = {
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "cliff_rmse_mean": float(np.nanmean(cliffs)) if np.any(np.isfinite(cliffs)) else float("nan"),
            "non_cliff_rmse_mean": float(np.nanmean(ncliffs)),
            "spearman_mean": float(np.mean(sps)),
        }

    mean_w_strat = {
        k: float(np.mean([sr["w_strat"][k] for sr in split_records]))
        for k in ("mlp", "film")
    }
    mean_w_group = {
        k: float(np.mean([sr["w_group"][k] for sr in split_records]))
        for k in ("mlp", "film")
    }
    elapsed = round(time.time() - t0, 2)

    result = {
        "target": target,
        "assay_type": "EC50" if target.endswith("_EC50") else "Ki",
        "n_total": int(len(ds.smiles)),
        "n_total_scaffolds": int(len(set(scaffolds))),
        "orig_mean_smax": orig_mean_smax,
        "scaf_mean_smax": float(np.mean([sr["mean_s_max"] for sr in split_records])),
        "scaf_frac_smax_lt_05": float(np.mean([sr["frac_smax_lt_05"] for sr in split_records])),
        "scaf_frac_smax_ge_07": float(np.mean([sr["frac_smax_ge_07"] for sr in split_records])),
        "mean_n_test": float(np.mean([sr["n_test"] for sr in split_records])),
        "total_test_mols": int(sum(sr["n_test"] for sr in split_records)),
        "total_test_cliff": int(sum(sr["n_test_cliff"] for sr in split_records)),
        "w_strat_mean": mean_w_strat,
        "w_group_mean": mean_w_group,
        "w_prior_mean": float(np.mean([sr["w_prior"] for sr in split_records])),
        "elapsed_sec": elapsed,
        "metrics": metrics_agg,
        "splits": split_records,
    }

    if save_cache and seeds == SCAFFOLD_SEEDS:
        npz_payload: dict[str, np.ndarray] = {
            "y_pooled": np.concatenate(pooled_y, axis=0),
            "c_pooled": np.concatenate(pooled_c, axis=0),
            "s_max_pooled": np.concatenate(pooled_smax, axis=0),
        }
        for m in SCAFFOLD_MODELS:
            npz_payload[f"{m}_pooled"] = np.concatenate(pooled_preds[m], axis=0)
        np.savez_compressed(npz_path, **npz_payload)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mm = metrics_agg
    print(
        f"[DONE {elapsed:.1f}s] {target} (n={result['n_total']}, scafs={result['n_total_scaffolds']}, "
        f"smax {orig_mean_smax:.3f}->{result['scaf_mean_smax']:.3f}): "
        f"RBF={mm['K_RBF_SVM']['rmse_mean']:.4f} | Multi={mm['K_Multi_SVR']['rmse_mean']:.4f} | "
        f"MLP_OptW={mm['E_Multi_DirectMLP_OptW']['rmse_mean']:.4f} | "
        f"FiLM_OptW={mm['E_Multi_FiLM_OptW']['rmse_mean']:.4f}(w={mean_w_strat['film']:.3f}) | "
        f"FiLM_GrpW={mm['E_Multi_FiLM_GroupOptW']['rmse_mean']:.4f}(wg={mean_w_group['film']:.3f})",
        flush=True,
    )
    return result


def _compute_smax_bins(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_dir = _out_root() / "per_target"
    ys, cs, smaxes = [], [], []
    preds_all: dict[str, list[np.ndarray]] = {m: [] for m in SCAFFOLD_MODELS}

    for r in results:
        z = np.load(out_dir / f"{r['target']}_scaffold_preds.npz")
        ys.append(z["y_pooled"])
        cs.append(z["c_pooled"])
        smaxes.append(z["s_max_pooled"])
        for m in SCAFFOLD_MODELS:
            preds_all[m].append(z[f"{m}_pooled"])

    y_vec = np.concatenate(ys, axis=0)
    c_vec = np.concatenate(cs, axis=0) == 1
    s_vec = np.concatenate(smaxes, axis=0)
    p_map = {m: np.concatenate(preds_all[m], axis=0) for m in SCAFFOLD_MODELS}

    bins = [
        ("s_max < 0.50 (Far OOD)", s_vec < 0.50),
        ("0.50 <= s_max < 0.70 (Scaffold Hop)", (s_vec >= 0.50) & (s_vec < 0.70)),
        ("s_max >= 0.70 (Near Bioisostere)", s_vec >= 0.70),
        ("Original cliff_mol == 1 (Pooled)", c_vec),
    ]
    rows: list[dict[str, Any]] = []
    for lbl, mask in bins:
        if np.sum(mask) == 0:
            continue
        row: dict[str, Any] = {"bin": lbl, "n_mols": int(np.sum(mask))}
        for m in SCAFFOLD_MODELS:
            row[f"{m}_rmse"] = compute_rmse(y_vec[mask], p_map[m][mask])
            row[f"{m}_mae"] = compute_mae(y_vec[mask], p_map[m][mask])
        rows.append(row)
    return rows


def _plot_scaffold_3panel(
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    smax_bins: list[dict[str, Any]],
    fig_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8))

    # Panel A: 10-Target Mean Overall RMSE under Strict Scaffold Split
    bar_specs = [
        ("A1_Direct_MLP", "DirectMLP", "#BBBBBB"),
        ("B1_CatBoost", "CatBoost", "#A6CEE3"),
        ("Y_FiLM_2DLFER", "Y_FiLM_Delta", "#F0E442"),
        ("K_RBF_SVM", "RBF-SVM", "#777777"),
        ("K_Multi_SVR", "MultiKernel-SVR", "#0072B2"),
        ("E_Multi_DirectMLP_OptW", "Multi+MLP(OptW)", "#CC79A7"),
        ("E_Multi_DirectMLP_GroupOptW", "Multi+MLP(GrpOptW)", "#8C6BB1"),
        ("E_Multi_FiLM", "Multi+FiLM(w=0.5)", "#E69F00"),
        ("E_Multi_FiLM_PriorN", "Multi+FiLM(PriorN)", "#009E73"),
        ("E_Multi_FiLM_OptW", "Multi+FiLM(OptW)", "#D55E00"),
        ("E_Multi_FiLM_GroupOptW", "Multi+FiLM(GrpOptW)", "#990000"),
    ]
    agg = df_long.groupby("model")[["rmse_mean", "mae_mean"]].mean()
    x = np.arange(len(bar_specs))
    ovals = [float(agg.loc[m, "rmse_mean"]) for m, _, _ in bar_specs]
    cols = [c for _, _, c in bar_specs]
    lbls = [l for _, l, _ in bar_specs]
    ax1.bar(x, ovals, width=0.65, color=cols, edgecolor="black", linewidth=0.5)
    for xi, ov in zip(x, ovals):
        ax1.text(xi, ov + 0.002, f"{ov:.4f}", ha="center", va="bottom", fontsize=6.8, rotation=45)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=30, ha="right", fontsize=7.5)
    ax1.set_ylim(min(ovals) - 0.04, max(ovals) + 0.04)
    ax1.set_ylabel("10-Target Mean Overall RMSE (Lower = Better)")
    ax1.set_title("A. Strict Murcko Scaffold Split Accuracy (N=10 x 2)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel B: Pooled RMSE across OOD s_max Similarity Regimes
    regimes = [r for r in smax_bins if r["bin"].startswith("s_max") or r["bin"].startswith("0.50")]
    xb = np.arange(len(regimes))
    xlbls = [f"{r['bin'].split(' (')[0]}\n(n={r['n_mols']})" for r in regimes]
    curves = [
        ("K_RBF_SVM", "K_RBF_SVM", "#777777", "--o"),
        ("K_Multi_SVR", "K_Multi_SVR", "#0072B2", "--s"),
        ("Y_FiLM_2DLFER", "Y_FiLM_Delta", "#F0E442", ":^"),
        ("E_Multi_DirectMLP_OptW", "Multi+MLP(OptW)", "#CC79A7", "-.^"),
        ("E_Multi_FiLM_OptW", "Multi+FiLM(OptW)", "#D55E00", "-D"),
        ("E_Multi_FiLM_GroupOptW", "Multi+FiLM(GrpOptW)", "#990000", "-P"),
    ]
    for m_key, lbl, col, fmt in curves:
        ys = [r[f"{m_key}_rmse"] for r in regimes]
        ax2.plot(xb, ys, fmt, color=col, linewidth=1.7, markersize=5, label=lbl)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(xlbls, fontsize=7.5)
    ax2.set_ylabel("Pooled RMSE")
    ax2.set_title("B. Performance Stratified by Unseen-Scaffold Similarity", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=7.2, loc="upper right")

    # Panel C: Learned Bates-Granger Weights vs OOD Mean s_max
    ax3.scatter(df_wide["scaf_mean_smax"], df_wide["w_strat_film"], color="#D55E00", s=55, edgecolor="black", label="Stratified OOF w*_FiLM")
    ax3.scatter(df_wide["scaf_mean_smax"], df_wide["w_group_film"], color="#0072B2", marker="^", s=60, edgecolor="black", label="Scaffold-GroupKFold w*_FiLM")
    for _, row in df_wide.iterrows():
        ax3.plot([row["scaf_mean_smax"], row["scaf_mean_smax"]], [row["w_strat_film"], row["w_group_film"]], color="gray", alpha=0.45, linewidth=0.9)
        ax3.text(row["scaf_mean_smax"] + 0.002, row["w_group_film"] + 0.01, row["target"].replace("CHEMBL", "C"), fontsize=6.5)
    ax3.set_xlabel("Scaffold Test-to-Train Mean Max Tanimoto (s_max)")
    ax3.set_ylabel("Learned Nested-OOF Weight w*_FiLM")
    ax3.set_title("C. Self-Limiting Weight Calibration Under OOD Shift", fontweight="bold")
    ax3.set_ylim(-0.02, 0.72)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def _generate_report(
    results: list[dict[str, Any]],
    smax_bins: list[dict[str, Any]],
    total_elapsed: float,
) -> None:
    out_dir = _out_root()
    fig_dir = _fig_dir()

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for r in results:
        tgt = r["target"]
        wrow: dict[str, Any] = {
            "target": tgt,
            "assay_type": r["assay_type"],
            "n_total": r["n_total"],
            "n_total_scaffolds": r["n_total_scaffolds"],
            "orig_mean_smax": r["orig_mean_smax"],
            "scaf_mean_smax": r["scaf_mean_smax"],
            "scaf_frac_smax_lt_05": r["scaf_frac_smax_lt_05"],
            "total_test_cliff": r["total_test_cliff"],
            "w_strat_film": r["w_strat_mean"]["film"],
            "w_strat_mlp": r["w_strat_mean"]["mlp"],
            "w_group_film": r["w_group_mean"]["film"],
            "w_group_mlp": r["w_group_mean"]["mlp"],
            "w_prior": r["w_prior_mean"],
        }
        for m in SCAFFOLD_MODELS:
            mm = r["metrics"][m]
            long_rows.append({"target": tgt, "assay_type": r["assay_type"], "n_total": r["n_total"], "model": m, **mm})
            wrow[f"{m}_rmse"] = mm["rmse_mean"]
            wrow[f"{m}_cliff"] = mm["cliff_rmse_mean"]
            wrow[f"{m}_spearman"] = mm["spearman_mean"]
        wide_rows.append(wrow)

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)
    csv_path = out_dir / "scaffold_summary.csv"
    wide_csv = out_dir / "scaffold_wide.csv"
    df_long.to_csv(csv_path, index=False)
    df_wide.to_csv(wide_csv, index=False)

    fig_path = fig_dir / "scaffold_robustness.png"
    _plot_scaffold_3panel(df_long, df_wide, smax_bins, fig_path)

    agg = df_long.groupby("model")[["rmse_mean", "mae_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "spearman_mean"]].mean()
    agg_sorted = agg.sort_values("rmse_mean")

    md_path = ROOT_DIR / "refine-logs" / "SCAFFOLD_ROBUSTNESS_AUDIT.md"
    lines: list[str] = [
        "# Strict Bemis-Murcko Scaffold Split External Robustness Audit",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Targets**: {len(results)} Wave-1 Diverse Targets × 2 Scaffold Splits (0% Murcko scaffold overlap asserted) | **Elapsed**: {total_elapsed:.1f}s",
        "",
        f"- **Structural Shift Verified**: Mean test-to-train `s_max` drops from `{df_wide['orig_mean_smax'].mean():.3f}` (MoleculeACE stratified split) to **`{df_wide['scaf_mean_smax'].mean():.3f}`** (strict Murcko scaffold split), with `{df_wide['scaf_frac_smax_lt_05'].mean():.1%}` of test molecules at `s_max < 0.50`.",
        f"- **Weight Self-Limiting Check**: Mean learned `w*_FiLM` under Stratified OOF = `{df_wide['w_strat_film'].mean():.3f}`; under Scaffold-Aware `GroupKFold` OOF = `{df_wide['w_group_film'].mean():.3f}`.",
        "",
        "## 1. 10-Target Scaffold Split Leaderboard (Sorted by Overall RMSE)",
        "",
        "| Model | Overall RMSE | MAE | Original-Cliff RMSE | Non-Cliff RMSE | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, row in agg_sorted.iterrows():
        lines.append(
            f"| `{m_name}` | {row['rmse_mean']:.4f} | {row['mae_mean']:.4f} | "
            f"{row['cliff_rmse_mean']:.4f} | {row['non_cliff_rmse_mean']:.4f} | {row['spearman_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Cross-Target Wilcoxon & 95% Bootstrap CI Under Scaffold Split (`N = 10` Targets)",
        "",
        "### A. Overall RMSE (Primary Under Unseen Scaffolds)",
        "",
        "| Comparison | Win / Tie / Loss | Mean Diff | Median Diff | 95% Bootstrap CI | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in [
        ("OptW_FiLM vs K_RBF_SVM", "E_Multi_FiLM_OptW_rmse", "K_RBF_SVM_rmse"),
        ("OptW_FiLM vs K_Multi_SVR", "E_Multi_FiLM_OptW_rmse", "K_Multi_SVR_rmse"),
        ("OptW_FiLM vs OptW_DirectMLP", "E_Multi_FiLM_OptW_rmse", "E_Multi_DirectMLP_OptW_rmse"),
        ("OptW_FiLM vs E_Multi_FiLM(0.5)", "E_Multi_FiLM_OptW_rmse", "E_Multi_FiLM_rmse"),
        ("GrpOptW_FiLM vs K_RBF_SVM", "E_Multi_FiLM_GroupOptW_rmse", "K_RBF_SVM_rmse"),
        ("GrpOptW_FiLM vs K_Multi_SVR", "E_Multi_FiLM_GroupOptW_rmse", "K_Multi_SVR_rmse"),
        ("GrpOptW_FiLM vs GrpOptW_MLP", "E_Multi_FiLM_GroupOptW_rmse", "E_Multi_DirectMLP_GroupOptW_rmse"),
        ("PriorN_FiLM vs K_RBF_SVM", "E_Multi_FiLM_PriorN_rmse", "K_RBF_SVM_rmse"),
        ("PriorN_FiLM vs K_Multi_SVR", "E_Multi_FiLM_PriorN_rmse", "K_Multi_SVR_rmse"),
    ]:
        st = _cross_target_stats(df_wide, ca, cb)
        lines.append(
            f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | "
            f"{st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] | `{st['p_value']:.4e}` |"
        )

    valid_cliff_wide = df_wide.dropna(subset=["E_Multi_FiLM_OptW_cliff", "K_RBF_SVM_cliff"])
    lines.extend([
        "",
        f"### B. Original Cliff-Labeled Test Subset Under Scaffold Split (`N = {len(valid_cliff_wide)}` Targets with `n_test_cliff > 0`)",
        "",
        "| Comparison | Win / Tie / Loss | Mean Diff | Median Diff | 95% Bootstrap CI | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in [
        ("OptW_FiLM vs K_RBF_SVM", "E_Multi_FiLM_OptW_cliff", "K_RBF_SVM_cliff"),
        ("OptW_FiLM vs K_Multi_SVR", "E_Multi_FiLM_OptW_cliff", "K_Multi_SVR_cliff"),
        ("OptW_FiLM vs OptW_DirectMLP", "E_Multi_FiLM_OptW_cliff", "E_Multi_DirectMLP_OptW_cliff"),
        ("GrpOptW_FiLM vs K_Multi_SVR", "E_Multi_FiLM_GroupOptW_cliff", "K_Multi_SVR_cliff"),
    ]:
        st = _cross_target_stats(valid_cliff_wide, ca, cb)
        lines.append(
            f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | "
            f"{st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | "
            f"[{st['ci95_lo']:+.5f}, {st['ci95_hi']:+.5f}] | `{st['p_value']:.4e}` |"
        )

    lines.extend([
        "",
        "## 3. Pooled OOD Similarity-Bin Breakdown (`s_max` Regimes Across 10 Targets × 2 Splits)",
        "",
        "| Similarity Regime | N Mols | RBF_SVM | Multi_SVR | Y_FiLM | Multi+MLP_OptW | Multi+FiLM(0.5) | Multi+FiLM_OptW | Multi+FiLM_GrpOptW |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in smax_bins:
        lines.append(
            f"| `{r['bin']}` | {r['n_mols']} | {r['K_RBF_SVM_rmse']:.4f} | {r['K_Multi_SVR_rmse']:.4f} | "
            f"{r['Y_FiLM_2DLFER_rmse']:.4f} | {r['E_Multi_DirectMLP_OptW_rmse']:.4f} | "
            f"{r['E_Multi_FiLM_rmse']:.4f} | {r['E_Multi_FiLM_OptW_rmse']:.4f} | {r['E_Multi_FiLM_GroupOptW_rmse']:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Per-Target Scaffold Split Summary",
        "",
        "| Target | n (Scafs) | `s_max` (Orig -> Scaf) | Cliff Mols | `w*_Strat` | `w*_Group` | RBF_SVM | Multi_SVR | OptW_MLP | OptW_FiLM | GrpOptW_FiLM |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, row in df_wide.iterrows():
        lines.append(
            f"| `{row['target']}` | {int(row['n_total'])} ({int(row['n_total_scaffolds'])}) | "
            f"{row['orig_mean_smax']:.3f} -> {row['scaf_mean_smax']:.3f} | {int(row['total_test_cliff'])} | "
            f"{row['w_strat_film']:.3f} | {row['w_group_film']:.3f} | "
            f"{row['K_RBF_SVM_rmse']:.4f} | {row['K_Multi_SVR_rmse']:.4f} | "
            f"{row['E_Multi_DirectMLP_OptW_rmse']:.4f} | {row['E_Multi_FiLM_OptW_rmse']:.4f} | {row['E_Multi_FiLM_GroupOptW_rmse']:.4f} |"
        )

    lines.extend([
        "",
        f"Summary CSV: `{csv_path.as_posix()}`",
        f"Wide CSV: `{wide_csv.as_posix()}`",
        f"Figure: `{fig_path.as_posix()}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {md_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10-target Bemis-Murcko Scaffold Split external robustness check.")
    parser.add_argument("--targets", type=str, default="", help="Optional comma-separated targets.")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated scaffold split seeds.")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()

    target_list = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else list(WAVE_1_TARGETS)
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip()) if args.seeds.strip() else SCAFFOLD_SEEDS
    save_cache = not args.no_save

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting Bemis-Murcko Scaffold Split check "
        f"({len(target_list)} targets x {len(seeds)} splits, save_cache={save_cache})",
        flush=True,
    )
    t0 = time.time()
    results = [_evaluate_target_scaffold(tgt, seeds=seeds, force=args.force, save_cache=save_cache) for tgt in target_list]
    elapsed = time.time() - t0

    if save_cache and len(results) == len(WAVE_1_TARGETS) and seeds == SCAFFOLD_SEEDS:
        smax_bins = _compute_smax_bins(results)
        _generate_report(results, smax_bins, total_elapsed=elapsed)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Smoke/partial scaffold check completed in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
