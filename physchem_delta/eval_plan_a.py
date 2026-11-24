"""Scheme A evaluation: RBF-SVR base + orthogonal LFER local delta + e_span/AIMNet2 physical gate."""

from __future__ import annotations

import copy
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
import torch
import torch.nn as nn
from scipy import optimize
from sklearn.svm import SVR

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import MVP_TARGETS, load_target_dataset
from physchem_delta.featurize import (
    TargetFeatures,
    compute_local_pair_deltas,
    extract_or_load_target_features,
    precompute_mol_env_counters,
)
from physchem_delta.metrics import compute_rmse, evaluate_predictions, paired_wilcoxon
from physchem_delta.models import (
    AntisymmetricPairwiseDeltaNet,
    _select_topk_anchors,
    build_tier_matrices,
    compute_tanimoto_matrix,
    get_device,
    make_inner_split,
    set_global_seed,
)

SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

# Exact indices in the 41-dim local LFER vector:
#   [0:25]  desc2d diff
#   [25]    fp_bit_diff
#   [26:33] sum_diff of 7 atomic props: [aimnet_eff_q, crip_logp, crip_mr, tpsa_c, 3d_dist, 3d_crowd, chiral]
#   [33:40] mean_diff of same 7 atomic props
#   [40]    diff_atom_count
LFER_CHANNELS: dict[str, list[int]] = {
    "AIMNet2_Delta_q": [26, 33],
    "Crippen_Delta_LogP": [27, 34],
    "Crippen_Delta_MR": [28, 35],
    "Delta_TPSA": [29, 36],
    "3D_eSpan_Geom": [30, 31, 37, 38],
    "Chiral_Stereo": [32, 39],
}


def _refine_dir() -> Path:
    return ROOT_DIR / "refine-logs"


def _fig_dir() -> Path:
    d = _refine_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lfer_atom_matrix(feats: TargetFeatures) -> np.ndarray:
    """
    Build clean 7-column orthogonal medicinal-chemistry LFER atomic matrix:
      [0: aimnet_eff_q, 1: crip_logp, 2: crip_mr, 3: tpsa_c, 4: 3d_dist, 5: 3d_crowd, 6: chiral_flag]
    """
    elec = feats.atom_elec_props
    # When aimnet2_wb97m_d3 is active, elec has 8 cols: [aimnet_q, aimnet_eff_q, gast_q, gast_eff_q, logp, mr, tpsa, estate]
    if elec.shape[1] >= 8:
        aimnet_eff_q = elec[:, 1:2]
        crip_logp = elec[:, 4:5]
        crip_mr = elec[:, 5:6]
        tpsa_c = elec[:, 6:7]
    else:
        # Explicit fallback compatibility
        aimnet_eff_q = elec[:, 1:2]
        crip_logp = elec[:, 2:3]
        crip_mr = elec[:, 3:4]
        tpsa_c = elec[:, 4:5]
    return np.concatenate([aimnet_eff_q, crip_logp, crip_mr, tpsa_c, feats.atom_3d_props], axis=1).astype(np.float32)


def _build_pair_diff(
    feats: TargetFeatures,
    bundle_2d_cont: np.ndarray,
    lfer_atoms: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    env_counters: list[Any],
    mode: str,
    e0_kcal: float = 10.0,
) -> np.ndarray:
    """
    mode:
      - '2d': desc2d diff + fp_bit diff (26 dims)
      - 'lfer_raw': desc2d diff + fp_bit diff + 7-channel sum_diff & mean_diff + cnt_diff (41 dims)
      - 'lfer_espan': same as 'lfer_raw', with 3D geometric channels attenuated by exp(-max(e_span_i, e_span_j)/E0)
    """
    desc_diff = (bundle_2d_cont[i_idx] - bundle_2d_cont[j_idx]).astype(np.float32)
    fp_i = feats.ecfp_chiral[i_idx]
    fp_j = feats.ecfp_chiral[j_idx]
    n_add = np.sum((fp_i > 0.5) & (fp_j < 0.5), axis=1, keepdims=True).astype(np.float32)
    n_rem = np.sum((fp_j > 0.5) & (fp_i < 0.5), axis=1, keepdims=True).astype(np.float32)
    fp_diff = n_add - n_rem

    if mode == "2d":
        return np.concatenate([desc_diff, fp_diff], axis=1).astype(np.float32)

    raw_lfer = compute_local_pair_deltas(
        i_idx, j_idx, feats.atom_ptr, feats.atom_env_hash, lfer_atoms, env_counters=env_counters
    )
    d = lfer_atoms.shape[1]  # 7
    sum_mean = raw_lfer[:, : 2 * d].copy()  # 14 dims
    cnt_diff = raw_lfer[:, 3 * d : 3 * d + 1]  # 1 dim

    if mode == "lfer_espan":
        e_span = feats.feat_3d[:, 1]
        e_pair = np.maximum(e_span[i_idx], e_span[j_idx]).astype(np.float32)
        c_conf = np.exp(-np.clip(e_pair, 0.0, 100.0) / e0_kcal)[:, None]
        # Attenuate ONLY 3D conformational geometry (cols 4, 5 in sum_diff and 11, 12 in mean_diff); keep chiral_flag (6, 13) untouched
        geom_cols = [4, 5, 11, 12]
        sum_mean[:, geom_cols] = sum_mean[:, geom_cols] * c_conf

    return np.concatenate([desc_diff, fp_diff, sum_mean, cnt_diff], axis=1).astype(np.float32)


def _predict_delta_backbone(
    model: AntisymmetricPairwiseDeltaNet,
    feats: TargetFeatures,
    bundle_2d: Any,
    lfer_atoms: np.ndarray,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    y: np.ndarray,
    y_std: float,
    diff_scale: np.ndarray,
    env_counters: list[Any],
    mode: str,
    device: torch.device,
    k_eval: int = 10,
    tau: float = 10.0,
    zero_cols: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[query_idx], feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(sim_mat, anchor_pool_idx, k=k_eval)
    s_max = anchor_sims[:, 0].copy()

    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    s_flat = anchor_sims.reshape(-1)

    raw_diff = _build_pair_diff(feats, bundle_2d.x_cont_scaled, lfer_atoms, q_flat, a_flat, env_counters, mode=mode)
    scaled_diff = np.clip(raw_diff / diff_scale, -10.0, 10.0).astype(np.float32)
    if zero_cols:
        scaled_diff[:, zero_cols] = 0.0

    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for start in range(0, q_flat.size, 2048):
            sl = slice(start, start + 2048)
            out = model(
                torch.tensor(bundle_2d.x_mol[q_flat[sl]], dtype=torch.float32, device=device),
                torch.tensor(bundle_2d.x_mol[a_flat[sl]], dtype=torch.float32, device=device),
                torch.tensor(scaled_diff[sl], dtype=torch.float32, device=device),
                torch.tensor(s_flat[sl], dtype=torch.float32, device=device),
            ).cpu().numpy()
            preds_scaled.append(out)

    delta_pred = (np.concatenate(preds_scaled, axis=0) * y_std).reshape(n_q, k_eff)
    y_anchors = y[anchor_g_idx]
    logits = tau * anchor_sims
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    y_hat = np.sum(weights * (y_anchors + delta_pred), axis=1).astype(np.float64)
    return y_hat, s_max.astype(np.float64)


def _train_delta_backbone(
    feats: TargetFeatures,
    lfer_atoms: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    mode: str,
    k_train: int = 15,
    k_eval: int = 10,
    tau: float = 10.0,
    max_epochs: int = 80,
    batch_size: int = 512,
    lr: float = 8e-4,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = get_device()

    inner_tr_rel, inner_va_rel = make_inner_split(y[train_idx], seed=seed, val_frac=0.15)
    inner_tr_idx = train_idx[inner_tr_rel]
    inner_va_idx = train_idx[inner_va_rel]

    bundle_2d = build_tier_matrices(feats, tier="2d", fit_idx=inner_tr_idx)
    env_counters = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)

    sim_tr = compute_tanimoto_matrix(feats.ecfp_chiral[inner_tr_idx], feats.ecfp_chiral[inner_tr_idx])
    nbr_g, nbr_sims = _select_topk_anchors(
        sim_tr, inner_tr_idx, k=k_train, exclude_self_global_idx=inner_tr_idx
    )
    pair_i = np.repeat(inner_tr_idx, k_train)
    pair_j = nbr_g.reshape(-1)
    pair_s = nbr_sims.reshape(-1)

    y_std = float(max(np.std(y[inner_tr_idx]), 1e-6))
    pair_dy = ((y[pair_i] - y[pair_j]) / y_std).astype(np.float32)

    raw_diff_tr = _build_pair_diff(
        feats, bundle_2d.x_cont_scaled, lfer_atoms, pair_i, pair_j, env_counters, mode=mode
    )
    diff_scale = np.std(raw_diff_tr, axis=0)
    diff_scale = np.where(diff_scale < 1e-6, 1.0, diff_scale).astype(np.float32)
    scaled_diff_tr = np.clip(raw_diff_tr / diff_scale, -10.0, 10.0).astype(np.float32)

    model = AntisymmetricPairwiseDeltaNet(
        mol_dim=bundle_2d.x_mol.shape[1],
        pair_diff_dim=scaled_diff_tr.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    x_mol_all_t = torch.tensor(bundle_2d.x_mol, dtype=torch.float32, device=device)
    pair_i_t = torch.tensor(pair_i, dtype=torch.long, device=device)
    pair_j_t = torch.tensor(pair_j, dtype=torch.long, device=device)
    diff_tr_t = torch.tensor(scaled_diff_tr, dtype=torch.float32, device=device)
    sim_tr_t = torch.tensor(pair_s, dtype=torch.float32, device=device)
    dy_tr_t = torch.tensor(pair_dy, dtype=torch.float32, device=device)

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    patience = 12
    n_pairs = pair_i.size

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n_pairs, device=device)
        for start in range(0, n_pairs, batch_size):
            b = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            pred_ij = model(x_mol_all_t[pair_i_t[b]], x_mol_all_t[pair_j_t[b]], diff_tr_t[b], sim_tr_t[b])
            loss = loss_fn(pred_ij, dy_tr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        y_va_pred, _ = _predict_delta_backbone(
            model, feats, bundle_2d, lfer_atoms, inner_va_idx, inner_tr_idx, y, y_std,
            diff_scale, env_counters, mode, device, k_eval=k_eval, tau=tau
        )
        va_rmse = float(np.sqrt(np.mean((y[inner_va_idx] - y_va_pred) ** 2)))
        if va_rmse < best_val_rmse - 1e-5:
            best_val_rmse = va_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    y_pred_val, s_max_val = _predict_delta_backbone(
        model, feats, bundle_2d, lfer_atoms, inner_va_idx, inner_tr_idx, y, y_std,
        diff_scale, env_counters, mode, device, k_eval=k_eval, tau=tau
    )
    y_pred_test, s_max_test = _predict_delta_backbone(
        model, feats, bundle_2d, lfer_atoms, test_idx, train_idx, y, y_std,
        diff_scale, env_counters, mode, device, k_eval=k_eval, tau=tau
    )
    return {
        "model": model,
        "bundle_2d": bundle_2d,
        "y_std": y_std,
        "diff_scale": diff_scale,
        "env_counters": env_counters,
        "inner_tr_idx": inner_tr_idx,
        "inner_va_idx": inner_va_idx,
        "y_pred_val": y_pred_val,
        "s_max_val": s_max_val,
        "y_pred_test": y_pred_test,
        "s_max_test": s_max_test,
    }


def _fit_rbf_svr(fp_all: np.ndarray, y_all: np.ndarray, fit_idx: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    svr = SVR(kernel="rbf", C=3.0, epsilon=0.1)
    svr.fit(fp_all[fit_idx], y_all[fit_idx])
    return svr.predict(fp_all[query_idx]).astype(np.float64)


def _compute_local_aimnet_dq(
    feats: TargetFeatures,
    lfer_atoms: np.ndarray,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    env_counters: list[Any],
    exclude_self: bool,
    k_top: int = 5,
    tau: float = 10.0,
) -> np.ndarray:
    """Similarity-weighted |Delta q_AIMNet2| at the modified substituent relative to top-5 anchors."""
    sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[query_idx], feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(
        sim_mat, anchor_pool_idx, k=k_top, exclude_self_global_idx=query_idx if exclude_self else None
    )
    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    # Col 0 of lfer_atoms is aimnet_eff_q
    raw_dq = compute_local_pair_deltas(
        q_flat, a_flat, feats.atom_ptr, feats.atom_env_hash, lfer_atoms[:, 0:1], env_counters=env_counters
    )[:, 0]
    abs_dq = np.abs(raw_dq).reshape(n_q, k_eff)

    logits = tau * anchor_sims
    logits -= np.max(logits, axis=1, keepdims=True)
    w = np.exp(logits)
    w /= np.sum(w, axis=1, keepdims=True)
    return np.sum(w * abs_dq, axis=1).astype(np.float64)


def _fit_2param_gate(
    y_val_true: np.ndarray,
    y_val_delta: np.ndarray,
    y_val_svm: np.ndarray,
    s_max_val: np.ndarray,
    y_test_delta: np.ndarray,
    y_test_svm: np.ndarray,
    s_max_test: np.ndarray,
) -> dict[str, Any]:
    def sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def obj(p: np.ndarray) -> float:
        ws, b = float(p[0]), float(p[1])
        a = sig(ws * (s_max_val - b))
        yb = a * y_val_delta + (1.0 - a) * y_val_svm
        return float(np.mean((y_val_true - yb) ** 2)) + 1e-4 * (ws ** 2)

    res = optimize.minimize(obj, x0=np.array([6.0, 0.70]), bounds=[(0.0, 25.0), (0.20, 0.95)], method="L-BFGS-B")
    ws, b = float(res.x[0]), float(res.x[1])
    alpha_test = sig(ws * (s_max_test - b))
    y_pred_test = (alpha_test * y_test_delta + (1.0 - alpha_test) * y_test_svm).astype(np.float64)
    return {
        "y_pred_test": y_pred_test,
        "alpha_test": alpha_test,
        "ws": ws,
        "b": b,
        "loss_val": float(res.fun),
    }


def _fit_phys_gate_g2(
    feats: TargetFeatures,
    lfer_atoms: np.ndarray,
    env_counters: list[Any],
    inner_tr_idx: np.ndarray,
    inner_va_idx: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_val_true: np.ndarray,
    y_val_delta: np.ndarray,
    y_val_svm: np.ndarray,
    s_max_val: np.ndarray,
    y_test_delta: np.ndarray,
    y_test_svm: np.ndarray,
    s_max_test: np.ndarray,
    g1_ws: float,
    g1_b: float,
) -> dict[str, Any]:
    """
    Sign-constrained Monotonic Physical-Chemistry Gate:
      alpha = sigmoid(w_s * (s_max - b) - w_e * e_tilde + w_q * dq_tilde)
      with w_s >= 0, w_e >= 0, w_q >= 0, all standardizers fit STRICTLY on inner_tr_idx.
    """
    e_span = feats.feat_3d[:, 1].astype(np.float64)
    e_mean = float(np.mean(e_span[inner_tr_idx]))
    e_std = float(max(np.std(e_span[inner_tr_idx]), 1e-6))
    e_val_tilde = np.clip((e_span[inner_va_idx] - e_mean) / e_std, -5.0, 5.0)
    e_test_tilde = np.clip((e_span[test_idx] - e_mean) / e_std, -5.0, 5.0)

    dq_tr = _compute_local_aimnet_dq(feats, lfer_atoms, inner_tr_idx, inner_tr_idx, env_counters, exclude_self=True)
    dq_mean = float(np.mean(dq_tr))
    dq_std = float(max(np.std(dq_tr), 1e-6))

    dq_val = _compute_local_aimnet_dq(feats, lfer_atoms, inner_va_idx, inner_tr_idx, env_counters, exclude_self=False)
    dq_test = _compute_local_aimnet_dq(feats, lfer_atoms, test_idx, train_idx, env_counters, exclude_self=False)
    dq_val_tilde = np.clip((dq_val - dq_mean) / dq_std, -5.0, 5.0)
    dq_test_tilde = np.clip((dq_test - dq_mean) / dq_std, -5.0, 5.0)

    def sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def obj(p: np.ndarray) -> float:
        ws, b, we, wq = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        a = sig(ws * (s_max_val - b) - we * e_val_tilde + wq * dq_val_tilde)
        yb = a * y_val_delta + (1.0 - a) * y_val_svm
        return float(np.mean((y_val_true - yb) ** 2)) + 1e-4 * (ws ** 2) + 1e-3 * (we ** 2 + wq ** 2)

    bounds = [(0.0, 25.0), (0.20, 0.95), (0.0, 5.0), (0.0, 5.0)]
    inits = [
        np.array([g1_ws, g1_b, 0.05, 0.05]),
        np.array([6.0, 0.70, 0.25, 0.25]),
        np.array([4.0, 0.65, 0.50, 0.50]),
    ]
    best_res = None
    for x0 in inits:
        r = optimize.minimize(obj, x0=x0, bounds=bounds, method="L-BFGS-B")
        if best_res is None or r.fun < best_res.fun:
            best_res = r

    ws, b, we, wq = [float(v) for v in best_res.x]
    alpha_test = sig(ws * (s_max_test - b) - we * e_test_tilde + wq * dq_test_tilde)
    y_pred_test = (alpha_test * y_test_delta + (1.0 - alpha_test) * y_test_svm).astype(np.float64)
    return {
        "y_pred_test": y_pred_test,
        "alpha_test": alpha_test,
        "dq_test": dq_test,
        "ws": ws,
        "b": b,
        "we": we,
        "wq": wq,
        "alpha_mean": float(np.mean(alpha_test)),
    }


def _plot_plan_a(summary_df: pd.DataFrame, attr_df: pd.DataFrame) -> Path:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.9))

    models = [
        ("B1c_RBF_SVM", "B1c RBF-SVM", "#999999"),
        ("G0_RBFSVM_Gate_A2", "G0 SVM+2D", "#56B4E9"),
        ("G1_RBFSVM_Gate_L2", "G1 SVM+L2", "#009E73"),
        ("G2_RBFSVM_PhysGate_L2", "G2 +PhysGate", "#D55E00"),
    ]
    x = np.arange(len(MVP_TARGETS))
    w = 0.19
    for i, (m_key, m_lbl, col) in enumerate(models):
        means, stds = [], []
        for tgt in MVP_TARGETS:
            sub = summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == m_key)]
            means.append(float(sub["cliff_rmse_mean"].iloc[0]))
            stds.append(float(sub["cliff_rmse_std"].iloc[0]))
        ax1.bar(x + (i - 1.5) * w, means, yerr=stds, capsize=3, width=w, label=m_lbl, color=col, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(MVP_TARGETS), fontweight="bold")
    ax1.set_ylabel("Activity Cliff RMSE (5-seed mean ± std)")
    ax1.set_title("R02 Control: Kernel Base + 2D (G0) vs Orthogonal PhysChem (G1/G2)", fontweight="bold")
    ax1.set_ylim(0.50, 0.90)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)

    # Right panel: 5-channel LFER attribution heatmap (Delta Cliff RMSE when channel zeroed; >0 means important)
    ch_names = list(LFER_CHANNELS.keys())
    mat = np.zeros((len(ch_names), len(MVP_TARGETS)), dtype=np.float64)
    for r_i, ch in enumerate(ch_names):
        for c_i, tgt in enumerate(MVP_TARGETS):
            sub = attr_df[(attr_df["target"] == tgt) & (attr_df["channel"] == ch)]
            mat[r_i, c_i] = float(sub["delta_cliff_rmse_mean"].iloc[0])
    vmax = max(0.005, float(np.max(np.abs(mat))))
    im = ax2.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_xticks(np.arange(len(MVP_TARGETS)))
    ax2.set_xticklabels(list(MVP_TARGETS), fontweight="bold")
    ax2.set_yticks(np.arange(len(ch_names)))
    ax2.set_yticklabels(ch_names)
    for r_i in range(len(ch_names)):
        for c_i in range(len(MVP_TARGETS)):
            val = mat[r_i, c_i]
            ax2.text(c_i, r_i, f"{val:+.4f}", ha="center", va="center", color="black", fontsize=9)
    ax2.set_title("L2 Channel-Zeroing Attribution (Δ Cliff RMSE; >0 = Useful)", fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = _fig_dir() / "plan_a_r02_and_interpretability.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    t0 = time.time()
    device = get_device()
    datasets = {tgt: load_target_dataset(tgt) for tgt in MVP_TARGETS}
    features = {
        tgt: extract_or_load_target_features(tgt, datasets[tgt].smiles, prefer_aimnet=True, num_confs=5)
        for tgt in MVP_TARGETS
    }

    records: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    attr_records: list[dict[str, Any]] = []
    preds_store: dict[tuple[str, str, int], np.ndarray] = {}
    g2_case_data: dict[str, list[dict[str, Any]]] = {}
    alpha_strat_rows: list[dict[str, Any]] = []

    for tgt in MVP_TARGETS:
        ds = datasets[tgt]
        feats = features[tgt]
        lfer_atoms = _lfer_atom_matrix(feats)
        y_te = ds.y[ds.test_idx]
        c_te = ds.cliff_mol[ds.test_idx]
        cmask = c_te == 1

        # Exact B1c_RBF_SVM on train_idx -> test_idx
        y_test_svm = _fit_rbf_svr(feats.ecfp_chiral, ds.y, ds.train_idx, ds.test_idx)
        svm_m = evaluate_predictions(y_te, y_test_svm, c_te)

        # Track per-seed G2 test predictions and metadata for interpretability cases
        g2_preds_per_seed = []
        g0_preds_per_seed = []
        alpha_per_seed = []
        dq_last = None

        for seed in SEEDS:
            # Train 3 strictly-controlled delta backbones using identical inner split
            a2_out = _train_delta_backbone(feats, lfer_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="2d")
            l1_out = _train_delta_backbone(feats, lfer_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="lfer_raw")
            l2_out = _train_delta_backbone(feats, lfer_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="lfer_espan")

            # Leak-free inner_val RBF-SVR predictions for gate fitting
            y_val_svm = _fit_rbf_svr(feats.ecfp_chiral, ds.y, a2_out["inner_tr_idx"], a2_out["inner_va_idx"])
            y_val_true = ds.y[a2_out["inner_va_idx"]]

            # G0: RBF-SVR + 2-param gate(A2 2D-only) -> STRICT R02 CONTROL
            g0_out = _fit_2param_gate(
                y_val_true, a2_out["y_pred_val"], y_val_svm, a2_out["s_max_val"],
                a2_out["y_pred_test"], y_test_svm, a2_out["s_max_test"]
            )
            # G1: RBF-SVR + 2-param gate(L2 LFER+eSpan)
            g1_out = _fit_2param_gate(
                y_val_true, l2_out["y_pred_val"], y_val_svm, l2_out["s_max_val"],
                l2_out["y_pred_test"], y_test_svm, l2_out["s_max_test"]
            )
            # G2: RBF-SVR + Sign-constrained physical gate(s_max, e_span, |Delta q_AIMNet2|)
            g2_out = _fit_phys_gate_g2(
                feats, lfer_atoms, l2_out["env_counters"],
                l2_out["inner_tr_idx"], l2_out["inner_va_idx"],
                ds.train_idx, ds.test_idx,
                y_val_true, l2_out["y_pred_val"], y_val_svm, l2_out["s_max_val"],
                l2_out["y_pred_test"], y_test_svm, l2_out["s_max_test"],
                g1_ws=g1_out["ws"], g1_b=g1_out["b"],
            )

            gate_rows.append(
                {
                    "target": tgt,
                    "seed": seed,
                    "g0_ws": g0_out["ws"],
                    "g0_b": g0_out["b"],
                    "g1_ws": g1_out["ws"],
                    "g1_b": g1_out["b"],
                    "g2_ws": g2_out["ws"],
                    "g2_b": g2_out["b"],
                    "g2_we": g2_out["we"],
                    "g2_wq": g2_out["wq"],
                    "g2_alpha_mean": g2_out["alpha_mean"],
                }
            )

            model_preds = {
                "B1c_RBF_SVM": y_test_svm,
                "A2_2D_Delta": a2_out["y_pred_test"],
                "L1_LFER_Delta": l1_out["y_pred_test"],
                "L2_eSpan_LFER_Delta": l2_out["y_pred_test"],
                "G0_RBFSVM_Gate_A2": g0_out["y_pred_test"],
                "G1_RBFSVM_Gate_L2": g1_out["y_pred_test"],
                "G2_RBFSVM_PhysGate_L2": g2_out["y_pred_test"],
            }
            for m_name, yp in model_preds.items():
                met = evaluate_predictions(y_te, yp, c_te)
                records.append({"target": tgt, "model": m_name, "seed": seed, **met})
                preds_store[(tgt, m_name, seed)] = yp

            g2_preds_per_seed.append(g2_out["y_pred_test"])
            g0_preds_per_seed.append(g0_out["y_pred_test"])
            alpha_per_seed.append(g2_out["alpha_test"])
            dq_last = g2_out["dq_test"]

            # 5-Channel LFER Zeroing Attribution on L2 test cliff molecules
            base_cliff = evaluate_predictions(y_te, l2_out["y_pred_test"], c_te)["cliff_rmse"]
            for ch_name, cols in LFER_CHANNELS.items():
                yp_zero, _ = _predict_delta_backbone(
                    l2_out["model"], feats, l2_out["bundle_2d"], lfer_atoms, ds.test_idx, ds.train_idx,
                    ds.y, l2_out["y_std"], l2_out["diff_scale"], l2_out["env_counters"],
                    mode="lfer_espan", device=device, zero_cols=cols
                )
                zero_cliff = compute_rmse(y_te[cmask], yp_zero[cmask])
                attr_records.append(
                    {
                        "target": tgt,
                        "seed": seed,
                        "channel": ch_name,
                        "delta_cliff_rmse": float(zero_cliff - base_cliff),
                    }
                )

            print(
                f"[{tgt} | seed={seed}] "
                f"SVM={svm_m['cliff_rmse']:.4f} | A2={evaluate_predictions(y_te, a2_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"L1={evaluate_predictions(y_te, l1_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"L2={evaluate_predictions(y_te, l2_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"G0={evaluate_predictions(y_te, g0_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"G1={evaluate_predictions(y_te, g1_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"G2={evaluate_predictions(y_te, g2_out['y_pred_test'], c_te)['cliff_rmse']:.4f} "
                f"(ws={g2_out['ws']:.2f}, we={g2_out['we']:.2f}, wq={g2_out['wq']:.2f})",
                flush=True,
            )

        # High-alpha vs Low-alpha stratification + concrete chemical case extraction
        g2_mean = np.mean(np.stack(g2_preds_per_seed, axis=0), axis=0)
        g0_mean = np.mean(np.stack(g0_preds_per_seed, axis=0), axis=0)
        alpha_mean = np.mean(np.stack(alpha_per_seed, axis=0), axis=0)

        sim_te_tr = compute_tanimoto_matrix(feats.ecfp_chiral[ds.test_idx], feats.ecfp_chiral[ds.train_idx])
        s_max_te = np.max(sim_te_tr, axis=1)
        nearest_tr = ds.train_idx[np.argmax(sim_te_tr, axis=1)]
        e_span_te = feats.feat_3d[ds.test_idx, 1]
        rotb_te = feats.desc2d[ds.test_idx, 5]

        # Cliff molecules routed top-25% alpha vs bottom-25% alpha
        a_cliff = alpha_mean[cmask]
        q25, q75 = float(np.percentile(a_cliff, 25)), float(np.percentile(a_cliff, 75))
        for grp, sub_m in [
            ("Bottom-25% alpha (routed to SVM)", cmask & (alpha_mean <= q25)),
            ("Top-25% alpha (routed to L2 PhysChem)", cmask & (alpha_mean >= q75)),
        ]:
            if np.sum(sub_m) == 0:
                continue
            alpha_strat_rows.append(
                {
                    "target": tgt,
                    "group": grp,
                    "n_cliff": int(np.sum(sub_m)),
                    "alpha_med": round(float(np.median(alpha_mean[sub_m])), 3),
                    "smax_med": round(float(np.median(s_max_te[sub_m])), 3),
                    "espan_med": round(float(np.median(e_span_te[sub_m])), 2),
                    "rotb_med": round(float(np.median(rotb_te[sub_m])), 1),
                    "dq_med": round(float(np.median(dq_last[sub_m])), 4),
                    "rmse_svm": round(compute_rmse(y_te[sub_m], y_test_svm[sub_m]), 4),
                    "rmse_g0": round(compute_rmse(y_te[sub_m], g0_mean[sub_m]), 4),
                    "rmse_g2": round(compute_rmse(y_te[sub_m], g2_mean[sub_m]), 4),
                }
            )

        # Extract 2 win cases (G2 beats BOTH SVM & G0 by largest margin) + 1 honest failure case (G2 worst vs SVM)
        err_svm = np.abs(y_te - y_test_svm)
        err_g0 = np.abs(y_te - g0_mean)
        err_g2 = np.abs(y_te - g2_mean)
        win_margin = np.minimum(err_svm - err_g2, err_g0 - err_g2)
        cliff_positions = np.where(cmask)[0]

        win_sorted = cliff_positions[np.argsort(-win_margin[cliff_positions])]
        fail_sorted = cliff_positions[np.argsort(err_svm[cliff_positions] - err_g2[cliff_positions])]

        cases_list = []
        for kind, pos_list in [("WIN", win_sorted[:2]), ("FAIL", fail_sorted[:1])]:
            for pos in pos_list:
                g_idx = int(ds.test_idx[pos])
                anc_g = int(nearest_tr[pos])
                cases_list.append(
                    {
                        "type": kind,
                        "test_idx": g_idx,
                        "test_smiles": ds.smiles[g_idx],
                        "anchor_smiles": ds.smiles[anc_g],
                        "y_true": round(float(y_te[pos]), 3),
                        "anchor_y": round(float(ds.y[anc_g]), 3),
                        "s_max": round(float(s_max_te[pos]), 3),
                        "e_span": round(float(e_span_te[pos]), 2),
                        "rotb": int(rotb_te[pos]),
                        "dq_aimnet": round(float(dq_last[pos]), 4),
                        "alpha": round(float(alpha_mean[pos]), 3),
                        "pred_svm": round(float(y_test_svm[pos]), 3),
                        "pred_g0": round(float(g0_mean[pos]), 3),
                        "pred_g2": round(float(g2_mean[pos]), 3),
                    }
                )
        g2_case_data[tgt] = cases_list

    # Summarize results
    rec_df = pd.DataFrame(records)
    model_order = [
        "B1c_RBF_SVM",
        "A2_2D_Delta",
        "L1_LFER_Delta",
        "L2_eSpan_LFER_Delta",
        "G0_RBFSVM_Gate_A2",
        "G1_RBFSVM_Gate_L2",
        "G2_RBFSVM_PhysGate_L2",
    ]
    summary_rows = []
    for tgt in MVP_TARGETS:
        for m in model_order:
            sub = rec_df[(rec_df["target"] == tgt) & (rec_df["model"] == m)]
            summary_rows.append(
                {
                    "target": tgt,
                    "model": m,
                    "rmse_mean": round(float(sub["rmse"].mean()), 4),
                    "rmse_std": round(float(sub["rmse"].std(ddof=1)), 4),
                    "cliff_rmse_mean": round(float(sub["cliff_rmse"].mean()), 4),
                    "cliff_rmse_std": round(float(sub["cliff_rmse"].std(ddof=1)), 4),
                    "non_cliff_rmse_mean": round(float(sub["non_cliff_rmse"].mean()), 4),
                    "spearman_mean": round(float(sub["spearman"].mean()), 4),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(_refine_dir() / "plan_a_summary.csv", index=False)

    # Paired Wilcoxon tests
    comparisons = [
        ("L1_LFER_Delta", "A2_2D_Delta"),
        ("L2_eSpan_LFER_Delta", "L1_LFER_Delta"),
        ("L2_eSpan_LFER_Delta", "A2_2D_Delta"),
        ("G0_RBFSVM_Gate_A2", "B1c_RBF_SVM"),
        ("G1_RBFSVM_Gate_L2", "G0_RBFSVM_Gate_A2"),
        ("G2_RBFSVM_PhysGate_L2", "G0_RBFSVM_Gate_A2"),
        ("G2_RBFSVM_PhysGate_L2", "B1c_RBF_SVM"),
        ("G2_RBFSVM_PhysGate_L2", "G1_RBFSVM_Gate_L2"),
    ]
    wil_rows = []
    for tgt in MVP_TARGETS:
        ds = datasets[tgt]
        y_te = ds.y[ds.test_idx]
        cmask = ds.cliff_mol[ds.test_idx] == 1
        for cand, ref in comparisons:
            ec = np.concatenate([np.abs(y_te[cmask] - preds_store[(tgt, cand, s)][cmask]) for s in SEEDS])
            er = np.concatenate([np.abs(y_te[cmask] - preds_store[(tgt, ref, s)][cmask]) for s in SEEDS])
            pval = paired_wilcoxon(ec, er)
            cv = float(summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == cand)]["cliff_rmse_mean"].iloc[0])
            rv = float(summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == ref)]["cliff_rmse_mean"].iloc[0])
            wil_rows.append(
                {
                    "target": tgt,
                    "candidate": cand,
                    "reference": ref,
                    "delta_cliff_rmse": round(cv - rv, 4),
                    "wilcoxon_p": float(f"{pval:.3e}"),
                }
            )

    # Gate stability & LFER attribution summary
    gate_df = pd.DataFrame(gate_rows)
    attr_df_raw = pd.DataFrame(attr_records)
    attr_sum = (
        attr_df_raw.groupby(["target", "channel"], as_index=False)
        .agg(
            delta_cliff_rmse_mean=("delta_cliff_rmse", "mean"),
            delta_cliff_rmse_std=("delta_cliff_rmse", "std"),
        )
    )
    fig_path = _plot_plan_a(summary_df, attr_sum)

    # Write PLAN_A_EVAL.md
    md: list[str] = [
        "# Scheme A Evaluation: R02 Fair Control & Chemical-Property Interpretability",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Elapsed**: {time.time() - t0:.1f}s",
        "",
        "## 1. Main & R02 Fair Control Table (3 Targets x 5 Seeds Mean ± Std)",
        "| Target | Model | RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in summary_df.iterrows():
        md.append(
            f"| {r['target']} | `{r['model']}` | "
            f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f} | "
            f"**{r['cliff_rmse_mean']:.4f} ± {r['cliff_rmse_std']:.4f}** | "
            f"{r['non_cliff_rmse_mean']:.4f} | {r['spearman_mean']:.4f} |"
        )

    md.extend(
        [
            "",
            "## 2. Paired Wilcoxon Significance on Cliff Subset (5 Seeds Pooled)",
            "| Target | Candidate | Reference | Δ Cliff RMSE | Two-sided p-val |",
            "|---|---|---|---:|---:|",
        ]
    )
    for w in wil_rows:
        md.append(
            f"| {w['target']} | `{w['candidate']}` | `{w['reference']}` | "
            f"{w['delta_cliff_rmse']:+.4f} | {w['wilcoxon_p']:.3e} |"
        )

    md.extend(
        [
            "",
            "## 3. Learned Physical Gate Parameters (G2 Monotonic Gate: w_s>=0, w_e>=0, w_q>=0)",
            "| Target | w_s (s_max) | b (threshold) | w_e (-e_span) | w_q (+|Δq_AIMNet2|) | Test alpha mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tgt in MVP_TARGETS:
        sub = gate_df[gate_df["target"] == tgt]
        md.append(
            f"| {tgt} | {sub['g2_ws'].mean():.2f} ± {sub['g2_ws'].std(ddof=1):.2f} | "
            f"{sub['g2_b'].mean():.3f} ± {sub['g2_b'].std(ddof=1):.3f} | "
            f"{sub['g2_we'].mean():.3f} ± {sub['g2_we'].std(ddof=1):.3f} | "
            f"{sub['g2_wq'].mean():.3f} ± {sub['g2_wq'].std(ddof=1):.3f} | "
            f"{sub['g2_alpha_mean'].mean():.3f} |"
        )

    md.extend(
        [
            "",
            "## 4. 5-Channel Orthogonal LFER Attribution (Channel-Zeroing Δ Cliff RMSE on L2)",
            "| Channel | CHEMBL204_Ki | CHEMBL2147_Ki | CHEMBL237_Ki |",
            "|---|---:|---:|---:|",
        ]
    )
    for ch in LFER_CHANNELS:
        vals = []
        for tgt in MVP_TARGETS:
            s = attr_sum[(attr_sum["target"] == tgt) & (attr_sum["channel"] == ch)].iloc[0]
            vals.append(f"{s['delta_cliff_rmse_mean']:+.4f} ± {s['delta_cliff_rmse_std']:.4f}")
        md.append(f"| `{ch}` | " + " | ".join(vals) + " |")

    md.extend(
        [
            "",
            "## 5. High-alpha vs Low-alpha Chemical Stratification on Cliff Molecules",
            "| Target | Group | n_cliff | alpha_med | s_max_med | e_span_med | RotBonds_med | |Δq|_med | RMSE_SVM | RMSE_G0 | RMSE_G2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in alpha_strat_rows:
        md.append(
            f"| {row['target']} | {row['group']} | {row['n_cliff']} | {row['alpha_med']} | {row['smax_med']} | "
            f"{row['espan_med']} | {row['rotb_med']} | {row['dq_med']} | {row['rmse_svm']:.4f} | {row['rmse_g0']:.4f} | {row['rmse_g2']:.4f} |"
        )

    md.extend(["", "## 6. Concrete Cliff Pair Cases (2 Wins + 1 Honest Failure per Target)"])
    for tgt in MVP_TARGETS:
        md.append(f"### {tgt}")
        for cs in g2_case_data[tgt]:
            md.append(
                f"- **[{cs['type']}]** `test_idx={cs['test_idx']}`: `y={cs['y_true']}`, `anchor_y={cs['anchor_y']}`, "
                f"`s_max={cs['s_max']}`, `e_span={cs['e_span']}`, `RotBonds={cs['rotb']}`, `|Δq|={cs['dq_aimnet']}`, `alpha={cs['alpha']}` "
                f"-> `SVM={cs['pred_svm']}`, `G0={cs['pred_g0']}`, `G2={cs['pred_g2']}`\n"
                f"  - Query: `{cs['test_smiles']}`\n"
                f"  - Anchor: `{cs['anchor_smiles']}`"
            )

    md.extend(["", f"Figure: `{fig_path}`"])
    (_refine_dir() / "PLAN_A_EVAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Scheme A evaluation finished cleanly in {time.time() - t0:.1f}s -> PLAN_A_EVAL.md & {fig_path}")


if __name__ == "__main__":
    main()
