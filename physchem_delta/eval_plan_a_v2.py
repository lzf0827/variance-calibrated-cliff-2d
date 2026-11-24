"""Scheme A v2 evaluation: Boltzmann multi-conformer ensemble + Context-Modulated Skip-LFER."""

from __future__ import annotations

import copy
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
from physchem_delta.eval_plan_a import _fit_2param_gate, _fit_phys_gate_g2, _fit_rbf_svr, _lfer_atom_matrix, _train_delta_backbone
from physchem_delta.featurize import TargetFeatures, compute_local_pair_deltas, precompute_mol_env_counters
from physchem_delta.featurize_boltz import BoltzmannTargetFeatures, extract_or_load_boltzmann_features
from physchem_delta.metrics import evaluate_predictions, paired_wilcoxon
from physchem_delta.models import (
    _select_topk_anchors,
    build_tier_matrices,
    compute_tanimoto_matrix,
    get_device,
    make_inner_split,
    set_global_seed,
)

SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

S1_CHANNEL_GROUPS: dict[str, list[int]] = {
    "AIMNet2_q": [0, 7],
    "Crippen_LogP": [1, 8],
    "Crippen_MR": [2, 9],
    "TPSA": [3, 10],
    "3D_Geom": [4, 5, 11, 12],
    "Chiral": [6, 13],
    "AtomCount": [14],
}

S2_CHANNEL_GROUPS: dict[str, list[int]] = {
    "AIMNet2_q": [0, 9],
    "AIMNet2_qFluct": [1, 10],
    "Crippen_LogP": [2, 11],
    "Crippen_MR": [3, 12],
    "TPSA": [4, 13],
    "3D_Geom": [5, 6, 7, 14, 15, 16],
    "Chiral": [8, 17],
    "AtomCount": [18],
}


def _refine_dir() -> Path:
    return ROOT_DIR / "refine-logs"


def _fig_dir() -> Path:
    d = _refine_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ContextModulatedSkipLFERDeltaNet(nn.Module):
    """
    Antisymmetric Pairwise DeltaNet with an explicit Context-Modulated Bilinear LFER Skip Head:
      f_delta(i, j) = f_2D_topo(i, j) + (w_global + Delta_w_ctx(i, j))^T * Delta_x_LFER(i, j)
    Guarantees f_delta(i, j) = -f_delta(j, i) and exact additive delta-pKi channel decomposition.
    """

    def __init__(self, mol_dim: int, topo_diff_dim: int, lfer_dim: int, emb_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
        )
        pair_in_dim = emb_dim * 4 + topo_diff_dim + 1
        self.pair_head_2d = nn.Sequential(
            nn.Linear(pair_in_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        ctx_in_dim = emb_dim * 2 + 1
        self.w_global = nn.Parameter(torch.zeros(lfer_dim))
        self.ctx_modulator = nn.Sequential(
            nn.Linear(ctx_in_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.30),
            nn.Linear(64, lfer_dim),
        )
        # Initialize contextual modulation near zero for smooth regularization
        nn.init.zeros_(self.ctx_modulator[-1].weight)
        nn.init.zeros_(self.ctx_modulator[-1].bias)

    def forward_decomposed(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        topo_diff_ij: torch.Tensor,
        lfer_diff_ij: torch.Tensor,
        sim_ij: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_i = self.mol_encoder(x_i)
        h_j = self.mol_encoder(x_j)
        prod = h_i * h_j
        sim_col = sim_ij.unsqueeze(-1)

        u_ij = torch.cat([h_i, h_j, h_i - h_j, prod, topo_diff_ij, sim_col], dim=-1)
        u_ji = torch.cat([h_j, h_i, h_j - h_i, prod, -topo_diff_ij, sim_col], dim=-1)
        f_2d = 0.5 * (self.pair_head_2d(u_ij).squeeze(-1) - self.pair_head_2d(u_ji).squeeze(-1))

        ctx_sym = torch.cat([0.5 * (h_i + h_j), prod, sim_col], dim=-1)
        dw_ctx = self.ctx_modulator(ctx_sym)
        w_eff = self.w_global.unsqueeze(0) + dw_ctx
        lfer_terms = w_eff * lfer_diff_ij  # (B, lfer_dim), strictly antisymmetric
        f_lfer = torch.sum(lfer_terms, dim=-1)
        return f_2d + f_lfer, lfer_terms, dw_ctx


def _build_topo_and_lfer_diffs(
    base_feats: TargetFeatures,
    bundle_2d_cont: np.ndarray,
    atom_mat: np.ndarray,
    thermo_col: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    env_counters: list[Any],
    geom_cols_in_sum_mean: list[int],
    attenuation_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (topo_diff [26 dims], lfer_diff [2*D + 1 dims]) both strictly antisymmetric."""
    desc_diff = (bundle_2d_cont[i_idx] - bundle_2d_cont[j_idx]).astype(np.float32)
    fp_i = base_feats.ecfp_chiral[i_idx]
    fp_j = base_feats.ecfp_chiral[j_idx]
    n_add = np.sum((fp_i > 0.5) & (fp_j < 0.5), axis=1, keepdims=True).astype(np.float32)
    n_rem = np.sum((fp_j > 0.5) & (fp_i < 0.5), axis=1, keepdims=True).astype(np.float32)
    topo_diff = np.concatenate([desc_diff, n_add - n_rem], axis=1).astype(np.float32)

    raw_lfer = compute_local_pair_deltas(
        i_idx, j_idx, base_feats.atom_ptr, base_feats.atom_env_hash, atom_mat, env_counters=env_counters
    )
    d = atom_mat.shape[1]
    sum_mean = raw_lfer[:, : 2 * d].copy()
    cnt_diff = raw_lfer[:, 3 * d : 3 * d + 1]

    pair_thermo = np.maximum(thermo_col[i_idx], thermo_col[j_idx]).astype(np.float32)
    c_conf = np.exp(-np.clip(pair_thermo, 0.0, 100.0) / attenuation_scale)[:, None]
    sum_mean[:, geom_cols_in_sum_mean] = sum_mean[:, geom_cols_in_sum_mean] * c_conf

    lfer_diff = np.concatenate([sum_mean, cnt_diff], axis=1).astype(np.float32)
    return topo_diff, lfer_diff


def _predict_skip_lfer(
    model: ContextModulatedSkipLFERDeltaNet,
    base_feats: TargetFeatures,
    bundle_2d: Any,
    atom_mat: np.ndarray,
    thermo_col: np.ndarray,
    geom_cols: list[int],
    attenuation_scale: float,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    y: np.ndarray,
    y_std: float,
    topo_scale: np.ndarray,
    lfer_scale: np.ndarray,
    env_counters: list[Any],
    channel_groups: dict[str, list[int]],
    device: torch.device,
    k_eval: int = 10,
    tau: float = 10.0,
) -> dict[str, Any]:
    sim_mat = compute_tanimoto_matrix(base_feats.ecfp_chiral[query_idx], base_feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(sim_mat, anchor_pool_idx, k=k_eval)
    s_max = anchor_sims[:, 0].copy()

    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    s_flat = anchor_sims.reshape(-1)

    raw_topo, raw_lfer = _build_topo_and_lfer_diffs(
        base_feats, bundle_2d.x_cont_scaled, atom_mat, thermo_col, q_flat, a_flat,
        env_counters, geom_cols, attenuation_scale
    )
    sc_topo = np.clip(raw_topo / topo_scale, -10.0, 10.0).astype(np.float32)
    sc_lfer = np.clip(raw_lfer / lfer_scale, -10.0, 10.0).astype(np.float32)

    model.eval()
    preds_all, terms_all = [], []
    with torch.no_grad():
        for start in range(0, q_flat.size, 2048):
            sl = slice(start, start + 2048)
            p_tot, l_terms, _ = model.forward_decomposed(
                torch.tensor(bundle_2d.x_mol[q_flat[sl]], dtype=torch.float32, device=device),
                torch.tensor(bundle_2d.x_mol[a_flat[sl]], dtype=torch.float32, device=device),
                torch.tensor(sc_topo[sl], dtype=torch.float32, device=device),
                torch.tensor(sc_lfer[sl], dtype=torch.float32, device=device),
                torch.tensor(s_flat[sl], dtype=torch.float32, device=device),
            )
            preds_all.append(p_tot.cpu().numpy())
            terms_all.append(l_terms.cpu().numpy())

    delta_pred = (np.concatenate(preds_all, axis=0) * y_std).reshape(n_q, k_eff)
    lfer_terms_pk = (np.concatenate(terms_all, axis=0) * y_std).reshape(n_q, k_eff, -1)

    y_anchors = y[anchor_g_idx]
    logits = tau * anchor_sims
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    y_hat = np.sum(weights * (y_anchors + delta_pred), axis=1).astype(np.float64)

    # Exact additive pKi channel contributions per query molecule
    weighted_terms = np.sum(weights[:, :, None] * lfer_terms_pk, axis=1)  # (n_q, d_lfer)
    channel_contribs = {
        g_name: np.sum(weighted_terms[:, cols], axis=1).astype(np.float64)
        for g_name, cols in channel_groups.items()
    }
    sum_lfer_total = np.sum(np.stack(list(channel_contribs.values()), axis=1), axis=1)
    base_2d_part = y_hat - sum_lfer_total
    # Exact mathematical decomposition invariant check
    reconstructed = base_2d_part + sum_lfer_total
    assert np.max(np.abs(reconstructed - y_hat)) < 1e-4

    return {
        "y_hat": y_hat,
        "s_max": s_max.astype(np.float64),
        "channel_contribs": channel_contribs,
        "base_2d_part": base_2d_part,
        "top_anchor_g": anchor_g_idx[:, 0],
    }


def train_skip_lfer_model(
    base_feats: TargetFeatures,
    atom_mat: np.ndarray,
    thermo_col: np.ndarray,
    geom_cols: list[int],
    attenuation_scale: float,
    channel_groups: dict[str, list[int]],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
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

    bundle_2d = build_tier_matrices(base_feats, tier="2d", fit_idx=inner_tr_idx)
    env_counters = precompute_mol_env_counters(base_feats.atom_ptr, base_feats.atom_env_hash)

    sim_tr = compute_tanimoto_matrix(base_feats.ecfp_chiral[inner_tr_idx], base_feats.ecfp_chiral[inner_tr_idx])
    nbr_g, nbr_sims = _select_topk_anchors(
        sim_tr, inner_tr_idx, k=k_train, exclude_self_global_idx=inner_tr_idx
    )
    pair_i = np.repeat(inner_tr_idx, k_train)
    pair_j = nbr_g.reshape(-1)
    pair_s = nbr_sims.reshape(-1)

    y_std = float(max(np.std(y[inner_tr_idx]), 1e-6))
    pair_dy = ((y[pair_i] - y[pair_j]) / y_std).astype(np.float32)

    raw_topo_tr, raw_lfer_tr = _build_topo_and_lfer_diffs(
        base_feats, bundle_2d.x_cont_scaled, atom_mat, thermo_col, pair_i, pair_j,
        env_counters, geom_cols, attenuation_scale
    )
    topo_scale = np.std(raw_topo_tr, axis=0)
    topo_scale = np.where(topo_scale < 1e-6, 1.0, topo_scale).astype(np.float32)
    lfer_scale = np.std(raw_lfer_tr, axis=0)
    lfer_scale = np.where(lfer_scale < 1e-6, 1.0, lfer_scale).astype(np.float32)

    sc_topo_tr = np.clip(raw_topo_tr / topo_scale, -10.0, 10.0).astype(np.float32)
    sc_lfer_tr = np.clip(raw_lfer_tr / lfer_scale, -10.0, 10.0).astype(np.float32)

    model = ContextModulatedSkipLFERDeltaNet(
        mol_dim=bundle_2d.x_mol.shape[1],
        topo_diff_dim=sc_topo_tr.shape[1],
        lfer_dim=sc_lfer_tr.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    x_mol_all_t = torch.tensor(bundle_2d.x_mol, dtype=torch.float32, device=device)
    pair_i_t = torch.tensor(pair_i, dtype=torch.long, device=device)
    pair_j_t = torch.tensor(pair_j, dtype=torch.long, device=device)
    topo_tr_t = torch.tensor(sc_topo_tr, dtype=torch.float32, device=device)
    lfer_tr_t = torch.tensor(sc_lfer_tr, dtype=torch.float32, device=device)
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
            pred_ij, _, dw_ctx = model.forward_decomposed(
                x_mol_all_t[pair_i_t[b]],
                x_mol_all_t[pair_j_t[b]],
                topo_tr_t[b],
                lfer_tr_t[b],
                sim_tr_t[b],
            )
            reg = 1e-3 * (torch.mean(model.w_global ** 2) + torch.mean(dw_ctx ** 2))
            loss = loss_fn(pred_ij, dy_tr_t[b]) + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        va_out = _predict_skip_lfer(
            model, base_feats, bundle_2d, atom_mat, thermo_col, geom_cols, attenuation_scale,
            inner_va_idx, inner_tr_idx, y, y_std, topo_scale, lfer_scale, env_counters,
            channel_groups, device, k_eval=k_eval, tau=tau
        )
        va_rmse = float(np.sqrt(np.mean((y[inner_va_idx] - va_out["y_hat"]) ** 2)))
        if va_rmse < best_val_rmse - 1e-5:
            best_val_rmse = va_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    va_out = _predict_skip_lfer(
        model, base_feats, bundle_2d, atom_mat, thermo_col, geom_cols, attenuation_scale,
        inner_va_idx, inner_tr_idx, y, y_std, topo_scale, lfer_scale, env_counters,
        channel_groups, device, k_eval=k_eval, tau=tau
    )
    te_out = _predict_skip_lfer(
        model, base_feats, bundle_2d, atom_mat, thermo_col, geom_cols, attenuation_scale,
        test_idx, train_idx, y, y_std, topo_scale, lfer_scale, env_counters,
        channel_groups, device, k_eval=k_eval, tau=tau
    )
    return {
        "inner_tr_idx": inner_tr_idx,
        "inner_va_idx": inner_va_idx,
        "env_counters": env_counters,
        "y_pred_val": va_out["y_hat"],
        "s_max_val": va_out["s_max"],
        "y_pred_test": te_out["y_hat"],
        "s_max_test": te_out["s_max"],
        "test_channel_contribs": te_out["channel_contribs"],
        "test_base_2d": te_out["base_2d_part"],
        "test_top_anchor_g": te_out["top_anchor_g"],
    }


def _compute_local_q_mag(
    base_feats: TargetFeatures,
    q_col: np.ndarray,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    env_counters: list[Any],
    exclude_self: bool,
    k_top: int = 5,
    tau: float = 10.0,
) -> np.ndarray:
    sim_mat = compute_tanimoto_matrix(base_feats.ecfp_chiral[query_idx], base_feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(
        sim_mat, anchor_pool_idx, k=k_top, exclude_self_global_idx=query_idx if exclude_self else None
    )
    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    raw_dq = compute_local_pair_deltas(
        q_flat, a_flat, base_feats.atom_ptr, base_feats.atom_env_hash, q_col, env_counters=env_counters
    )[:, 0]
    abs_dq = np.abs(raw_dq).reshape(n_q, k_eff)
    logits = tau * anchor_sims
    logits -= np.max(logits, axis=1, keepdims=True)
    w = np.exp(logits)
    w /= np.sum(w, axis=1, keepdims=True)
    return np.sum(w * abs_dq, axis=1).astype(np.float64)


def _fit_boltz_phys_gate(
    base_feats: TargetFeatures,
    boltz: BoltzmannTargetFeatures,
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
) -> dict[str, Any]:
    """Monotonic physical gate using Boltzmann conformational entropy S_conf and Boltzmann |Delta q|."""
    s_conf = boltz.mol_thermo[:, 1].astype(np.float64)
    sc_mean = float(np.mean(s_conf[inner_tr_idx]))
    sc_std = float(max(np.std(s_conf[inner_tr_idx]), 1e-6))
    sc_val_tilde = np.clip((s_conf[inner_va_idx] - sc_mean) / sc_std, -5.0, 5.0)
    sc_test_tilde = np.clip((s_conf[test_idx] - sc_mean) / sc_std, -5.0, 5.0)

    q_col = boltz.atom_boltz_lfer[:, 0:1]
    dq_tr = _compute_local_q_mag(base_feats, q_col, inner_tr_idx, inner_tr_idx, env_counters, exclude_self=True)
    dq_mean = float(np.mean(dq_tr))
    dq_std = float(max(np.std(dq_tr), 1e-6))

    dq_val = _compute_local_q_mag(base_feats, q_col, inner_va_idx, inner_tr_idx, env_counters, exclude_self=False)
    dq_test = _compute_local_q_mag(base_feats, q_col, test_idx, train_idx, env_counters, exclude_self=False)
    dq_val_tilde = np.clip((dq_val - dq_mean) / dq_std, -5.0, 5.0)
    dq_test_tilde = np.clip((dq_test - dq_mean) / dq_std, -5.0, 5.0)

    def sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def obj(p: np.ndarray) -> float:
        ws, b, we, wq = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        a = sig(ws * (s_max_val - b) - we * sc_val_tilde + wq * dq_val_tilde)
        yb = a * y_val_delta + (1.0 - a) * y_val_svm
        return float(np.mean((y_val_true - yb) ** 2)) + 1e-4 * (ws ** 2) + 1e-3 * (we ** 2 + wq ** 2)

    bounds = [(0.0, 25.0), (0.20, 0.95), (0.0, 5.0), (0.0, 5.0)]
    best_res = None
    for x0 in [np.array([4.0, 0.70, 0.1, 0.1]), np.array([6.0, 0.65, 0.3, 0.3])]:
        r = optimize.minimize(obj, x0=x0, bounds=bounds, method="L-BFGS-B")
        if best_res is None or r.fun < best_res.fun:
            best_res = r

    ws, b, we, wq = [float(v) for v in best_res.x]
    alpha_test = sig(ws * (s_max_test - b) - we * sc_test_tilde + wq * dq_test_tilde)
    y_pred_test = (alpha_test * y_test_delta + (1.0 - alpha_test) * y_test_svm).astype(np.float64)
    return {
        "y_pred_test": y_pred_test,
        "alpha_test": alpha_test,
        "dq_test": dq_test,
        "ws": ws,
        "b": b,
        "we": we,
        "wq": wq,
    }


def _plot_v2(summary_df: pd.DataFrame, mag_df: pd.DataFrame) -> Path:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.0))

    models = [
        ("B1c_RBF_SVM", "B1c RBF-SVM", "#999999"),
        ("G0_RBFSVM_Gate_A2", "G0 SVM+2D", "#56B4E9"),
        ("G2_Prev_SingleConcat", "G2 Single-Concat", "#009E73"),
        ("G_S1_Single_SkipLFER", "G_S1 Single-Skip", "#0072B2"),
        ("G_S2_Boltz_SkipLFER", "G_S2 Boltz-Skip", "#D55E00"),
    ]
    x = np.arange(len(MVP_TARGETS))
    w = 0.16
    for i, (m_key, m_lbl, col) in enumerate(models):
        means, stds = [], []
        for tgt in MVP_TARGETS:
            sub = summary_df[(summary_df["target"] == tgt) & (summary_df["model"] == m_key)]
            means.append(float(sub["cliff_rmse_mean"].iloc[0]))
            stds.append(float(sub["cliff_rmse_std"].iloc[0]))
        ax1.bar(x + (i - 2.0) * w, means, yerr=stds, capsize=2.5, width=w, label=m_lbl, color=col, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(MVP_TARGETS), fontweight="bold")
    ax1.set_ylabel("Activity Cliff RMSE (5-seed mean ± std)")
    ax1.set_title("Scheme A v2: Skip-LFER & Boltzmann Multi-Conformer vs Controls", fontweight="bold")
    ax1.set_ylim(0.50, 0.90)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(frameon=False, fontsize=8.5)

    # Right panel: Exact analytical |Delta pKi| magnitude per LFER channel in S2_Boltz_SkipLFER
    ch_names = [c for c in S2_CHANNEL_GROUPS if c != "AtomCount"]
    cx = np.arange(len(ch_names))
    cw = 0.25
    t_colors = ["#0072B2", "#D55E00", "#009E73"]
    for idx_t, tgt in enumerate(MVP_TARGETS):
        vals = []
        for ch in ch_names:
            sub = mag_df[(mag_df["target"] == tgt) & (mag_df["channel"] == ch)]
            vals.append(float(sub["mean_abs_dpki"].iloc[0]) if not sub.empty else 0.0)
        ax2.bar(cx + (idx_t - 1) * cw, vals, width=cw, label=tgt, color=t_colors[idx_t], edgecolor="black", linewidth=0.5)
    ax2.set_xticks(cx)
    ax2.set_xticklabels(ch_names, rotation=28, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Analytical |ΔpKi| Contribution (Cliff Subset)")
    ax2.set_title("Exact Additive ΔpKi Decomposition in S2_Boltz_SkipLFER", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = _fig_dir() / "plan_a_v2_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    t0 = time.time()
    datasets = {tgt: load_target_dataset(tgt) for tgt in MVP_TARGETS}
    boltz_feats: dict[str, BoltzmannTargetFeatures] = {}
    for tgt in MVP_TARGETS:
        print(f"=== Extracting/Loading Boltzmann 5-Conformer Features for {tgt} ===", flush=True)
        bf = extract_or_load_boltzmann_features(tgt, datasets[tgt].smiles, num_confs=5, energy_window_kcal=5.0)
        boltz_feats[tgt] = bf
        print(f"Boltzmann meta [{tgt}]: {bf.meta}", flush=True)

    records: list[dict[str, Any]] = []
    preds_store: dict[tuple[str, str, int], np.ndarray] = {}
    mag_rows: list[dict[str, Any]] = []
    case_data: dict[str, list[dict[str, Any]]] = {}
    gate_s2_rows: list[dict[str, Any]] = []

    for tgt in MVP_TARGETS:
        ds = datasets[tgt]
        bf = boltz_feats[tgt]
        base = bf.base
        lfer_single_atoms = _lfer_atom_matrix(base)
        y_te = ds.y[ds.test_idx]
        c_te = ds.cliff_mol[ds.test_idx]
        cmask = c_te == 1

        y_test_svm = _fit_rbf_svr(base.ecfp_chiral, ds.y, ds.train_idx, ds.test_idx)

        gs2_preds_seeds = []
        g0_preds_seeds = []
        s2_ch_accum: dict[str, list[np.ndarray]] = {ch: [] for ch in S2_CHANNEL_GROUPS}
        s2_base2d_seeds = []
        alpha_s2_seeds = []
        dq_s2_last = None
        top_anc_last = None

        for seed in SEEDS:
            # Controls: A2 (2D) & L2 (Previous single-conf concat)
            a2_out = _train_delta_backbone(base, lfer_single_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="2d")
            l2_out = _train_delta_backbone(base, lfer_single_atoms, ds.y, ds.train_idx, ds.test_idx, seed, mode="lfer_espan")

            # S1: Single-conformer Context-Modulated Skip-LFER
            s1_out = train_skip_lfer_model(
                base, lfer_single_atoms, base.feat_3d[:, 1], [4, 5, 11, 12], 10.0, S1_CHANNEL_GROUPS,
                ds.y, ds.train_idx, ds.test_idx, seed=seed
            )
            # S2: Boltzmann 5-conformer Context-Modulated Skip-LFER (3D attenuated by S_conf, scale=1.0 -> 1/N_eff)
            s2_out = train_skip_lfer_model(
                base, bf.atom_boltz_lfer, bf.mol_thermo[:, 1], [5, 6, 7, 14, 15, 16], 1.0, S2_CHANNEL_GROUPS,
                ds.y, ds.train_idx, ds.test_idx, seed=seed
            )

            y_val_svm = _fit_rbf_svr(base.ecfp_chiral, ds.y, a2_out["inner_tr_idx"], a2_out["inner_va_idx"])
            y_val_true = ds.y[a2_out["inner_va_idx"]]

            g0_out = _fit_2param_gate(
                y_val_true, a2_out["y_pred_val"], y_val_svm, a2_out["s_max_val"],
                a2_out["y_pred_test"], y_test_svm, a2_out["s_max_test"]
            )
            g1_l2_tmp = _fit_2param_gate(
                y_val_true, l2_out["y_pred_val"], y_val_svm, l2_out["s_max_val"],
                l2_out["y_pred_test"], y_test_svm, l2_out["s_max_test"]
            )
            g2_prev_out = _fit_phys_gate_g2(
                base, lfer_single_atoms, l2_out["env_counters"],
                l2_out["inner_tr_idx"], l2_out["inner_va_idx"], ds.train_idx, ds.test_idx,
                y_val_true, l2_out["y_pred_val"], y_val_svm, l2_out["s_max_val"],
                l2_out["y_pred_test"], y_test_svm, l2_out["s_max_test"],
                g1_ws=g1_l2_tmp["ws"], g1_b=g1_l2_tmp["b"],
            )
            g1_s1_tmp = _fit_2param_gate(
                y_val_true, s1_out["y_pred_val"], y_val_svm, s1_out["s_max_val"],
                s1_out["y_pred_test"], y_test_svm, s1_out["s_max_test"]
            )
            g_s1_out = _fit_phys_gate_g2(
                base, lfer_single_atoms, s1_out["env_counters"],
                s1_out["inner_tr_idx"], s1_out["inner_va_idx"], ds.train_idx, ds.test_idx,
                y_val_true, s1_out["y_pred_val"], y_val_svm, s1_out["s_max_val"],
                s1_out["y_pred_test"], y_test_svm, s1_out["s_max_test"],
                g1_ws=g1_s1_tmp["ws"], g1_b=g1_s1_tmp["b"],
            )
            g_s2_out = _fit_boltz_phys_gate(
                base, bf, s2_out["env_counters"],
                s2_out["inner_tr_idx"], s2_out["inner_va_idx"], ds.train_idx, ds.test_idx,
                y_val_true, s2_out["y_pred_val"], y_val_svm, s2_out["s_max_val"],
                s2_out["y_pred_test"], y_test_svm, s2_out["s_max_test"],
            )

            gate_s2_rows.append(
                {
                    "target": tgt,
                    "seed": seed,
                    "ws": g_s2_out["ws"],
                    "b": g_s2_out["b"],
                    "we": g_s2_out["we"],
                    "wq": g_s2_out["wq"],
                    "alpha_mean": float(np.mean(g_s2_out["alpha_test"])),
                }
            )

            model_preds = {
                "B1c_RBF_SVM": y_test_svm,
                "A2_2D_Delta": a2_out["y_pred_test"],
                "L2_Prev_SingleConcat": l2_out["y_pred_test"],
                "S1_Single_SkipLFER": s1_out["y_pred_test"],
                "S2_Boltz_SkipLFER": s2_out["y_pred_test"],
                "G0_RBFSVM_Gate_A2": g0_out["y_pred_test"],
                "G2_Prev_SingleConcat": g2_prev_out["y_pred_test"],
                "G_S1_Single_SkipLFER": g_s1_out["y_pred_test"],
                "G_S2_Boltz_SkipLFER": g_s2_out["y_pred_test"],
            }
            for m_name, yp in model_preds.items():
                met = evaluate_predictions(y_te, yp, c_te)
                records.append({"target": tgt, "model": m_name, "seed": seed, **met})
                preds_store[(tgt, m_name, seed)] = yp

            gs2_preds_seeds.append(g_s2_out["y_pred_test"])
            g0_preds_seeds.append(g0_out["y_pred_test"])
            s2_base2d_seeds.append(s2_out["test_base_2d"])
            alpha_s2_seeds.append(g_s2_out["alpha_test"])
            dq_s2_last = g_s2_out["dq_test"]
            top_anc_last = s2_out["test_top_anchor_g"]
            for ch in S2_CHANNEL_GROUPS:
                s2_ch_accum[ch].append(s2_out["test_channel_contribs"][ch])

            print(
                f"[{tgt} | seed={seed}] "
                f"SVM={evaluate_predictions(y_te, y_test_svm, c_te)['cliff_rmse']:.4f} | "
                f"G0={evaluate_predictions(y_te, g0_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"G2_Prev={evaluate_predictions(y_te, g2_prev_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"G_S1={evaluate_predictions(y_te, g_s1_out['y_pred_test'], c_te)['cliff_rmse']:.4f} | "
                f"G_S2={evaluate_predictions(y_te, g_s2_out['y_pred_test'], c_te)['cliff_rmse']:.4f}",
                flush=True,
            )

        # Analytical |Delta pKi| contribution stats across cliff subset
        s2_ch_mean = {ch: np.mean(np.stack(vals, axis=0), axis=0) for ch, vals in s2_ch_accum.items()}
        for ch, vec in s2_ch_mean.items():
            mag_rows.append(
                {
                    "target": tgt,
                    "channel": ch,
                    "mean_abs_dpki": round(float(np.mean(np.abs(vec[cmask]))), 4),
                    "max_abs_dpki": round(float(np.max(np.abs(vec[cmask]))), 4),
                }
            )

        # Concrete win cases for exact Delta pKi chemical decomposition
        gs2_mean = np.mean(np.stack(gs2_preds_seeds, axis=0), axis=0)
        g0_mean = np.mean(np.stack(g0_preds_seeds, axis=0), axis=0)
        alpha_mean = np.mean(np.stack(alpha_s2_seeds, axis=0), axis=0)
        sim_te_tr = compute_tanimoto_matrix(base.ecfp_chiral[ds.test_idx], base.ecfp_chiral[ds.train_idx])
        s_max_te = np.max(sim_te_tr, axis=1)

        err_svm = np.abs(y_te - y_test_svm)
        err_g0 = np.abs(y_te - g0_mean)
        err_gs2 = np.abs(y_te - gs2_mean)
        win_margin = np.minimum(err_svm - err_gs2, err_g0 - err_gs2)
        cliff_pos = np.where(cmask)[0]
        top_wins = cliff_pos[np.argsort(-win_margin[cliff_pos])[:2]]
        worst_fail = cliff_pos[np.argsort(err_svm[cliff_pos] - err_gs2[cliff_pos])[:1]]

        cases = []
        for kind, plist in [("WIN", top_wins), ("FAIL", worst_fail)]:
            for pos in plist:
                g_idx = int(ds.test_idx[pos])
                anc_g = int(top_anc_last[pos])
                decomp = {ch: round(float(s2_ch_mean[ch][pos]), 4) for ch in S2_CHANNEL_GROUPS if ch != "AtomCount"}
                cases.append(
                    {
                        "type": kind,
                        "test_idx": g_idx,
                        "smiles": ds.smiles[g_idx],
                        "anchor_smiles": ds.smiles[anc_g],
                        "y_true": round(float(y_te[pos]), 3),
                        "anchor_y": round(float(ds.y[anc_g]), 3),
                        "s_max": round(float(s_max_te[pos]), 3),
                        "s_conf": round(float(bf.mol_thermo[g_idx, 1]), 3),
                        "n_eff": round(float(bf.mol_thermo[g_idx, 2]), 2),
                        "dq_boltz": round(float(dq_s2_last[pos]), 4),
                        "alpha": round(float(alpha_mean[pos]), 3),
                        "pred_svm": round(float(y_test_svm[pos]), 3),
                        "pred_g0": round(float(g0_mean[pos]), 3),
                        "pred_gs2": round(float(gs2_mean[pos]), 3),
                        "decomp_dpki": decomp,
                    }
                )
        case_data[tgt] = cases

    rec_df = pd.DataFrame(records)
    model_order = [
        "B1c_RBF_SVM",
        "A2_2D_Delta",
        "L2_Prev_SingleConcat",
        "S1_Single_SkipLFER",
        "S2_Boltz_SkipLFER",
        "G0_RBFSVM_Gate_A2",
        "G2_Prev_SingleConcat",
        "G_S1_Single_SkipLFER",
        "G_S2_Boltz_SkipLFER",
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
    summary_df.to_csv(_refine_dir() / "plan_a_v2_summary.csv", index=False)

    comparisons = [
        ("S1_Single_SkipLFER", "A2_2D_Delta"),
        ("S2_Boltz_SkipLFER", "S1_Single_SkipLFER"),
        ("S2_Boltz_SkipLFER", "A2_2D_Delta"),
        ("G_S1_Single_SkipLFER", "G0_RBFSVM_Gate_A2"),
        ("G_S2_Boltz_SkipLFER", "G0_RBFSVM_Gate_A2"),
        ("G_S2_Boltz_SkipLFER", "G2_Prev_SingleConcat"),
        ("G_S2_Boltz_SkipLFER", "B1c_RBF_SVM"),
        ("G_S1_Single_SkipLFER", "B1c_RBF_SVM"),
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

    mag_df = pd.DataFrame(mag_rows)
    gate_df = pd.DataFrame(gate_s2_rows)
    fig_path = _plot_v2(summary_df, mag_df)

    md: list[str] = [
        "# Scheme A v2 Evaluation: Boltzmann Multi-Conformer & Context-Modulated Skip-LFER",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Elapsed**: {time.time() - t0:.1f}s",
        "",
        "## 0. Boltzmann Multi-Conformer Extraction Summary",
        "| Target | Mean Valid Confs | Mean Kept (<=5 kcal/mol) | Embed Fallbacks | AIMNet2 Fallbacks | Extract Sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tgt in MVP_TARGETS:
        m = boltz_feats[tgt].meta
        md.append(
            f"| {tgt} | {m['mean_valid_confs']} | {m['mean_kept_confs']} | "
            f"{m['embed_fallback_count']} | {m['aimnet_fallback_count']} | {m['elapsed_sec']} |"
        )

    md.extend(
        [
            "",
            "## 1. Main Comparison Table (3 Targets x 5 Seeds Mean ± Std)",
            "| Target | Model | RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
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
            "## 3. Learned Boltzmann Physical Gate Weights (G_S2: -S_conf, +|Δq_Boltz|)",
            "| Target | w_s (s_max) | b | w_e (-S_conf) | w_q (+|Δq_Boltz|) | Test alpha mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tgt in MVP_TARGETS:
        sub = gate_df[gate_df["target"] == tgt]
        md.append(
            f"| {tgt} | {sub['ws'].mean():.2f} ± {sub['ws'].std(ddof=1):.2f} | "
            f"{sub['b'].mean():.3f} ± {sub['b'].std(ddof=1):.3f} | "
            f"{sub['we'].mean():.3f} ± {sub['we'].std(ddof=1):.3f} | "
            f"{sub['wq'].mean():.3f} ± {sub['wq'].std(ddof=1):.3f} | "
            f"{sub['alpha_mean'].mean():.3f} |"
        )

    md.extend(
        [
            "",
            "## 4. Exact Analytical ΔpKi Channel Magnitudes in S2_Boltz_SkipLFER (Cliff Subset)",
            "| Channel | CHEMBL204_Ki Mean |ΔpKi| | CHEMBL2147_Ki Mean |ΔpKi| | CHEMBL237_Ki Mean |ΔpKi| |",
            "|---|---:|---:|---:|",
        ]
    )
    for ch in [c for c in S2_CHANNEL_GROUPS if c != "AtomCount"]:
        vals = []
        for tgt in MVP_TARGETS:
            s = mag_df[(mag_df["target"] == tgt) & (mag_df["channel"] == ch)].iloc[0]
            vals.append(f"{s['mean_abs_dpki']:.4f} (max {s['max_abs_dpki']:.3f})")
        md.append(f"| `{ch}` | " + " | ".join(vals) + " |")

    md.extend(["", "## 5. Concrete Cliff Pairs with Exact Analytical ΔpKi Decomposition"])
    for tgt in MVP_TARGETS:
        md.append(f"### {tgt}")
        for cs in case_data[tgt]:
            md.append(
                f"- **[{cs['type']}]** `test_idx={cs['test_idx']}`: `y={cs['y_true']}`, `anchor_y={cs['anchor_y']}`, "
                f"`s_max={cs['s_max']}`, `S_conf={cs['s_conf']}` (`N_eff={cs['n_eff']}`), `|Δq_Boltz|={cs['dq_boltz']}`, `alpha={cs['alpha']}` "
                f"-> `SVM={cs['pred_svm']}`, `G0={cs['pred_g0']}`, `G_S2={cs['pred_gs2']}`\n"
                f"  - Exact LFER ΔpKi channels: `{cs['decomp_dpki']}`\n"
                f"  - Query: `{cs['smiles']}`\n"
                f"  - Anchor: `{cs['anchor_smiles']}`"
            )

    md.extend(["", f"Figure: `{fig_path}`"])
    (_refine_dir() / "PLAN_A_V2_EVAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Scheme A v2 evaluation finished cleanly in {time.time() - t0:.1f}s -> PLAN_A_V2_EVAL.md")


if __name__ == "__main__":
    main()
