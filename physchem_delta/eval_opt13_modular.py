"""Option 1 + Option 3 Modular Exploration in Direct y-Space over 10 Wave-1 Targets."""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from rdkit import Chem
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physchem_delta.data import load_target_dataset
from physchem_delta.eval_path_a_modular import (
    FiLMGated3DQMResidualNet,
    FiLMKernelResidualDeltaNet,
    _build_modular_pair_features,
)
from physchem_delta.eval_plan_a import SEEDS, _lfer_atom_matrix
from physchem_delta.featurize import (
    TargetFeatures,
    extract_or_load_target_features,
    precompute_mol_env_counters,
)
from physchem_delta.metrics import evaluate_predictions
from physchem_delta.models import (
    _select_topk_anchors,
    build_tier_matrices,
    compute_tanimoto_matrix,
    get_device,
    make_inner_split,
    set_global_seed,
)
from physchem_delta.run_30_targets import WAVE_1_TARGETS

ATOM_Z_VOCAB = (6, 7, 8, 16, 9, 17, 35, 53, 15, 5, 14)
HYB_VOCAB = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
CHIRAL_VOCAB = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
BOND_TYPE_VOCAB = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
STEREO_VOCAB = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
)

DELTA_MODELS = (
    "Y_A2_2D",
    "Y_FiLM_2DLFER",
    "Y_FiLM_Gated3DQM",
    "Y_FiLM_CrossAttn",
    "Y_GraphFP_FiLM_Attn",
)
KERNEL_MODELS = ("K_RBF_SVM", "K_Tan_SVR", "K_Multi_SVR")

ALL_MODELS = (
    "K_RBF_SVM",
    "K_Tan_SVR",
    "K_Multi_SVR",
    "B1_CatBoost",
    "A1_Direct_MLP",
    "G2_Prev",
    "Y_A2_2D",
    "Y_FiLM_2DLFER",
    "Y_FiLM_Gated3DQM",
    "Y_FiLM_CrossAttn",
    "Y_GraphFP_FiLM_Attn",
    "E_SVM_CatBoost",
    "E_SVM_DirectMLP",
    "E_Tan_CatBoost",
    "E_Tan_DirectMLP",
    "E_Multi_CatBoost",
    "E_Multi_DirectMLP",
    "E_SVM_A2Delta",
    "E_Tan_A2Delta",
    "E_Multi_A2Delta",
    "E_SVM_FiLM",
    "E_Tan_FiLM",
    "E_Multi_FiLM",
    "E_SVM_3DQM",
    "E_Tan_3DQM",
    "E_Multi_3DQM",
    "E_SVM_Attn",
    "E_Tan_Attn",
    "E_Multi_Attn",
    "E_SVM_GraphAttn",
    "E_Tan_GraphAttn",
    "E_Multi_GraphAttn",
)


def _out_root() -> Path:
    d = ROOT_DIR / "refine-logs" / "opt13_modular_results"
    (d / "per_target").mkdir(parents=True, exist_ok=True)
    return d


def _fig_dir() -> Path:
    d = ROOT_DIR / "refine-logs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _onehot(idx: int, n_bins: int) -> np.ndarray:
    v = np.zeros((n_bins,), dtype=np.float32)
    v[min(max(idx, 0), n_bins - 1)] = 1.0
    return v


def _vocab_onehot(val: Any, vocab: tuple[Any, ...]) -> np.ndarray:
    v = np.zeros((len(vocab) + 1,), dtype=np.float32)
    try:
        idx = vocab.index(val)
    except ValueError:
        idx = len(vocab)
    v[idx] = 1.0
    return v


def _extract_target_graphs(smiles_list: list[str], feats: TargetFeatures) -> list[dict[str, np.ndarray]]:
    """Pre-extract RDKit 2D molecular graphs + 3 atomic Crippen/TPSA channels per molecule."""
    lfer_atoms = _lfer_atom_matrix(feats)
    graphs: list[dict[str, np.ndarray]] = []
    for m_idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        p0 = int(feats.atom_ptr[m_idx])
        p1 = int(feats.atom_ptr[m_idx + 1])
        atom_lfer = lfer_atoms[p0:p1, [1, 2, 3]]  # [crip_logp, crip_mr, tpsa_c]

        if mol is None or mol.GetNumAtoms() == 0:
            x_cat = np.zeros((1, 41), dtype=np.float32)
            x_cat[0, len(ATOM_Z_VOCAB)] = 1.0
            x_cont = np.zeros((1, 3), dtype=np.float32)
            edge_src = np.zeros((1,), dtype=np.int64)
            edge_dst = np.zeros((1,), dtype=np.int64)
            edge_attr = np.zeros((1, 11), dtype=np.float32)
            edge_attr[0, 4] = 1.0
            graphs.append(
                {
                    "x_cat": x_cat,
                    "x_cont": x_cont,
                    "edge_src": edge_src,
                    "edge_dst": edge_dst,
                    "edge_attr": edge_attr,
                }
            )
            continue

        n_atoms = mol.GetNumAtoms()
        cat_rows = []
        for a in mol.GetAtoms():
            z_oh = _vocab_onehot(a.GetAtomicNum(), ATOM_Z_VOCAB)  # 12
            deg_oh = _onehot(min(a.GetTotalDegree(), 5), 6)  # 6
            fc = a.GetFormalCharge()
            fc_idx = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}.get(fc, 5)
            fc_oh = _onehot(fc_idx, 6)  # 6
            hyb_oh = _vocab_onehot(a.GetHybridization(), HYB_VOCAB)  # 6
            hs_oh = _onehot(min(a.GetTotalNumHs(), 4), 5)  # 5
            ch_oh = _vocab_onehot(a.GetChiralTag(), CHIRAL_VOCAB)  # 4
            flags = np.asarray([float(a.GetIsAromatic()), float(a.IsInRing())], dtype=np.float32)  # 2
            cat_rows.append(np.concatenate([z_oh, deg_oh, fc_oh, hyb_oh, hs_oh, ch_oh, flags], axis=0))

        x_cat = np.stack(cat_rows, axis=0).astype(np.float32)
        if atom_lfer.shape[0] == n_atoms:
            x_cont = atom_lfer.astype(np.float32)
        else:
            x_cont = np.zeros((n_atoms, 3), dtype=np.float32)

        src_l: list[int] = []
        dst_l: list[int] = []
        eattr_l: list[np.ndarray] = []
        for b in mol.GetBonds():
            u = b.GetBeginAtomIdx()
            v = b.GetEndAtomIdx()
            bt_oh = _vocab_onehot(b.GetBondType(), BOND_TYPE_VOCAB)  # 5
            bflags = np.asarray([float(b.GetIsConjugated()), float(b.IsInRing())], dtype=np.float32)  # 2
            st_oh = _vocab_onehot(b.GetStereo(), STEREO_VOCAB)  # 4
            ef = np.concatenate([bt_oh, bflags, st_oh], axis=0).astype(np.float32)
            src_l.extend([u, v])
            dst_l.extend([v, u])
            eattr_l.extend([ef, ef])

        if not src_l:
            # Self-loop fallback for single-atom molecules
            src_l = [0]
            dst_l = [0]
            ef = np.zeros((11,), dtype=np.float32)
            ef[4] = 1.0
            eattr_l = [ef]

        graphs.append(
            {
                "x_cat": x_cat,
                "x_cont": x_cont,
                "edge_src": np.asarray(src_l, dtype=np.int64),
                "edge_dst": np.asarray(dst_l, dtype=np.int64),
                "edge_attr": np.stack(eattr_l, axis=0).astype(np.float32),
            }
        )
    return graphs


def _collate_mol_graphs(
    graphs: list[dict[str, np.ndarray]],
    mol_ids: np.ndarray,
    node_mean: np.ndarray,
    node_std: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    node_cats = []
    node_conts = []
    srcs = []
    dsts = []
    eattrs = []
    mol_ptr = []
    offset = 0
    for b_idx, m_id in enumerate(mol_ids.tolist()):
        g = graphs[m_id]
        nv = g["x_cat"].shape[0]
        node_cats.append(g["x_cat"])
        node_conts.append(g["x_cont"])
        srcs.append(g["edge_src"] + offset)
        dsts.append(g["edge_dst"] + offset)
        eattrs.append(g["edge_attr"])
        mol_ptr.append(np.full((nv,), b_idx, dtype=np.int64))
        offset += nv

    x_cat = np.concatenate(node_cats, axis=0)
    x_cont = np.clip((np.concatenate(node_conts, axis=0) - node_mean) / node_std, -10.0, 10.0).astype(np.float32)
    x_node = torch.tensor(np.concatenate([x_cat, x_cont], axis=1), dtype=torch.float32, device=device)
    edge_src = torch.tensor(np.concatenate(srcs, axis=0), dtype=torch.long, device=device)
    edge_dst = torch.tensor(np.concatenate(dsts, axis=0), dtype=torch.long, device=device)
    edge_attr = torch.tensor(np.concatenate(eattrs, axis=0), dtype=torch.float32, device=device)
    batch_vec = torch.tensor(np.concatenate(mol_ptr, axis=0), dtype=torch.long, device=device)
    return x_node, edge_src, edge_dst, edge_attr, batch_vec, int(mol_ids.size)


class FiLMAnchorCrossAttnNet(FiLMKernelResidualDeltaNet):
    """y-Space FiLM 2D-LFER Pairwise Delta + Zero-Init Scaled Dot-Product Anchor Cross-Attention."""

    def __init__(
        self,
        mol_dim: int,
        diff2d_dim: int = 26,
        lfer2d_dim: int = 7,
        emb_dim: int = 128,
        d_attn: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__(mol_dim=mol_dim, diff2d_dim=diff2d_dim, lfer2d_dim=lfer2d_dim, emb_dim=emb_dim, dropout=dropout)
        self.d_attn = d_attn
        self.w_q = nn.Linear(emb_dim, d_attn)
        nn.init.zeros_(self.w_q.weight)
        nn.init.zeros_(self.w_q.bias)
        self.w_k = nn.Linear(emb_dim + diff2d_dim + lfer2d_dim + 1, d_attn)

    def forward_batch(
        self,
        x_q_all: torch.Tensor,   # (B, mol_dim)
        x_a_all: torch.Tensor,   # (B, K, mol_dim)
        d2d: torch.Tensor,       # (B, K, 26)
        sim: torch.Tensor,       # (B, K)
        dlfer2d: torch.Tensor,   # (B, K, 7)
        tau: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = x_a_all.shape
        h_q = self.mol_encoder(x_q_all)  # (B, emb)
        h_q_exp = h_q.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        h_a_flat = self.mol_encoder(x_a_all.reshape(B * K, -1))
        prod = h_q_exp * h_a_flat
        d2d_flat = d2d.reshape(B * K, -1)
        dlf_flat = dlfer2d.reshape(B * K, -1)
        s_flat = sim.reshape(B * K, 1)

        out_ij = self._encode_pair_half(h_q_exp, h_a_flat, prod, d2d_flat, s_flat, dlf_flat)
        out_ji = self._encode_pair_half(h_a_flat, h_q_exp, prod, -d2d_flat, s_flat, -dlf_flat)
        delta_mat = (0.5 * (out_ij - out_ji)).view(B, K)

        q_vec = self.w_q(h_q)  # (B, d_attn)
        k_in = torch.cat([h_a_flat, d2d_flat, dlf_flat, s_flat], dim=-1)
        k_vec = self.w_k(k_in).view(B, K, self.d_attn)
        attn_dot = torch.sum(q_vec.unsqueeze(1) * k_vec, dim=-1) / math.sqrt(self.d_attn)
        weights = torch.softmax(tau * sim + attn_dot, dim=1)
        return delta_mat, weights


class GINE2DEncoder(nn.Module):
    """Pure-PyTorch 3-layer Edge-Conditioned GINE + Mean/Max Readout."""

    def __init__(self, node_dim: int = 44, edge_dim: int = 11, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.node_in = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_projs = nn.ModuleList([nn.Linear(edge_dim, hidden_dim) for _ in range(3)])
        self.eps = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(3)])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x_node: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_vec: torch.Tensor,
        num_mols: int,
    ) -> torch.Tensor:
        h = self.node_in(x_node)
        dim = h.shape[1]
        for l_idx in range(3):
            msg = F.relu(h[edge_src] + self.edge_projs[l_idx](edge_attr))
            agg = torch.zeros_like(h)
            agg.index_add_(0, edge_dst, msg)
            h_upd = self.mlps[l_idx]((1.0 + self.eps[l_idx]) * h + agg)
            h = h + h_upd

        idx_exp = batch_vec.unsqueeze(-1).expand(-1, dim)
        mean_pool = torch.zeros((num_mols, dim), dtype=h.dtype, device=h.device)
        mean_pool.scatter_reduce_(0, idx_exp, h, reduce="mean", include_self=False)
        max_pool = torch.zeros((num_mols, dim), dtype=h.dtype, device=h.device)
        max_pool.scatter_reduce_(0, idx_exp, h, reduce="amax", include_self=False)
        return self.readout_proj(torch.cat([mean_pool, max_pool], dim=-1))


class GraphFPDualFiLMAttnNet(nn.Module):
    """3-Layer GINE Graph + ECFP/MACCS/Desc2D Dual Representation (256-dim) + FiLM + CrossAttn."""

    def __init__(
        self,
        mol_dim: int,
        node_dim: int = 44,
        edge_dim: int = 11,
        diff2d_dim: int = 26,
        lfer2d_dim: int = 7,
        fp_dim: int = 128,
        graph_dim: int = 128,
        d_attn: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        emb_dim = fp_dim + graph_dim  # 256
        self.d_attn = d_attn
        self.fp_encoder = nn.Sequential(
            nn.Linear(mol_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, fp_dim),
            nn.LayerNorm(fp_dim),
            nn.SiLU(),
        )
        self.graph_encoder = GINE2DEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=graph_dim, dropout=dropout * 0.8)

        pair_in_dim = emb_dim * 4 + diff2d_dim + 1
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_in_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout * 0.8),
        )
        self.film_gamma = nn.Linear(lfer2d_dim, 192)
        self.film_beta = nn.Linear(lfer2d_dim, 192)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.readout = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.lfer_skip = nn.Linear(lfer2d_dim, 1, bias=False)
        nn.init.zeros_(self.lfer_skip.weight)

        self.w_q = nn.Linear(emb_dim, d_attn)
        nn.init.zeros_(self.w_q.weight)
        nn.init.zeros_(self.w_q.bias)
        self.w_k = nn.Linear(emb_dim + diff2d_dim + lfer2d_dim + 1, d_attn)

    def encode_molecules(
        self,
        x_mol_uniq: torch.Tensor,
        x_node: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_vec: torch.Tensor,
        num_mols: int,
    ) -> torch.Tensor:
        h_fp = self.fp_encoder(x_mol_uniq)
        h_g = self.graph_encoder(x_node, edge_src, edge_dst, edge_attr, batch_vec, num_mols)
        return torch.cat([h_fp, h_g], dim=-1)

    def _encode_pair_half(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        prod: torch.Tensor,
        diff2d: torch.Tensor,
        sim_col: torch.Tensor,
        p_lfer2d: torch.Tensor,
    ) -> torch.Tensor:
        u = torch.cat([h_a, h_b, h_a - h_b, prod, diff2d, sim_col], dim=-1)
        h_tr = self.pair_trunk(u)
        gamma = 1.0 + torch.tanh(self.film_gamma(p_lfer2d))
        beta = self.film_beta(p_lfer2d)
        h_mod = gamma * h_tr + beta
        return self.readout(h_mod).squeeze(-1) + self.lfer_skip(p_lfer2d).squeeze(-1)

    def forward_from_embeddings(
        self,
        h_q: torch.Tensor,       # (B, 256)
        h_a: torch.Tensor,       # (B, K, 256)
        d2d: torch.Tensor,       # (B, K, 26)
        sim: torch.Tensor,       # (B, K)
        dlfer2d: torch.Tensor,   # (B, K, 7)
        tau: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = h_a.shape
        h_q_exp = h_q.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        h_a_flat = h_a.reshape(B * K, -1)
        prod = h_q_exp * h_a_flat
        d2d_flat = d2d.reshape(B * K, -1)
        dlf_flat = dlfer2d.reshape(B * K, -1)
        s_flat = sim.reshape(B * K, 1)

        out_ij = self._encode_pair_half(h_q_exp, h_a_flat, prod, d2d_flat, s_flat, dlf_flat)
        out_ji = self._encode_pair_half(h_a_flat, h_q_exp, prod, -d2d_flat, s_flat, -dlf_flat)
        delta_mat = (0.5 * (out_ij - out_ji)).view(B, K)

        q_vec = self.w_q(h_q)
        k_in = torch.cat([h_a_flat, d2d_flat, dlf_flat, s_flat], dim=-1)
        k_vec = self.w_k(k_in).view(B, K, self.d_attn)
        attn_dot = torch.sum(q_vec.unsqueeze(1) * k_vec, dim=-1) / math.sqrt(self.d_attn)
        weights = torch.softmax(tau * sim + attn_dot, dim=1)
        return delta_mat, weights


def _fit_kernel_bases(feats: TargetFeatures, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit Tanimoto-SVR and Multi-Kernel PSD SVR strictly on train_idx."""
    # 1. Precomputed Tanimoto SVR on chiral ECFP4
    k_tan_tr = compute_tanimoto_matrix(feats.ecfp_chiral[train_idx], feats.ecfp_chiral[train_idx])
    k_tan_te = compute_tanimoto_matrix(feats.ecfp_chiral[test_idx], feats.ecfp_chiral[train_idx])
    svr_tan = SVR(kernel="precomputed", C=3.0, epsilon=0.1)
    svr_tan.fit(k_tan_tr, y[train_idx])
    pred_tan = svr_tan.predict(k_tan_te).astype(np.float64)

    # 2. Composite Multi-Kernel PSD SVR: 0.70*ECFP_Tan + 0.15*MACCS_Tan + 0.15*RBF(Desc2D)
    k_mac_tr = compute_tanimoto_matrix(feats.maccs[train_idx], feats.maccs[train_idx])
    k_mac_te = compute_tanimoto_matrix(feats.maccs[test_idx], feats.maccs[train_idx])

    sc = StandardScaler()
    sc.fit(feats.desc2d[train_idx])
    scale = np.where(sc.scale_ < 1e-8, 1.0, sc.scale_)
    d_tr = np.clip((feats.desc2d[train_idx] - sc.mean_) / scale, -10.0, 10.0)
    d_te = np.clip((feats.desc2d[test_idx] - sc.mean_) / scale, -10.0, 10.0)
    gamma = 1.0 / float(feats.desc2d.shape[1])
    k_rbf_tr = rbf_kernel(d_tr, d_tr, gamma=gamma)
    k_rbf_te = rbf_kernel(d_te, d_tr, gamma=gamma)

    k_multi_tr = 0.70 * k_tan_tr + 0.15 * k_mac_tr + 0.15 * k_rbf_tr
    k_multi_te = 0.70 * k_tan_te + 0.15 * k_mac_te + 0.15 * k_rbf_te
    svr_multi = SVR(kernel="precomputed", C=3.0, epsilon=0.1)
    svr_multi.fit(k_multi_tr, y[train_idx])
    pred_multi = svr_multi.predict(k_multi_te).astype(np.float64)
    return pred_tan, pred_multi


def _predict_pair_yspace(
    model: nn.Module,
    variant: str,
    feats: TargetFeatures,
    bundle_2d: Any,
    lfer_atoms: np.ndarray,
    query_idx: np.ndarray,
    anchor_pool_idx: np.ndarray,
    y: np.ndarray,
    y_std: float,
    scales: tuple[np.ndarray, np.ndarray, np.ndarray],
    env_counters: list[Any],
    device: torch.device,
    graphs: list[dict[str, np.ndarray]] | None = None,
    node_stats: tuple[np.ndarray, np.ndarray] | None = None,
    k_eval: int = 10,
    tau: float = 10.0,
) -> np.ndarray:
    sim_mat = compute_tanimoto_matrix(feats.ecfp_chiral[query_idx], feats.ecfp_chiral[anchor_pool_idx])
    anchor_g_idx, anchor_sims = _select_topk_anchors(sim_mat, anchor_pool_idx, k=k_eval)

    n_q, k_eff = anchor_g_idx.shape
    q_flat = np.repeat(query_idx, k_eff)
    a_flat = anchor_g_idx.reshape(-1)
    s_flat = anchor_sims.reshape(-1)

    d2d_raw, dlf_raw, d3d_raw = _build_modular_pair_features(
        feats, bundle_2d.x_cont_scaled, lfer_atoms, q_flat, a_flat, env_counters
    )
    sc_2d, sc_lf, sc_3d = scales
    d2d = np.clip(d2d_raw / sc_2d, -10.0, 10.0).astype(np.float32)
    dlf = np.clip(dlf_raw / sc_lf, -10.0, 10.0).astype(np.float32)
    d3d = np.clip(d3d_raw / sc_3d, -10.0, 10.0).astype(np.float32)

    model.eval()
    if variant in ("Y_FiLM_2DLFER", "Y_FiLM_Gated3DQM"):
        preds_scaled = []
        with torch.no_grad():
            for start in range(0, q_flat.size, 2048):
                sl = slice(start, start + 2048)
                xi_t = torch.tensor(bundle_2d.x_mol[q_flat[sl]], dtype=torch.float32, device=device)
                xj_t = torch.tensor(bundle_2d.x_mol[a_flat[sl]], dtype=torch.float32, device=device)
                s_t = torch.tensor(s_flat[sl], dtype=torch.float32, device=device)
                d2d_t = torch.tensor(d2d[sl], dtype=torch.float32, device=device)
                dlf_t = torch.tensor(dlf[sl], dtype=torch.float32, device=device)
                if variant == "Y_FiLM_2DLFER":
                    out = model(xi_t, xj_t, d2d_t, s_t, dlf_t).cpu().numpy()
                else:
                    d3d_t = torch.tensor(d3d[sl], dtype=torch.float32, device=device)
                    out = model(xi_t, xj_t, d2d_t, s_t, dlf_t, d3d_t).cpu().numpy()
                preds_scaled.append(out)
        delta_mat = (np.concatenate(preds_scaled, axis=0) * y_std).reshape(n_q, k_eff)
        logits = tau * anchor_sims
        logits = logits - np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return np.sum(weights * (y[anchor_g_idx] + delta_mat), axis=1).astype(np.float64)

    # Query-batched Cross-Attention variants
    d2d_q = d2d.reshape(n_q, k_eff, -1)
    dlf_q = dlf.reshape(n_q, k_eff, -1)
    preds_all = np.zeros((n_q,), dtype=np.float64)
    q_batch = 128

    with torch.no_grad():
        for start in range(0, n_q, q_batch):
            end = min(start + q_batch, n_q)
            q_sub = query_idx[start:end]
            a_sub = anchor_g_idx[start:end]
            s_t = torch.tensor(anchor_sims[start:end], dtype=torch.float32, device=device)
            d2d_t = torch.tensor(d2d_q[start:end], dtype=torch.float32, device=device)
            dlf_t = torch.tensor(dlf_q[start:end], dtype=torch.float32, device=device)

            if variant == "Y_FiLM_CrossAttn":
                xq_t = torch.tensor(bundle_2d.x_mol[q_sub], dtype=torch.float32, device=device)
                xa_t = torch.tensor(bundle_2d.x_mol[a_sub], dtype=torch.float32, device=device)
                delta_mat, w_mat = model.forward_batch(xq_t, xa_t, d2d_t, s_t, dlf_t, tau=tau)
            else:
                assert graphs is not None and node_stats is not None
                uniq_mols, inv = np.unique(np.concatenate([q_sub, a_sub.reshape(-1)]), return_inverse=True)
                q_loc = torch.tensor(inv[: q_sub.size], dtype=torch.long, device=device)
                a_loc = torch.tensor(inv[q_sub.size :].reshape(q_sub.size, k_eff), dtype=torch.long, device=device)
                xn, es, ed, ea, bv, nm = _collate_mol_graphs(graphs, uniq_mols, node_stats[0], node_stats[1], device)
                xmol_u = torch.tensor(bundle_2d.x_mol[uniq_mols], dtype=torch.float32, device=device)
                h_uniq = model.encode_molecules(xmol_u, xn, es, ed, ea, bv, nm)
                delta_mat, w_mat = model.forward_from_embeddings(
                    h_uniq[q_loc], h_uniq[a_loc], d2d_t, s_t, dlf_t, tau=tau
                )

            w_np = w_mat.cpu().numpy()
            d_np = delta_mat.cpu().numpy() * y_std
            preds_all[start:end] = np.sum(w_np * (y[a_sub] + d_np), axis=1)
    return preds_all


def _train_yspace_modular(
    variant: str,
    feats: TargetFeatures,
    graphs: list[dict[str, np.ndarray]],
    lfer_atoms: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    inner_tr_idx: np.ndarray,
    inner_va_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    max_epochs: int = 80,
    lr: float = 8e-4,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = get_device()
    bundle_2d = build_tier_matrices(feats, tier="2d", fit_idx=inner_tr_idx)
    env_counters = precompute_mol_env_counters(feats.atom_ptr, feats.atom_env_hash)

    # Strictly fit graph atom continuous scaler on inner_tr_idx nodes only
    tr_cont = np.concatenate([graphs[m]["x_cont"] for m in inner_tr_idx.tolist()], axis=0)
    node_mean = np.mean(tr_cont, axis=0, keepdims=True).astype(np.float32)
    node_std = np.std(tr_cont, axis=0, keepdims=True).astype(np.float32)
    node_std = np.where(node_std < 1e-6, 1.0, node_std).astype(np.float32)
    node_stats = (node_mean, node_std)

    is_attn_mode = variant in ("Y_FiLM_CrossAttn", "Y_GraphFP_FiLM_Attn")
    k_train = 10 if is_attn_mode else 15
    k_eval = 10

    sim_tr = compute_tanimoto_matrix(feats.ecfp_chiral[inner_tr_idx], feats.ecfp_chiral[inner_tr_idx])
    nbr_g, nbr_sims = _select_topk_anchors(
        sim_tr, inner_tr_idx, k=k_train, exclude_self_global_idx=inner_tr_idx
    )
    pair_i = np.repeat(inner_tr_idx, k_train)
    pair_j = nbr_g.reshape(-1)
    pair_s = nbr_sims.reshape(-1)

    y_std = float(max(np.std(y[inner_tr_idx]), 1e-6))
    pair_dy = ((y[pair_i] - y[pair_j]) / y_std).astype(np.float32)

    d2d_raw, dlf_raw, d3d_raw = _build_modular_pair_features(
        feats, bundle_2d.x_cont_scaled, lfer_atoms, pair_i, pair_j, env_counters
    )
    sc_2d = np.where(np.std(d2d_raw, axis=0) < 1e-6, 1.0, np.std(d2d_raw, axis=0)).astype(np.float32)
    sc_lf = np.where(np.std(dlf_raw, axis=0) < 1e-6, 1.0, np.std(dlf_raw, axis=0)).astype(np.float32)
    sc_3d = np.where(np.std(d3d_raw, axis=0) < 1e-6, 1.0, np.std(d3d_raw, axis=0)).astype(np.float32)
    scales = (sc_2d, sc_lf, sc_3d)

    d2d_tr = np.clip(d2d_raw / sc_2d, -10.0, 10.0).astype(np.float32)
    dlf_tr = np.clip(dlf_raw / sc_lf, -10.0, 10.0).astype(np.float32)
    d3d_tr = np.clip(d3d_raw / sc_3d, -10.0, 10.0).astype(np.float32)

    mol_dim = bundle_2d.x_mol.shape[1]
    if variant == "Y_FiLM_2DLFER":
        model: nn.Module = FiLMKernelResidualDeltaNet(mol_dim=mol_dim, diff2d_dim=26, lfer2d_dim=7).to(device)
    elif variant == "Y_FiLM_Gated3DQM":
        model = FiLMGated3DQMResidualNet(mol_dim=mol_dim, diff2d_dim=26, lfer2d_dim=7, p3d_dim=8).to(device)
    elif variant == "Y_FiLM_CrossAttn":
        model = FiLMAnchorCrossAttnNet(mol_dim=mol_dim, diff2d_dim=26, lfer2d_dim=7).to(device)
    elif variant == "Y_GraphFP_FiLM_Attn":
        model = GraphFPDualFiLMAttnNet(mol_dim=mol_dim, node_dim=44, edge_dim=11, diff2d_dim=26, lfer2d_dim=7).to(device)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    y_va_true = y[inner_va_idx]
    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    patience = 12

    if not is_attn_mode:
        x_mol_all_t = torch.tensor(bundle_2d.x_mol, dtype=torch.float32, device=device)
        pair_i_t = torch.tensor(pair_i, dtype=torch.long, device=device)
        pair_j_t = torch.tensor(pair_j, dtype=torch.long, device=device)
        d2d_t = torch.tensor(d2d_tr, dtype=torch.float32, device=device)
        dlf_t = torch.tensor(dlf_tr, dtype=torch.float32, device=device)
        d3d_t = torch.tensor(d3d_tr, dtype=torch.float32, device=device)
        sim_t = torch.tensor(pair_s, dtype=torch.float32, device=device)
        dy_t = torch.tensor(pair_dy, dtype=torch.float32, device=device)
        n_pairs = pair_i.size
        batch_size = 512

        for _ in range(max_epochs):
            model.train()
            perm = torch.randperm(n_pairs, device=device)
            for start in range(0, n_pairs, batch_size):
                b = perm[start : start + batch_size]
                opt.zero_grad(set_to_none=True)
                xi, xj = x_mol_all_t[pair_i_t[b]], x_mol_all_t[pair_j_t[b]]
                if variant == "Y_FiLM_2DLFER":
                    pred_ij = model(xi, xj, d2d_t[b], sim_t[b], dlf_t[b])
                    loss = loss_fn(pred_ij, dy_t[b])
                else:
                    pred_ij = model(xi, xj, d2d_t[b], sim_t[b], dlf_t[b], d3d_t[b])
                    loss = loss_fn(pred_ij, dy_t[b]) + 5e-3 * torch.mean(torch.sigmoid(model.gate_logits))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()

            y_va_pred = _predict_pair_yspace(
                model, variant, feats, bundle_2d, lfer_atoms, inner_va_idx, inner_tr_idx,
                y, y_std, scales, env_counters, device, graphs=graphs, node_stats=node_stats, k_eval=k_eval,
            )
            va_rmse = float(np.sqrt(np.mean((y_va_true - y_va_pred) ** 2)))
            if va_rmse < best_val_rmse - 1e-5:
                best_val_rmse = va_rmse
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
    else:
        n_q_tr = inner_tr_idx.size
        d2d_q = d2d_tr.reshape(n_q_tr, k_train, -1)
        dlf_q = dlf_tr.reshape(n_q_tr, k_train, -1)
        dy_q = pair_dy.reshape(n_q_tr, k_train)
        q_batch = 64

        x_mol_all_t = torch.tensor(bundle_2d.x_mol, dtype=torch.float32, device=device) if variant == "Y_FiLM_CrossAttn" else None
        rng = np.random.default_rng(seed)

        for _ in range(max_epochs):
            model.train()
            perm_q = rng.permutation(n_q_tr)
            for start in range(0, n_q_tr, q_batch):
                qb = perm_q[start : start + q_batch]
                q_g = inner_tr_idx[qb]
                a_g = nbr_g[qb]
                d2d_b = torch.tensor(d2d_q[qb], dtype=torch.float32, device=device)
                dlf_b = torch.tensor(dlf_q[qb], dtype=torch.float32, device=device)
                sim_b = torch.tensor(nbr_sims[qb], dtype=torch.float32, device=device)
                dy_b = torch.tensor(dy_q[qb], dtype=torch.float32, device=device)

                opt.zero_grad(set_to_none=True)
                if variant == "Y_FiLM_CrossAttn":
                    assert x_mol_all_t is not None
                    xq = x_mol_all_t[torch.tensor(q_g, dtype=torch.long, device=device)]
                    xa = x_mol_all_t[torch.tensor(a_g, dtype=torch.long, device=device)]
                    delta_mat, w_mat = model.forward_batch(xq, xa, d2d_b, sim_b, dlf_b, tau=10.0)
                else:
                    uniq_mols, inv = np.unique(np.concatenate([q_g, a_g.reshape(-1)]), return_inverse=True)
                    q_loc = torch.tensor(inv[: q_g.size], dtype=torch.long, device=device)
                    a_loc = torch.tensor(inv[q_g.size :].reshape(q_g.size, k_train), dtype=torch.long, device=device)
                    xn, es, ed, ea, bv, nm = _collate_mol_graphs(graphs, uniq_mols, node_mean, node_std, device)
                    xmol_u = torch.tensor(bundle_2d.x_mol[uniq_mols], dtype=torch.float32, device=device)
                    h_uniq = model.encode_molecules(xmol_u, xn, es, ed, ea, bv, nm)
                    delta_mat, w_mat = model.forward_from_embeddings(
                        h_uniq[q_loc], h_uniq[a_loc], d2d_b, sim_b, dlf_b, tau=10.0
                    )

                loss_pair = loss_fn(delta_mat.reshape(-1), dy_b.reshape(-1))
                agg_err = torch.sum(w_mat * (delta_mat - dy_b), dim=1)
                loss_agg = loss_fn(agg_err, torch.zeros_like(agg_err))
                loss = 0.5 * loss_pair + 0.5 * loss_agg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()

            y_va_pred = _predict_pair_yspace(
                model, variant, feats, bundle_2d, lfer_atoms, inner_va_idx, inner_tr_idx,
                y, y_std, scales, env_counters, device, graphs=graphs, node_stats=node_stats, k_eval=k_eval,
            )
            va_rmse = float(np.sqrt(np.mean((y_va_true - y_va_pred) ** 2)))
            if va_rmse < best_val_rmse - 1e-5:
                best_val_rmse = va_rmse
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    model.load_state_dict(best_state)
    y_pred_test = _predict_pair_yspace(
        model, variant, feats, bundle_2d, lfer_atoms, test_idx, train_idx,
        y, y_std, scales, env_counters, device, graphs=graphs, node_stats=node_stats, k_eval=k_eval,
    )
    if not np.all(np.isfinite(y_pred_test)):
        raise RuntimeError(f"Non-finite predictions encountered in {variant} (seed={seed})")

    gate_info = {}
    if variant == "Y_FiLM_Gated3DQM":
        with torch.no_grad():
            g = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
            beta = float(torch.abs(model.beta_3d).detach().cpu().item())
        gate_info = {
            "g_aimnet": float(g[0]),
            "g_3dgeom": float(g[1]),
            "g_chiral": float(g[2]),
            "abs_beta_3d": beta,
        }
    return {"y_pred_test": y_pred_test, "gate_info": gate_info, "best_val_rmse": best_val_rmse}


def _evaluate_target(
    target: str,
    seeds: tuple[int, ...],
    force: bool,
    save_cache: bool,
) -> dict[str, Any]:
    out_dir = _out_root() / "per_target"
    json_path = out_dir / f"{target}.json"
    npz_path = out_dir / f"{target}_preds.npz"

    if save_cache and len(seeds) == len(SEEDS) and json_path.exists() and not force:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        if all(m in cached.get("metrics", {}) for m in ALL_MODELS):
            print(f"[CACHE] {target}: loaded existing Option 1+3 checkpoint", flush=True)
            return cached

    t0 = time.time()
    ds = load_target_dataset(target)
    feats = extract_or_load_target_features(target, ds.smiles, prefer_aimnet=True, num_confs=5, n_workers=8)
    lfer_atoms = _lfer_atom_matrix(feats)
    graphs = _extract_target_graphs(ds.smiles, feats)

    pa_npz_path = ROOT_DIR / "refine-logs" / "path_a_modular_results" / "per_target" / f"{target}_preds.npz"
    if not pa_npz_path.exists():
        raise FileNotFoundError(f"Missing Path-A cache: {pa_npz_path}")
    pa = np.load(pa_npz_path)

    y_te = ds.y[ds.test_idx]
    c_te = ds.cliff_mol[ds.test_idx]
    assert np.allclose(y_te, pa["y_te"]) and np.array_equal(c_te, pa["c_te"]), f"Split mismatch on {target}"

    pred_tan, pred_multi = _fit_kernel_bases(feats, ds.y, ds.train_idx, ds.test_idx)

    seed_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in ALL_MODELS}
    seed_preds: dict[str, list[np.ndarray]] = {m: [] for m in ALL_MODELS}
    gate_3dqm_list: list[dict[str, float]] = []

    for seed in seeds:
        s_idx = SEEDS.index(seed)
        inner_tr_rel, inner_va_rel = make_inner_split(ds.y[ds.train_idx], seed=seed, val_frac=0.15)
        inner_tr_idx = ds.train_idx[inner_tr_rel]
        inner_va_idx = ds.train_idx[inner_va_rel]

        y_svm = pa["B1c_RBF_SVM_seeds"][s_idx].astype(np.float64)
        y_cat = pa["B1_CatBoost_seeds"][s_idx].astype(np.float64)
        y_mlp = pa["A1_Direct_MLP_seeds"][s_idx].astype(np.float64)
        y_a2 = pa["A2_2D_Delta_seeds"][s_idx].astype(np.float64)
        y_g2 = pa["G2_Prev_seeds"][s_idx].astype(np.float64)

        out_film = _train_yspace_modular("Y_FiLM_2DLFER", feats, graphs, lfer_atoms, ds.y, ds.train_idx, inner_tr_idx, inner_va_idx, ds.test_idx, seed)
        out_3dqm = _train_yspace_modular("Y_FiLM_Gated3DQM", feats, graphs, lfer_atoms, ds.y, ds.train_idx, inner_tr_idx, inner_va_idx, ds.test_idx, seed)
        out_attn = _train_yspace_modular("Y_FiLM_CrossAttn", feats, graphs, lfer_atoms, ds.y, ds.train_idx, inner_tr_idx, inner_va_idx, ds.test_idx, seed)
        out_gattn = _train_yspace_modular("Y_GraphFP_FiLM_Attn", feats, graphs, lfer_atoms, ds.y, ds.train_idx, inner_tr_idx, inner_va_idx, ds.test_idx, seed)
        gate_3dqm_list.append(out_3dqm["gate_info"])

        y_film = out_film["y_pred_test"]
        y_3dqm = out_3dqm["y_pred_test"]
        y_attn = out_attn["y_pred_test"]
        y_gattn = out_gattn["y_pred_test"]

        preds_map = {
            "K_RBF_SVM": y_svm,
            "K_Tan_SVR": pred_tan,
            "K_Multi_SVR": pred_multi,
            "B1_CatBoost": y_cat,
            "A1_Direct_MLP": y_mlp,
            "G2_Prev": y_g2,
            "Y_A2_2D": y_a2,
            "Y_FiLM_2DLFER": y_film,
            "Y_FiLM_Gated3DQM": y_3dqm,
            "Y_FiLM_CrossAttn": y_attn,
            "Y_GraphFP_FiLM_Attn": y_gattn,
            "E_SVM_CatBoost": 0.5 * y_svm + 0.5 * y_cat,
            "E_SVM_DirectMLP": 0.5 * y_svm + 0.5 * y_mlp,
            "E_Tan_CatBoost": 0.5 * pred_tan + 0.5 * y_cat,
            "E_Tan_DirectMLP": 0.5 * pred_tan + 0.5 * y_mlp,
            "E_Multi_CatBoost": 0.5 * pred_multi + 0.5 * y_cat,
            "E_Multi_DirectMLP": 0.5 * pred_multi + 0.5 * y_mlp,
            "E_SVM_A2Delta": 0.5 * y_svm + 0.5 * y_a2,
            "E_Tan_A2Delta": 0.5 * pred_tan + 0.5 * y_a2,
            "E_Multi_A2Delta": 0.5 * pred_multi + 0.5 * y_a2,
            "E_SVM_FiLM": 0.5 * y_svm + 0.5 * y_film,
            "E_Tan_FiLM": 0.5 * pred_tan + 0.5 * y_film,
            "E_Multi_FiLM": 0.5 * pred_multi + 0.5 * y_film,
            "E_SVM_3DQM": 0.5 * y_svm + 0.5 * y_3dqm,
            "E_Tan_3DQM": 0.5 * pred_tan + 0.5 * y_3dqm,
            "E_Multi_3DQM": 0.5 * pred_multi + 0.5 * y_3dqm,
            "E_SVM_Attn": 0.5 * y_svm + 0.5 * y_attn,
            "E_Tan_Attn": 0.5 * pred_tan + 0.5 * y_attn,
            "E_Multi_Attn": 0.5 * pred_multi + 0.5 * y_attn,
            "E_SVM_GraphAttn": 0.5 * y_svm + 0.5 * y_gattn,
            "E_Tan_GraphAttn": 0.5 * pred_tan + 0.5 * y_gattn,
            "E_Multi_GraphAttn": 0.5 * pred_multi + 0.5 * y_gattn,
        }
        for m_name, yp in preds_map.items():
            seed_metrics[m_name].append(evaluate_predictions(y_te, yp, c_te))
            seed_preds[m_name].append(yp)

    metrics_summary: dict[str, dict[str, float]] = {}
    npz_payload: dict[str, np.ndarray] = {"y_te": y_te, "c_te": c_te}
    for m_name in ALL_MODELS:
        rmse_arr = np.asarray([r["rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        cliff_arr = np.asarray([r["cliff_rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        ncliff_arr = np.asarray([r["non_cliff_rmse"] for r in seed_metrics[m_name]], dtype=np.float64)
        sp_arr = np.asarray([r["spearman"] for r in seed_metrics[m_name]], dtype=np.float64)
        metrics_summary[m_name] = {
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),
            "cliff_rmse_mean": float(np.mean(cliff_arr)),
            "cliff_rmse_std": float(np.std(cliff_arr)),
            "non_cliff_rmse_mean": float(np.mean(ncliff_arr)),
            "spearman_mean": float(np.mean(sp_arr)),
        }
        stacked = np.stack(seed_preds[m_name], axis=0)
        npz_payload[f"{m_name}_seeds"] = stacked
        npz_payload[f"{m_name}_mean"] = np.mean(stacked, axis=0)

    gate_summary = {
        k: float(np.mean([g[k] for g in gate_3dqm_list]))
        for k in ("g_aimnet", "g_3dgeom", "g_chiral", "abs_beta_3d")
    }
    elapsed = round(time.time() - t0, 2)
    result = {
        "target": target,
        "elapsed_sec": elapsed,
        "n_test": int(y_te.size),
        "n_test_cliff": int(np.sum(c_te == 1)),
        "metrics": metrics_summary,
        "gate_3dqm": gate_summary,
    }

    if save_cache and len(seeds) == len(SEEDS):
        np.savez_compressed(npz_path, **npz_payload)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = metrics_summary
    print(
        f"[DONE {elapsed:.1f}s] {target}: "
        f"RBF={m['K_RBF_SVM']['cliff_rmse_mean']:.4f} | Tan={m['K_Tan_SVR']['cliff_rmse_mean']:.4f} | Multi={m['K_Multi_SVR']['cliff_rmse_mean']:.4f} | "
        f"E_SVM_A2={m['E_SVM_A2Delta']['cliff_rmse_mean']:.4f} | "
        f"E_Tan_FiLM={m['E_Tan_FiLM']['cliff_rmse_mean']:.4f} | "
        f"E_Tan_Attn={m['E_Tan_Attn']['cliff_rmse_mean']:.4f} | "
        f"E_Tan_GAttn={m['E_Tan_GraphAttn']['cliff_rmse_mean']:.4f} | "
        f"E_Multi_GAttn={m['E_Multi_GraphAttn']['cliff_rmse_mean']:.4f} "
        f"(g=[{gate_summary['g_aimnet']:.2f},{gate_summary['g_3dgeom']:.2f},{gate_summary['g_chiral']:.2f}], |beta|={gate_summary['abs_beta_3d']:.3f})",
        flush=True,
    )
    return result


def _wilcoxon_pair(df_wide: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    diff = (df_wide[col_a] - df_wide[col_b]).to_numpy(dtype=np.float64)
    wins = int(np.sum(diff < -1e-9))
    ties = int(np.sum(np.abs(diff) <= 1e-9))
    losses = int(np.sum(diff > 1e-9))
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
        "p_value": p,
    }


def _plot_opt13(df_long: pd.DataFrame, df_wide: pd.DataFrame, fig_path: Path) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 6.0))

    bar_models = [
        ("K_RBF_SVM", "RBF-SVM", "#888888"),
        ("K_Tan_SVR", "Tanimoto-SVR", "#56B4E9"),
        ("K_Multi_SVR", "MultiKernel-SVR", "#0072B2"),
        ("Y_A2_2D", "Y_A2_2D", "#BBBBBB"),
        ("Y_FiLM_2DLFER", "Y_FiLM_2DLFER", "#F0E442"),
        ("Y_FiLM_CrossAttn", "Y_FiLM_CrossAttn", "#E69F00"),
        ("Y_GraphFP_FiLM_Attn", "Y_GraphFP_FiLM_Attn", "#CC79A7"),
        ("E_SVM_A2Delta", "E_SVM_A2 (Ref 0.7361)", "#999999"),
        ("E_SVM_FiLM", "E_SVM_FiLM", "#009E73"),
        ("E_Tan_FiLM", "E_Tan_FiLM", "#228B22"),
        ("E_Tan_Attn", "E_Tan_Attn", "#D55E00"),
        ("E_Tan_GraphAttn", "E_Tan_GraphAttn", "#B22222"),
        ("E_Multi_GraphAttn", "E_Multi_GraphAttn", "#6A0DAD"),
    ]
    agg = df_long.groupby("model")[["cliff_rmse_mean", "rmse_mean"]].mean()
    x = np.arange(len(bar_models))
    w = 0.38
    cliff_vals = [float(agg.loc[m, "cliff_rmse_mean"]) for m, _, _ in bar_models]
    all_vals = [float(agg.loc[m, "rmse_mean"]) for m, _, _ in bar_models]
    cols = [c for _, _, c in bar_models]
    lbls = [l for _, l, _ in bar_models]

    ax1.bar(x - w / 2, cliff_vals, width=w, color=cols, edgecolor="black", linewidth=0.5, label="Cliff RMSE")
    ax1.bar(x + w / 2, all_vals, width=w, color=cols, alpha=0.45, edgecolor="black", linewidth=0.5, hatch="//", label="Overall RMSE")
    ax1.axhline(float(agg.loc["E_SVM_A2Delta", "cliff_rmse_mean"]), color="black", linestyle="--", linewidth=0.9, label="E_SVM_A2 Ref")
    for xi, cv in zip(x, cliff_vals):
        ax1.text(xi - w / 2, cv + 0.002, f"{cv:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=50)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, rotation=32, ha="right", fontsize=7.5)
    ax1.set_ylim(0.58, max(cliff_vals) + 0.03)
    ax1.set_ylabel("10-Target Mean RMSE (Lower = Better)")
    ax1.set_title("Option 1 + Option 3 Modular Ablation (10 Wave-1 Targets)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")

    # Right panel: Per-target Cliff RMSE gain vs E_SVM_A2Delta for best Opt-1 (E_SVM_FiLM / E_SVM_Attn) & Opt-1+3 (E_Tan_Attn / E_Multi_GraphAttn)
    cand_cols = [
        ("E_SVM_FiLM", "E_SVM_FiLM - Ref", "#009E73"),
        ("E_Tan_Attn", "E_Tan_Attn - Ref", "#D55E00"),
        ("E_Multi_GraphAttn", "E_Multi_GraphAttn - Ref", "#6A0DAD"),
    ]
    # Pick whichever has lowest mean cliff RMSE among Opt1+3 blends to guarantee best candidate is shown
    blend_means = {m: float(agg.loc[m, "cliff_rmse_mean"]) for m in ALL_MODELS if m.startswith("E_")}
    best_opt1_svm = min(["E_SVM_FiLM", "E_SVM_3DQM", "E_SVM_Attn", "E_SVM_GraphAttn"], key=lambda k: blend_means[k])
    best_opt13 = min([k for k in blend_means if k.startswith("E_Tan_") or k.startswith("E_Multi_")], key=lambda k: blend_means[k])

    d1 = (df_wide[f"{best_opt1_svm}_cliff"] - df_wide["E_SVM_A2Delta_cliff"]).to_numpy()
    d2 = (df_wide[f"{best_opt13}_cliff"] - df_wide["E_SVM_A2Delta_cliff"]).to_numpy()
    order = np.argsort(d2)
    targets_s = df_wide["target"].to_numpy()[order]
    d1_s = d1[order]
    d2_s = d2[order]

    yp = np.arange(len(targets_s))
    bh = 0.35
    ax2.barh(yp - bh / 2, d1_s, height=bh, color="#009E73", label=f"{best_opt1_svm} - E_SVM_A2")
    ax2.barh(yp + bh / 2, d2_s, height=bh, color="#B22222", label=f"{best_opt13} - E_SVM_A2")
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_yticks(yp)
    ax2.set_yticklabels(targets_s, fontsize=8)
    ax2.set_xlabel("Per-Target Delta Cliff RMSE vs E_SVM_A2Delta (< 0 = Better)")
    ax2.set_title("Per-Target Gains of Best Option-1 and Option-1+3 Blends", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Option 1 + Option 3 Composable Modular Evaluation.")
    parser.add_argument("--targets", type=str, default="", help="Optional comma-separated target names.")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seeds (must be subset of 42..46).")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()

    target_list = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else list(WAVE_1_TARGETS)
    seeds_tuple = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip()) if args.seeds.strip() else SEEDS
    save_cache = not args.no_save

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting Option 1+3 Modular Eval "
        f"({len(target_list)} targets, seeds={seeds_tuple}, save_cache={save_cache})",
        flush=True,
    )
    t0 = time.time()
    results = [_evaluate_target(tgt, seeds=seeds_tuple, force=args.force, save_cache=save_cache) for tgt in target_list]
    elapsed = time.time() - t0

    if not (save_cache and len(results) == len(WAVE_1_TARGETS) and len(seeds_tuple) == len(SEEDS)):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Partial/smoke run finished in {elapsed:.1f}s.", flush=True)
        return

    long_rows = []
    wide_rows = []
    for r in results:
        tgt = r["target"]
        wrow: dict[str, Any] = {"target": tgt, **r["gate_3dqm"]}
        for m_name in ALL_MODELS:
            mm = r["metrics"][m_name]
            long_rows.append({"target": tgt, "model": m_name, **mm})
            wrow[f"{m_name}_cliff"] = mm["cliff_rmse_mean"]
            wrow[f"{m_name}_rmse"] = mm["rmse_mean"]
        wide_rows.append(wrow)

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)
    csv_path = _out_root() / "opt13_summary.csv"
    df_long.to_csv(csv_path, index=False)

    fig_path = _fig_dir() / "opt13_modular_comparison.png"
    _plot_opt13(df_long, df_wide, fig_path)

    agg = df_long.groupby("model")[["rmse_mean", "cliff_rmse_mean", "non_cliff_rmse_mean", "spearman_mean"]].mean()
    md_path = ROOT_DIR / "refine-logs" / "OPT13_MODULAR_EVAL.md"
    lines = [
        "# Option 1 + Option 3: Composable Modular Architecture Evaluation (10 Targets x 5 Seeds)",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Targets**: {len(results)} | **Elapsed**: {elapsed:.1f}s",
        "",
        "## 1. Kernel Base Comparison (Option 3A)",
        "",
        "| Kernel Model | Overall RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for m in KERNEL_MODELS:
        r = agg.loc[m]
        lines.append(f"| `{m}` | {r['rmse_mean']:.4f} | {r['cliff_rmse_mean']:.4f} | {r['non_cliff_rmse_mean']:.4f} | {r['spearman_mean']:.4f} |")

    lines.extend([
        "",
        "| Kernel Wilcoxon Comparison (N=10) | Win / Tie / Loss | Mean Diff | Median Diff | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in [
        ("Cliff: K_Tan_SVR vs K_RBF_SVM", "K_Tan_SVR_cliff", "K_RBF_SVM_cliff"),
        ("Cliff: K_Multi_SVR vs K_RBF_SVM", "K_Multi_SVR_cliff", "K_RBF_SVM_cliff"),
        ("Cliff: K_Multi_SVR vs K_Tan_SVR", "K_Multi_SVR_cliff", "K_Tan_SVR_cliff"),
        ("Overall: K_Tan_SVR vs K_RBF_SVM", "K_Tan_SVR_rmse", "K_RBF_SVM_rmse"),
        ("Overall: K_Multi_SVR vs K_RBF_SVM", "K_Multi_SVR_rmse", "K_RBF_SVM_rmse"),
    ]:
        st = _wilcoxon_pair(df_wide, ca, cb)
        lines.append(f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | {st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | `{st['p_value']:.4e}` |")

    lines.extend([
        "",
        "## 2. Standalone y-Space Modular Delta Comparison (Option 1 + Option 3B GINE)",
        "",
        "| Delta Model | Overall RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for m in DELTA_MODELS:
        r = agg.loc[m]
        lines.append(f"| `{m}` | {r['rmse_mean']:.4f} | {r['cliff_rmse_mean']:.4f} | {r['non_cliff_rmse_mean']:.4f} | {r['spearman_mean']:.4f} |")

    lines.extend([
        "",
        "| Standalone Delta Wilcoxon (N=10) | Win / Tie / Loss | Mean Diff | Median Diff | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in [
        ("Cliff: Y_FiLM_2DLFER vs Y_A2_2D", "Y_FiLM_2DLFER_cliff", "Y_A2_2D_cliff"),
        ("Cliff: Y_FiLM_Gated3DQM vs Y_FiLM_2DLFER", "Y_FiLM_Gated3DQM_cliff", "Y_FiLM_2DLFER_cliff"),
        ("Cliff: Y_FiLM_CrossAttn vs Y_FiLM_2DLFER", "Y_FiLM_CrossAttn_cliff", "Y_FiLM_2DLFER_cliff"),
        ("Cliff: Y_GraphFP_FiLM_Attn vs Y_FiLM_CrossAttn", "Y_GraphFP_FiLM_Attn_cliff", "Y_FiLM_CrossAttn_cliff"),
        ("Cliff: Y_GraphFP_FiLM_Attn vs Y_A2_2D", "Y_GraphFP_FiLM_Attn_cliff", "Y_A2_2D_cliff"),
        ("Overall: Y_FiLM_2DLFER vs Y_A2_2D", "Y_FiLM_2DLFER_rmse", "Y_A2_2D_rmse"),
        ("Overall: Y_GraphFP_FiLM_Attn vs Y_A2_2D", "Y_GraphFP_FiLM_Attn_rmse", "Y_A2_2D_rmse"),
    ]:
        st = _wilcoxon_pair(df_wide, ca, cb)
        lines.append(f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | {st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | `{st['p_value']:.4e}` |")

    lines.extend([
        "",
        "### Learned 3D/QM Gate Shrinkage in y-Space (`Y_FiLM_Gated3DQM`)",
        "",
        "| Target | g_aimnet | g_3dgeom | g_chiral | |beta_3d| | Y_FiLM_2DLFER Cliff | Y_FiLM_Gated3DQM Cliff |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in results:
        g = r["gate_3dqm"]
        mm = r["metrics"]
        lines.append(
            f"| `{r['target']}` | {g['g_aimnet']:.3f} | {g['g_3dgeom']:.3f} | {g['g_chiral']:.3f} | "
            f"{g['abs_beta_3d']:.4f} | {mm['Y_FiLM_2DLFER']['cliff_rmse_mean']:.4f} | {mm['Y_FiLM_Gated3DQM']['cliff_rmse_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Full `0.5 * Kernel + 0.5 * Model` Blend Leaderboard (Sorted by Cliff RMSE)",
        "",
        "| Blend / Baseline | Overall RMSE | Cliff RMSE | Non-Cliff RMSE | Spearman |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    blend_names = [m for m in ALL_MODELS if m.startswith("E_")] + ["G2_Prev", "K_RBF_SVM", "K_Tan_SVR", "K_Multi_SVR"]
    blend_sorted = sorted(blend_names, key=lambda m: float(agg.loc[m, "cliff_rmse_mean"]))
    for m in blend_sorted:
        r = agg.loc[m]
        lines.append(f"| `{m}` | {r['rmse_mean']:.4f} | {r['cliff_rmse_mean']:.4f} | {r['non_cliff_rmse_mean']:.4f} | {r['spearman_mean']:.4f} |")

    best_delta_blend = min([m for m in ALL_MODELS if m.startswith("E_") and "CatBoost" not in m and "DirectMLP" not in m], key=lambda m: float(agg.loc[m, "cliff_rmse_mean"]))
    best_opt1_svm = min(["E_SVM_FiLM", "E_SVM_3DQM", "E_SVM_Attn", "E_SVM_GraphAttn"], key=lambda m: float(agg.loc[m, "cliff_rmse_mean"]))
    best_non_delta_tan = "E_Tan_DirectMLP" if float(agg.loc["E_Tan_DirectMLP", "cliff_rmse_mean"]) <= float(agg.loc["E_Tan_CatBoost", "cliff_rmse_mean"]) else "E_Tan_CatBoost"
    best_non_delta_multi = "E_Multi_DirectMLP" if float(agg.loc["E_Multi_DirectMLP", "cliff_rmse_mean"]) <= float(agg.loc["E_Multi_CatBoost", "cliff_rmse_mean"]) else "E_Multi_CatBoost"

    lines.extend([
        "",
        "| Blend Cross-Target Wilcoxon (N=10) | Win / Tie / Loss | Mean Diff | Median Diff | Two-Sided p |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for lbl, ca, cb in [
        (f"Cliff: {best_delta_blend} vs E_SVM_A2Delta", f"{best_delta_blend}_cliff", "E_SVM_A2Delta_cliff"),
        (f"Cliff: {best_delta_blend} vs K_RBF_SVM", f"{best_delta_blend}_cliff", "K_RBF_SVM_cliff"),
        (f"Cliff: {best_delta_blend} vs K_Tan_SVR", f"{best_delta_blend}_cliff", "K_Tan_SVR_cliff"),
        (f"Cliff: {best_delta_blend} vs K_Multi_SVR", f"{best_delta_blend}_cliff", "K_Multi_SVR_cliff"),
        (f"Cliff: {best_delta_blend} vs {best_non_delta_tan}", f"{best_delta_blend}_cliff", f"{best_non_delta_tan}_cliff"),
        (f"Cliff: {best_delta_blend} vs {best_non_delta_multi}", f"{best_delta_blend}_cliff", f"{best_non_delta_multi}_cliff"),
        (f"Cliff: {best_opt1_svm} vs E_SVM_A2Delta", f"{best_opt1_svm}_cliff", "E_SVM_A2Delta_cliff"),
        ("Cliff: E_Tan_FiLM vs E_Tan_A2Delta", "E_Tan_FiLM_cliff", "E_Tan_A2Delta_cliff"),
        ("Cliff: E_Tan_Attn vs E_Tan_A2Delta", "E_Tan_Attn_cliff", "E_Tan_A2Delta_cliff"),
        ("Cliff: E_Tan_GraphAttn vs E_Tan_A2Delta", "E_Tan_GraphAttn_cliff", "E_Tan_A2Delta_cliff"),
        ("Cliff: E_Multi_GraphAttn vs E_Multi_A2Delta", "E_Multi_GraphAttn_cliff", "E_Multi_A2Delta_cliff"),
        (f"Overall: {best_delta_blend} vs E_SVM_A2Delta", f"{best_delta_blend}_rmse", "E_SVM_A2Delta_rmse"),
        (f"Overall: {best_delta_blend} vs K_Tan_SVR", f"{best_delta_blend}_rmse", "K_Tan_SVR_rmse"),
        (f"Overall: {best_delta_blend} vs K_Multi_SVR", f"{best_delta_blend}_rmse", "K_Multi_SVR_rmse"),
    ]:
        st = _wilcoxon_pair(df_wide, ca, cb)
        lines.append(f"| {lbl} | {st['wins']} / {st['ties']} / {st['losses']} | {st['mean_diff']:+.5f} | {st['median_diff']:+.5f} | `{st['p_value']:.4e}` |")

    top_show = ["K_RBF_SVM", "K_Tan_SVR", "K_Multi_SVR", "E_SVM_A2Delta", best_opt1_svm, "E_Tan_FiLM", "E_Tan_Attn", "E_Tan_GraphAttn", "E_Multi_GraphAttn"]
    top_show = list(dict.fromkeys(top_show))
    lines.extend([
        "",
        "## 4. Per-Target Cliff RMSE Matrix",
        "",
        "| Target | " + " | ".join(top_show) + " |",
        "| :--- | " + " | ".join([":---:"] * len(top_show)) + " |",
    ])
    for r in results:
        mm = r["metrics"]
        vals = " | ".join(f"{mm[m]['cliff_rmse_mean']:.4f}" for m in top_show)
        lines.append(f"| `{r['target']}` | {vals} |")

    lines.extend([
        "",
        f"Summary CSV: `{csv_path.as_posix()}`",
        f"Figure: `{fig_path.as_posix()}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {md_path}", flush=True)


if __name__ == "__main__":
    main()
