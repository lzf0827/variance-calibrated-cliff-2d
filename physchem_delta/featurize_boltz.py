"""Boltzmann-weighted multi-conformer 3D geometry and AIMNet2 atomic charge featurizer."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from ._mp_rdkit import run_isolated_rdkit_pool

from .featurize import (
    AIMNET2_SUPPORTED_Z,
    TargetFeatures,
    _sanitize_array,
    detect_aimnet2_calculator,
    extract_or_load_target_features,
)

RDLogger.DisableLog("rdApp.*")

KB_T_300K_KCAL = 0.59616  # R * 300 K in kcal/mol


@dataclass
class BoltzmannTargetFeatures:
    base: TargetFeatures
    mol_thermo: np.ndarray        # (N, 4): [e_span_raw, boltz_s_conf, boltz_n_eff, boltz_e_std]
    atom_boltz_lfer: np.ndarray   # (Total_heavy, 9): [boltz_q_mean, boltz_q_std, crip_logp, crip_mr, tpsa_c, 3d_dist_mean, 3d_crowd_mean, 3d_std, chiral_flag]
    meta: dict[str, Any]


def get_boltz_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "feature_cache_boltz"


def _embed_multi_confs(
    mol: Chem.Mol,
    seed: int,
    num_confs: int = 5,
    max_iters: int = 150,
    timeout_sec: int = 2,
    num_threads: int = 1,
) -> tuple[Chem.Mol, list[tuple[int, float]], bool]:
    """Generate and MMFF94/UFF-optimize up to 5 conformers with tight iteration/timeout caps."""
    mol_h = Chem.AddHs(mol, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.maxIterations = max_iters
    params.timeout = timeout_sec
    params.numThreads = num_threads

    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))
    if not cids:
        params.useRandomCoords = True
        params.maxIterations = 100
        params.timeout = timeout_sec
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))

    if not cids:
        AllChem.Compute2DCoords(mol_h)
        return mol_h, [(0, 0.0)], True

    energies: list[tuple[int, float]] = []
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=num_threads, maxIters=150)
            energies = [(cid, float(e)) for cid, (_conv, e) in zip(cids, res) if np.isfinite(e)]
    except Exception:
        energies = []

    if not energies:
        try:
            res = AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=num_threads, maxIters=150)
            energies = [(cid, float(e)) for cid, (_conv, e) in zip(cids, res) if np.isfinite(e)]
        except Exception:
            energies = []

    if not energies:
        return mol_h, [(cids[0], 0.0)], False

    energies.sort(key=lambda x: x[1])
    return mol_h, energies, False


def _conf_heavy_geom(mol_h: Chem.Mol, cid: int, n_orig: int) -> np.ndarray:
    """Compute [dist_to_centroid, mean_nearest3_dist] for original heavy atoms on conformer cid."""
    geom = np.zeros((n_orig, 2), dtype=np.float32)
    if mol_h.GetNumConformers() == 0 or n_orig == 0:
        return geom
    conf = mol_h.GetConformer(cid)
    coords = np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n_orig)],
        dtype=np.float32,
    )
    centroid = coords.mean(axis=0, keepdims=True)
    geom[:, 0] = np.linalg.norm(coords - centroid, axis=1)
    if n_orig > 1:
        pdist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(pdist, np.inf)
        k = min(3, n_orig - 1)
        nearest_k = np.partition(pdist, k - 1, axis=1)[:, :k]
        geom[:, 1] = np.mean(nearest_k, axis=1)
    return _sanitize_array(geom)


def _boltz_single_mol_worker(task: tuple[int, str, int, int, float, bool]) -> dict[str, Any]:
    """Windows-spawn-safe worker generating multi-conformer RDKit geometry + AIMNet2 inputs."""
    idx, smi, seed, num_confs, energy_window_kcal, prepare_aimnet = task
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES at {idx}: {smi}")

    n_orig = mol.GetNumAtoms()
    mol_h, energies, failed = _embed_multi_confs(
        mol, seed=seed + idx, num_confs=num_confs, num_threads=1
    )

    min_e = energies[0][1]
    max_e = energies[-1][1]
    e_span_raw = float(max(0.0, max_e - min_e))

    kept = [(cid, e - min_e) for cid, e in energies if (e - min_e) <= energy_window_kcal]
    if not kept:
        kept = [(energies[0][0], 0.0)]

    rel_e = np.asarray([de for _cid, de in kept], dtype=np.float64)
    boltz_unnorm = np.exp(-np.clip(rel_e / KB_T_300K_KCAL, 0.0, 50.0))
    weights = boltz_unnorm / np.sum(boltz_unnorm)

    s_conf = float(-np.sum(weights * np.log(weights + 1e-12)))
    n_eff = float(np.exp(s_conf))
    e_mean = float(np.sum(weights * rel_e))
    e_std = float(np.sqrt(np.sum(weights * (rel_e - e_mean) ** 2)))
    thermo = _sanitize_array(np.asarray([e_span_raw, s_conf, n_eff, e_std], dtype=np.float32))

    geom_confs = np.stack([_conf_heavy_geom(mol_h, cid, n_orig) for cid, _ in kept], axis=0)
    w_3d = weights[:, None, None].astype(np.float32)
    g_mean = np.sum(w_3d * geom_confs, axis=0)
    g_var = np.sum(w_3d * (geom_confs - g_mean[None, :, :]) ** 2, axis=0)
    g_std = np.sqrt(np.sum(g_var, axis=1))

    aimnet_payload = None
    if prepare_aimnet and mol_h.GetNumConformers() > 0:
        nums = np.asarray([a.GetAtomicNum() for a in mol_h.GetAtoms()], dtype=np.int64)
        if set(nums.tolist()).issubset(AIMNET2_SUPPORTED_Z):
            heavy_idx = list(range(n_orig))
            h_nbrs = [
                [nbr.GetIdx() for nbr in mol_h.GetAtomWithIdx(hi).GetNeighbors() if nbr.GetIdx() >= n_orig]
                for hi in heavy_idx
            ]
            coords_confs = []
            for cid, _ in kept:
                conf = mol_h.GetConformer(cid)
                coords_confs.append(
                    np.asarray(
                        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol_h.GetNumAtoms())],
                        dtype=np.float32,
                    )
                )
            aimnet_payload = (coords_confs, nums, float(Chem.GetFormalCharge(mol)), heavy_idx, h_nbrs)

    return {
        "idx": idx,
        "failed": failed,
        "n_valid_confs": len(energies),
        "n_kept_confs": len(kept),
        "weights": weights.astype(np.float32),
        "thermo": thermo,
        "g_mean": g_mean.astype(np.float32),
        "g_std": g_std.astype(np.float32),
        "aimnet_payload": aimnet_payload,
    }


def extract_or_load_boltzmann_features(
    target: str,
    smiles_list: list[str],
    num_confs: int = 5,
    energy_window_kcal: float = 5.0,
    seed: int = 42,
    cache_dir: Path | None = None,
    base_cache_dir: Path | None = None,
    n_workers: int | None = None,
) -> BoltzmannTargetFeatures:
    """Compute or load Boltzmann-weighted multi-conformer 3D + AIMNet2 atomic LFER features."""
    base = extract_or_load_target_features(
        target,
        smiles_list,
        cache_dir=base_cache_dir,
        prefer_aimnet=True,
        num_confs=num_confs,
        seed=seed,
        n_workers=n_workers,
    )
    cdir = cache_dir or get_boltz_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    npz_path = cdir / f"{target}_boltz.npz"
    meta_path = cdir / f"{target}_boltz_meta.json"

    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n_mols") == len(smiles_list):
            data = np.load(npz_path)
            return BoltzmannTargetFeatures(
                base=base,
                mol_thermo=data["mol_thermo"],
                atom_boltz_lfer=data["atom_boltz_lfer"],
                meta=meta,
            )

    t0 = time.time()
    aimnet_calc, backend = detect_aimnet2_calculator()
    import torch

    n = len(smiles_list)
    if n_workers is None:
        n_workers = max(1, min((os.cpu_count() or 4) - 2, 8)) if n >= 64 else 1

    tasks = [(idx, smi, seed, num_confs, energy_window_kcal, aimnet_calc is not None) for idx, smi in enumerate(smiles_list)]
    if n_workers > 1:
        worker_results = run_isolated_rdkit_pool("boltz", tasks, n_workers)
    else:
        worker_results = [_boltz_single_mol_worker(t) for t in tasks]

    mol_thermo = np.zeros((n, 4), dtype=np.float32)
    atom_boltz_chunks: list[np.ndarray] = []

    embed_fail_count = 0
    aimnet_fail_count = 0
    total_valid_confs = 0
    total_kept_confs = 0

    for idx, res in enumerate(worker_results):
        if res["failed"]:
            embed_fail_count += 1
        total_valid_confs += res["n_valid_confs"]
        total_kept_confs += res["n_kept_confs"]
        mol_thermo[idx, :] = res["thermo"]
        weights = res["weights"]

        s_ptr, e_ptr = int(base.atom_ptr[idx]), int(base.atom_ptr[idx + 1])
        base_elec = base.atom_elec_props[s_ptr:e_ptr]
        base_3d = base.atom_3d_props[s_ptr:e_ptr]

        if base_elec.shape[1] >= 8:
            fallback_q = base_elec[:, 1]
            crip_logp = base_elec[:, 4]
            crip_mr = base_elec[:, 5]
            tpsa_c = base_elec[:, 6]
        else:
            fallback_q = base_elec[:, 1]
            crip_logp = base_elec[:, 2]
            crip_mr = base_elec[:, 3]
            tpsa_c = base_elec[:, 4]
        chiral_flag = base_3d[:, 2]

        q_confs_list: list[np.ndarray] = []
        payload = res["aimnet_payload"]
        if aimnet_calc is not None and payload is not None:
            coords_confs, nums, net_q, heavy_idx, h_nbrs = payload
            calc_dev = getattr(aimnet_calc, "device", "cpu")
            nums_t = torch.tensor(nums, dtype=torch.long, device=calc_dev)
            chg_t = torch.tensor(net_q, dtype=torch.float32, device=calc_dev)
            for coords in coords_confs:
                try:
                    out = aimnet_calc(
                        {
                            "coord": torch.tensor(coords, dtype=torch.float32, device=calc_dev),
                            "numbers": nums_t,
                            "charge": chg_t,
                        }
                    )
                    q_all = out["charges"].detach().cpu().numpy().reshape(-1)
                    q_eff = np.asarray([q_all[hi] + np.sum(q_all[hn]) for hi, hn in zip(heavy_idx, h_nbrs)], dtype=np.float32)
                    if np.all(np.isfinite(q_eff)):
                        q_confs_list.append(q_eff)
                except Exception:
                    break

        n_kept = res["n_kept_confs"]
        if len(q_confs_list) == n_kept:
            q_confs = np.stack(q_confs_list, axis=0)
        else:
            aimnet_fail_count += 1
            q_confs = np.repeat(fallback_q[None, :], n_kept, axis=0)

        w_col = weights[:, None].astype(np.float32)
        q_mean = np.sum(w_col * q_confs, axis=0)
        q_std = np.sqrt(np.sum(w_col * (q_confs - q_mean[None, :]) ** 2, axis=0))

        g_mean = res["g_mean"]
        g_std = res["g_std"]
        atom_chunk = np.stack(
            [
                q_mean,
                q_std,
                crip_logp,
                crip_mr,
                tpsa_c,
                g_mean[:, 0],
                g_mean[:, 1],
                g_std,
                chiral_flag,
            ],
            axis=1,
        )
        atom_boltz_chunks.append(_sanitize_array(atom_chunk))

        if (idx + 1) % 300 == 0 or (idx + 1) == n:
            print(
                f"  [{target}] Boltzmann multi-conf + AIMNet2 (workers={n_workers}): {idx + 1}/{n} ({time.time() - t0:.1f}s)",
                flush=True,
            )

    atom_boltz_lfer = np.concatenate(atom_boltz_chunks, axis=0)
    elapsed = round(time.time() - t0, 2)
    meta = {
        "target": target,
        "n_mols": n,
        "electronic_backend": backend,
        "num_confs_requested": num_confs,
        "energy_window_kcal": energy_window_kcal,
        "mean_valid_confs": round(total_valid_confs / max(n, 1), 2),
        "mean_kept_confs": round(total_kept_confs / max(n, 1), 2),
        "embed_fallback_count": int(embed_fail_count),
        "aimnet_fallback_count": int(aimnet_fail_count),
        "elapsed_sec": elapsed,
    }
    np.savez_compressed(npz_path, mol_thermo=mol_thermo, atom_boltz_lfer=atom_boltz_lfer)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return BoltzmannTargetFeatures(
        base=base,
        mol_thermo=mol_thermo,
        atom_boltz_lfer=atom_boltz_lfer,
        meta=meta,
    )
