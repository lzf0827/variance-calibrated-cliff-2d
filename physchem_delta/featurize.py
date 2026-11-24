"""Three-tier molecular featurization and fast local substituent perturbation deltas."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.EState import EState
from rdkit.Chem import rdFingerprintGenerator

from ._mp_rdkit import run_isolated_rdkit_pool

RDLogger.DisableLog("rdApp.*")

# Elements natively covered by standard AIMNet2 (wB97M-D3) neutral/closed-shell parameters
AIMNET2_SUPPORTED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}


@dataclass
class TargetFeatures:
    target: str
    electronic_backend: str
    ecfp_nochiral: np.ndarray   # (N, 2048) float32
    ecfp_chiral: np.ndarray     # (N, 2048) float32
    maccs: np.ndarray           # (N, 167) float32
    desc2d: np.ndarray          # (N, D2) float32
    feat_3d: np.ndarray         # (N, D3) float32
    feat_elec: np.ndarray       # (N, De) float32
    atom_ptr: np.ndarray        # (N + 1,) int64
    atom_env_hash: np.ndarray   # (Total_heavy,) int64
    atom_3d_props: np.ndarray   # (Total_heavy, 3) float32
    atom_elec_props: np.ndarray # (Total_heavy, Da) float32
    meta: dict[str, Any]

    @property
    def feat_2d(self) -> np.ndarray:
        return np.concatenate([self.ecfp_chiral, self.maccs, self.desc2d], axis=1)


def get_feature_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "feature_cache"


def _morgan_bit_array(mol: Chem.Mol, use_chirality: bool, n_bits: int = 2048) -> np.ndarray:
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=n_bits, includeChirality=use_chirality
        )
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=n_bits, useChirality=use_chirality
        )
        arr = np.zeros((n_bits,), dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr


def _maccs_array(mol: Chem.Mol) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.float32)
    for idx in fp.GetOnBits():
        if idx < 167:
            arr[idx] = 1.0
    return arr


def _sanitize_array(arr: np.ndarray, clip: float = 1e6) -> np.ndarray:
    out = np.nan_to_num(arr, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(out, -clip, clip).astype(np.float32)


def _compute_desc2d(mol: Chem.Mol) -> np.ndarray:
    vals = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumRings(mol),
        float(mol.GetNumHeavyAtoms()),
        rdMolDescriptors.CalcLabuteASA(mol),
        float(Chem.GetFormalCharge(mol)),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.Kappa3(mol),
        Descriptors.Chi0v(mol),
        Descriptors.Chi1v(mol),
        Descriptors.NumValenceElectrons(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        Descriptors.NOCount(mol),
    ]
    return _sanitize_array(np.asarray(vals, dtype=np.float32))


def _atom_env_hashes(mol: Chem.Mol) -> np.ndarray:
    """Compute canonical radius-1 chiral atom-environment hashes for each heavy atom."""
    hashes = []
    for atom in mol.GetAtoms():
        nbrs = sorted(
            (
                int(b.GetBondTypeAsDouble() * 2),
                n.GetAtomicNum(),
                n.GetTotalDegree(),
                int(n.GetIsAromatic()),
                n.GetTotalNumHs(),
            )
            for b in atom.GetBonds()
            for n in [b.GetOtherAtom(atom)]
        )
        sig = (
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            int(atom.GetChiralTag()),
            tuple(nbrs),
        )
        digest = hashlib.blake2b(repr(sig).encode("utf-8"), digest_size=8).digest()
        hashes.append(int.from_bytes(digest, byteorder="little", signed=True))
    return np.asarray(hashes, dtype=np.int64)


def _embed_and_optimize_3d(
    mol: Chem.Mol,
    seed: int,
    num_confs: int = 5,
    max_iters: int = 150,
    timeout_sec: int = 2,
    num_threads: int = 1,
) -> tuple[Chem.Mol, int, float, float, bool]:
    """Generate 3D conformers via ETKDGv3 + MMFF94/UFF. Returns (mol_h, best_cid, min_e, e_span, failed)."""
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
        return mol_h, 0, 0.0, 0.0, True

    energies: list[tuple[int, float]] = []
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=num_threads, maxIters=150)
            energies = [(cid, float(e)) for cid, (conv, e) in zip(cids, res) if np.isfinite(e)]
    except Exception:
        energies = []

    if not energies:
        try:
            res = AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=num_threads, maxIters=150)
            energies = [(cid, float(e)) for cid, (conv, e) in zip(cids, res) if np.isfinite(e)]
        except Exception:
            energies = []

    if not energies:
        return mol_h, cids[0], 0.0, 0.0, False

    energies.sort(key=lambda x: x[1])
    best_cid, min_e = energies[0]
    e_span = float(energies[-1][1] - min_e)
    return mol_h, best_cid, float(min_e), float(e_span), False


def _extract_3d_descriptors(
    mol_no_h: Chem.Mol,
    mol_h: Chem.Mol,
    best_cid: int,
    min_e: float,
    e_span: float,
    embed_failed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return molecule-level 3D descriptors and per-heavy-atom 3D descriptors."""
    n_heavy = max(mol_no_h.GetNumHeavyAtoms(), 1)
    # Build a heavy-atom copy preserving the lowest-energy 3D conformer
    mol_3d_heavy = Chem.RemoveHs(mol_h, sanitize=False)
    conf_id = best_cid if mol_3d_heavy.GetNumConformers() > best_cid else 0

    try:
        pmi1 = rdMolDescriptors.CalcPMI1(mol_h, confId=best_cid)
        pmi2 = rdMolDescriptors.CalcPMI2(mol_h, confId=best_cid)
        pmi3 = rdMolDescriptors.CalcPMI3(mol_h, confId=best_cid)
        npr1 = rdMolDescriptors.CalcNPR1(mol_h, confId=best_cid)
        npr2 = rdMolDescriptors.CalcNPR2(mol_h, confId=best_cid)
        asph = rdMolDescriptors.CalcAsphericity(mol_h, confId=best_cid)
        ecc = rdMolDescriptors.CalcEccentricity(mol_h, confId=best_cid)
        isf = rdMolDescriptors.CalcInertialShapeFactor(mol_h, confId=best_cid)
        rog = rdMolDescriptors.CalcRadiusOfGyration(mol_h, confId=best_cid)
        sph = rdMolDescriptors.CalcSpherocityIndex(mol_h, confId=best_cid)
    except Exception:
        pmi1 = pmi2 = pmi3 = npr1 = npr2 = asph = ecc = isf = rog = sph = 0.0

    chiral_assigned = len(Chem.FindMolChiralCenters(mol_no_h, includeUnassigned=False))
    chiral_all = len(Chem.FindMolChiralCenters(mol_no_h, includeUnassigned=True))
    ez_count = sum(
        1
        for b in mol_no_h.GetBonds()
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
    )

    mol_vec = _sanitize_array(
        np.asarray(
            [
                min_e / n_heavy,
                e_span,
                pmi1,
                pmi2,
                pmi3,
                npr1,
                npr2,
                asph,
                ecc,
                isf,
                rog,
                sph,
                float(chiral_assigned),
                float(chiral_all),
                float(ez_count),
                float(embed_failed),
            ],
            dtype=np.float32,
        )
    )

    # Per-heavy-atom 3D properties: [dist_to_centroid, mean_nearest3_dist, chiral_flag]
    n_atoms_no_h = mol_no_h.GetNumAtoms()
    atom_3d = np.zeros((n_atoms_no_h, 3), dtype=np.float32)
    if mol_h.GetNumConformers() > 0 and n_atoms_no_h > 0:
        conf = mol_h.GetConformer(best_cid)
        # AddHs appends hydrogens after original atoms [0 .. n_atoms_no_h - 1]
        orig_indices = list(range(n_atoms_no_h))
        coords = np.asarray(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in orig_indices],
            dtype=np.float32,
        )
        centroid = coords.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(coords - centroid, axis=1)
        atom_3d[:, 0] = dists
        if n_atoms_no_h > 1:
            pdist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            np.fill_diagonal(pdist, np.inf)
            k = min(3, n_atoms_no_h - 1)
            nearest_k = np.partition(pdist, k - 1, axis=1)[:, :k]
            atom_3d[:, 1] = np.mean(nearest_k, axis=1)
    for a in mol_no_h.GetAtoms():
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            atom_3d[a.GetIdx(), 2] = 1.0

    return mol_vec, _sanitize_array(atom_3d)


def _summarize_atom_vector(vec: np.ndarray) -> list[float]:
    if vec.size == 0:
        return [0.0] * 10
    v = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    sorted_v = np.sort(v)
    top3_neg = float(np.mean(sorted_v[: min(3, v.size)]))
    top3_pos = float(np.mean(sorted_v[-min(3, v.size) :]))
    return [
        float(np.mean(v)),
        float(np.std(v)),
        float(np.min(v)),
        float(np.max(v)),
        float(np.percentile(v, 10)),
        float(np.percentile(v, 90)),
        float(np.sum(np.abs(v))),
        top3_neg,
        top3_pos,
        float(np.sum(v > 0) - np.sum(v < 0)),
    ]


def _rdkit_electronic_for_mol(mol_no_h: Chem.Mol, mol_h: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    """Compute verified RDKit Gasteiger + Crippen + TPSA + EState electronic features."""
    mol_h_copy = Chem.Mol(mol_h)
    try:
        AllChem.ComputeGasteigerCharges(mol_h_copy)
        h_charges = np.asarray(
            [float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0 for a in mol_h_copy.GetAtoms()],
            dtype=np.float32,
        )
        h_charges = np.nan_to_num(h_charges, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        h_charges = np.zeros((mol_h_copy.GetNumAtoms(),), dtype=np.float32)

    n_heavy = mol_no_h.GetNumAtoms()
    gast_raw = np.zeros((n_heavy,), dtype=np.float32)
    gast_eff = np.zeros((n_heavy,), dtype=np.float32)
    if mol_h_copy.GetNumAtoms() >= n_heavy:
        for local_i in range(n_heavy):
            q_self = float(h_charges[local_i])
            atom_h = mol_h_copy.GetAtomWithIdx(local_i)
            q_hs = sum(
                float(h_charges[nbr.GetIdx()])
                for nbr in atom_h.GetNeighbors()
                if nbr.GetIdx() >= n_heavy
            )
            gast_raw[local_i] = q_self
            gast_eff[local_i] = q_self + q_hs

    try:
        crip = np.asarray(rdMolDescriptors._CalcCrippenContribs(mol_no_h), dtype=np.float32)
        crip_logp = crip[:, 0]
        crip_mr = crip[:, 1]
    except Exception:
        crip_logp = np.zeros((n_heavy,), dtype=np.float32)
        crip_mr = np.zeros((n_heavy,), dtype=np.float32)

    try:
        tpsa_c = np.asarray(rdMolDescriptors._CalcTPSAContribs(mol_no_h), dtype=np.float32)
    except Exception:
        tpsa_c = np.zeros((n_heavy,), dtype=np.float32)

    try:
        estate = np.asarray(EState.EStateIndices(mol_no_h), dtype=np.float32)
    except Exception:
        estate = np.zeros((n_heavy,), dtype=np.float32)

    atom_elec = np.stack([gast_raw, gast_eff, crip_logp, crip_mr, tpsa_c, estate], axis=1)
    atom_elec = _sanitize_array(atom_elec)

    mol_summary: list[float] = []
    for col in range(atom_elec.shape[1]):
        mol_summary.extend(_summarize_atom_vector(atom_elec[:, col]))
    # Global polarity ratios
    mol_summary.append(float(np.sum(np.abs(gast_raw[tpsa_c > 0]))))
    mol_summary.append(float(Crippen.MolLogP(mol_no_h)))
    mol_summary.append(float(Crippen.MolMR(mol_no_h)))

    return _sanitize_array(np.asarray(mol_summary, dtype=np.float32)), atom_elec


def detect_aimnet2_calculator() -> tuple[Any | None, str]:
    """Attempt to initialize AIMNet2 (trying CUDA first, then CPU) without altering torch."""
    try:
        import torch
        from aimnet.calculators import AIMNet2Calculator

        for dev in ("cuda", "cpu"):
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            try:
                calc = AIMNet2Calculator("aimnet2", device=dev)
                calc_dev = getattr(calc, "device", dev)
                water = {
                    "coord": torch.tensor(
                        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
                        dtype=torch.float32,
                        device=calc_dev,
                    ),
                    "numbers": torch.tensor([8, 1, 1], dtype=torch.long, device=calc_dev),
                    "charge": torch.tensor(0.0, dtype=torch.float32, device=calc_dev),
                }
                out = calc(water)
                e = float(out["energy"].detach().cpu().item())
                q = out["charges"].detach().cpu().numpy()
                if np.isfinite(e) and np.all(np.isfinite(q)):
                    return calc, "aimnet2_wb97m_d3"
            except Exception:
                continue
        return None, "rdkit_gasteiger_crippen_estate"
    except Exception:
        return None, "rdkit_gasteiger_crippen_estate"


def _featurize_single_mol_worker(task: tuple[int, str, int, int, bool]) -> dict[str, Any]:
    """Windows-spawn-safe worker computing RDKit 2D, 3D conformer, and RDKit electronic props."""
    idx, smi, seed, num_confs, prepare_aimnet = task
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES at index {idx}: {smi}")

    ecfp_nc = _morgan_bit_array(mol, use_chirality=False)
    ecfp_c = _morgan_bit_array(mol, use_chirality=True)
    maccs = _maccs_array(mol)
    desc2d = _compute_desc2d(mol)
    env_h = _atom_env_hashes(mol)

    mol_h, best_cid, min_e, e_span, failed = _embed_and_optimize_3d(
        mol, seed=seed + idx, num_confs=num_confs, num_threads=1
    )
    f3d_vec, a3d = _extract_3d_descriptors(mol, mol_h, best_cid, min_e, e_span, failed)
    felec_r, aelec_r = _rdkit_electronic_for_mol(mol, mol_h)

    aimnet_payload = None
    if prepare_aimnet and mol_h.GetNumConformers() > 0:
        nums = np.asarray([a.GetAtomicNum() for a in mol_h.GetAtoms()], dtype=np.int64)
        if set(nums.tolist()).issubset(AIMNET2_SUPPORTED_Z):
            conf = mol_h.GetConformer(best_cid)
            coords = np.asarray(
                [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol_h.GetNumAtoms())],
                dtype=np.float32,
            )
            n_orig = mol.GetNumAtoms()
            heavy_idx = list(range(n_orig))
            h_nbrs = [
                [nbr.GetIdx() for nbr in mol_h.GetAtomWithIdx(hi).GetNeighbors() if nbr.GetIdx() >= n_orig]
                for hi in heavy_idx
            ]
            aimnet_payload = (coords, nums, float(Chem.GetFormalCharge(mol)), heavy_idx, h_nbrs)

    return {
        "idx": idx,
        "ecfp_nc": ecfp_nc,
        "ecfp_c": ecfp_c,
        "maccs": maccs,
        "desc2d": desc2d,
        "env_h": env_h,
        "f3d": f3d_vec,
        "a3d": a3d,
        "felec_r": felec_r,
        "aelec_r": aelec_r,
        "n_atoms": mol.GetNumAtoms(),
        "failed": failed,
        "aimnet_payload": aimnet_payload,
    }


def extract_or_load_target_features(
    target: str,
    smiles_list: list[str],
    cache_dir: Path | None = None,
    prefer_aimnet: bool = True,
    num_confs: int = 5,
    seed: int = 42,
    n_workers: int | None = None,
) -> TargetFeatures:
    """Compute or load cached 3-tier molecular features for a MoleculeACE target."""
    cache_dir = cache_dir or get_feature_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{target}_features.npz"
    meta_path = cache_dir / f"{target}_meta.json"

    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n_mols") == len(smiles_list):
            data = np.load(npz_path)
            return TargetFeatures(
                target=target,
                electronic_backend=meta["electronic_backend"],
                ecfp_nochiral=data["ecfp_nochiral"],
                ecfp_chiral=data["ecfp_chiral"],
                maccs=data["maccs"],
                desc2d=data["desc2d"],
                feat_3d=data["feat_3d"],
                feat_elec=data["feat_elec"],
                atom_ptr=data["atom_ptr"],
                atom_env_hash=data["atom_env_hash"],
                atom_3d_props=data["atom_3d_props"],
                atom_elec_props=data["atom_elec_props"],
                meta=meta,
            )

    t0 = time.time()
    aimnet_calc, backend = (detect_aimnet2_calculator() if prefer_aimnet else (None, "rdkit_gasteiger_crippen_estate"))

    n = len(smiles_list)
    if n_workers is None:
        n_workers = max(1, min((os.cpu_count() or 4) - 2, 8)) if n >= 64 else 1

    tasks = [(idx, smi, seed, num_confs, aimnet_calc is not None) for idx, smi in enumerate(smiles_list)]
    ecfp_nc_list: list[np.ndarray] = []
    ecfp_c_list: list[np.ndarray] = []
    maccs_list: list[np.ndarray] = []
    desc2d_list: list[np.ndarray] = []
    f3d_list: list[np.ndarray] = []
    felec_rdkit_list: list[np.ndarray] = []

    atom_ptr = [0]
    atom_env_chunks: list[np.ndarray] = []
    atom_3d_chunks: list[np.ndarray] = []
    atom_elec_rdkit_chunks: list[np.ndarray] = []

    embed_fail_count = 0
    aimnet_inputs: list[tuple[int, np.ndarray, np.ndarray, float, list[int], list[list[int]]]] = []

    if n_workers > 1:
        worker_results = run_isolated_rdkit_pool("single", tasks, n_workers)
    else:
        worker_results = [_featurize_single_mol_worker(task) for task in tasks]

    for count, res in enumerate(worker_results, start=1):
        ecfp_nc_list.append(res["ecfp_nc"])
        ecfp_c_list.append(res["ecfp_c"])
        maccs_list.append(res["maccs"])
        desc2d_list.append(res["desc2d"])
        atom_env_chunks.append(res["env_h"])
        f3d_list.append(res["f3d"])
        atom_3d_chunks.append(res["a3d"])
        felec_rdkit_list.append(res["felec_r"])
        atom_elec_rdkit_chunks.append(res["aelec_r"])
        atom_ptr.append(atom_ptr[-1] + res["n_atoms"])
        if res["failed"]:
            embed_fail_count += 1
        if res["aimnet_payload"] is not None:
            aimnet_inputs.append((res["idx"], *res["aimnet_payload"]))
        if count % 300 == 0 or count == n:
            print(f"  [{target}] 3D conformers + RDKit props (workers={n_workers}): {count}/{n} ({time.time() - t0:.1f}s)", flush=True)

    # Optional AIMNet2 enrichment if verified active
    aimnet_fallback_count = 0
    if aimnet_calc is not None:
        import torch

        aimnet_mol_feats = np.zeros((n, 14), dtype=np.float32)
        aimnet_atom_qs: list[np.ndarray | None] = [None] * n
        supported_map = {item[0]: item for item in aimnet_inputs}
        for idx in range(n):
            if idx not in supported_map:
                aimnet_fallback_count += 1
                continue
            _, coords, nums, net_q, heavy_idx, h_nbrs = supported_map[idx]
            try:
                calc_dev = getattr(aimnet_calc, "device", "cpu")
                out = aimnet_calc(
                    {
                        "coord": torch.tensor(coords, dtype=torch.float32, device=calc_dev),
                        "numbers": torch.tensor(nums, dtype=torch.long, device=calc_dev),
                        "charge": torch.tensor(net_q, dtype=torch.float32, device=calc_dev),
                    }
                )
                e_ev = float(out["energy"].detach().cpu().item())
                q_all = out["charges"].detach().cpu().numpy().reshape(-1)
                if "dipole" in out:
                    dip = out["dipole"].detach().cpu().numpy().reshape(-1)
                else:
                    centered = coords - coords.mean(axis=0, keepdims=True)
                    dip = np.sum(q_all[:, None] * centered, axis=0)
                dip_norm = float(np.linalg.norm(dip))
                q_heavy = q_all[heavy_idx]
                q_eff = np.asarray([q_all[hi] + np.sum(q_all[hn]) for hi, hn in zip(heavy_idx, h_nbrs)], dtype=np.float32)
                if np.isfinite(e_ev) and np.all(np.isfinite(q_eff)):
                    n_heavy = max(len(heavy_idx), 1)
                    aimnet_mol_feats[idx, :] = _sanitize_array(
                        np.asarray(
                            [
                                e_ev / n_heavy,
                                dip_norm,
                                *_summarize_atom_vector(q_heavy),
                                float(np.min(q_eff)),
                                float(np.max(q_eff)),
                            ],
                            dtype=np.float32,
                        )
                    )
                    aimnet_atom_qs[idx] = np.stack([q_heavy.astype(np.float32), q_eff], axis=1)
                else:
                    aimnet_fallback_count += 1
            except Exception:
                aimnet_fallback_count += 1

        if aimnet_fallback_count == n:
            backend = "rdkit_gasteiger_crippen_estate"
            feat_elec = np.stack(felec_rdkit_list, axis=0)
            atom_elec_props = np.concatenate(atom_elec_rdkit_chunks, axis=0)
        else:
            feat_elec = np.concatenate([aimnet_mol_feats, np.stack(felec_rdkit_list, axis=0)], axis=1)
            merged_atom_elec = []
            for idx in range(n):
                rd_a = atom_elec_rdkit_chunks[idx]
                aq = aimnet_atom_qs[idx]
                if aq is None or aq.shape[0] != rd_a.shape[0]:
                    aq = rd_a[:, :2]
                merged_atom_elec.append(np.concatenate([aq, rd_a], axis=1))
            atom_elec_props = np.concatenate(merged_atom_elec, axis=0)
    else:
        feat_elec = np.stack(felec_rdkit_list, axis=0)
        atom_elec_props = np.concatenate(atom_elec_rdkit_chunks, axis=0)

    ecfp_nochiral = np.stack(ecfp_nc_list, axis=0)
    ecfp_chiral = np.stack(ecfp_c_list, axis=0)
    maccs = np.stack(maccs_list, axis=0)
    desc2d = np.stack(desc2d_list, axis=0)
    feat_3d = np.stack(f3d_list, axis=0)
    atom_ptr_arr = np.asarray(atom_ptr, dtype=np.int64)
    atom_env_hash = np.concatenate(atom_env_chunks, axis=0)
    atom_3d_props = np.concatenate(atom_3d_chunks, axis=0)

    elapsed = round(time.time() - t0, 2)
    meta = {
        "target": target,
        "n_mols": n,
        "electronic_backend": backend,
        "embed_fallback_count": int(embed_fail_count),
        "aimnet_fallback_count": int(aimnet_fallback_count),
        "dims": {
            "ecfp": int(ecfp_chiral.shape[1]),
            "maccs": int(maccs.shape[1]),
            "desc2d": int(desc2d.shape[1]),
            "feat_3d": int(feat_3d.shape[1]),
            "feat_elec": int(feat_elec.shape[1]),
            "atom_3d": int(atom_3d_props.shape[1]),
            "atom_elec": int(atom_elec_props.shape[1]),
        },
        "elapsed_sec": elapsed,
    }

    np.savez_compressed(
        npz_path,
        ecfp_nochiral=ecfp_nochiral,
        ecfp_chiral=ecfp_chiral,
        maccs=maccs,
        desc2d=desc2d,
        feat_3d=feat_3d,
        feat_elec=feat_elec,
        atom_ptr=atom_ptr_arr,
        atom_env_hash=atom_env_hash,
        atom_3d_props=atom_3d_props,
        atom_elec_props=atom_elec_props,
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TargetFeatures(
        target=target,
        electronic_backend=backend,
        ecfp_nochiral=ecfp_nochiral,
        ecfp_chiral=ecfp_chiral,
        maccs=maccs,
        desc2d=desc2d,
        feat_3d=feat_3d,
        feat_elec=feat_elec,
        atom_ptr=atom_ptr_arr,
        atom_env_hash=atom_env_hash,
        atom_3d_props=atom_3d_props,
        atom_elec_props=atom_elec_props,
        meta=meta,
    )


def precompute_mol_env_counters(atom_ptr: np.ndarray, atom_env_hash: np.ndarray) -> list[Counter[int]]:
    """Precompute atom-environment multisets for fast local perturbation matching."""
    n = atom_ptr.size - 1
    return [Counter(atom_env_hash[atom_ptr[i] : atom_ptr[i + 1]].tolist()) for i in range(n)]


def compute_local_pair_deltas(
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    atom_ptr: np.ndarray,
    atom_env_hash: np.ndarray,
    atom_props: np.ndarray,
    env_counters: list[Counter[int]] | None = None,
) -> np.ndarray:
    """
    Compute antisymmetric local substituent perturbation deltas for pairs (i, j).
    For each pair, match shared circular atom-environment hashes and summarize
    the unmatched (modified) substituent atoms in i vs j.
    Output shape: (len(i_indices), 3 * D_atom + 1), strictly antisymmetric under i <-> j.
    """
    if env_counters is None:
        env_counters = precompute_mol_env_counters(atom_ptr, atom_env_hash)

    d_atom = atom_props.shape[1]
    out = np.zeros((i_indices.size, 3 * d_atom + 1), dtype=np.float32)

    for p_idx, (i, j) in enumerate(zip(i_indices.tolist(), j_indices.tolist())):
        if i == j:
            continue
        ci = env_counters[i]
        cj = env_counters[j]
        shared = ci & cj

        # Unmatched atoms in i (added/changed substituent in i)
        rem_i = dict(shared)
        mask_i = []
        for h in atom_env_hash[atom_ptr[i] : atom_ptr[i + 1]].tolist():
            if rem_i.get(h, 0) > 0:
                rem_i[h] -= 1
                mask_i.append(False)
            else:
                mask_i.append(True)

        rem_j = dict(shared)
        mask_j = []
        for h in atom_env_hash[atom_ptr[j] : atom_ptr[j + 1]].tolist():
            if rem_j.get(h, 0) > 0:
                rem_j[h] -= 1
                mask_j.append(False)
            else:
                mask_j.append(True)

        props_i = atom_props[atom_ptr[i] : atom_ptr[i + 1]]
        props_j = atom_props[atom_ptr[j] : atom_ptr[j + 1]]

        diff_i = props_i[np.asarray(mask_i, dtype=bool)] if any(mask_i) else np.zeros((0, d_atom), dtype=np.float32)
        diff_j = props_j[np.asarray(mask_j, dtype=bool)] if any(mask_j) else np.zeros((0, d_atom), dtype=np.float32)

        sum_i = diff_i.sum(axis=0) if diff_i.shape[0] > 0 else np.zeros((d_atom,), dtype=np.float32)
        sum_j = diff_j.sum(axis=0) if diff_j.shape[0] > 0 else np.zeros((d_atom,), dtype=np.float32)
        mean_i = diff_i.mean(axis=0) if diff_i.shape[0] > 0 else np.zeros((d_atom,), dtype=np.float32)
        mean_j = diff_j.mean(axis=0) if diff_j.shape[0] > 0 else np.zeros((d_atom,), dtype=np.float32)
        maxabs_i = np.max(np.abs(diff_i), axis=0) if diff_i.shape[0] > 0 else np.zeros((d_atom,), dtype=np.float32)
        maxabs_j = np.max(np.abs(diff_j), axis=0) if diff_j.shape[0] > 0 else np.zeros((d_atom,), dtype=np.float32)

        out[p_idx, :d_atom] = sum_i - sum_j
        out[p_idx, d_atom : 2 * d_atom] = mean_i - mean_j
        out[p_idx, 2 * d_atom : 3 * d_atom] = maxabs_i - maxabs_j
        out[p_idx, 3 * d_atom] = float(diff_i.shape[0] - diff_j.shape[0])

    return _sanitize_array(out)
