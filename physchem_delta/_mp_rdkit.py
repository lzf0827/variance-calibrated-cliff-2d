"""Isolated subprocess coordinator for RDKit molecule-level multiprocessing.

On Windows (spawn), ProcessPoolExecutor re-executes the caller's __main__ script
inside every worker. Launching the worker pool from this clean RDKit-only module
prevents child workers from importing torch, scipy, sklearn, or aimnet/warp.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_isolated_rdkit_pool(mode: str, tasks: list[Any], n_workers: int) -> list[Any]:
    """Run RDKit worker tasks inside a clean subprocess pool without heavy ML imports."""
    with tempfile.TemporaryDirectory(prefix="rdkit_mp_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        in_pkl = tmp_path / "in.pkl"
        out_pkl = tmp_path / "out.pkl"
        in_pkl.write_bytes(pickle.dumps((mode, tasks, int(n_workers)), protocol=pickle.HIGHEST_PROTOCOL))

        env = os.environ.copy()
        prev_pp = env.get("PYTHONPATH", "")
        root_str = str(ROOT_DIR)
        env["PYTHONPATH"] = f"{root_str}{os.pathsep}{prev_pp}" if prev_pp else root_str

        subprocess.run(
            [sys.executable, "-m", "physchem_delta._mp_rdkit", str(in_pkl), str(out_pkl)],
            cwd=root_str,
            env=env,
            check=True,
        )
        return pickle.loads(out_pkl.read_bytes())


def _run_worker_main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m physchem_delta._mp_rdkit <in.pkl> <out.pkl>")
    in_pkl = Path(sys.argv[1])
    out_pkl = Path(sys.argv[2])

    mode, tasks, n_workers = pickle.loads(in_pkl.read_bytes())
    if mode == "single":
        from physchem_delta.featurize import _featurize_single_mol_worker as worker_fn
    elif mode == "boltz":
        from physchem_delta.featurize_boltz import _boltz_single_mol_worker as worker_fn
    else:
        raise ValueError(f"Unsupported isolated RDKit pool mode: {mode}")

    with ProcessPoolExecutor(max_workers=int(n_workers)) as pool:
        results = list(pool.map(worker_fn, tasks, chunksize=16))

    out_pkl.write_bytes(pickle.dumps(results, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    _run_worker_main()
