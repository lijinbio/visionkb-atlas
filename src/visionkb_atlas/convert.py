"""h5ad -> TileDB-SOMA Experiment, streaming X in CSR row blocks.

``tiledbsoma.io.from_h5ad`` (2.3.0) fails with anndata 0.13 in backed mode (``layers`` holds a ``None`` key that
``sanitize_key`` rejects), so X is written here block by block: obs/var go through ``from_anndata`` on an
X-less AnnData, then measurement ``RNA`` gets the sparse layer ``counts`` (float32; values are integer counts).
Peak memory is O(block nnz), not O(matrix).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from .h5ad_io import CSRReader, file_fingerprint, read_obs_var, x_info

MEASUREMENT = "RNA"
COUNTS = "counts"


def versions() -> dict:
    from importlib.metadata import version

    from . import __version__

    out = {"visionkb-atlas": __version__}
    for p in ("tiledbsoma", "anndata", "h5py", "numpy", "pyarrow"):
        out[p] = version(p)
    return out


def _clean_obs(obs):
    """String-typed object columns and a plain string index, as SOMA needs."""
    import pandas as pd

    obs = obs.copy()
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].astype(str)
    obs.index = pd.Index([str(v) for v in obs.index], name=None)
    return obs


def convert(h5ad: str | Path, uri: str | Path, *, goal_nnz: int = 100_000_000, sha256: bool = True, log=print) -> dict:
    import anndata as ad
    import tiledbsoma as soma
    import tiledbsoma.io as sio

    uri = Path(uri)
    if uri.exists():
        raise FileExistsError(f"{uri} exists; remove it to re-convert")
    t0 = time.time()
    info = x_info(h5ad)
    n_obs, n_var = info["shape"]
    obs, var = read_obs_var(h5ad)
    if obs.shape[0] != n_obs or var.shape[0] != n_var:
        raise ValueError("obs/var length does not match X shape")
    if not var.index.is_unique:
        raise ValueError("var index is not unique")
    log(f"obs {obs.shape}, var {var.shape}, X nnz {info['nnz']:,} ({info['dtype']})")
    shell = ad.AnnData(obs=_clean_obs(obs), var=var.copy())
    tmp = uri.with_name(uri.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    sio.from_anndata(str(tmp), shell, MEASUREMENT, uns_keys=[])
    del shell, obs
    written = 0
    nonint = 0
    with soma.Experiment.open(str(tmp), "w") as exp:
        X = exp.ms[MEASUREMENT].X
        arr = X.add_new_sparse_ndarray(COUNTS, type=pa.float32(), shape=(n_obs, n_var))
        with CSRReader(h5ad) as rd:
            for a, b in rd.blocks(goal_nnz):
                m = rd.rows(a, b)
                coo = m.tocoo()
                data = coo.data
                nonint += int(np.count_nonzero(data != np.round(data)))
                tbl = pa.Table.from_pydict(
                    {
                        "soma_dim_0": pa.array(coo.row.astype(np.int64) + a),
                        "soma_dim_1": pa.array(coo.col.astype(np.int64)),
                        "soma_data": pa.array(data.astype(np.float32)),
                    }
                )
                arr.write(tbl)
                written += coo.nnz
                log(f"rows {a:,}-{b:,} nnz {coo.nnz:,} total {written:,} ({time.time() - t0:.0f}s)")
                del m, coo, tbl, data
        exp.metadata.update({"visionkb_counts_layer": COUNTS, "visionkb_source_h5ad_nnz": info["nnz"]})
    if written != info["nnz"]:
        raise RuntimeError(f"wrote {written} values, h5ad has {info['nnz']}")
    tmp.rename(uri)
    src = file_fingerprint(h5ad, sha256=sha256)
    src.update(
        n_obs=n_obs,
        n_var=n_var,
        nnz=info["nnz"],
        x_dtype=info["dtype"],
        n_noninteger_values=nonint,
        soma_uri=str(uri),
        measurement=MEASUREMENT,
        x_layer=COUNTS,
        converter="visionkb_atlas.convert (h5py CSR row blocks)",
        goal_nnz_per_block=goal_nnz,
        seconds=round(time.time() - t0, 1),
        tool_versions=versions(),
    )
    return src


def write_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
