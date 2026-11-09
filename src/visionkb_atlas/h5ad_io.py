"""Cheap h5ad access with h5py: obs/var metadata and CSR row blocks, never the whole matrix."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_obs_var(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """obs and var data frames via anndata backed mode (handles legacy encodings; X is not read)."""
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    try:
        return a.obs.copy(), a.var.copy()
    finally:
        a.file.close()


def x_info(path: str | Path) -> dict:
    with h5py.File(path, "r") as h:
        x = h["X"]
        enc = x.attrs.get("encoding-type")
        if isinstance(enc, bytes):
            enc = enc.decode()
        if enc != "csr_matrix":
            raise ValueError(f"{path}: X must be CSR, found {enc}")
        shape = tuple(int(v) for v in x.attrs["shape"])
        return {"shape": shape, "nnz": int(x["data"].shape[0]), "dtype": str(x["data"].dtype)}


class CSRReader:
    """Row-block reader over an h5ad CSR X; memory O(block nnz)."""

    def __init__(self, path: str | Path):
        self.h = h5py.File(path, "r")
        self.x = self.h["X"]
        self.shape = tuple(int(v) for v in self.x.attrs["shape"])
        self.indptr = self.x["indptr"][:].astype(np.int64)

    def rows(self, a: int, b: int) -> sp.csr_matrix:
        s, e = int(self.indptr[a]), int(self.indptr[b])
        data = self.x["data"][s:e]
        idx = self.x["indices"][s:e]
        ptr = self.indptr[a : b + 1] - s
        return sp.csr_matrix((data, idx, ptr), shape=(b - a, self.shape[1]))

    def blocks(self, goal_nnz: int = 200_000_000):
        """Yield (start, stop) row ranges holding about goal_nnz stored values each."""
        n = self.shape[0]
        a = 0
        while a < n:
            target = self.indptr[a] + goal_nnz
            b = int(np.searchsorted(self.indptr, target, side="right")) - 1
            b = min(max(b, a + 1), n)
            yield a, b
            a = b

    def close(self):
        self.h.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def file_fingerprint(path: str | Path, sha256: bool = True, bufsize: int = 16 << 20) -> dict:
    st = os.stat(path)
    out = {
        "path": str(path),
        "bytes": st.st_size,
        "mtime_utc": pd.Timestamp(st.st_mtime, unit="s", tz="UTC").isoformat(),
    }
    if sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(bufsize):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out
