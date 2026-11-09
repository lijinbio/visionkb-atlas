"""Streamed per gene x group expression summaries from a SOMA experiment (contracts §3).

Cells are read in obs blocks; for each level the block's counts are aggregated into (group, donor) combinations
with sparse indicator products, so memory is O(combinations x genes), never O(cells x genes). All statistics are
additive over donors, so group values are sums over the group's donor combinations.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import Atlas
from .convert import COUNTS, MEASUREMENT

DONOR_EXPR_MIN_PCT = 0.10
MISSING = {"", "nan", "NaN", "None", "NA", "N/A"}


def _labels(s: pd.Series) -> pd.Series:
    """Object labels with missing values as None (including literal 'nan' categories)."""
    v = s.astype(object).where(s.notna(), None)
    return v.map(lambda x: None if x is None or str(x).strip() in MISSING else str(x))


def group_table(obs: pd.DataFrame, atlas: Atlas) -> dict[str, pd.Series]:
    """Per level, the group label of each cell (None = excluded)."""
    mc = _labels(obs[atlas.majorclass_col])
    out = {"majorclass": mc}
    if atlas.celltype_col:
        out["celltype"] = _labels(obs[atlas.celltype_col])
    if atlas.region_col:
        reg = _labels(obs[atlas.region_col]).map(lambda x: None if x is None else atlas.map_label("region", x))
        reg = reg.map(lambda x: None if x is None or str(x) in MISSING else x)
        rm = pd.Series([f"{r}|{m}" if r is not None and m is not None else None for r, m in zip(reg, mc, strict=True)])
        rm.index = obs.index
        out["region_majorclass"] = rm
    return out


class _Level:
    def __init__(self, name: str, groups: pd.Series, donors: pd.Series, samples: pd.Series, n_var: int):
        self.name = name
        g = groups.to_numpy(dtype=object)
        d = donors.to_numpy(dtype=object)
        valid = np.array([x is not None for x in g])
        self.group_labels = sorted({x for x in g[valid]})
        gidx = {k: i for i, k in enumerate(self.group_labels)}
        dnames = sorted({x for x in d if x is not None})
        didx = {k: i for i, k in enumerate(dnames)}
        gi = np.array([gidx[x] if x is not None else -1 for x in g], dtype=np.int64)
        di = np.array([didx[x] if x is not None else len(dnames) for x in d], dtype=np.int64)  # missing donor bucket
        key = np.where(valid, gi * (len(dnames) + 1) + di, -1)
        uniq, inv = np.unique(key[valid], return_inverse=True)
        self.combo = np.full(len(g), -1, dtype=np.int64)
        self.combo[valid] = inv
        self.K = len(uniq)
        self.combo_group = (uniq // (len(dnames) + 1)).astype(np.int64)
        self.combo_donor_known = (uniq % (len(dnames) + 1)) < len(dnames)
        self.combo_cells = np.bincount(inv, minlength=self.K).astype(np.int64)
        self.n_excluded = int((~valid).sum())
        # per group: cells, donors, samples
        sv = samples.to_numpy(dtype=object)
        self.group_cells = np.bincount(gi[valid], minlength=len(self.group_labels))
        self.group_donors = np.zeros(len(self.group_labels), dtype=np.int64)
        np.add.at(self.group_donors, self.combo_group[self.combo_donor_known], 1)
        gs = pd.DataFrame({"g": gi[valid], "s": sv[valid]}).dropna()
        self.group_samples = (
            gs.groupby("g")["s"].nunique().reindex(range(len(self.group_labels)), fill_value=0).to_numpy()
        )
        self.n_expr = np.zeros((self.K, n_var), dtype=np.int32)
        self.sum_log = np.zeros((self.K, n_var), dtype=np.float64)
        self.sum_cnt = np.zeros((self.K, n_var), dtype=np.float64)
        self.combo_total = np.zeros(self.K, dtype=np.float64)

    def add(self, a: int, xb: sp.csr_matrix, xbin: sp.csr_matrix, xlog: sp.csr_matrix, totals: np.ndarray):
        c = self.combo[a : a + xb.shape[0]]
        ok = c >= 0
        if not ok.any():
            return
        rows = np.nonzero(ok)[0]
        G = sp.csr_matrix((np.ones(len(rows)), (c[ok], rows)), shape=(self.K, xb.shape[0]))
        for acc, m in ((self.n_expr, xbin), (self.sum_log, xlog), (self.sum_cnt, xb)):
            r = (G @ m).tocoo()
            r.sum_duplicates()
            vals = r.data if acc.dtype == np.float64 else np.rint(r.data).astype(acc.dtype)
            acc[r.row, r.col] += vals
        np.add.at(self.combo_total, c[ok], totals[ok])

    def result(self, var_symbols: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        ng = len(self.group_labels)
        M = sp.csr_matrix((np.ones(self.K), (self.combo_group, np.arange(self.K))), shape=(ng, self.K))
        n_expr = np.asarray(M @ self.n_expr)
        sum_log = np.asarray(M @ self.sum_log)
        sum_cnt = np.asarray(M @ self.sum_cnt)
        total = np.asarray(M @ self.combo_total)
        frac = self.n_expr / self.combo_cells[:, None]
        ind = (frac >= DONOR_EXPR_MIN_PCT) & self.combo_donor_known[:, None]
        n_de = np.asarray(M @ ind.astype(np.int64))
        cells = self.group_cells.astype(np.float64)
        nvar = len(var_symbols)
        with np.errstate(invalid="ignore", divide="ignore"):
            pb = np.where(total[:, None] > 0, sum_cnt / total[:, None] * 1e4, 0.0)
        df = pd.DataFrame(
            {
                "level": self.name,
                "group_label": np.repeat(np.array(self.group_labels, dtype=object), nvar),
                "var_symbol": np.tile(var_symbols.astype(object), ng),
                "n_cells": np.repeat(self.group_cells, nvar).astype(np.int64),
                "n_donors": np.repeat(self.group_donors, nvar).astype(np.int64),
                "n_expressing": n_expr.ravel().astype(np.int64),
                "pct_expressing": (n_expr / cells[:, None]).ravel(),
                "mean_log1p_cp10k": (sum_log / cells[:, None]).ravel(),
                "pseudobulk_cp10k": pb.ravel(),
                "n_donors_expressing": n_de.ravel().astype(np.int64),
            }
        )
        groups = pd.DataFrame(
            {
                "level": self.name,
                "group_label": self.group_labels,
                "n_cells": self.group_cells.astype(np.int64),
                "n_donors": self.group_donors,
                "n_samples": self.group_samples.astype(np.int64),
                "total_counts": total,
            }
        )
        return df, groups


def summarize(
    uri: str, atlas: Atlas, *, goal_nnz: int = 100_000_000, log=print
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import tiledbsoma as soma

    t0 = time.time()
    cols = [atlas.donor_col, atlas.sample_col, atlas.majorclass_col]
    cols += [c for c in (atlas.celltype_col, atlas.region_col) if c and c not in cols]
    with soma.Experiment.open(uri) as exp:
        obs = exp.obs.read(column_names=["soma_joinid", *cols]).concat().to_pandas().sort_values("soma_joinid")
        n_obs = len(obs)
        if not (obs["soma_joinid"].to_numpy() == np.arange(n_obs)).all():
            raise ValueError("obs soma_joinid is not 0..n-1")
        var = exp.ms[MEASUREMENT].var.read(column_names=["soma_joinid", "var_id"]).concat().to_pandas()
        var = var.sort_values("soma_joinid")
        n_var = len(var)
        gt = group_table(obs, atlas)
        donors = _labels(obs[atlas.donor_col])
        samples = _labels(obs[atlas.sample_col])
        levels = [_Level(k, v, donors, samples, n_var) for k, v in gt.items()]
        for lv in levels:
            log(
                f"level {lv.name}: {len(lv.group_labels)} groups, {lv.K} group x donor combos, {lv.n_excluded} cells excluded"
            )
        X = exp.ms[MEASUREMENT].X[COUNTS]
        # block boundaries from per-row nnz would need a pass; use a fixed row count sized by mean nnz per row
        mean_nnz = max(1.0, X.nnz / max(1, n_obs))
        step = max(1000, int(goal_nnz / mean_nnz))
        seen = 0
        for a in range(0, n_obs, step):
            b = min(n_obs, a + step)
            tbl = X.read(coords=(slice(a, b - 1),)).tables().concat()
            r = tbl["soma_dim_0"].to_numpy() - a
            c = tbl["soma_dim_1"].to_numpy()
            d = tbl["soma_data"].to_numpy().astype(np.float64)
            del tbl
            xb = sp.csr_matrix((d, (r, c)), shape=(b - a, n_var))
            xb.sum_duplicates()
            seen += xb.nnz
            totals = np.asarray(xb.sum(axis=1)).ravel()
            xbin = xb.copy()
            xbin.data = (xbin.data > 0).astype(np.float64)
            xlog = xb.copy()
            row_tot = np.repeat(totals, np.diff(xb.indptr))
            with np.errstate(divide="ignore", invalid="ignore"):
                xlog.data = np.log1p(np.where(row_tot > 0, xlog.data / row_tot * 1e4, 0.0))
            for lv in levels:
                lv.add(a, xb, xbin, xlog, totals)
            log(f"cells {b:,}/{n_obs:,} nnz {seen:,} ({time.time() - t0:.0f}s)")
            del xb, xbin, xlog, r, c, d
        if seen != X.nnz:
            raise RuntimeError(f"streamed {seen} values, array has {X.nnz}")
    var_symbols = var["var_id"].astype(str).to_numpy()
    parts, gparts = [], []
    for lv in levels:
        df, g = lv.result(var_symbols)
        if lv.name == "celltype":
            # dominant majorclass of each cell type (for the CellState parent)
            ct = gt["celltype"]
            mc = gt["majorclass"]
            tab = pd.crosstab(ct, mc)
            g["majorclass"] = [tab.loc[x].idxmax() if x in tab.index else None for x in g["group_label"]]
            g["majorclass_fraction"] = [
                float(tab.loc[x].max() / tab.loc[x].sum()) if x in tab.index else None for x in g["group_label"]
            ]
        elif lv.name == "region_majorclass":
            g["region"] = [x.split("|", 1)[0] for x in g["group_label"]]
            g["majorclass"] = [x.split("|", 1)[1] for x in g["group_label"]]
            df["region"] = df["group_label"].str.split("|", n=1).str[0]
        else:
            g["majorclass"] = g["group_label"]
        parts.append(df)
        gparts.append(g)
        log(f"level {lv.name}: {len(df):,} rows")
    out = pd.concat(parts, ignore_index=True)
    if "region" not in out:
        out["region"] = None
    out["region"] = out["region"].where(out["level"] == "region_majorclass", None)
    groups = pd.concat(gparts, ignore_index=True)
    stats = {
        "n_obs": n_obs,
        "n_var": n_var,
        "nnz_streamed": int(seen),
        "block_rows": step,
        "n_donors": int(donors.dropna().nunique()),
        "n_samples": int(samples.dropna().nunique()),
        "excluded_cells": {lv.name: lv.n_excluded for lv in levels},
        "seconds": round(time.time() - t0, 1),
    }
    return out, groups, stats
