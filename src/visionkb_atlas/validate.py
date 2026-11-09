"""Round-trip checks h5ad vs SOMA (contracts §1 validation.json)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .convert import COUNTS, MEASUREMENT
from .h5ad_io import CSRReader, read_obs_var


def validate(h5ad: str | Path, uri: str | Path, *, n_sample: int = 200, seed: int = 0, check_cols=()) -> dict:
    import tiledbsoma as soma

    checks = {}
    obs, var = read_obs_var(h5ad)
    with soma.Experiment.open(str(uri)) as exp, CSRReader(h5ad) as rd:
        X = exp.ms[MEASUREMENT].X[COUNTS]
        n_obs, n_var = rd.shape
        checks["n_obs"] = {"h5ad": n_obs, "soma": int(exp.obs.count)}
        checks["n_var"] = {"h5ad": n_var, "soma": int(exp.ms[MEASUREMENT].var.count)}
        checks["nnz"] = {"h5ad": int(rd.indptr[-1]), "soma": int(X.nnz)}
        soma_cols = [f.name for f in exp.obs.schema if f.name not in ("soma_joinid", "obs_id")]
        checks["obs_columns"] = {"h5ad": sorted(map(str, obs.columns)), "soma": sorted(soma_cols)}
        var_ids = exp.ms[MEASUREMENT].var.read(column_names=["var_id"]).concat()["var_id"].to_pylist()
        checks["var_ids_equal"] = var_ids == [str(v) for v in var.index]
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n_obs, size=min(n_sample, n_obs), replace=False))
        ok_sum = ok_vals = 0
        sums = []
        for r in rows:
            m = rd.rows(int(r), int(r) + 1)
            t = X.read(coords=(int(r),)).tables().concat()
            s = t["soma_data"].to_numpy().astype(np.float64)
            hs = float(m.data.astype(np.float64).sum())
            ok_sum += int(np.isclose(hs, s.sum()))
            a = sorted(zip(m.indices.tolist(), m.data.astype(np.float32).tolist(), strict=True))
            b = sorted(zip(t["soma_dim_1"].to_numpy().tolist(), s.astype(np.float32).tolist(), strict=True))
            ok_vals += int(a == b)
            sums.append([int(r), hs, float(s.sum())])
        checks["sampled_rows"] = {"n": len(rows), "row_sum_equal": ok_sum, "values_equal": ok_vals, "first": sums[:5]}
        if check_cols:
            sub = exp.obs.read(coords=(rows.tolist(),), column_names=["soma_joinid", *check_cols]).concat().to_pandas()
            sub = sub.set_index("soma_joinid").loc[rows]
            checks["obs_values_equal"] = {
                c: bool((sub[c].astype(str).to_numpy() == obs[c].iloc[rows].astype(str).to_numpy()).all())
                for c in check_cols
            }
    passed = (
        checks["n_obs"]["h5ad"] == checks["n_obs"]["soma"]
        and checks["n_var"]["h5ad"] == checks["n_var"]["soma"]
        and checks["nnz"]["h5ad"] == checks["nnz"]["soma"]
        and checks["obs_columns"]["h5ad"] == checks["obs_columns"]["soma"]
        and checks["var_ids_equal"]
        and checks["sampled_rows"]["row_sum_equal"] == len(rows)
        and checks["sampled_rows"]["values_equal"] == len(rows)
        and all(checks.get("obs_values_equal", {}).values())
    )
    return {"status": "pass" if passed else "fail", "checks": checks}
