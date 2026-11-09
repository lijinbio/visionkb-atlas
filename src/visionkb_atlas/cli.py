"""visionkb-atlas command line: inventory, convert, validate, summarize, annotate, crosscheck, run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .config import atlas_dir, data_root, dirname_of, get_atlas, load_atlases, work_dir
from .convert import write_json


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cfg(args) -> Path:
    return Path(args.config) if args.config else data_root() / "config" / "atlases.yaml"


def _work(a) -> Path:
    p = work_dir() / dirname_of(a.dataset_id) / a.atlas_version
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_inventory(args):
    import h5py
    import numpy as np

    from .h5ad_io import CSRReader, x_info

    out = {}
    for a in load_atlases(_cfg(args)):
        rec = {"x": x_info(a.h5ad)}
        with h5py.File(a.h5ad, "r") as h:
            rec["obs_columns"] = sorted(h["obs"].keys())
            rec["layers"] = sorted(h["layers"].keys()) if "layers" in h else []
            rec["has_raw"] = "raw" in h
        with CSRReader(a.h5ad) as rd:
            rows = np.sort(np.random.default_rng(0).choice(rd.shape[0], 50, replace=False))
            vals = np.concatenate([rd.rows(int(r), int(r) + 1).data for r in rows])
            rec["sampled_rows"] = {
                "n": 50,
                "n_values": int(vals.size),
                "n_noninteger": int(np.sum(vals != np.round(vals))),
            }
        for c in ("donor_col", "sample_col", "majorclass_col", "celltype_col", "region_col"):
            col = getattr(a, c)
            if col and col.replace("/", "") not in rec["obs_columns"]:
                raise SystemExit(f"{a.dataset_id}: {c}={col} not in obs")
        out[a.dataset_id] = rec
        _log(f"{a.dataset_id}: {rec['x']} noninteger={rec['sampled_rows']['n_noninteger']}")
    if args.out:
        write_json(out, args.out)


def cmd_convert(args):
    from .convert import convert

    a = get_atlas(_cfg(args), args.dataset)
    final = atlas_dir(a)
    w = _work(a)
    tmp = w / "soma"
    if (final / "soma").exists():
        raise SystemExit(f"{final / 'soma'} exists (convert once); delete it explicitly to redo")
    if tmp.exists():
        shutil.rmtree(tmp)
    src = convert(a.h5ad, tmp, goal_nnz=args.goal_nnz, sha256=not args.no_sha256, log=_log)
    final.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(final / "soma"))
    src["soma_uri"] = str(final / "soma")
    write_json(src, final / "source.json")
    _log(f"wrote {final / 'soma'}")


def cmd_validate(args):
    from .validate import validate

    a = get_atlas(_cfg(args), args.dataset)
    final = atlas_dir(a)
    cols = [c for c in (a.donor_col, a.sample_col, a.majorclass_col, a.celltype_col, a.region_col) if c]
    res = validate(a.h5ad, final / "soma", n_sample=args.n_sample, check_cols=sorted(set(cols)))
    write_json(res, final / "validation.json")
    _log(f"validation {res['status']}")
    if res["status"] != "pass":
        sys.exit(1)


def cmd_summarize(args):
    from .summarize import summarize

    a = get_atlas(_cfg(args), args.dataset)
    w = _work(a)
    raw, groups, stats = summarize(str(atlas_dir(a) / "soma"), a, goal_nnz=args.goal_nnz, log=_log)
    raw.to_parquet(w / "summary_raw.parquet", index=False)
    groups.to_parquet(w / "groups_raw.parquet", index=False)
    write_json(stats, w / "summarize_stats.json")
    _log(f"wrote {w / 'summary_raw.parquet'} ({len(raw):,} rows)")


def cmd_annotate(args):
    import pandas as pd

    from .annotate import annotate, map_groups, read_sssom
    from .genes import load_hgnc, resolution_report, resolve
    from .registry import registry_record

    a = get_atlas(_cfg(args), args.dataset)
    w = _work(a)
    final = atlas_dir(a)
    raw = pd.read_parquet(w / "summary_raw.parquet")
    groups = pd.read_parquet(w / "groups_raw.parquet")
    stats = json.loads((w / "summarize_stats.json").read_text())
    hgnc = load_hgnc(args.hgnc)
    genes = resolve(sorted(raw["var_symbol"].unique()), hgnc)
    genes.to_csv(w / "gene_resolution.tsv", sep="\t", index=False)
    rep = resolution_report(genes, hgnc)
    groups = map_groups(groups, a, read_sssom(args.crosswalk))
    df = annotate(raw, groups, genes, a)
    (final / "summaries").mkdir(parents=True, exist_ok=True)
    df.to_parquet(final / "summaries" / "expr_summary.parquet", index=False)
    groups.insert(0, "dataset_id", a.dataset_id)
    groups.insert(1, "atlas_version", a.atlas_version)
    groups.to_parquet(final / "summaries" / "groups.parquet", index=False)
    src = json.loads((final / "source.json").read_text())
    src.update(n_donors=stats["n_donors"], n_samples=stats["n_samples"])
    reg = registry_record(a, groups, src, str(final / "soma"))
    write_json(reg, final / "registry.json")
    write_json(
        {"gene_resolution": rep, "groups_by_mapping_source": groups["mapping_source"].value_counts().to_dict()},
        w / "annotate_report.json",
    )
    _log(
        f"{a.dataset_id}: {len(df):,} rows; genes {rep['by_resolution']}; groups {groups['mapping_source'].value_counts().to_dict()}"
    )


def cmd_crosscheck(args):
    import pandas as pd

    from .crosscheck import crosscheck

    a = get_atlas(_cfg(args), args.dataset)
    s = pd.read_parquet(
        atlas_dir(a) / "summaries" / "expr_summary.parquet",
        columns=["level", "group_label", "var_symbol", "n_cells", "pseudobulk_cp10k", "mean_log1p_cp10k"],
    )
    level = args.level
    if level == "auto":
        level = "majorclass"
    res = crosscheck(s, args.mean, args.labelsummary, level=level)
    res.to_csv(_work(a) / "crosscheck.tsv", sep="\t", index=False)
    print(res.to_string(index=False))


def cmd_list(args):
    import os

    for a in load_atlases(_cfg(args)):
        print("\t".join([a.slug, a.atlas_version, str(os.path.getsize(a.h5ad)), str(atlas_dir(a))]))


def cmd_report(args):
    """Exclusions and filters of one atlas (nothing dropped silently): cells without labels/donor/sample/region,
    region labels without UBERON, groups below the KG n_cells threshold, gene resolution."""
    import pandas as pd
    import tiledbsoma as soma

    from .summarize import MISSING, _labels

    a = get_atlas(_cfg(args), args.dataset)
    w = _work(a)
    cols = sorted({c for c in (a.donor_col, a.sample_col, a.majorclass_col, a.celltype_col, a.region_col) if c})
    with soma.Experiment.open(str(atlas_dir(a) / "soma")) as exp:
        obs = exp.obs.read(column_names=cols).concat().to_pandas()
    rep = {"dataset_id": a.dataset_id, "atlas_version": a.atlas_version, "n_cells": len(obs), "cells_missing": {}}
    for role in ("donor_col", "sample_col", "majorclass_col", "celltype_col", "region_col"):
        c = getattr(a, role)
        if not c:
            continue
        lab = _labels(obs[c])
        if role == "region_col":
            raw = lab.copy()
            lab = lab.map(lambda x: None if x is None else a.map_label("region", x))
            lab = lab.map(lambda x: None if x is None or str(x) in MISSING else x)
            rep["region_labels_mapped_to_missing"] = sorted(
                {str(r) for r, m in zip(raw, lab, strict=True) if r is not None and m is None}
            )
            counts = lab.value_counts().to_dict()
            rep["region_cells"] = {k: int(v) for k, v in sorted(counts.items())}
            rep["region_labels_without_uberon"] = {
                k: int(v) for k, v in sorted(counts.items()) if k not in a.region_uberon
            }
        rep["cells_missing"][f"{role[:-4]} ({c})"] = int(lab.isna().sum())
    g = pd.read_parquet(atlas_dir(a) / "summaries" / "groups.parquet")
    rep["groups"] = {k: int(v) for k, v in g.groupby("level").size().items()}
    small = g[g.n_cells < args.min_cells]
    rep[f"groups_below_{args.min_cells}_cells"] = {
        lv: sorted(f"{x} ({n})" for x, n in zip(s.group_label, s.n_cells, strict=True))
        for lv, s in small.groupby("level")
    }
    rep["groups_by_mapping_source"] = (
        g.groupby(["level", "mapping_source"]).size().rename("n").reset_index().to_dict("records")
    )
    rep["gene_resolution"] = json.loads((w / "annotate_report.json").read_text())["gene_resolution"]
    rep["summarize_excluded_cells_by_level"] = json.loads((w / "summarize_stats.json").read_text())["excluded_cells"]
    write_json(rep, w / "exclusions.json")
    _log(
        f"{a.dataset_id}: missing {rep['cells_missing']}; groups below {args.min_cells} cells: {sum(map(len, rep[f'groups_below_{args.min_cells}_cells'].values()))}"
    )


def cmd_run(args):
    cmd_convert(args)
    cmd_validate(args)
    cmd_summarize(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="visionkb-atlas", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", help="atlases.yaml (default $VISIONKB_DATA_ROOT/config/atlases.yaml)")
    p.add_argument(
        "--work-dir", help="working-file root (default $VISIONKB_ATLAS_WORK or $VISIONKB_WORK_DIR/visionkb-atlas)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inventory")
    s.add_argument("--out")
    s.set_defaults(func=cmd_inventory)
    for name, fn in (
        ("convert", cmd_convert),
        ("validate", cmd_validate),
        ("summarize", cmd_summarize),
        ("run", cmd_run),
    ):
        s = sub.add_parser(name)
        s.add_argument("dataset")
        s.add_argument("--goal-nnz", type=int, default=100_000_000)
        s.add_argument("--no-sha256", action="store_true")
        s.add_argument("--n-sample", type=int, default=200)
        s.set_defaults(func=fn)
    s = sub.add_parser("list", help="dataset slug, atlas version, h5ad bytes, final dir")
    s.set_defaults(func=cmd_list)
    s = sub.add_parser("report", help="exclusions and filters of one atlas")
    s.add_argument("dataset")
    s.add_argument("--min-cells", type=int, default=20)
    s.set_defaults(func=cmd_report)
    s = sub.add_parser("annotate")
    s.add_argument("dataset")
    s.add_argument("--crosswalk", required=True)
    s.add_argument("--hgnc", required=True)
    s.set_defaults(func=cmd_annotate)
    s = sub.add_parser("crosscheck")
    s.add_argument("dataset")
    s.add_argument("--mean", required=True)
    s.add_argument("--labelsummary", required=True)
    s.add_argument("--level", default="auto")
    s.set_defaults(func=cmd_crosscheck)
    args = p.parse_args(argv)
    if args.work_dir:
        import os

        os.environ["VISIONKB_ATLAS_WORK"] = str(Path(args.work_dir).resolve())
    args.func(args)


if __name__ == "__main__":
    main()
