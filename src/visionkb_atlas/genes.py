"""Atlas gene names -> HGNC (approved symbol, alias, previous symbol, Ensembl id), with collision checks."""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

ENSG = re.compile(r"^ENSG\d{11}(\.\d+)?$")


def load_hgnc(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    need = {"hgnc_id", "symbol", "status", "alias_symbol", "prev_symbol", "ensembl_gene_id"}
    if missing := need - set(df.columns):
        raise ValueError(f"HGNC file lacks columns {missing}")
    return df[df["status"] == "Approved"].copy()


def _split(v) -> list[str]:
    if not isinstance(v, str) or not v:
        return []
    return [x.strip().strip('"') for x in v.split("|") if x.strip()]


def resolve(symbols, hgnc: pd.DataFrame) -> pd.DataFrame:
    """One row per input symbol: gene_id, gene_symbol, resolution.

    resolution: approved | alias | previous | ensembl | ambiguous | collision | unresolved.
    Unresolved/ambiguous/collision rows get ``visionkb:gene-<symbol>`` and keep the atlas symbol.
    """
    approved = dict(zip(hgnc["symbol"], hgnc["hgnc_id"], strict=True))
    sym_of = dict(zip(hgnc["hgnc_id"], hgnc["symbol"], strict=True))
    alias, prev, ens = defaultdict(set), defaultdict(set), defaultdict(set)
    for hid, a, p, e in zip(
        hgnc["hgnc_id"], hgnc["alias_symbol"], hgnc["prev_symbol"], hgnc["ensembl_gene_id"], strict=True
    ):
        for x in _split(a):
            alias[x].add(hid)
        for x in _split(p):
            prev[x].add(hid)
        if isinstance(e, str) and e:
            ens[e].add(hid)
    symbols = [str(s) for s in symbols]
    rows = []
    for s in symbols:
        if s in approved:
            rows.append((s, approved[s], "approved"))
            continue
        if ENSG.match(s):
            hits = ens.get(s.split(".")[0], set())
            if len(hits) == 1:
                rows.append((s, next(iter(hits)), "ensembl"))
            else:
                rows.append((s, f"ENSEMBL:{s.split('.')[0]}", "ensembl_unmapped" if not hits else "ambiguous"))
            continue
        cand, how = None, "unresolved"
        for table, name in ((alias, "alias"), (prev, "previous")):
            hits = table.get(s, set())
            if len(hits) == 1:
                cand, how = next(iter(hits)), name
                break
            if len(hits) > 1:
                how = "ambiguous"
        rows.append((s, cand, how))
    df = pd.DataFrame(rows, columns=["var_symbol", "gene_id", "resolution"])
    # an HGNC id may be claimed once: approved matches win; other claimants become collisions
    taken = set(df.loc[df.resolution == "approved", "gene_id"])
    nonexact = df[df.resolution.isin(["alias", "previous", "ensembl"])]
    dup = nonexact["gene_id"].duplicated(keep=False)
    bad = nonexact.index[(nonexact["gene_id"].isin(taken)) | dup]
    df.loc[bad, "resolution"] = "collision"
    df.loc[bad, "gene_id"] = None
    unres = df["gene_id"].isna()
    df.loc[unres, "gene_id"] = [
        "visionkb:gene-" + re.sub(r"[^0-9A-Za-z._-]", "_", s) for s in df.loc[unres, "var_symbol"]
    ]
    df["gene_symbol"] = [sym_of.get(g, s) for g, s in zip(df["gene_id"], df["var_symbol"], strict=True)]
    df["flagged"] = ~df["gene_id"].str.startswith("HGNC:")
    return df


def resolution_report(df: pd.DataFrame, hgnc: pd.DataFrame | None = None) -> dict:
    out = {"n_var": len(df), "by_resolution": df["resolution"].value_counts().sort_index().to_dict()}
    out["resolved_to_hgnc"] = int((~df["flagged"]).sum())
    out["fraction_resolved"] = round(out["resolved_to_hgnc"] / max(1, len(df)), 4)
    if hgnc is not None and "locus_group" in hgnc:
        pc = set(hgnc.loc[hgnc["locus_group"] == "protein-coding gene", "hgnc_id"])
        out["protein_coding_hgnc_in_atlas"] = len(pc & set(df["gene_id"]))
        out["protein_coding_hgnc_total"] = len(pc)
    return out
