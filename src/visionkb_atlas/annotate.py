"""Attach CL / CellState ids (SSSOM crosswalk) and HGNC gene ids to raw summaries (contracts §3)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .config import Atlas, slugify

DIRECT = {"skos:exactMatch", "skos:broadMatch"}
COLUMNS = [
    "dataset_id",
    "atlas_version",
    "level",
    "group_label",
    "cell_type_id",
    "region",
    "gene_id",
    "gene_symbol",
    "var_symbol",
    "n_cells",
    "n_donors",
    "n_expressing",
    "pct_expressing",
    "mean_log1p_cp10k",
    "pseudobulk_cp10k",
    "n_donors_expressing",
]


def read_sssom(path) -> pd.DataFrame:
    text = Path(path).read_text()
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return pd.read_csv(io.StringIO(body), sep="\t", dtype=str).fillna("")


def cellstate_id(atlas: Atlas, label: str) -> str:
    return f"visionkb:cs-{atlas.slug}-{slugify(label)}"


def map_groups(groups: pd.DataFrame, atlas: Atlas, xw: pd.DataFrame) -> pd.DataFrame:
    """Add cell_type_id and mapping columns to groups (one row per level x group)."""
    xw = xw[xw["subject_source"] == atlas.dataset_id]
    look = {(r.annotation_level, r.subject_label): r for r in xw.itertuples(index=False)}
    g = groups.copy()

    def one(level, label):
        r = look.get((level, label))
        if r is None:
            return None
        return r.predicate_id, r.object_id, r.object_label, r.confidence

    mc_map = {}
    for lab in sorted(set(g.loc[g.level == "majorclass", "group_label"])):
        mc_map[lab] = one("majorclass", lab)

    def mc_ids(lab):
        m = mc_map.get(lab)
        if m and m[0] in DIRECT:
            return m[1], None
        return cellstate_id(atlas, lab), (m[1] if m else None)

    out = []
    for r in g.itertuples(index=False):
        if r.level == "region_majorclass":
            cid, parent = mc_ids(r.majorclass)
            m = mc_map.get(r.majorclass)
            out.append(
                (
                    cid,
                    parent,
                    m[0] if m else "",
                    m[1] if m else "",
                    m[3] if m else "",
                    "majorclass_crosswalk" if m else "unmapped",
                )
            )
            continue
        m = one(r.level, r.group_label)
        if m and m[0] in DIRECT:
            out.append((m[1], None, m[0], m[1], m[3], "crosswalk"))
        elif m:  # closeMatch / relatedMatch: dataset-specific state of the mapped CL class
            out.append((cellstate_id(atlas, r.group_label), m[1], m[0], m[1], m[3], "crosswalk"))
        elif r.level == "celltype" and isinstance(r.majorclass, str):
            pid, pparent = mc_ids(r.majorclass)
            parent = pid if pid.startswith("CL:") else pparent
            out.append((cellstate_id(atlas, r.group_label), parent, "", "", "", "majorclass_parent"))
        else:
            out.append((cellstate_id(atlas, r.group_label), None, "", "", "", "unmapped"))
    g[["cell_type_id", "parent_cl", "mapping_predicate", "mapping_object", "mapping_confidence", "mapping_source"]] = (
        pd.DataFrame(out, index=g.index)
    )
    return g


def annotate(raw: pd.DataFrame, groups: pd.DataFrame, genes: pd.DataFrame, atlas: Atlas) -> pd.DataFrame:
    df = raw.merge(
        groups[["level", "group_label", "cell_type_id"]], on=["level", "group_label"], how="left", validate="m:1"
    )
    df = df.merge(genes[["var_symbol", "gene_id", "gene_symbol"]], on="var_symbol", how="left", validate="m:1")
    if df["cell_type_id"].isna().any() or df["gene_id"].isna().any():
        raise RuntimeError("unmapped groups or genes after annotation")
    df["dataset_id"] = atlas.dataset_id
    df["atlas_version"] = atlas.atlas_version
    df = df[COLUMNS].sort_values(["level", "group_label", "var_symbol"], kind="mergesort").reset_index(drop=True)
    return df
