"""Atlas column map (contracts §2) and environment roots (contracts §9)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REQUIRED = ("dataset_id", "title", "h5ad", "species", "tissue", "assay", "donor_col", "sample_col", "majorclass_col")


def data_root() -> Path:
    v = os.environ.get("VISIONKB_DATA_ROOT")
    if not v:
        raise RuntimeError("VISIONKB_DATA_ROOT is not set")
    return Path(v)


def work_dir() -> Path:
    """Scratch/working files: $VISIONKB_ATLAS_WORK if set (e.g. a workflow step folder), else $VISIONKB_WORK_DIR/visionkb-atlas."""
    if v := os.environ.get("VISIONKB_ATLAS_WORK"):
        return Path(v)
    v = os.environ.get("VISIONKB_WORK_DIR")
    if not v:
        raise RuntimeError("VISIONKB_WORK_DIR (or VISIONKB_ATLAS_WORK) is not set")
    return Path(v) / "visionkb-atlas"


def slugify(label: str) -> str:
    """Lower-case slug used in CellState ids. Marker signs are kept apart so that distinct labels stay distinct:
    '+' -> 'pos', a trailing '-' (e.g. 'CD16-') -> 'neg', '*' -> 'star'; other runs of non-alphanumerics -> one hyphen.
    (Collisions that remain are detected in annotate.map_groups.)"""
    s = str(label).replace("+", " pos ").replace("*", " star ")
    s = re.sub(r"(?<=[0-9A-Za-z])-(?=$|[^0-9A-Za-z])", " neg ", s)
    s = re.sub(r"[^0-9a-zA-Z]+", "-", s).strip("-").lower()
    if not s:
        raise ValueError(f"label {label!r} has no slug")
    return s


@dataclass
class Atlas:
    dataset_id: str
    title: str
    h5ad: str
    species: str
    tissue: list[str]
    assay: str
    donor_col: str
    sample_col: str
    majorclass_col: str
    atlas_version: str
    counts_source: str = "X"
    celltype_col: str | None = None
    region_col: str | None = None
    label_maps: dict = field(default_factory=dict)
    region_uberon: dict = field(default_factory=dict)
    stage: str | None = None
    disease: str | None = None
    license: str | None = None
    citation: str | None = None
    access_level: str = "lab"

    @property
    def slug(self) -> str:
        if not self.dataset_id.startswith("visionkb:ds-"):
            raise ValueError(f"dataset_id must look like visionkb:ds-<slug>: {self.dataset_id}")
        return self.dataset_id.removeprefix("visionkb:ds-")

    def map_label(self, kind: str, value):
        """Apply an optional label map (e.g. region spelling variants) to a raw obs label."""
        return self.label_maps.get(kind, {}).get(value, value)

    def levels(self) -> list[str]:
        out = ["majorclass"]
        if self.celltype_col:
            out.append("celltype")
        if self.region_col:
            out.append("region_majorclass")
        return out


def load_atlases(path: str | Path) -> list[Atlas]:
    cfg = yaml.safe_load(Path(path).read_text())
    default_version = cfg.get("atlas_version")
    out = []
    for rec in cfg["atlases"]:
        rec = dict(rec)
        rec.setdefault("atlas_version", default_version)
        missing = [k for k in REQUIRED if not rec.get(k)]
        if missing or not rec.get("atlas_version"):
            raise ValueError(f"atlas {rec.get('dataset_id')}: missing {missing or ['atlas_version']}")
        if rec.get("counts_source", "X") != "X":
            raise NotImplementedError("only counts_source: X is implemented")
        if isinstance(rec["tissue"], str):
            rec["tissue"] = [rec["tissue"]]
        a = Atlas(**rec)
        _ = a.slug
        out.append(a)
    ids = [a.dataset_id for a in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate dataset_id in atlases config")
    return out


def get_atlas(path: str | Path, dataset_id: str) -> Atlas:
    for a in load_atlases(path):
        if dataset_id in (a.dataset_id, a.slug):
            return a
    raise KeyError(dataset_id)


def dirname_of(curie: str) -> str:
    """File/directory name for a CURIE: the local part without prefix (no ':' in paths; contracts v0.4).

    visionkb:ds-hrca-scrna -> ds-hrca-scrna. The CURIE itself stays unchanged inside files.
    """
    if ":" not in curie:
        raise ValueError(f"not a CURIE: {curie}")
    name = curie.split(":", 1)[1]
    if not name or "/" in name or ":" in name or name in (".", ".."):
        raise ValueError(f"CURIE {curie!r} has no safe file name")
    return name


def atlas_dir(a: Atlas, root: Path | None = None) -> Path:
    """Final location: $VISIONKB_DATA_ROOT/atlas/<dataset slug>/<atlas_version>/ (e.g. atlas/ds-hrca-scrna/heca-2026.05)."""
    return (root or data_root()) / "atlas" / dirname_of(a.dataset_id) / a.atlas_version
