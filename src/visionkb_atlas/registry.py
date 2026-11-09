"""Dataset registry record (contracts §4)."""

from __future__ import annotations

import pandas as pd

from .config import Atlas


def registry_record(atlas: Atlas, groups: pd.DataFrame, source: dict, soma_uri: str) -> dict:
    mc = groups[groups.level == "majorclass"]
    return {
        "id": atlas.dataset_id,
        "version": atlas.atlas_version,
        "title": atlas.title,
        "species": atlas.species,
        "tissue": list(atlas.tissue),
        "assay": atlas.assay,
        "disease": atlas.disease,
        "stage": atlas.stage,
        "n_cells": int(source["n_obs"]),
        "n_cells_annotated": int(mc["n_cells"].sum()),
        "n_donors": int(source.get("n_donors", 0)),
        "n_samples": int(source.get("n_samples", 0)),
        "n_genes": int(source["n_var"]),
        "levels": atlas.levels(),
        "cell_types": sorted(set(groups.loc[groups.level != "region_majorclass", "cell_type_id"])),
        "format": "soma",
        "h5ad_path": atlas.h5ad,
        "soma_uri": soma_uri,
        "checksum": f"sha256:{source['sha256']}"
        if source.get("sha256")
        else f"size+mtime:{source['bytes']}+{source['mtime_utc']}",
        "access_level": atlas.access_level,
        "license": atlas.license,
        "citation": atlas.citation,
        "kg_release": None,
        "bind": {"path": atlas.h5ad, "name": atlas.slug, "role": "reference"},
    }
