import json

import numpy as np
import pandas as pd

from visionkb_atlas.annotate import annotate, map_groups
from visionkb_atlas.config import get_atlas, slugify
from visionkb_atlas.convert import convert
from visionkb_atlas.genes import resolve
from visionkb_atlas.summarize import summarize
from visionkb_atlas.validate import validate


def _expected(dense, mask, donors):
    cells = dense[mask]
    tot = dense.sum(1)[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        cp = np.where(tot[:, None] > 0, cells / tot[:, None] * 1e4, 0)
    pct = (cells > 0).mean(0)
    mean_log = np.log1p(cp).mean(0)
    pb = cells.sum(0) / tot.sum() * 1e4
    nde = np.zeros(dense.shape[1], dtype=int)
    for d in np.unique(donors[mask]):
        m2 = donors[mask] == d
        nde += ((cells[m2] > 0).mean(0) >= 0.10).astype(int)
    return pct, mean_log, pb, nde


def test_convert_validate_summarize(synthetic, tmp_path):
    a = get_atlas(synthetic["config"], "syn-scrna")
    uri = tmp_path / "soma"
    src = convert(synthetic["path"], uri, goal_nnz=500, log=lambda *_: None)
    assert src["n_obs"] == 600 and src["nnz"] == int((synthetic["dense"] > 0).sum())
    assert src["n_noninteger_values"] == 0
    v = validate(synthetic["path"], uri, n_sample=50, check_cols=["majorclass", "donorid"])
    assert v["status"] == "pass", json.dumps(v)
    raw, _groups, stats = summarize(str(uri), a, goal_nnz=700, log=lambda *_: None)
    assert stats["nnz_streamed"] == src["nnz"]
    dense, obs = synthetic["dense"], synthetic["obs"]
    donors = obs["donorid"].astype(str).to_numpy()
    for level, labels in (("majorclass", obs["majorclass"].astype(str)), ("celltype", obs["celltype"].astype(str))):
        for lab in sorted(set(labels) - {"nan"}):
            mask = (labels == lab).to_numpy()
            pct, ml, pb, nde = _expected(dense, mask, donors)
            sub = (
                raw[(raw.level == level) & (raw.group_label == lab)].set_index("var_symbol").loc[synthetic["var"].index]
            )
            assert (sub.n_cells == mask.sum()).all()
            assert (sub.n_donors == len(set(donors[mask]))).all()
            np.testing.assert_allclose(sub.pct_expressing, pct, rtol=1e-9)
            np.testing.assert_allclose(sub.mean_log1p_cp10k, ml, rtol=1e-6)
            np.testing.assert_allclose(sub.pseudobulk_cp10k, pb, rtol=1e-6)
            np.testing.assert_array_equal(sub.n_donors_expressing, nde)
    rm = raw[raw.level == "region_majorclass"]
    assert set(rm.region) == {"fovea", "periphery"}  # 'Unknown' mapped to missing
    lab = "fovea|Rod"
    mask = ((obs["tissue"] == "fovea") & (obs["majorclass"] == "Rod")).to_numpy()
    sub = rm[rm.group_label == lab].set_index("var_symbol").loc[synthetic["var"].index]
    np.testing.assert_allclose(sub.pct_expressing, _expected(dense, mask, donors)[0])
    assert stats["excluded_cells"]["celltype"] == 7


def test_annotate_crosswalk_and_genes(synthetic, tmp_path):
    a = get_atlas(synthetic["config"], "visionkb:ds-syn-scrna")
    uri = tmp_path / "soma"
    convert(synthetic["path"], uri, log=lambda *_: None)
    raw, groups, _ = summarize(str(uri), a, log=lambda *_: None)
    xw = pd.DataFrame(
        [
            ("visionkb:ds-syn-scrna", "majorclass", "Rod", "skos:exactMatch", "CL:0000604", "retinal rod cell", "0.95"),
            (
                "visionkb:ds-syn-scrna",
                "majorclass",
                "Cone",
                "skos:closeMatch",
                "CL:0000573",
                "retinal cone cell",
                "0.5",
            ),
            ("visionkb:ds-syn-scrna", "celltype", "Rod_1", "skos:broadMatch", "CL:0000604", "retinal rod cell", "0.8"),
        ],
        columns=[
            "subject_source",
            "annotation_level",
            "subject_label",
            "predicate_id",
            "object_id",
            "object_label",
            "confidence",
        ],
    )
    g = map_groups(groups, a, xw)
    get = g.set_index(["level", "group_label"])
    assert get.loc[("majorclass", "Rod"), "cell_type_id"] == "CL:0000604"
    assert get.loc[("majorclass", "Cone"), "cell_type_id"] == "visionkb:cs-syn-scrna-cone"
    assert get.loc[("majorclass", "Cone"), "parent_cl"] == "CL:0000573"
    assert get.loc[("majorclass", "MG"), "mapping_source"] == "unmapped"
    assert get.loc[("celltype", "Rod_1"), "cell_type_id"] == "CL:0000604"
    assert get.loc[("celltype", "Rod_0"), "parent_cl"] == "CL:0000604"
    assert get.loc[("celltype", "Rod_0"), "mapping_source"] == "majorclass_parent"
    assert get.loc[("region_majorclass", "fovea|Rod"), "cell_type_id"] == "CL:0000604"
    hgnc = pd.DataFrame(
        {
            "hgnc_id": ["HGNC:10012", "HGNC:2", "HGNC:3", "HGNC:4"],
            "symbol": ["RHO", "NEWSYM", "GENEA", "GENEB"],
            "status": ["Approved"] * 4,
            "alias_symbol": ["", "", "ALIASX", "ALIASX"],
            "prev_symbol": ["", "OLDSYM", "", ""],
            "ensembl_gene_id": ["ENSG00000163914", "", "", ""],
        }
    )
    genes = resolve(synthetic["var"].index, hgnc)
    r = genes.set_index("var_symbol")
    assert r.loc["RHO", "gene_id"] == "HGNC:10012"
    assert r.loc["OLDSYM", "gene_id"] == "HGNC:2" and r.loc["OLDSYM", "gene_symbol"] == "NEWSYM"
    assert r.loc["ALIASX", "resolution"] == "ambiguous" and r.loc["ALIASX", "gene_id"] == "visionkb:gene-ALIASX"
    df = annotate(raw, g, genes, a)
    assert list(df.columns)[:5] == ["dataset_id", "atlas_version", "level", "group_label", "cell_type_id"]
    assert df["region"].notna().sum() == (df.level == "region_majorclass").sum()


def test_slugify():
    assert slugify("Corneal Stromal Keratocytes - C1") == "corneal-stromal-keratocytes-c1"
    assert slugify("NK/T Cell") == "nk-t-cell"


def test_dirname_of():
    import pytest

    from visionkb_atlas.config import dirname_of

    assert dirname_of("visionkb:ds-hrca-scrna") == "ds-hrca-scrna"
    for bad in ("nocolon", "visionkb:", "visionkb:a/b", "visionkb:.."):
        with pytest.raises(ValueError):
            dirname_of(bad)
