import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml


@pytest.fixture
def synthetic(tmp_path):
    """A small raw-count h5ad with majorclass, celltype, region and donor columns (synthetic values)."""
    rng = np.random.default_rng(42)
    n, g = 600, 40
    dense = rng.poisson(0.6, size=(n, g)).astype(np.float32)
    dense[:, 0] = 0  # a never-expressed gene
    dense[:5] = 0  # some empty cells
    mc = rng.choice(["Rod", "Cone", "MG"], size=n)
    ct = np.array([f"{m}_{rng.integers(2)}" for m in mc], dtype=object)
    ct[:7] = np.nan
    obs = pd.DataFrame(
        {
            "donorid": pd.Categorical(rng.choice(["d1", "d2", "d3", "d4"], size=n)),
            "sampleid": pd.Categorical(rng.choice(["s1", "s2", "s3", "s4", "s5"], size=n)),
            "majorclass": pd.Categorical(mc),
            "celltype": pd.Categorical(ct),
            "tissue": pd.Categorical(rng.choice(["fovea", "periphery", "Unknown"], size=n)),
            "nCount_RNA": dense.sum(1).astype(np.float64),
        },
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(g - 3)] + ["RHO", "OLDSYM", "ALIASX"])
    path = tmp_path / "syn.h5ad"
    ad.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var).write_h5ad(path)
    cfg = {
        "atlas_version": "test-0",
        "atlases": [
            {
                "dataset_id": "visionkb:ds-syn-scrna",
                "title": "synthetic",
                "h5ad": str(path),
                "species": "NCBITaxon:9606",
                "tissue": ["UBERON:0000966"],
                "assay": "EFO:0008913",
                "donor_col": "donorid",
                "sample_col": "sampleid",
                "majorclass_col": "majorclass",
                "celltype_col": "celltype",
                "region_col": "tissue",
                "label_maps": {"region": {"Unknown": ""}},
            }
        ],
    }
    cfgp = tmp_path / "atlases.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    return {"path": path, "dense": dense, "obs": obs, "var": var, "config": cfgp, "tmp": tmp_path}
