# visionkb-atlas

Atlas store for VisionKB: converts single-cell `.h5ad` atlases once to on-disk
[TileDB-SOMA](https://github.com/single-cell-data/TileDB-SOMA) experiments, checks the round trip, and computes
per gene × cell group expression summaries by streaming the SOMA store in blocks. Only the summaries go into the
VisionKB knowledge graph ([visionkb-kg](https://github.com/lijinbio/visionkb-kg)); matrices stay on disk (decision D7).

## What it does

| Command | Output |
|---|---|
| `visionkb-atlas inventory` | obs columns, layers, X shape/nnz and an integer-count check per atlas (h5py; X is not loaded) |
| `visionkb-atlas convert <ds>` | `<root>/atlas/<ds-slug>/<version>/soma/` (measurement `RNA`, X layer `counts`) and `source.json` |
| `visionkb-atlas validate <ds>` | `validation.json`: n_obs, n_var, nnz, obs columns, var ids, values and sums of sampled rows (h5ad vs SOMA) |
| `visionkb-atlas summarize <ds>` | per gene × group statistics (majorclass, celltype, region × majorclass) in the working dir |
| `visionkb-atlas annotate <ds> --crosswalk <sssom> --hgnc <file>` | `summaries/expr_summary.parquet`, `summaries/groups.parquet`, `registry.json` |
| `visionkb-atlas report <ds>` | exclusions: cells without label/donor/sample/region, regions without UBERON, groups < 20 cells, gene resolution |
| `visionkb-atlas crosscheck <ds> --mean <tsv> --labelsummary <tsv>` | Spearman agreement with an earlier per-label pseudo-bulk CPM table |

Columns of `expr_summary.parquet` follow `docs/contracts.md` §3: `n_cells`, `n_donors`, `n_expressing`, `pct_expressing`,
`mean_log1p_cp10k` (mean over all cells of log1p(count / total × 1e4)), `pseudobulk_cp10k` (Σ counts / Σ totals × 1e4),
`n_donors_expressing` (donors in which ≥ 10 % of that donor's group cells express the gene), plus `cell_type_id` (CL id for
exact/broad crosswalk matches, else a `visionkb:cs-…` CellState) and `gene_id` (HGNC by approved symbol, alias, previous symbol
or Ensembl id; collisions and ambiguous aliases stay unresolved and flagged).

## Why a custom converter

`tiledbsoma.io.from_h5ad` 2.3.0 with anndata 0.13 fails in backed mode: anndata 0.13 exposes `X` as `layers[None]`, and
`tiledbsoma._util.sanitize_key(None, ...)` raises `TypeError: quote_from_bytes() expected bytes` (it had already used ~8 GB for a
0.3 GB file before failing). `visionkb_atlas.convert` writes obs/var through `from_anndata` on an X-less AnnData and streams X from
the h5ad CSR arrays in blocks of ~1e8 stored values into a `SparseNDArray`. Peak memory is set by the block size, not the file:
on 13 atlases of 0.3–19.5 GB the measured peak RSS of convert + validate + summarize was ~9–15 GB.

## Install

```bash
pip install git+https://github.com/lijinbio/visionkb-atlas.git@v0.1.0
# or, for development:
git clone https://github.com/lijinbio/visionkb-atlas.git && cd visionkb-atlas
uv sync && uv run pytest
```

## Configuration

Copy `config/atlases.example.yaml` to `$VISIONKB_DATA_ROOT/config/atlases.yaml` and set the real h5ad paths. Per atlas:
`dataset_id`, `title`, `h5ad`, `species`, `tissue` (UBERON list), `assay` (EFO), `counts_source` (`X`), `donor_col`,
`sample_col`, `majorclass_col`, optional `celltype_col`, `region_col`, `label_maps` (e.g. region spelling variants; map to `""` for
missing) and `region_uberon`.

Environment: `VISIONKB_DATA_ROOT` (published outputs), `VISIONKB_WORK_DIR` or `VISIONKB_ATLAS_WORK` (scratch, working files).
Directory names use the dataset slug without prefix (`atlas/ds-hrca-scrna/…`); the CURIE stays inside the files.

## Tests

Unit tests use a synthetic h5ad and compare every streamed statistic with a dense computation (`pytest`). No lab data is in this repository.

## License

MIT, see [LICENSE](LICENSE).
