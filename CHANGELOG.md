# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-09

### Added

- Block-streaming h5ad → TileDB-SOMA converter (`convert`), replacing `tiledbsoma.io.from_h5ad`, which fails with anndata 0.13 in
  backed mode (`sanitize_key(None)`).
- Round-trip validation (`validate`), streamed summaries at majorclass, celltype and region × majorclass levels (`summarize`),
  CL/CellState and HGNC annotation and registry records (`annotate`), exclusion report (`report`), pseudo-bulk CPM cross-check
  (`crosscheck`), atlas listing (`list`), h5py inventory (`inventory`).
- Directory names from CURIEs through one slug function (`dirname_of`; contracts v0.4).
- CI: lint and tests on Python 3.12 and 3.13, pip install check.
