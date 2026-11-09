# visionkb-atlas

h5ad → TileDB-SOMA conversion, round-trip validation and streamed expression summaries for VisionKB.

- Contracts in `docs/contracts.md` (sanitized copy of the project contracts) define file layouts and columns; change them there first.
- Never load a whole matrix: stream X in row blocks (`h5ad_io.CSRReader`, SOMA `X.read(coords=(slice,))`).
- Every filter or exclusion is counted and reported (`report` command); never drop silently.
- No lab data, personal paths, account names or tokens in the repo: real paths live in `$VISIONKB_DATA_ROOT/config/*.yaml`.
- Directory and file names derive from CURIEs only via `config.dirname_of` (no `:` in paths).
- Before a commit: `ruff check . && ruff format . && pytest`.
- Version in `pyproject.toml` and `src/visionkb_atlas/__init__.py`; CHANGELOG line; tag `vX.Y.Z`.
