# Contributing

```bash
git clone https://github.com/lijinbio/visionkb-atlas.git && cd visionkb-atlas
uv sync && uv run pytest          # or: pip install -e . pytest ruff
pre-commit install                # optional
```

| Topic | Rule |
| --- | --- |
| Tests | unit tests on synthetic data only; streamed statistics checked against a dense computation |
| Lint | `ruff check .` and `ruff format --check .` (settings in `pyproject.toml`) |
| Memory | no step may densify or fully load a cell × gene matrix |
| Exclusions | every dropped cell, group or gene is counted in a report |
| Hygiene | no personal paths, emails, cluster accounts, tokens or lab data; use `/path/to/...`, `<partition>`, `<account>` |
| Versions | bump `pyproject.toml` and `__init__.py`, add a CHANGELOG line, tag `vX.Y.Z` |
