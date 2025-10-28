# meta-prov-fixer

A small toolkit and pipeline to detect and fix provenance issues in the OpenCitations "Meta" dataset.

This repository provides a set of fixers that run detection queries (either against a SPARQL endpoint or reading local RDF dump files) and apply corrective updates to the triplestore. The pipeline coordinates the fixers, supports checkpointing, logging and a dry-run mode for safe testing.

## Features

- Pipeline orchestration (ordered fixers with checkpointing and timing)
- Multiple fixers implemented:
  - `FillerFixer` — remove filler snapshots and rename/adjust remaining snapshots
  - `DateTimeFixer` — normalize ill-formed datetime values (make them offset-aware with a consistent format and remove microseconds)
  - `MissingPrimSourceFixer` — add primary source quads for creation snapshots missing them
  - `MultiPAFixer` — normalize snapshots with multiple `prov:wasAttributedTo` values
  - `MultiObjectFixer` — reset graphs where snapshots have too many objects for single-valued properties (creating a new creation snapshot)

- Two operating modes:
  - SPARQL endpoint mode (detect issues by querying an endpoint and apply fixes)
  - File-based detection mode (read RDF/JSON-LD dumps locally for detection and still apply fixes to the endpoint)

## Requirements

The project uses Python 3.11 (see `pyproject.toml`). Key runtime dependencies are:

- rdflib
- SPARQLWrapper
- tqdm
- tzdata

Note: the project provides development dependencies (pytest, notebook, pandas) in `pyproject.toml`.

## Quick usage

The main CLI entrypoint is `meta_prov_fixer/main.py`. It accepts the following options (brief):

- `-e, --endpoint` (required) — SPARQL endpoint URL to update.
- `-m, --meta-dumps` (required) — Path to a JSON file containing a list of published meta-dump records; this must be a JSON list of 2-item arrays: `["YYYY-MM-DD", "<dump-doi-or-url>"]`.
- `-i, --issues-log-dir` — Directory where detected issues (JSON Lines) will be written. Required when `--dump-dir` is used.
- `-d, --dump-dir` — If provided, detection will read RDF dump provenance files from this directory instead of querying the SPARQL endpoint.
- `-c, --checkpoint` — Path to a checkpoint file (default: `checkpoint.json`).
- `--dry-run` — Run pipeline in dry-run mode (no updates applied; useful for debugging).
- `-l, --log-fp` — File path for pipeline logs. Defaults to `provenance_fix_<today>.log`.

Examples:

Detect issues from the SPARQL endpoint and apply fixes (only applicable with small datasets, due to memory limits):

```shell
python -m meta_prov_fixer.main -e http://localhost:8890/sparql/ -m meta_dumps.json
```

Detect issues from the SPARQL endpoint, save them to disk, and apply fixes (might incur in timeout errors if the dataset is very large):

```shell
python -m meta_prov_fixer.main -e http://localhost:8890/sparql/ -m meta_dumps.json -i ./data_to_fix
```

Detect issues by reading RDF dump files, store issues to disk, then apply fixes to the endpoint (particularly useful with large datasets):

```shell
python -m meta_prov_fixer.main -e http://localhost:8890/sparql/ -m meta_dumps.json -i ./data_to_fix -d C:/path/to/rdf/dumps
```

## Input format for `--meta-dumps`

The `--meta-dumps` argument expects a JSON file containing a top-level array of two-item arrays (date and URL). Example (`meta_dumps.json`):

```json
[
  ["2022-12-19", "https://doi.org/10.6084/m9.figshare.21747536.v1"],
  ["2022-12-20", "https://doi.org/10.6084/m9.figshare.21747536.v2"],
  ["2023-02-15", "https://doi.org/10.6084/m9.figshare.21747536.v3"],
  ["2023-06-28", "https://doi.org/10.6084/m9.figshare.21747536.v4"],
  ["2023-10-26", "https://doi.org/10.6084/m9.figshare.21747536.v5"],
  ["2024-04-06", "https://doi.org/10.6084/m9.figshare.21747536.v6"],
  ["2024-06-17", "https://doi.org/10.6084/m9.figshare.21747536.v7"],
  ["2025-02-02", "https://doi.org/10.6084/m9.figshare.21747536.v8"],
  ["2025-06-06", "https://doi.org/10.5281/zenodo.15855112"]
]
```

The date format must be ISO-style (YYYY-MM-DD). The CLI loader validates the structure and will raise an error for invalid files.

## Output and logging

- A log file is written to the path supplied with `-l/--log-fp` (default includes date in filename).
- When `--issues-log-dir` is provided, each fixer writes detected issues as JSON Lines files to that directory. Following fixes will stream-read these files for updating the endpoint.
- A checkpoint file (default: `checkpoint.json`) is used to resume the pipeline if interrupted. The pipeline clears the checkpoint after successful completion.

## Project layout

- `meta_prov_fixer/` — package code
  - `main.py` — CLI entrypoint and argument parsing (this file)
  - `fix_via_sparql.py` — fixer implementations and the pipeline orchestration (detection and update logic)
  - `utils.py` — shared helpers (checkpoint management, RDF dump reading, small utilities)
- `tests/` — unit tests (pytest)

## Developer notes

- Use `--dry-run` to validate detection and simulate the pipeline without executing queries.
- When using file-based detection (`--dump-dir`), supply `--issues-log-dir` so detected issues are stored as JSONL files; those files can be inspected or edited and then used by the same pipeline to apply updates.
- The pipeline uses a per-fixer checkpointing mechanism so long-running runs can be resumed after interruptions.
