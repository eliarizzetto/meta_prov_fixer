# meta-prov-fixer

A toolkit to detect and fix issues in the OpenCitations Meta provenance dataset.

This repository provides a set of fixers that detect issues reading local RDF dump files and apply corrective updates to the triplestore and the RDF files. The pipeline coordinates the fixers, supports checkpointing, and logging.

## Features

- Pipeline orchestration (ordered fixers with checkpointing and progress bar)
- Multiple fixers implemented:
  - `FillerFixer` — remove filler snapshots and rename/adjust remaining snapshots
  - `DateTimeFixer` — normalize ill-formed datetime values (make them offset-aware with a consistent format and remove microseconds)
  - `MissingPrimSourceFixer` — add primary source quads for creation snapshots missing them
  - `MultiPAFixer` — normalize snapshots with multiple `prov:wasAttributedTo` values
  - `MultiObjectFixer` — reset graphs where snapshots have too many objects for single-valued properties (creating a new creation snapshot)

## Requirements

The project uses Python 3.11 and manages dependencies with Python Poetry (see `pyproject.toml`). Key runtime dependencies are:

- rdflib
- SPARQLWrapper
- tqdm
- tzdata
- docker

Make sure you have Poetry installed. Then, install dependencies with:

```bash
poetry install
```

and activate the virtual environment with:

```bash
poetry env activate # for Poetry >= 2.2
```

## Quick usage

The main CLI entrypoint is `meta_prov_fixer/main.py`. It accepts the following options (brief):

- `-e`, `--endpoint` **(required)** — SPARQL endpoint URL.
- `-i`, `--data-dir` **(required)** — Path to the directory containing the RDF files to process.
- `-o`, `--out-dir` **(required)** — Directory where fixed files will be written. If this is the same as `--data-dir` and `--overwrite-ok` is not set, an error will be raised.
- `-m`, `--meta-dumps` **(required)** — Path to a JSON file with a list of `[date, URL]` pairs; the loader validates the structure (see "Input format for `--meta-dumps`" below).
- `--chunk-size` — Number of detected issues included in each SPARQL update (default: `100`).
- `--failed-queries-fp` — File path to log failed SPARQL update queries (default: `prov_fix_failed_queries_<YYYY-MM-DD>.txt`).
- `-l`, `--log-fp` — File path for the run log (default: `provenance_fix_<YYYY-MM-DD>.log`).
- `--overwrite-ok` — Allow overwriting input files when `--out-dir` equals `--data-dir` and the input files are decompressed `.json` (default: not set).
- `--checkpoint-fp` — Path for the checkpoint file used to resume a run (default: `fix_prov.checkpoint.json`).
- `--cache-fp` — Path for the issues cache file (default: `filler_issues.cache.json`).

### Example

Detect issues from RDF files and fix them on the triplestore, and new correct copies of invalid files:

```shell
poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -i "../meta_prov/br" -o "../fixed/br" -m meta_dumps.json 
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
- A checkpoint file (default: `fix_prov.checkpoint.json`) is used to resume the pipeline if interrupted. The pipeline clears the checkpoint after successful completion.

## Developer notes

- Use `--dry-run` to get detected issues and simulate the pipeline without executing queries or updates to files. Dry runs require a callback function to be specified as the value of `dry_run_callback` parameter in `src.fix_provenance_process()`.
- The pipeline uses a per-file and per-fixer checkpointing mechanism so long-running runs can be resumed after interruptions.
