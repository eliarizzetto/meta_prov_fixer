#!/usr/bin/env python3
import argparse
import json
import logging
import datetime

from meta_prov_fixer.src import fix_provenance_process
from meta_prov_fixer.virtuoso_watchdog import start_watchdog_thread
from meta_prov_fixer.dry_run_utils import create_dry_run_issues_callback

def load_meta_dumps(json_path: str):
    """
    Load meta_dumps_pub_dates from a JSON file.
    The JSON file should contain a list of [date, url] pairs.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not all(isinstance(t, list) and len(t) == 2 for t in data):
            raise ValueError
        return [(str(d[0]), str(d[1])) for d in data]
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Failed to load meta_dumps_pub_dates from '{json_path}': {e}"
        )

def main():
    parser = argparse.ArgumentParser(
        description="Run the pipeline for fixing Meta provenance triplestore and RDF files."
    )

    parser.add_argument(
        "-e", "--endpoint", type=str, required=True,
        help="SPARQL endpoint URL"
    )

    parser.add_argument(
        "-i", "--data-dir", type=str, required=True,
        help="Path to directory containing the RDF files to process."
    )

    parser.add_argument(
        "-o", "--out-dir", type=str, required=True,
        help="Directory where to save fixed files. If it is the same as data-dir and 'overwrite' is False, an Error will be raised."
    )

    parser.add_argument(
        "-m", "--meta-dumps", type=load_meta_dumps, required=True,
        help="Path to JSON file with list of [date, URL] pairs"
    )

    parser.add_argument(
        "--chunk-size", type=int, default=100,
        help="Number of detected issues to process in each SPARQL update query. Default is 100."
    )

    parser.add_argument(
        "--failed-queries-fp", type=str, default=f"prov_fix_failed_queries_{datetime.date.today().strftime('%Y-%m-%d')}.txt",
        help="File path to log failed SPARQL update queries. Default is 'prov_fix_failed_queries_<today's date>.txt'."
    )

    parser.add_argument(
        "-l", "--log-fp", type=str,
        default=f"provenance_fix_{datetime.date.today().strftime('%Y-%m-%d')}.log",
        help="File path to log file. Default is 'provenance_fix_<today's date>.log'."
    )

    parser.add_argument(
        "--overwrite-ok", action="store_true",
        help="If specified, allows overwriting the input file with the fixed output without raising errors. "
            "To be overwritten, the input file must still be a decompressed .json file and '--out-dir' must be "
            "the same as '--data-dir'. Default is False."
    )

    parser.add_argument(
        "--checkpoint-fp", type=str, default="fix_prov.checkpoint.json",
        help="File path to store checkpoint information for resuming the process. Default is 'fix_prov.checkpoint.json'."
    )

    parser.add_argument(
        "--cache-fp", type=str, default="filler_issues.cache.json",
        help="File path to store cache of detected issues. Default is 'filler_issues.cache.json'."
    )

    parser.add_argument(
        "-r", "--auto-restart-container", action="store_true",
        help="Enable memory watchdog to auto-restart the Virtuoso Docker container when memory usage is too high."
    )

    parser.add_argument(
        "-v", "--virtuoso-container", type=str, default=None,
        help="Name of the Virtuoso Docker container (required when --auto-restart-container is used)."
    )

    parser.add_argument(
        "--dry-run-db", action="store_true",
        help="If specified, no SPARQL updates are sent to the endpoint. Useful for testing or when you only want to write fixed files."
    )

    parser.add_argument(
        "--dry-run-files", action="store_true",
        help="If specified, no output files are written to out-dir. Useful when you only want to update the database."
    )

    parser.add_argument(
        "--dry-run-issues-dir", type=str, default=None,
        help="Directory where to write issues found during dry-run. If specified with --dry-run-db, creates JSON-Lines files "
             "with issues found in each processed file. Each file contains at most 1000 lines. The callback is only used when "
             "--dry-run-db is enabled."
    )

    parser.add_argument(
        "--dry-run-process-id", type=str, default=None,
        help="Optional identifier for parallel execution (e.g., directory name like 'br', 'ar'). Used to create unique filenames "
             "when running multiple processes with --dry-run-issues-dir to avoid file conflicts."
    )

    args = parser.parse_args()

    if args.auto_restart_container:
        if not args.virtuoso_container:
            parser.error(
                "--virtuoso-container is required when using --auto-restart-container"
            )


    # --- Logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(funcName)s, %(filename)s:%(lineno)d] - %(message)s",
        filename=args.log_fp
    )


    # --- Start the Virtuoso memory watchdog thread if enabled ---
    if args.auto_restart_container:
        logging.info("Starting Virtuoso memory watchdog thread...")
        start_watchdog_thread(
            container_name=args.virtuoso_container,
            endpoint=args.endpoint
        )
    else:
        logging.info("Auto-restart watchdog disabled.")

    # --- Setup dry-run callback if needed ---
    dry_run_callback = None
    if args.dry_run_db and args.dry_run_issues_dir:
        logging.info(f"Creating dry-run issues callback writing to: {args.dry_run_issues_dir}")
        dry_run_callback = create_dry_run_issues_callback(
            output_dir=args.dry_run_issues_dir,
            max_lines_per_file=1000,
            process_id=args.dry_run_process_id
        )

    fix_provenance_process(
        endpoint=args.endpoint,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        meta_dumps_register=args.meta_dumps,
        dry_run_db=args.dry_run_db,
        dry_run_files=args.dry_run_files,
        dry_run_callback=dry_run_callback,
        chunk_size=args.chunk_size,
        failed_queries_fp=args.failed_queries_fp,
        overwrite_ok=args.overwrite_ok,
        resume=True,
        checkpoint_fp=args.checkpoint_fp,
        cache_fp=args.cache_fp
    )


if __name__ == "__main__":
    main()



## Detect and fix provenance issues (with auto-restart watchdog for Virtuoso):
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -i "../meta_prov/br" -o "../fixed/br" -m meta_dumps.json -r -v oc-meta-prov

## Run in dry-run mode: only write fixed files, don't update database:
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -i "../meta_prov/br" -o "../fixed/br" -m meta_dumps.json --dry-run-db

## Run in dry-run mode with issues logging: only write fixed files and log issues to JSON-Lines:
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -i "../meta_prov/br" -o "../fixed/br" -m meta_dumps.json --dry-run-db --dry-run-issues-dir "issues_output"

## Run in dry-run mode with issues logging for parallel execution (avoiding filename conflicts):
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -i "../meta_prov/br" -o "../fixed/br" -m meta_dumps.json --dry-run-db --dry-run-issues-dir "issues_output" --dry-run-process-id "br"
