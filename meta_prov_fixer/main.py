#!/usr/bin/env python3
import argparse
import json
import logging
import datetime

from meta_prov_fixer.fix_via_sparql import fix_process, fix_process_reading_from_files
from meta_prov_fixer.virtuoso_watchdog import start_watchdog_thread

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
        description="Run the pipeline for fixing Meta provenance triplestore"
    )

    parser.add_argument(
        "-e", "--endpoint", type=str, required=True,
        help="SPARQL endpoint URL"
    )

    parser.add_argument(
        "-m", "--meta-dumps", type=load_meta_dumps, required=True,
        help="Path to JSON file with list of [date, URL] pairs"
    )

    parser.add_argument(
        "-i", "--issues-log-dir", type=str, default=None,
        help="Directory to save data to fix. Required if using --dump-dir."
    )

    parser.add_argument(
        "-c", "--checkpoint", type=str, default="checkpoint.json",
        help="Path to checkpoint file"
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run in dry-run mode (no update queries made to the database)."
    )

    parser.add_argument(
        "-d", "--dump-dir", type=str, default=None,
        help="Path to directory containing RDF dumps. If provided, "
             "the pipeline reads from files instead of querying the endpoint."
    )

    parser.add_argument(
        "-l", "--log-fp", type=str,
        default=f"provenance_fix_{datetime.date.today().strftime('%Y-%m-%d')}.log",
        help="File path to log file."
    )

    parser.add_argument(
        "-r", "--auto-restart-container", action="store_true",
        help="Enable memory watchdog to auto-restart the Virtuoso Docker container when memory usage is too high."
    )

    parser.add_argument(
        "-v", "--virtuoso-container", type=str, default=None,
        help="Name of the Virtuoso Docker container (required when --auto-restart-container is used)."
    )

    args = parser.parse_args()

    if args.auto_restart_container:
        if not args.virtuoso_container:
            parser.error(
                "--virtuoso-container is required when using --auto-restart-container"
            )

    # --- Enforce issues_log_dir if dump_dir is used ---
    if args.dump_dir and not args.issues_log_dir:
        parser.error("--issues-log-dir (-i) is required when using --dump-dir")

    # --- Logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=args.log_fp
    )

    logging.info("Starting provenance fixing pipeline…")
    logging.info(f"Endpoint: {args.endpoint}")
    logging.info(f"Dump dir: {args.dump_dir or 'None (SPARQL endpoint mode)'}")
    logging.info(f"Issues log dir: {args.issues_log_dir}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info(f"Auto-restart enabled: {args.auto_restart_container}")
    logging.info(f"Virtuoso container: {args.virtuoso_container}")

    # --- Start the Virtuoso memory watchdog thread if enabled ---
    if args.auto_restart_container:
        logging.info("Starting Virtuoso memory watchdog thread...")
        start_watchdog_thread(
            container_name=args.virtuoso_container,
            endpoint=args.endpoint
        )
    else:
        logging.info("Auto-restart watchdog disabled.")

    # --- Choose processing mode ---
    if args.dump_dir:
        logging.info("Running pipeline in 'file-based' mode (reading from RDF dumps).")
        fix_process_reading_from_files(
            endpoint=args.endpoint,
            dump_dir=args.dump_dir,
            issues_log_dir=args.issues_log_dir,
            meta_dumps_pub_dates=args.meta_dumps,
            dry_run=args.dry_run,
            checkpoint_fp=args.checkpoint
        )
    else:
        logging.info("Running pipeline in 'SPARQL endpoint' mode.")
        fix_process(
            endpoint=args.endpoint,
            meta_dumps_pub_dates=args.meta_dumps,
            issues_log_dir=args.issues_log_dir or "data_to_fix",
            dry_run=args.dry_run,
            checkpoint_fp=args.checkpoint
        )

    logging.info("Provenance fixing pipeline completed successfully.")


if __name__ == "__main__":
    main()


## Detect issues from DB and fix on DB (storing errors in memory only):
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -m meta_dumps.json 


## Detect issues from RDF files and fix on DB:
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -m meta_dumps.json -i "./data_to_fix" -d "/meta/dump/directory/"


## Detect issues from RDF files and fix on DB, automatically restarting Virtuoso Docker container if memory usage exceeds 98%
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -m meta_dumps.json -i ./data_to_fix -d "/meta/dump/directory/" --auto-restart-container --virtuoso-container <container_name>



