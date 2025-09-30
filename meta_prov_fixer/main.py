#!/usr/bin/env python3
import argparse
import json
import logging
from meta_prov_fixer.fix_via_sparql import fix_process
import datetime

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
        "-i", "--issues-log-dir", type=str, default="data_to_fix",
        help="Directory to save data to fix"
    )

    parser.add_argument(
        "-c", "--checkpoint", type=str, default="checkpoint.json",
        help="Path to checkpoint file"
    )

    parser.add_argument(
        "-d", "--dry-run", action="store_true",
        help="Run in dry-run mode (no queries made to database, just for debugging purposes)"
    )

    parser.add_argument(
        "-l", "--log-fp", type=str, default=f"provenance_fix_{datetime.date.today().strftime('%Y-%m-%d')}.log",
        help="File path to log file."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG, # INFO
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=args.log_fp
    )

    fix_process(
        endpoint=args.endpoint,
        meta_dumps_pub_dates=args.meta_dumps,
        issues_log_dir=args.issues_log_dir,
        dry_run=args.dry_run,
        checkpoint=args.checkpoint
    )

if __name__ == "__main__":
    main()



# poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -m meta_dumps.json 

