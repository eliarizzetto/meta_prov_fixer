#!/usr/bin/env python3
import argparse
import json
import logging
import datetime

from meta_prov_fixer.src import fix_provenance_process

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

    args = parser.parse_args()


    # --- Logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=args.log_fp
    )

    fix_provenance_process(
        endpoint_url=args.endpoint,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        meta_dumps_pub_dates=args.meta_dumps,
        chunk_size=args.chunk_size,
        failed_queries_fp=args.failed_queries_fp,
        overwrite_ok=args.overwrite_ok,
        resume=True,
        checkpoint_fp=args.checkpoint_fp,
        cache_fp=args.cache_fp
    )


if __name__ == "__main__":
    main()



## Detect and fix provenance issues:
## poetry run python meta_prov_fixer/main.py -e http://localhost:8890/sparql/ -i "../meta_prov/br" -o "../fixed/br" -m meta_dumps.json 
