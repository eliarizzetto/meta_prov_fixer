import subprocess
import sys
from pathlib import Path
import argparse
import os
from typing import Optional

from meta_prov_fixer.virtuoso_watchdog import start_watchdog_thread

# ================== CONFIG DEFAULT ==================

META_DUMPS_DEFAULT = "meta_dumps.json"

ENDPOINT_DEFAULT = "http://localhost:8890/sparql/"
# CONTAINER_DEFAULT = "oc-meta-prov"

DIRS = ["br", "ar", "re", "ra", "id"]

LOGS_DIR = Path("logs")
CHECKPOINTS_DIR = Path("checkpoints")
CACHES_DIR = Path("caches")
FAILED_DIR = Path("failed_queries")

for d in (LOGS_DIR, CHECKPOINTS_DIR, CACHES_DIR, FAILED_DIR):
    d.mkdir(exist_ok=True)


env = os.environ.copy()  # env for subprocesses
env["TQDM_DISABLE"] = "1"  # disable tqdm in subprocesses

# ================== FIXER LAUNCH ==================

def launch_fixer(
    endpoint: str,
    base_in: str,
    base_out: str,
    meta_dumps_fp: str,
    dir_name: str,
    dry_run_db: bool = False,
    dry_run_files: bool = False,
    dry_run_issues_dir: Optional[str] = None
) -> subprocess.Popen:
    print(f">>> Launching fixer for '{dir_name}'")

    cmd = [
        "poetry", "run", "python", "meta_prov_fixer/main.py",
        "-e", endpoint,
        "-i", f"{base_in}/{dir_name}",
        "-o", f"{base_out}/{dir_name}",
        "-m", meta_dumps_fp,
        # file isolati
        "--checkpoint-fp", str(CHECKPOINTS_DIR / f"fix_prov_{dir_name}.checkpoint.json"),
        "--cache-fp", str(CACHES_DIR / f"filler_issues_{dir_name}.cache.json"),
        "--failed-queries-fp", str(FAILED_DIR / f"failed_{dir_name}.txt"),
        "-l", str(LOGS_DIR / f"provenance_fix_{dir_name}.log"),
    ]

    # Add dry-run flags if specified
    if dry_run_db:
        cmd.append("--dry-run-db")
    if dry_run_files:
        cmd.append("--dry-run-files")
    if dry_run_issues_dir:
        cmd.extend(["--dry-run-issues-dir", dry_run_issues_dir])
        cmd.extend(["--dry-run-process-id", dir_name])  # Use dir_name as process_id for uniqueness

    return subprocess.Popen(cmd, env=env)

# ================== MAIN ==================

def main():
    parser = argparse.ArgumentParser(description="Run multiple fixer processes in parallel.")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT_DEFAULT,
                        help="SPARQL endpoint URL")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Directory containing input subfolders")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Directory where fixed outputs will be saved (in subfolders)")
    parser.add_argument("--meta-dumps", type=str, default=META_DUMPS_DEFAULT,
                        help="Path to meta dumps register JSON file")
    
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
        help="Skip SPARQL updates to endpoint (write only fixed files)."
    )

    parser.add_argument(
        "--dry-run-files", action="store_true",
        help="Skip writing fixed files to out-dir (update only database)."
    )

    parser.add_argument(
        "--dry-run-issues-dir", type=str, default=None,
        help="Directory where to write issues found during dry-run as JSON-Lines files."
    )

    args = parser.parse_args()

    if args.auto_restart_container:
        if not args.virtuoso_container:
            parser.error(
                "--virtuoso-container is required when using --auto-restart-container"
            )

    if args.auto_restart_container:
        print("Starting single Virtuoso watchdog (launcher-controlled)")
        start_watchdog_thread(args.virtuoso_container, args.endpoint)
    else:
        print("Watchdog disabled. Processes will not be auto-restarted.")

    # Launch fixers
    processes = [
        (d, launch_fixer(
            args.endpoint,
            args.base_dir,
            args.out_dir,
            args.meta_dumps,
            d,
            dry_run_db=args.dry_run_db,
            dry_run_files=args.dry_run_files,
            dry_run_issues_dir=args.dry_run_issues_dir
        ))
        for d in DIRS
    ]
    exit_code = 0

    try:
        for d, proc in processes:
            ret = proc.wait()
            if ret != 0:
                print(f"Fixer '{d}' exited with code {ret}")
                exit_code = ret
            else:
                print(f"Fixer '{d}' completed successfully")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received: terminating all fixer processes")
        for _, p in processes:
            p.terminate()
        sys.exit(1)

    print("All fixer processes completed")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
