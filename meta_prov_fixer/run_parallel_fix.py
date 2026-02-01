import subprocess
import sys
from pathlib import Path
import argparse
import os

from virtuoso_watchdog import start_watchdog_thread

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

def launch_fixer(endpoint: str, base_in: str, base_out: str, meta_dumps_fp: str, dir_name: str) -> subprocess.Popen:
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

    processes = [(d, launch_fixer(args.endpoint, args.base_dir, args.out_dir, args.meta_dumps, d)) for d in DIRS]
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
