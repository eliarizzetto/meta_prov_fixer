import time
import threading
import logging
import docker
from SPARQLWrapper import SPARQLWrapper, JSON
import traceback

def wait_for_sparql(endpoint: str, timeout: int = 120):
    """Wait until SPARQL endpoint responds."""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery("ASK {}")
    sparql.setReturnFormat(JSON)

    start = time.time()
    while time.time() - start < timeout:
        try:
            sparql.query()
            return True
        except Exception:
            time.sleep(2)
    return False


def monitor_and_restart(
    container_name: str,
    endpoint: str,
    threshold: float = 0.98,
    interval: int = 180
):
    """
    Background thread: monitor Docker memory usage and restart Virtuoso
    when usage exceeds `threshold` of limit.

    threshold=0.98 means 98% of allowed container memory.
    """
    client = docker.from_env()

    while True:
        try:
            container = client.containers.get(container_name)
            stats = container.stats(stream=False)

            ## Unlike the output of Docker CLI "stats" command, Docker API stats include 
            ## cached memory in "usage", so subtract it out for measuring actual memory use.
            used = stats["memory_stats"]["usage"]
            limit = stats["memory_stats"]["limit"]
            cache = stats["memory_stats"]["stats"]["inactive_file"] # stores cached data
            effective_used = used - cache
            ratio = effective_used / limit

            logging.info(f"[Virtuoso watchdog] Mem use: {effective_used/1e9:.2f}GB / {limit/1e9:.2f}GB ({ratio*100:.1f}%)")
            if ratio > threshold:
                logging.warning(f"[Virtuoso watchdog] Memory above {threshold*100}% → restarting Virtuoso container")

                container.restart()
                logging.info("[Virtuoso watchdog] Waiting for SPARQL endpoint to come back online…")

                if wait_for_sparql(endpoint):
                    logging.info("[Virtuoso watchdog] SPARQL endpoint is back online")
                else:
                    logging.error("[Virtuoso watchdog] SPARQL endpoint DID NOT recover within timeout!")

        except Exception as e:
            print(traceback.format_exc())
            logging.error(f"[Virtuoso watchdog] Error: {e}")

        time.sleep(interval)


def start_watchdog_thread(container_name: str, endpoint: str):
    t = threading.Thread(
        target=monitor_and_restart,
        args=(container_name, endpoint),
        daemon=True
    )
    t.start()
