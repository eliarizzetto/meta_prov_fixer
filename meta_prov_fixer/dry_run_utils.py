"""
Utilities for dry-run mode, including callbacks for writing issues to JSON-Lines files.
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional
from meta_prov_fixer.utils import make_json_safe


def create_dry_run_issues_callback(
    output_dir: str,
    max_lines_per_file: int = 100,
    process_id: Optional[str] = None
) -> Callable:
    """
    Creates a callback function for dry_run mode that writes issues to JSON-Lines files.

    Each line in the JSON-Lines file has the structure:
    {filepath:str, ff:list, dt:list, mps:list, pa:list, mo:list}

    This implementation is safe for parallel execution when used with run_parallel_fix.py,
    as it uses unique file naming based on process_id and atomic writes.

    Args:
        output_dir: Directory where issues files will be written.
        max_lines_per_file: Maximum number of lines per file. Default is 100.
        process_id: Optional identifier for parallel execution (e.g., directory name like 'br', 'ar').
                    If provided, will be included in the output filename for uniqueness.

    Returns:
        A callback function compatible with fix_provenance_process() that accepts
        (file_path, (ff_issues, dt_issues, mps_issues, pa_issues, mo_issues))
    """
    # Create output directory
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique session ID when callback is created (not per file)
    # This ensures all files from this callback instance share the same session identifier
    # Use microseconds to ensure uniqueness even when callbacks are created rapidly
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session_pid = os.getpid()
    
    # Create a unique session ID that combines timestamp, PID, and optionally process_id
    # This prevents any possibility of filename collisions, even with parallel execution
    if process_id:
        session_id = f"{process_id}_{session_timestamp}_{session_pid}"
    else:
        session_id = f"{session_timestamp}_{session_pid}"

    # State variables for the callback
    state = {
        'lines_written': 0,
        'current_file_number': 0,
        'current_file_path': None
    }

    def _get_new_filename() -> Path:
        """
        Generate a new unique filename for the next chunk.
        
        The filename includes the session_id (generated once when callback is created),
        ensuring all files from the same session are grouped together and
        preventing any possibility of overwriting files from other sessions,
        even with parallel execution.
        """
        filename = f"dry_run_issues_{session_id}_chunk{state['current_file_number']}.jsonl"
        return output_dir / filename

    def _atomic_write_line(filepath: Path, line: str):
        """
        Append a line to a file atomically using a temporary file and os.replace().
        This ensures thread/process safety for concurrent writes.
        """
        # Read existing content if file exists
        existing_lines = []
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        
        # Add new line
        existing_lines.append(line + '\n')
        
        # Write to temporary file
        tmp_path = filepath.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.writelines(existing_lines)
        
        # Atomically replace (os.replace is atomic on Unix and Windows)
        os.replace(tmp_path, filepath)

    def dry_run_callback(
        file_path: str,
        issues: tuple
    ):
        """
        Callback function that writes issues to JSON-Lines files.

        Args:
            file_path: Path to the processed file.
            issues: Tuple of (ff_issues, dt_issues, mps_issues, pa_issues, mo_issues)
        """
        ff_issues, dt_issues, mps_issues, pa_issues, mo_issues = issues

        # Create JSON-safe version of issues using make_json_safe
        # This converts rdflib.URIRef and other objects to strings
        safe_ff = make_json_safe(ff_issues)
        safe_dt = make_json_safe(dt_issues)
        safe_mps = make_json_safe(mps_issues)
        safe_pa = make_json_safe(pa_issues)
        safe_mo = make_json_safe(mo_issues)

        # Prepare the record
        record = {
            'filepath': file_path,
            'ff': safe_ff,
            'dt': safe_dt,
            'mps': safe_mps,
            'pa': safe_pa,
            'mo': safe_mo
        }

        # Convert to JSON line
        json_line = json.dumps(record, ensure_ascii=False)

        # Check if we need a new file
        if (state['current_file_path'] is None or 
            state['lines_written'] >= max_lines_per_file):
            
            # Update file number
            state['current_file_number'] += 1
            state['current_file_path'] = _get_new_filename()
            state['lines_written'] = 0

        # Write the line atomically
        _atomic_write_line(state['current_file_path'], json_line)
        state['lines_written'] += 1

    return dry_run_callback