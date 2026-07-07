"""Execute example notebooks locally before a release.

Each marimo notebook is a plain Python file and is executed as a script from the
repository root. Run all notebooks, or pass substring filters to select a subset:

    uv run python examples/run_notebooks.py                # all notebooks
    uv run python examples/run_notebooks.py centrality     # matching paths only

Network-dependent notebooks (OSM downloads) require connectivity; a short delay
between runs keeps request rates polite. Exits nonzero if any notebook fails.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 1800
DELAY_S = 5

# class/ lessons are excluded: they contain deliberately failing teaching cells
# (error demos) and are meant for interactive use, not scripted execution
notebooks = sorted(p for p in REPO_ROOT.glob("examples/recipes/**/*.py") if p.name != "run_notebooks.py")

filters = sys.argv[1:]
if filters:
    notebooks = [p for p in notebooks if any(f in str(p) for f in filters)]

failed: list[tuple[Path, str]] = []
for nb in notebooks:
    rel = nb.relative_to(REPO_ROOT)
    print(f"=== {rel}", flush=True)
    try:
        # run from the notebook's own directory: notebook-relative paths (../../data,
        # images/...) then resolve identically to marimo's edit mode
        result = subprocess.run(
            [sys.executable, str(nb)],
            cwd=nb.parent,
            env={**os.environ, "CITYSEER_QUIET_MODE": "true", "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        if result.returncode != 0:
            tail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr"
            failed.append((rel, tail))
            print(f"FAILED: {rel} — {tail}", flush=True)
        else:
            print(f"OK: {rel}", flush=True)
    except subprocess.TimeoutExpired:
        failed.append((rel, f"timeout after {TIMEOUT_S}s"))
        print(f"FAILED: {rel} — timeout", flush=True)
    if nb is not notebooks[-1]:
        time.sleep(DELAY_S)

print(f"\n{len(notebooks) - len(failed)}/{len(notebooks)} notebooks passed")
for rel, msg in failed:
    print(f"  FAILED {rel}: {msg}")
sys.exit(1 if failed else 0)
