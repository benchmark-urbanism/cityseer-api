"""Export example notebooks to static HTML for the documentation site.

Runs each marimo notebook and exports it (with outputs) to docs/public/examples/notebooks/,
copies the raw .py alongside for download, and writes a manifest consumed by the Astro
dynamic routes. All outputs are untracked build artifacts; this runs locally as part of the
release/deploy flow, never on CI.

    uv run --group examples python docs/export_notebooks.py            # changed notebooks only
    uv run --group examples python docs/export_notebooks.py centrality # filtered
    uv run --group examples python docs/export_notebooks.py --force    # full re-export

Incremental by default: a notebook is skipped when its exported HTML is newer than its
source (its manifest entry is still refreshed). Exits nonzero if any export fails.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "docs" / "public" / "examples" / "notebooks"
MANIFEST_PATH = REPO_ROOT / "docs" / "src" / "generated" / "notebooks.json"
TIMEOUT_S = 1800
IMAGE_EXTS = {".png", ".jpg", ".webp"}


def finalize_export(nb: Path, html_out: Path) -> None:
    """Post-process an exported notebook. Idempotent; runs on fresh exports and on the
    incremental skip path so previously exported HTML picks up fixes without re-execution.

    - Injects <base target="_top"> so links inside the notebook iframe open at the top
      level instead of navigating the iframe itself (skipped if a <base> already exists).
    - Publishes any local images/ directory next to the notebook source (png/jpg/webp)
      alongside the exported HTML, since notebooks reference them relatively.
    """
    html = html_out.read_text()
    if "<base" not in html and "<head>" in html:
        html_out.write_text(html.replace("<head>", '<head><base target="_top">', 1))
    src_images = nb.parent / "images"
    if src_images.is_dir():
        dst = html_out.parent / "images"
        dst.mkdir(exist_ok=True)
        for img in src_images.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                shutil.copy2(img, dst / img.name)


# lesson number -> slug: 1_notebooks.py -> 1-notebooks
def slugify(stem: str) -> str:
    return stem.replace("_", "-")


def plain_text(md: str) -> str:
    """Reduce inline markdown to plain text: links keep their text, code spans keep
    their contents (only the backticks are dropped), emphasis markers are removed
    without touching underscores inside identifiers."""
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    md = md.replace("`", "")
    md = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", md)
    md = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", md)
    return md.strip()


def truncate(text: str, limit: int = 200) -> str:
    """Truncate to ~limit chars at a word boundary for meta descriptions."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}…"


def first_title_and_description(nb_path: Path) -> tuple[str, str]:
    """Pull the first markdown H1 and the first prose paragraph from a marimo notebook.

    Falls back to a title-cased filename when no H1 is present. Blockquotes and
    directive fences are skipped when picking the description.
    """
    text = nb_path.read_text()
    title = nb_path.stem.replace("_", " ").title()
    description = ""
    md_blocks = re.findall(r'mo\.md\(\s*r?"""(.*?)"""\s*\)', text, re.DOTALL)
    seen_h1 = any(ln.strip().startswith("# ") for block in md_blocks for ln in block.split("\n"))
    started = not seen_h1  # no H1: fall back to the first prose paragraph anywhere
    for block in md_blocks:
        for ln in (ln.strip() for ln in block.split("\n")):
            if not started:
                if ln.startswith("# "):
                    title = plain_text(ln[2:])
                    started = True
                continue
            if not ln or ln.startswith(("#", ">", ":::")):
                continue
            description = ln
            break
        if description:
            break
    return title, truncate(plain_text(description))


def collect() -> list[dict]:
    entries = []
    for nb in sorted((REPO_ROOT / "examples" / "recipes").rglob("*.py")):
        topic = nb.parent.name if nb.parent.name != "recipes" else "recipes"
        entries.append({"source": nb, "section": "examples", "topic": topic})
    for nb in sorted((REPO_ROOT / "examples" / "class").glob("*.py")):
        entries.append({"source": nb, "section": "learn", "topic": "class"})
    return entries


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    filters = [a for a in args if a != "--force"]
    entries = collect()
    if filters:
        entries = [e for e in entries if any(f in str(e["source"]) for f in filters)]

    manifest = []
    failed: list[tuple[Path, str]] = []
    for entry in entries:
        nb: Path = entry["source"]
        rel = nb.relative_to(REPO_ROOT / "examples")
        slug = slugify(nb.stem)
        out_dir = OUT_ROOT / entry["topic"] if entry["section"] == "examples" else OUT_ROOT / "learn"
        out_dir.mkdir(parents=True, exist_ok=True)
        html_out = out_dir / f"{slug}.html"
        # incremental: skip execution when the export is newer than the source, but still
        # refresh the manifest entry (metadata extraction reads the source without executing)
        if not force and html_out.exists() and html_out.stat().st_mtime >= nb.stat().st_mtime:
            finalize_export(nb, html_out)
            shutil.copy2(nb, out_dir / f"{slug}.py")
            title, description = first_title_and_description(nb)
            manifest.append(
                {
                    "slug": slug,
                    "section": entry["section"],
                    "topic": entry["topic"],
                    "title": title,
                    "description": description,
                    "html": f"/examples/notebooks/{out_dir.name}/{slug}.html",
                    "download": f"/examples/notebooks/{out_dir.name}/{slug}.py",
                    "source": str(rel),
                }
            )
            print(f"current: {rel}", flush=True)
            continue
        print(f"=== exporting {rel}", flush=True)
        result = None
        for attempt in (1, 2):
            try:
                result = subprocess.run(
                    ["uv", "run", "--group", "examples", "marimo", "export", "html", nb.name, "-o", str(html_out)],
                    cwd=nb.parent,
                    env={**os.environ, "CITYSEER_QUIET_MODE": "true", "MPLBACKEND": "Agg"},
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and (result.returncode == 0 or entry["section"] == "learn"):
                break
            # transient failures (OSM rate limiting) usually clear after a pause
            if attempt == 1:
                print(f"  retrying {rel} after a pause...", flush=True)
                time.sleep(60)
        if result is None:
            failed.append((rel, f"timeout after {TIMEOUT_S}s"))
            print(f"FAILED: {rel} (timeout)", flush=True)
            continue
        # learn lessons contain deliberately failing teaching cells (error demos), so a
        # partial execution still exports; recipes must execute cleanly, and a failed
        # recipe's HTML is removed so a traceback page can never reach the deployed site
        partial_ok = entry["section"] == "learn"
        if not html_out.exists() or (result.returncode != 0 and not partial_ok):
            log_path = OUT_ROOT / f"{slug}.failed.log"
            log_path.write_text(result.stderr or "no stderr")
            err_lines = [ln for ln in (result.stderr or "").splitlines() if "Error" in ln or "error" in ln]
            tail = err_lines[-1] if err_lines else "no error line captured"
            failed.append((rel, tail))
            html_out.unlink(missing_ok=True)
            print(f"FAILED: {rel} — {tail} (full stderr: {log_path})", flush=True)
            continue
        if result.returncode != 0:
            print(f"  note: {rel} exported with failing cells (allowed for lessons)", flush=True)
        finalize_export(nb, html_out)
        shutil.copy2(nb, out_dir / f"{slug}.py")
        title, description = first_title_and_description(nb)
        manifest.append(
            {
                "slug": slug,
                "section": entry["section"],
                "topic": entry["topic"],
                "title": title,
                "description": description,
                "html": f"/examples/notebooks/{out_dir.name}/{slug}.html",
                "download": f"/examples/notebooks/{out_dir.name}/{slug}.py",
                "source": str(rel),
            }
        )
        print(f"OK: {rel}", flush=True)

    if manifest:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        # merge with any prior manifest so filtered runs don't drop other entries
        prior = []
        if MANIFEST_PATH.exists():
            prior = [m for m in json.loads(MANIFEST_PATH.read_text()) if m["slug"] not in {n["slug"] for n in manifest}]
        merged = sorted(prior + manifest, key=lambda m: (m["section"], m["topic"], m["slug"]))
        MANIFEST_PATH.write_text(json.dumps(merged, indent=2))
        print(f"manifest: {MANIFEST_PATH} ({len(merged)} entries)")

    print(f"\n{len(entries) - len(failed)}/{len(entries)} notebooks exported")
    for rel, msg in failed:
        print(f"  FAILED {rel}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
