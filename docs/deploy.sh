#!/usr/bin/env bash
# Publish docs/dist to the gh-pages branch (single-commit orphan, force-pushed).
# Run after `poe export_notebooks` and `poe docs_build`; see `poe deploy_docs`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$REPO_ROOT/docs/dist"

[ -f "$DIST/index.html" ] || { echo "docs/dist missing or incomplete — run 'uv run poe docs_build' first" >&2; exit 1; }

# GitHub Pages: disable Jekyll so /_astro assets are served
touch "$DIST/.nojekyll"

WORKTREE="$(mktemp -d)"
cleanup() {
  cd "$REPO_ROOT"
  git worktree remove --force "$WORKTREE" 2>/dev/null || true
  git branch -D gh-pages-deploy 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO_ROOT"
git worktree add --detach "$WORKTREE"
cd "$WORKTREE"
git checkout --orphan gh-pages-deploy
git rm -rf --quiet . 2>/dev/null || true
cp -R "$DIST/." .
git add -A
git commit --quiet -m "deploy docs $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push --force origin HEAD:gh-pages
echo "deployed to gh-pages"
