#!/usr/bin/env bash
# Render the book and publish it to GitHub Pages (master -> docs/).
# Run setup.sh once per machine before using this.
set -euo pipefail
cd "$(dirname "$0")"

# --- the merge driver must exist, or docs/ conflicts come straight back
git config --get merge.keeptheirs.driver >/dev/null 2>&1 \
  || { echo "Run 'bash setup.sh' once on this machine first."; exit 1; }

# --- check Quarto version is recent enough (1.8+)
have=$(quarto --version)
want=$(tr -d '[:space:]' < .quarto-version)
min_version="1.8.0"
if [ "$(printf '%s\n' "$min_version" "$have" | sort -V | head -n1)" != "$min_version" ]; then
  echo "This book requires Quarto $min_version or newer; you have $have."
  echo "Install or update Quarto via: brew install quarto"
  exit 1
fi
if [ "$want" != "$have" ]; then
  echo "Note: this book was tested with Quarto $want; you have $have."
fi

# --- start from what is actually on GitHub
git pull --no-edit || {
  if git status | grep -q "unmerged"; then
    echo "Merge conflict detected. Aborting merge."
    git merge --abort
    echo "Fix the conflicts in your working directory and try again."
    exit 1
  fi
  exit 1
}

# Render HTML and process dependencies (PDF will fail without LaTeX, but that's OK)
quarto render 2>&1 | grep -E "^\[|Output created" || true

# --- mirror into docs/, don't delete-and-recreate all 140 files
# Note: exclude _extensions so quarto-live extension is preserved
# Use --checksum to verify content, not just timestamps (prevents timestamp sync issues)
rsync -a --delete --checksum --exclude='_extensions' _book/ docs/
# Ensure _extensions is in docs/ for GitHub Pages
cp -r _extensions docs/

# --- pin the one line Quarto randomises on every single render
find docs -name '*.html' -exec perl -pi -e \
  's/GLightbox\(\{[^}]*\}\)/GLightbox({"closeEffect":"zoom","descPosition":"bottom","loop":false,"openEffect":"zoom","selector":".lightbox"})/g' {} +

# --- .gitignore is the gatekeeper now, so -A is safe
git add -A
if git diff --cached --quiet; then
  echo "Nothing to publish."
  exit 0
fi
git commit -m "Dispatch book $(date '+%Y-%m-%d')"
git push
