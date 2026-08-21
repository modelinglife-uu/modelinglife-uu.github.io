#!/usr/bin/env bash
# Render the book and publish it to GitHub Pages (master -> docs/).
# Run setup.sh once per machine before using this.
set -euo pipefail
cd "$(dirname "$0")"

# --- the merge driver must exist, or docs/ conflicts come straight back
git config --get merge.keeptheirs.driver >/dev/null 2>&1 \
  || { echo "Run 'bash setup.sh' once on this machine first."; exit 1; }

# --- everyone must build with the same Quarto, or the whole site churns
want=$(tr -d '[:space:]' < .quarto-version)
have=$(quarto --version)
if [ "$want" != "$have" ]; then
  echo "This book is built with Quarto $want; you have $have."
  echo "Upgrade, or edit .quarto-version if the team agreed to move."
  exit 1
fi

# --- start from what is actually on GitHub
git pull --no-edit

quarto render

# --- mirror into docs/, don't delete-and-recreate all 140 files
rsync -a --delete _book/ docs/

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
