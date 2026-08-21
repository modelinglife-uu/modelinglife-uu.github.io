#!/usr/bin/env bash
# Run this ONCE on every machine that works on the book, right after cloning.
# It installs the merge driver that stops docs/ from producing conflicts.
set -euo pipefail
cd "$(dirname "$0")"

git config merge.keeptheirs.name "take the incoming copy of generated files"
git config merge.keeptheirs.driver 'cp -f "%B" "%A"'
git config pull.rebase false

echo "Merge driver installed."

want=$(tr -d '[:space:]' < .quarto-version)
have=$(quarto --version 2>/dev/null || echo "not installed")
if [ "$want" != "$have" ]; then
  echo
  echo "NOTE: this book is built with Quarto $want, you have $have."
  echo "      Install $want before publishing, or update_book.sh will refuse to run."
fi
