#!/usr/bin/env bash
# Run this ONCE on every machine that works on the book, right after cloning.
# It installs the merge driver that stops docs/ from producing conflicts.
set -euo pipefail
cd "$(dirname "$0")"

git config merge.keeptheirs.name "take the incoming copy of generated files"
git config merge.keeptheirs.driver 'cp -f "%B" "%A"'
git config pull.rebase false

echo "Merge driver installed."

version=$(tr -d '[:space:]' < .quarto-version)
have=$(quarto --version 2>/dev/null || echo "")

if [ -z "$have" ]; then
  echo
  echo "Quarto is not installed."
  echo
  echo "To install Quarto, run:"
  echo "  brew install quarto"
  echo
  echo "After installation, run this script again."
  echo
elif [ "$version" != "$have" ]; then
  echo
  echo "Quarto version: you have $have, and this project uses $version. I don't think minor differences shouldn't matter."
  echo
  echo "To update, run:"
  echo "  brew uninstall quarto"
  echo "  brew install quarto"
  echo
else
  echo "Quarto version: $version ✓"
fi
