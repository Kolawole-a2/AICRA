#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${AICRA_EMBER2024_DIR:-data/ember2024_real}"

echo "AICRA EMBER-2024 data check"
echo "Expected directory: ${TARGET_DIR}"

if [[ -d "${TARGET_DIR}" ]]; then
  SAMPLE="$(ls -1 "${TARGET_DIR}"/*_train.jsonl 2>/dev/null | head -n 1 || true)"
  if [[ -n "${SAMPLE}" ]]; then
    echo "OK: Found dataset directory and a sample file:"
    echo "  ${SAMPLE}"
    exit 0
  else
    echo "WARNING: Directory exists but no '*_train.jsonl' files found."
    echo "Place your EMBER-2024 JSONL split files here."
    exit 2
  fi
else
  echo "MISSING: EMBER-2024 directory not found."
  echo ""
  echo "To set up:"
  echo "1) Obtain EMBER-2024 JSONL split files via your approved source."
  echo "2) Create the directory: ${TARGET_DIR}"
  echo "3) Place the JSONL files inside it (example: '*_train.jsonl')."
  echo ""
  echo "Optional: export AICRA_EMBER2024_DIR to your dataset location."
  echo "See docs/DATA.md for details."
  exit 1
fi
