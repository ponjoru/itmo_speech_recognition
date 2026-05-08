#!/usr/bin/env bash
# Build a 4-gram KenLM model over all Russian number phrases 1000–999999.
# Requirements: lmplz and build_binary must be on PATH (from KenLM build).
set -euo pipefail

DATA_DIR="checkpoints/lm"
mkdir -p "$DATA_DIR"

echo "[1/3] Generating corpus (999 000 sentences)..."
python -m src.lm.build_corpus > "$DATA_DIR/corpus.txt"

echo "[2/3] Training 4-gram KenLM..."
lmplz -o 4 --discount_fallback < "$DATA_DIR/corpus.txt" > "$DATA_DIR/lm.arpa"

echo "[3/3] Converting to binary format..."
build_binary "$DATA_DIR/lm.arpa" "$DATA_DIR/lm.bin"

echo "Done.  Model saved to $DATA_DIR/lm.bin"
