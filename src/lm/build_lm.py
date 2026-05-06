"""Build a word-level KenLM language model from all Russian number transcriptions.

Steps:
  1. Generate Russian word forms for all integers in [1000, 999999] using num2words.
  2. Write them to a text file (one utterance per line).
  3. Train KenLM with lmplz (n-gram order configurable, default 4).
  4. Compile the ARPA file to a binary with build_binary.

Prerequisites:
    pip install num2words
    # Install KenLM tools: https://github.com/kpu/kenlm
    # On Colab/Kaggle:  !pip install https://github.com/kpu/kenlm/archive/master.zip
    #                   (lmplz and build_binary will be in /usr/local/bin or similar)

Usage:
    python -m hw_3.lm.build_lm --output_dir hw_3/lm --order 4
    python -m hw_3.lm.build_lm --output_dir hw_3/lm --order 4 --verify
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from num2words import num2words


def generate_corpus(output_path: Path) -> None:
    """Write all 999000 Russian number transcriptions to a text file."""
    print(f"Generating corpus for n in [1000, 999999] → {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for n in range(1000, 1_000_000):
            f.write(num2words(n, lang="ru") + "\n")
    size_mb = output_path.stat().st_size / 1e6
    print(f"Corpus written: {size_mb:.1f} MB")


def train_kenlm(corpus_path: Path, output_dir: Path, order: int) -> Path:
    """Train KenLM with lmplz and compile to binary."""
    arpa_path = output_dir / f"numbers_{order}gram.arpa"
    binary_path = output_dir / f"numbers_{order}gram.binary"

    # Train ARPA
    print(f"Training {order}-gram KenLM...")
    with open(corpus_path) as corpus_f:
        result = subprocess.run(
            ["lmplz", "-o", str(order), "--prune", "0", "0", "0", "1"],
            stdin=corpus_f,
            stdout=open(arpa_path, "w"),
            stderr=sys.stderr,
        )
    if result.returncode != 0:
        raise RuntimeError("lmplz failed. Make sure KenLM tools are installed.")
    print(f"ARPA written to {arpa_path}")

    # Compile to binary (faster loading)
    print("Compiling to binary...")
    result = subprocess.run(["build_binary", str(arpa_path), str(binary_path)])
    if result.returncode != 0:
        raise RuntimeError("build_binary failed.")
    print(f"Binary written to {binary_path}")
    return binary_path


def verify_lm(binary_path: Path) -> None:
    """Quick sanity check: score a known valid and invalid string."""
    import kenlm
    model = kenlm.Model(str(binary_path))
    valid = "сто тридцать девять тысяч четыреста семьдесят три"
    invalid = "тысяча двести тысяча"   # impossible
    print(f"LM score (valid)  : {model.score(valid, bos=True, eos=True):.4f}")
    print(f"LM score (invalid): {model.score(invalid, bos=True, eos=True):.4f}")
    print("(valid should have a significantly higher score than invalid)")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "numbers_corpus.txt"
    if not corpus_path.exists() or args.regen:
        generate_corpus(corpus_path)
    else:
        print(f"Corpus already exists at {corpus_path}, skipping generation.")

    binary_path = train_kenlm(corpus_path, output_dir, order=args.order)

    if args.verify:
        verify_lm(binary_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build KenLM for Russian number recognition")
    p.add_argument("--output_dir", default="hw_3/lm")
    p.add_argument("--order", type=int, default=4, help="n-gram order")
    p.add_argument("--regen", action="store_true", help="Regenerate corpus even if it exists")
    p.add_argument("--verify", action="store_true", help="Verify LM after building")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
