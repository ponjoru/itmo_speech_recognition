"""Inference script: loads a checkpoint, runs decoding on test split, writes submission CSV."""
import argparse
import os

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from . import text_utils
from .dataset import TARGET_SR
from .decoder import BeamCTCDecoder, GreedyDecoder
from .model import ConformerASR


def _load_audio(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform.squeeze(0)  # (T,)


@torch.inference_mode()
def run_inference(
    model: ConformerASR,
    decoder,
    paths: list[str],
    batch_size: int,
    device: torch.device,
) -> list[str]:
    model.eval()
    results = []

    for start in tqdm(range(0, len(paths), batch_size), desc="Inference"):
        batch_paths = paths[start : start + batch_size]
        waveforms = [_load_audio(p) for p in batch_paths]

        wav_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        padded = torch.zeros(len(waveforms), wav_lengths.max().item())
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w

        log_probs, out_lengths = model(padded.to(device), wav_lengths.to(device))

        word_indices = decoder.decode_batch(log_probs, out_lengths)
        for idx in word_indices:
            text = text_utils.denormalize(idx)
            results.append(text if text else "1000")

    return results

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConformerASR.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)

    decoder = (
        GreedyDecoder()
        if args.greedy
        else BeamCTCDecoder(
            lm_path=args.lm_path,
            alpha=args.alpha,
            beta=args.beta,
            beam_width=args.beam_width,
        )
    )

    test_csv = args.test_csv or os.path.join(args.data_root, "test.csv")
    df = pd.read_csv(test_csv)
    paths = [os.path.join(args.data_root, f) for f in df["filename"]]

    transcriptions = run_inference(model, decoder, paths, args.batch_size, device)

    submission = pd.DataFrame({"filename": df["filename"], "transcription": transcriptions})
    submission.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output_csv}")

    
def main() -> None:
    p = argparse.ArgumentParser(description="Run ASR inference and produce a Kaggle submission CSV")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--data_root", required=True, help="Path to dataset root directory")
    p.add_argument("--test_csv", default=None, help="Override path to test CSV")
    p.add_argument("--output_csv", default="submission.csv")
    p.add_argument("--batch_size", type=int, default=16)
    # Decoding
    p.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of beam")
    p.add_argument("--lm_path", default=None, help="Path to KenLM .bin model")
    p.add_argument("--alpha", type=float, default=0.5, help="LM weight")
    p.add_argument("--beta", type=float, default=1.0, help="Word insertion bonus")
    p.add_argument("--beam_width", type=int, default=50)
    args = p.parse_args()
    
    predict(args)
   

if __name__ == "__main__":
    main()
