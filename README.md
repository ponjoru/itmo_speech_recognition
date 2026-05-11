# Russian Spoken Numbers ASR

Conformer-based CTC model for recognising Russian spoken numbers (1 000 – 999 999).
Word-level CTC over a 41-word closed vocabulary, with optional KenLM rescoring.

## Setup

```bash
pip install -r requirements.txt
```

For MP3 support and room-simulation augmentation, also install `ffmpeg` and `pyroomacoustics`.

## Data

Place the Kaggle competition data under `data/asr-2026-spoken-numbers-recognition-challenge/`:

```
data/asr-2026-spoken-numbers-recognition-challenge/
├── train.csv / dev.csv / test.csv
├── train/   dev/   test/
```

## Training

```bash
python -m src.train \
  --data_root data/asr-2026-spoken-numbers-recognition-challenge \
  --batch_size 32 --num_workers 4
```

Resume from a checkpoint with `--resume_from checkpoints/last.ckpt`.  
TensorBoard logs are written to `logs/`.

## Language model (optional)

Requires KenLM CLI tools (`lmplz`, `build_binary`) on `PATH`.

```bash
bash src/lm/build_lm.sh   # writes checkpoints/lm/lm.bin
```

## Evaluation (dev split)

```bash
# Greedy
python -m src.evaluate \
  --checkpoint checkpoints/last.ckpt \
  --data_root data/asr-2026-spoken-numbers-recognition-challenge \
  --greedy

# Beam + LM
python -m src.evaluate \
  --checkpoint checkpoints/last.ckpt \
  --data_root data/asr-2026-spoken-numbers-recognition-challenge \
  --lm_path checkpoints/lm/lm.bin
```

## Inference (test split → submission CSV)

```bash
python -m src.predict \
  --checkpoint checkpoints/last.ckpt \
  --data_root data/asr-2026-spoken-numbers-recognition-challenge \
  --lm_path checkpoints/lm/lm.bin \
  --output_csv submission.csv
```

## Model

| Component | Config |
|-----------|--------|
| Features | 80-dim log-mel, 25 ms window, 10 ms hop, 16 kHz |
| Subsampling | 2 × Conv2D stride-2 (4× time reduction) |
| Encoder | torchaudio Conformer — dim 160, 4 heads, FFN 320, 10 layers |
| Vocabulary | 42 tokens (blank + 41 Russian number words) |
| Parameters | ~4.05 M |
| Loss | CTC |
| LM | 4-gram KenLM trained on all 999 000 number phrases |
