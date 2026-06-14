"""
scripts/extract_features_whisper.py

Extracts Whisper-Large-v3 encoder hidden states from the MSP-Podcast
HuggingFace dataset and saves them as .npy files.

Για κάθε utterance αποθηκεύει:
    SLP/features/whisper-large-v3/<utt_id>.npy   shape: [T, 1280]

Χρήση στο Colab:
    !python scripts/extract_features_whisper.py \
        --dataset_dir  /content/drive/MyDrive/SLP/msp_podcast_hf \
        --output_dir   /content/drive/MyDrive/SLP/features/whisper-large-v3 \
        --batch_size   8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoFeatureExtractor, WhisperModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WHISPER_MODEL = "openai/whisper-large-v3"
SAMPLE_RATE   = 16_000
MAX_DURATION  = 15          # seconds — Whisper hard limit
MAX_SAMPLES   = MAX_DURATION * SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_audio_array(sample: dict) -> np.ndarray:
    """Extract numpy audio array from a HuggingFace audio sample."""
    audio = sample["audio"]
    arr   = np.array(audio["array"], dtype=np.float32)
    sr    = audio["sampling_rate"]

    # Resample if needed (rare — dataset is already 16kHz)
    if sr != SAMPLE_RATE:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Truncate to 15s max
    if len(arr) > MAX_SAMPLES:
        arr = arr[:MAX_SAMPLES]

    return arr


def extract_batch(
    audio_arrays: list[np.ndarray],
    feature_extractor,
    model: WhisperModel,
    device: torch.device,
) -> list[np.ndarray]:
    """Run Whisper encoder on a batch, return list of [T, 1280] arrays."""
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device)   # [B, 128, 3000]

    with torch.no_grad():
        hidden = model.encoder(input_features).last_hidden_state  # [B, T, 1280]

    return [hidden[i].cpu().numpy() for i in range(len(audio_arrays))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Whisper-Large-v3 features from MSP-Podcast HF dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/content/drive/MyDrive/SLP/msp_podcast_hf",
        help="Path to the HuggingFace dataset saved with save_to_disk()",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/SLP/features/whisper-large-v3",
        help="Directory where .npy feature files will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of utterances to process at once. Reduce if OOM.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip utterances whose .npy file already exists (safe resume).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_features_whisper] Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[extract_features_whisper] Loading {WHISPER_MODEL} ...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(WHISPER_MODEL)
    model = WhisperModel.from_pretrained(WHISPER_MODEL)
    model.eval()
    model.to(device)
    print("[extract_features_whisper] Model loaded.")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"[extract_features_whisper] Loading dataset from {args.dataset_dir} ...")
    ds = load_from_disk(args.dataset_dir)
    # load_from_disk returns a DatasetDict — get the train split
    if hasattr(ds, "keys"):
        ds = ds["train"]
    print(f"[extract_features_whisper] Dataset size: {len(ds)} utterances")

    # ── Extract features in batches ──────────────────────────────────────────
    total     = len(ds)
    batch_size = args.batch_size
    saved     = 0
    skipped   = 0
    errors    = 0

    for start in tqdm(range(0, total, batch_size), desc="Extracting"):
        end   = min(start + batch_size, total)
        batch = ds.select(range(start, end))

        # Determine output paths and skip already-done files
        file_names  = batch["file"]                        # e.g. MSP-PODCAST_0001_0008.wav
        out_paths   = [output_dir / (Path(f).stem + ".npy") for f in file_names]

        if args.skip_existing:
            pending = [(i, f) for i, (f, p) in enumerate(zip(file_names, out_paths)) if not p.exists()]
            if not pending:
                skipped += len(file_names)
                continue
            indices, _ = zip(*pending)
            batch      = batch.select(list(indices))
            out_paths  = [out_paths[i] for i in indices]

        # Extract audio arrays
        try:
            audio_arrays = [get_audio_array(sample) for sample in batch]
        except Exception as exc:
            print(f"\n[WARNING] Audio load error at batch {start}-{end}: {exc}")
            errors += len(batch)
            continue

        # Run encoder
        try:
            features = extract_batch(audio_arrays, feature_extractor, model, device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\n[OOM] Batch {start}-{end} — try reducing --batch_size. Skipping.")
                torch.cuda.empty_cache()
                errors += len(batch)
                continue
            raise

        # Save .npy files
        for feat, out_path in zip(features, out_paths):
            np.save(str(out_path), feat)
            saved += 1

    print(
        f"\n[extract_features_whisper] Done! "
        f"Saved: {saved} | Skipped (existing): {skipped} | Errors: {errors}"
    )
    print(f"Features saved to: {output_dir}")


if __name__ == "__main__":
    main()
