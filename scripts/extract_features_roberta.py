"""
scripts/extract_features_roberta.py

Extracts RoBERTa-Large encoder hidden states from the MSP-Podcast
HuggingFace dataset transcripts and saves them as .npy files.

Για κάθε utterance αποθηκεύει:
    SLP/features/roberta-large/<utt_id>.npy   shape: [1, 1024]

Συμβατό με src/dataset_features.py — τα .npy αρχεία έχουν ίδια
ονομασία με τα Whisper features (π.χ. MSP-PODCAST_0001_0008.npy)
αλλά αποθηκεύονται σε διαφορετικό φάκελο.

Χρήση στο Colab:
    !python scripts/extract_features_roberta.py \
        --dataset_dir  /content/drive/MyDrive/SLP/msp_podcast_hf \
        --output_dir   /content/drive/MyDrive/SLP/features/roberta-large \
        --batch_size   32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, RobertaModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROBERTA_MODEL = "roberta-large"
MAX_LENGTH    = 512   # RoBERTa max token length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_transcript(sample: dict) -> str:
    """Εξάγει το transcript από ένα HuggingFace sample."""
    transcript = sample.get("transcription") or sample.get("transcript") or ""
    return str(transcript).strip()


def extract_batch(
    texts: list[str],
    tokenizer,
    model: RobertaModel,
    device: torch.device,
) -> list[np.ndarray]:
    """
    Τρέχει το RoBERTa σε ένα batch κειμένων.
    Επιστρέφει λίστα από [1, 1024] arrays — ένα ανά κείμενο.

    Χρησιμοποιεί weighted average όλων των encoder layers,
    ακριβώς όπως περιγράφει το SAILER paper για το RoBERTa.
    """
    # Tokenization
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Weighted average όλων των hidden states (όπως SAILER)
    # outputs.hidden_states: tuple of [B, T, 1024] — ένα ανά layer + embedding layer
    hidden_states = torch.stack(outputs.hidden_states, dim=0)  # [num_layers, B, T, 1024]

    # Mean pooling πάνω στα layers
    weighted_avg = hidden_states.mean(dim=0)  # [B, T, 1024]

    # Temporal average (mean pooling πάνω στα tokens)
    # Χρησιμοποιούμε attention mask για να αγνοήσουμε τα padding tokens
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
    text_embeddings = (weighted_avg * attention_mask).sum(dim=1)     # [B, 1024]
    text_embeddings = text_embeddings / attention_mask.sum(dim=1).clamp_min(1.0)  # [B, 1024]

    # Προσθέτουμε διάσταση T=1 για συμβατότητα με dataset_features.py
    # που περιμένει [T, D] format (όπως τα Whisper features)
    text_embeddings = text_embeddings.unsqueeze(1)  # [B, 1, 1024]

    return [text_embeddings[i].cpu().numpy() for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RoBERTa-Large features from MSP-Podcast HF dataset transcripts."
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
        default="/content/drive/MyDrive/SLP/features/roberta-large",
        help="Directory where .npy feature files will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of utterances to process at once. RoBERTa είναι πιο ελαφρύ από Whisper.",
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
    print(f"[extract_features_roberta] Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[extract_features_roberta] Loading {ROBERTA_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model     = RobertaModel.from_pretrained(ROBERTA_MODEL)
    model.eval()
    model.to(device)
    print("[extract_features_roberta] Model loaded.")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"[extract_features_roberta] Loading dataset from {args.dataset_dir} ...")
    ds = load_from_disk(args.dataset_dir)
    if hasattr(ds, "keys"):
        ds = ds["train"]

    # Αφαίρεσε το audio column — δεν το χρειαζόμαστε
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])

    print(f"[extract_features_roberta] Dataset size: {len(ds)} utterances")

    # ── Extract features in batches ──────────────────────────────────────────
    total      = len(ds)
    batch_size = args.batch_size
    saved      = 0
    skipped    = 0
    errors     = 0
    empty      = 0   # utterances χωρίς transcript

    for start in tqdm(range(0, total, batch_size), desc="Extracting RoBERTa features"):
        end   = min(start + batch_size, total)
        batch = ds.select(range(start, end))

        # Ονόματα αρχείων και output paths
        file_names = batch["file"]
        out_paths  = [output_dir / (Path(f).stem + ".npy") for f in file_names]

        # Skip αν υπάρχουν ήδη
        if args.skip_existing:
            pending = [
                (i, f) for i, (f, p) in enumerate(zip(file_names, out_paths))
                if not p.exists()
            ]
            if not pending:
                skipped += len(file_names)
                continue
            indices, _ = zip(*pending)
            batch      = batch.select(list(indices))
            out_paths  = [out_paths[i] for i in indices]

        # Εξαγωγή transcripts
        texts = [get_transcript(sample) for sample in batch]

        # Αντικατάσταση κενών transcripts με placeholder
        processed_texts = []
        is_empty        = []
        for text in texts:
            if not text:
                processed_texts.append("[empty]")
                is_empty.append(True)
                empty += 1
            else:
                processed_texts.append(text)
                is_empty.append(False)

        # Εξαγωγή features
        try:
            features = extract_batch(processed_texts, tokenizer, model, device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\n[OOM] Batch {start}-{end} — try reducing --batch_size. Skipping.")
                torch.cuda.empty_cache()
                errors += len(batch)
                continue
            raise
        except Exception as exc:
            print(f"\n[WARNING] Error at batch {start}-{end}: {exc}")
            errors += len(batch)
            continue

        # Αποθήκευση .npy αρχείων
        for feat, out_path in zip(features, out_paths):
            np.save(str(out_path), feat)
            saved += 1

    print(
        f"\n[extract_features_roberta] Done! "
        f"Saved: {saved} | Skipped (existing): {skipped} | "
        f"Errors: {errors} | Empty transcripts: {empty}"
    )
    print(f"Features saved to: {output_dir}")


if __name__ == "__main__":
    main()
