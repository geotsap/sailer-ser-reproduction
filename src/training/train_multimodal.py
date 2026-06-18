"""
src/training/train_multimodal.py

Training script για multimodal Speech Emotion Recognition με
Whisper + RoBERTa features και KL Divergence loss.

Χρήση στο Colab:
    !python src/training/train_multimodal.py \
        --hf_dataset_path  /content/drive/MyDrive/SLP/msp_podcast_hf \
        --shard_dir        /content/drive/MyDrive/SLP/features/whisper_shards \
        --roberta_dir      /content/drive/MyDrive/SLP/features/roberta-large \
        --output_dir       /content/drive/MyDrive/SLP/checkpoints/multimodal \
        --epochs           20 \
        --batch_size       32 \
        --lr               1e-4
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

from src.dataset_features_multimodal import MultimodalShardDataset
from src.model.emotion_models_compact import WhisperRobertaEmotionModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def kl_divergence_loss(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """KL Divergence loss όπως στο SAILER paper."""
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, soft_labels, reduction="batchmean")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    emotion_cols: list[str],
    verbose: bool = False,
) -> dict:
    """Υπολογίζει Macro-F1, loss, και per-class F1 στο validation/test set."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            whisper     = batch["whisper"].to(device)
            roberta     = batch["roberta"].unsqueeze(1).to(device)  # [B, 1024] -> [B, 1, 1024]
            soft_labels = batch["soft_labels"].to(device)
            hard_labels = batch["hard_labels"]

            # speech_mask από lengths
            lengths = batch["whisper_lengths"]
            B, T, _ = whisper.shape
            speech_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)

            logits = model(
                whisper_hidden_states=whisper,
                roberta_hidden_states=roberta,
                speech_mask=speech_mask,
            )

            loss = kl_divergence_loss(logits, soft_labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(hard_labels.numpy().tolist())

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    avg_loss  = total_loss / max(len(loader), 1)

    if verbose:
        print(classification_report(
            all_targets, all_preds,
            target_names=emotion_cols,
            zero_division=0,
        ))

    return {"macro_f1": macro_f1, "loss": avg_loss}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_multimodal] Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = MultimodalShardDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.shard_dir,
        roberta_dir=args.roberta_dir,
        split="train",
        split_mode=args.split_mode,
        seed=args.seed,
    )
    val_ds = MultimodalShardDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.shard_dir,
        roberta_dir=args.roberta_dir,
        split="validation",
        split_mode=args.split_mode,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=MultimodalShardDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=MultimodalShardDataset.collate_fn,
    )

    emotion_cols = train_ds.emotion_cols
    num_emotions = len(emotion_cols)
    print(f"[train_multimodal] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[train_multimodal] Emotion classes ({num_emotions}): {emotion_cols}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = WhisperRobertaEmotionModel(
        num_emotions=num_emotions,
        whisper_dim=1280,
        roberta_dim=1024,
        conv_dim=256,
        classifier_hidden_dim=256,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_multimodal] Trainable parameters: {total_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training log CSV ─────────────────────────────────────────────────────
    log_path   = output_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # ── Training loop ────────────────────────────────────────────────────────
    best_macro_f1 = 0.0
    best_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            whisper     = batch["whisper"].to(device)       # [B, 750, 1280]
            roberta     = batch["roberta"].unsqueeze(1).to(device)  # [B, 1, 1024]
            soft_labels = batch["soft_labels"].to(device)   # [B, 9]
            lengths     = batch["whisper_lengths"]

            # speech_mask από lengths
            B, T, _ = whisper.shape
            speech_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(
                whisper_hidden_states=whisper,
                roberta_hidden_states=roberta,
                speech_mask=speech_mask,
            )
            loss = kl_divergence_loss(logits, soft_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if (step + 1) % args.log_every == 0:
                print(
                    f"  Epoch {epoch} | Step {step+1} "
                    f"| Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        current_lr     = scheduler.get_last_lr()[0]

        # Validation — verbose κάθε 5 epochs
        verbose_eval = (epoch % 5 == 0) or (epoch == args.epochs)
        val_metrics  = evaluate(model, val_loader, device, emotion_cols, verbose=verbose_eval)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val Macro-F1: {val_metrics['macro_f1']:.4f} "
            f"| LR: {current_lr:.2e}"
        )

        # ── Αποθήκευση log ───────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":        epoch,
                "train_loss":   f"{avg_train_loss:.6f}",
                "val_loss":     f"{val_metrics['loss']:.6f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
                "lr":           f"{current_lr:.2e}",
            })

        # ── Αποθήκευση last checkpoint ───────────────────────────────────────
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_macro_f1":         val_metrics["macro_f1"],
            "val_loss":             val_metrics["loss"],
            "emotion_cols":         emotion_cols,
            "args":                 vars(args),
        }, output_dir / "last_model.pt")

        # ── Αποθήκευση best checkpoint ───────────────────────────────────────
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch    = epoch
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_macro_f1":         best_macro_f1,
                "val_loss":             val_metrics["loss"],
                "emotion_cols":         emotion_cols,
                "args":                 vars(args),
            }, output_dir / "best_model.pt")
            print(f"  ✓ Νέο καλύτερο μοντέλο! Macro-F1: {best_macro_f1:.4f} → best_model.pt")

    print(f"\n[train_multimodal] Ολοκληρώθηκε! Καλύτερο Macro-F1: {best_macro_f1:.4f} (epoch {best_epoch})")
    print(f"[train_multimodal] Training log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multimodal SER classifier (Whisper + RoBERTa)."
    )
    parser.add_argument("--hf_dataset_path", type=str, required=True,
                        help="Path to the HuggingFace dataset (save_to_disk format)")
    parser.add_argument("--shard_dir",       type=str, required=True,
                        help="Directory with Whisper shard .npz files")
    parser.add_argument("--roberta_dir",     type=str, required=True,
                        help="Directory with RoBERTa .npy feature files")
    parser.add_argument("--output_dir",      type=str, required=True,
                        help="Where to save checkpoints and training log")
    parser.add_argument("--split_mode",  type=str,   default="podcast",
                        choices=["podcast", "random"])
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--log_every",   type=int,   default=50)
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
