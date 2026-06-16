"""
src/training/train_whisper.py

Training script για Speech Emotion Recognition χρησιμοποιώντας
pre-extracted Whisper-Large-v3 features και KL Divergence loss.

Χρήση στο Colab:
    !python src/training/train_whisper.py \
        --hf_dataset_path  /content/drive/MyDrive/SLP/msp_podcast_hf \
        --whisper_feat_dir /content/drive/MyDrive/SLP/features/whisper_shards \
        --output_dir       /content/drive/MyDrive/SLP/checkpoints/whisper \
        --epochs           15 \
        --batch_size       32 \
        --lr               5e-4
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# Προσθέτουμε το root directory στο path
sys.path.append(str(Path(__file__).parents[2]))

from src.dataset_features import ShardFeatureDataset, EMOTION_COLS


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
# Emotion classifier model
# ---------------------------------------------------------------------------

class WhisperEmotionClassifier(nn.Module):
    """
    Emotion classifier πάνω από Whisper hidden states.

    Αρχιτεκτονική (ίδια με SAILER):
        Whisper hidden states [B, T, 1280]
            → 3-layer pointwise Conv1D → [B, T, conv_dim]
            → masked mean pooling     → [B, conv_dim]
            → 2-layer MLP             → [B, num_emotions]
    """

    def __init__(
        self,
        input_dim: int = 1280,      # Whisper-Large-v3 hidden dim
        conv_dim: int = 256,
        num_emotions: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(conv_dim, conv_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv_dim, num_emotions),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : Tensor [B, T, 1280]
        lengths  : Tensor [B]  — πραγματικό μήκος κάθε utterance σε frames

        Returns
        -------
        logits : Tensor [B, num_emotions]
        """
        # [B, T, D] → [B, D, T] για Conv1d → [B, T, conv_dim]
        x = self.conv(features.transpose(1, 2)).transpose(1, 2)

        # Masked mean pooling
        B, T, D = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()                             # [B, T, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B, D]

        return self.classifier(x)                                     # [B, num_emotions]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def kl_divergence_loss(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence loss μεταξύ predicted distribution και soft labels.
    Όπως στο SAILER paper: KL(soft_labels || predicted)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, soft_labels, reduction="batchmean")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    emotion_cols: list[str],
    verbose: bool = False,
) -> dict:
    """Υπολογίζει Macro-F1, loss, και per-class F1 στο validation/test set."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            features    = batch["whisper"].to(device)
            lengths     = batch["whisper_lengths"].to(device)
            soft_labels = batch["soft_labels"].to(device)
            hard_labels = batch["hard_labels"]

            logits = model(features, lengths)
            loss   = kl_divergence_loss(logits, soft_labels)
            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(hard_labels.numpy().tolist())

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    avg_loss  = total_loss / max(num_batches, 1)

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
    print(f"[train] Device: {device}")

    # ── Datasets (streaming από τα shards) ───────────────────────────────────
    train_ds = ShardFeatureDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.whisper_feat_dir,
        split="train",
    )
    val_ds = ShardFeatureDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.whisper_feat_dir,
        split="validation",
    )

    # ΠΡΟΣΟΧΗ: με IterableDataset ΔΕΝ βάζουμε shuffle=True (το shuffle γίνεται
    # μέσα στο dataset). Επίσης num_workers=0 για να μη διπλασιάζονται samples.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=ShardFeatureDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=ShardFeatureDataset.collate_fn,
    )

    # Πόσα batches περίπου ανά epoch (για το tqdm / μέσο loss)
    n_train_steps = math.ceil(len(train_ds) / args.batch_size)

    emotion_cols = train_ds.emotion_cols
    num_emotions = len(emotion_cols)
    print(f"[train] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[train] Emotion classes ({num_emotions}): {emotion_cols}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = WhisperEmotionClassifier(
        input_dim=1280,
        conv_dim=256,
        num_emotions=num_emotions,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {total_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training log CSV ─────────────────────────────────────────────────────
    log_path = output_dir / "training_log.csv"
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                    total=n_train_steps, leave=False)
        for step, batch in enumerate(pbar):
            features    = batch["whisper"].to(device)
            lengths     = batch["whisper_lengths"].to(device)
            soft_labels = batch["soft_labels"].to(device)

            optimizer.zero_grad()
            logits = model(features, lengths)
            loss   = kl_divergence_loss(logits, soft_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if (step + 1) % args.log_every == 0:
                print(
                    f"  Epoch {epoch} | Step {step+1}/{n_train_steps} "
                    f"| Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        current_lr     = scheduler.get_last_lr()[0]

        # Validation — verbose (per-class F1) μόνο κάθε 5 epochs
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

        # ── Αποθήκευση last checkpoint (για resume) ──────────────────────────
        torch.save({
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_macro_f1":        val_metrics["macro_f1"],
            "val_loss":            val_metrics["loss"],
            "emotion_cols":        emotion_cols,
            "args":                vars(args),
        }, output_dir / "last_model.pt")

        # ── Αποθήκευση best checkpoint ───────────────────────────────────────
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch    = epoch
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_macro_f1":        best_macro_f1,
                "val_loss":            val_metrics["loss"],
                "emotion_cols":        emotion_cols,
                "args":                vars(args),
            }, output_dir / "best_model.pt")
            print(f"  ✓ Νέο καλύτερο μοντέλο! Macro-F1: {best_macro_f1:.4f} → best_model.pt")

    print(f"\n[train] Ολοκληρώθηκε! Καλύτερο Macro-F1: {best_macro_f1:.4f} (epoch {best_epoch})")
    print(f"[train] Training log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SER classifier on pre-extracted Whisper features."
    )
    parser.add_argument("--hf_dataset_path",  type=str, required=True,
                        help="Path to the HuggingFace dataset (save_to_disk format)")
    parser.add_argument("--whisper_feat_dir", type=str, required=True,
                        help="Directory with .npy Whisper features (one per utterance)")
    parser.add_argument("--output_dir",       type=str, required=True,
                        help="Where to save checkpoints and training log")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=0,
                        help="0 = ασφαλές για streaming. Δοκίμασε 2 για πιο γρήγορο Drive I/O.")
    parser.add_argument("--log_every",   type=int,   default=50,
                        help="Εκτύπωσε loss κάθε N steps")
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
