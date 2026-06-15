"""
src/training/train_whisper.py

Training script για Speech Emotion Recognition χρησιμοποιώντας
pre-extracted Whisper-Large-v3 features και KL Divergence loss.

Χρήση στο Colab:
    !python src/training/train_whisper.py \
        --hf_dataset_path  /content/drive/MyDrive/SLP/msp_podcast_hf \
        --whisper_feat_dir /content/drive/MyDrive/SLP/features/whisper-large-v3 \
        --output_dir       /content/drive/MyDrive/SLP/checkpoints/whisper \
        --epochs           20 \
        --batch_size       32 \
        --lr               1e-4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

# Προσθέτουμε το root directory στο path
sys.path.append(str(Path(__file__).parents[2]))

from src.dataset_features import FeatureDataset


# ---------------------------------------------------------------------------
# Emotion classifier model
# ---------------------------------------------------------------------------

class WhisperEmotionClassifier(nn.Module):
    """
    Απλό emotion classifier πάνω από Whisper hidden states.

    Αρχιτεκτονική (ίδια με SAILER):
        Whisper hidden states [B, T, 1280]
            → 3-layer pointwise Conv1D → [B, T, 256]
            → masked mean pooling    → [B, 256]
            → 2-layer MLP            → [B, num_emotions]
    """

    def __init__(
        self,
        input_dim: int = 1280,       # Whisper-Large-v3 hidden dim
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
        lengths  : Tensor [B]  — πραγματικό μήκος κάθε utterance

        Returns
        -------
        logits : Tensor [B, num_emotions]
        """
        # [B, T, D] → [B, D, T] για Conv1d → [B, T, conv_dim]
        x = self.conv(features.transpose(1, 2)).transpose(1, 2)

        # Masked mean pooling
        B, T, D = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()                    # [B, T, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B, D]

        return self.classifier(x)                            # [B, num_emotions]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def kl_divergence_loss(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence loss μεταξύ predicted distribution και soft labels.
    Όπως στο SAILER paper.

    KL(soft_labels || predicted) = Σ soft_labels * log(soft_labels / predicted)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # soft_labels είναι ήδη probability distribution (αθροίζουν σε 1)
    loss = F.kl_div(log_probs, soft_labels, reduction="batchmean")
    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Υπολογίζει Macro-F1 και loss στο validation/test set."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            features    = batch["whisper"].to(device)
            lengths     = batch["whisper_lengths"].to(device)
            soft_labels = batch["soft_labels"].to(device)
            hard_labels = batch["hard_labels"]

            logits = model(features, lengths)
            loss   = kl_divergence_loss(logits, soft_labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(hard_labels.numpy().tolist())

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    avg_loss  = total_loss / len(loader)

    return {"macro_f1": macro_f1, "loss": avg_loss}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    feature_dirs = {"whisper": args.whisper_feat_dir}

    train_ds = FeatureDataset(
        hf_dataset_path=args.hf_dataset_path,
        feature_dirs=feature_dirs,
        split="train",
    )
    val_ds = FeatureDataset(
        hf_dataset_path=args.hf_dataset_path,
        feature_dirs=feature_dirs,
        split="validation",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=FeatureDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=FeatureDataset.collate_fn,
    )

    print(f"[train] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    num_emotions = train_ds.soft_labels.shape[1] if hasattr(train_ds, "soft_labels") else 9
    model = WhisperEmotionClassifier(
        input_dim=1280,
        conv_dim=256,
        num_emotions=len(train_ds.emotion_cols),
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {total_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_macro_f1 = 0.0
    best_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
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

            if (step + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        # Αποθήκευση καλύτερου checkpoint
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch    = epoch
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_macro_f1,
                "val_loss":     val_metrics["loss"],
            }, checkpoint_path)
            print(f"  ✓ Νέο καλύτερο μοντέλο! Macro-F1: {best_macro_f1:.4f} → {checkpoint_path}")

    print(f"\n[train] Ολοκληρώθηκε! Καλύτερο Macro-F1: {best_macro_f1:.4f} (epoch {best_epoch})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SER classifier on pre-extracted Whisper features."
    )
    parser.add_argument("--hf_dataset_path",  type=str, required=True)
    parser.add_argument("--whisper_feat_dir", type=str, required=True)
    parser.add_argument("--output_dir",       type=str, required=True)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
