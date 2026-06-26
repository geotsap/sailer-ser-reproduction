"""
src/training/train_deepser.py

Training script για DeepSER — hierarchical deep fusion (Whisper + RoBERTa).
Βασισμένο ακριβώς στο train_multimodal.py, αλλάζει μόνο το μοντέλο.

Hyperparameters από το MEDUSA paper:
    lr = 1e-5  (ΟΧΙ 5e-4 — το DeepSER είναι πιο ευαίσθητο)
    batch_size = 16
    hidden_dim = 1024
    epochs = 15

Χρήση στο Colab:
    !python src/training/train_deepser.py \
        --hf_dataset_path  /content/drive/MyDrive/SLP/msp_podcast_hf \
        --shard_dir        /content/drive/MyDrive/SLP/features/whisper_shards \
        --roberta_npz      /content/drive/MyDrive/SLP/features/roberta_all.npz \
        --output_dir       /content/drive/MyDrive/SLP/checkpoints/deepser \
        --epochs           15 \
        --batch_size       16 \
        --lr               1e-5
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
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

from src.dataset_features_multimodal import MultimodalShardDataset
from src.model.deepser import DeepSERModel


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
    total_loss  = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            whisper     = batch["whisper"].to(device)          # [B, 750, 1280]
            roberta     = batch["roberta"].to(device)          # [B, 1024]
            soft_labels = batch["soft_labels"].to(device)
            hard_labels = batch["hard_labels"]
            lengths     = batch["whisper_lengths"].to(device)

            logits = model(
                whisper=whisper,
                roberta=roberta,
                whisper_lengths=lengths,
            )
            loss = kl_divergence_loss(logits, soft_labels)
            total_loss  += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(hard_labels.numpy().tolist())

    labels   = list(range(len(emotion_cols)))
    macro_f1 = f1_score(all_targets, all_preds, labels=labels,
                        average="macro", zero_division=0)
    avg_loss = total_loss / max(num_batches, 1)

    if verbose:
        print(classification_report(
            all_targets, all_preds,
            labels=labels,
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
    print(f"[train_deepser] Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = MultimodalShardDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.shard_dir,
        roberta_dir=args.roberta_npz,
        split="train",
        split_mode=args.split_mode,
        seed=args.seed,
        drop_other=args.drop_other,
        annotation_dropout=args.annotation_dropout,
        n_annotators=args.n_annotators,
        drop_rate=args.drop_rate,
        audio_mixing=args.audio_mixing,
        audio_mix_prob=args.audio_mix_prob,
    )
    val_ds = MultimodalShardDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.shard_dir,
        roberta_dir=args.roberta_npz,
        split="validation",
        split_mode=args.split_mode,
        seed=args.seed,
        drop_other=args.drop_other,
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

    n_train_steps = math.ceil(len(train_ds) / args.batch_size)

    emotion_cols = train_ds.emotion_cols
    num_emotions = len(emotion_cols)
    print(f"[train_deepser] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[train_deepser] Emotion classes ({num_emotions}): {emotion_cols}")

    # ── Distribution re-weighting ─────────────────────────────────────────────
    class_weights = None
    if args.reweight:
        q = np.zeros(num_emotions, dtype=np.float64)
        for uid in train_ds.split_ids:
            q += train_ds.soft_map[uid]
        q /= max(len(train_ds.split_ids), 1)
        w = 1.0 / (q + 1e-6)
        w = w / w.sum()
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[reweight] q: { {c: round(float(v), 4) for c, v in zip(emotion_cols, q)} }")
        print(f"[reweight] w: { {c: round(float(v), 4) for c, v in zip(emotion_cols, w)} }")

    # ── Model (DeepSER) ───────────────────────────────────────────────────────
    model = DeepSERModel(
        num_emotions=num_emotions,
        whisper_dim=1280,
        roberta_dim=1024,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        dropout=args.dropout,
        classifier_hidden_dim=args.classifier_hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_deepser] Trainable parameters: {total_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    # lr=1e-5 όπως το MEDUSA paper — ΟΧΙ 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_macro_f1 = 0.0
    best_epoch    = 0
    last_ckpt     = output_dir / "last_model.pt"
    if args.resume and last_ckpt.exists():
        print(f"[resume] Βρέθηκε checkpoint: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_macro_f1 = ckpt.get("best_macro_f1", 0.0)
        best_epoch    = ckpt.get("best_epoch", 0)
        print(f"[resume] Συνέχεια από epoch {start_epoch} "
              f"(καλύτερο: Macro-F1 {best_macro_f1:.4f} @ epoch {best_epoch})")

    # ── Training log CSV ──────────────────────────────────────────────────────
    log_path   = output_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "lr"]
    if start_epoch == 1:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    if start_epoch > args.epochs:
        print(f"[train_deepser] Ήδη ολοκληρώθηκε. Macro-F1: {best_macro_f1:.4f}")
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss  = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                    total=n_train_steps, leave=False)

        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            whisper     = batch["whisper"].to(device)       # [B, 750, 1280]
            roberta     = batch["roberta"].to(device)       # [B, 1024]
            soft_labels = batch["soft_labels"].to(device)
            lengths     = batch["whisper_lengths"].to(device)

            # Re-weighting — μόνο στο training
            if class_weights is not None:
                soft_labels = soft_labels * class_weights
                soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True).clamp_min(1e-8)

            optimizer.zero_grad()
            logits = model(
                whisper=whisper,
                roberta=roberta,
                whisper_lengths=lengths,
            )
            loss = kl_divergence_loss(logits, soft_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if (step + 1) % args.log_every == 0:
                print(f"  Epoch {epoch} | Step {step+1}/{n_train_steps} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)
        current_lr     = scheduler.get_last_lr()[0]

        verbose_eval = (epoch % 5 == 0) or (epoch == args.epochs)
        val_metrics  = evaluate(model, val_loader, device, emotion_cols, verbose=verbose_eval)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val Macro-F1: {val_metrics['macro_f1']:.4f} "
            f"| LR: {current_lr:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":        epoch,
                "train_loss":   f"{avg_train_loss:.6f}",
                "val_loss":     f"{val_metrics['loss']:.6f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
                "lr":           f"{current_lr:.2e}",
            })

        # Best checkpoint
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

        # Last checkpoint (για resume)
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_macro_f1":         val_metrics["macro_f1"],
            "val_loss":             val_metrics["loss"],
            "best_macro_f1":        best_macro_f1,
            "best_epoch":           best_epoch,
            "emotion_cols":         emotion_cols,
            "args":                 vars(args),
        }, output_dir / "last_model.pt")

    print(f"\n[train_deepser] Ολοκληρώθηκε! Macro-F1: {best_macro_f1:.4f} (epoch {best_epoch})")
    print(f"[train_deepser] Training log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DeepSER (hierarchical deep fusion, Whisper + RoBERTa)."
    )
    parser.add_argument("--hf_dataset_path", type=str, required=True)
    parser.add_argument("--shard_dir",       type=str, required=True,
                        help="Directory με τα shard_*.npz Whisper features")
    parser.add_argument("--roberta_npz",     type=str, required=True,
                        help="Path to roberta_all.npz")
    parser.add_argument("--output_dir",      type=str, required=True)
    parser.add_argument("--split_mode",  type=str,   default="podcast",
                        choices=["podcast", "random"])
    # Hyperparameters από το MEDUSA paper
    parser.add_argument("--epochs",           type=int,   default=15)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--lr",               type=float, default=1e-5)
    parser.add_argument("--hidden_dim",       type=int,   default=1024)
    parser.add_argument("--nhead",            type=int,   default=8)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--classifier_hidden_dim", type=int, default=256)
    parser.add_argument("--num_workers",      type=int,   default=0)
    parser.add_argument("--log_every",        type=int,   default=50)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--no_resume",        dest="resume", action="store_false")
    parser.add_argument("--drop_other",       action="store_true",
                        help="8-class mode: αφαίρεσε την κλάση Other")
    parser.add_argument("--reweight",         action="store_true",
                        help="Distribution re-weighting για imbalance")
    parser.add_argument("--annotation_dropout", action="store_true")
    parser.add_argument("--n_annotators",     type=int,   default=5)
    parser.add_argument("--drop_rate",        type=float, default=0.2)
    parser.add_argument("--audio_mixing",     action="store_true")
    parser.add_argument("--audio_mix_prob",   type=float, default=0.5)
    parser.set_defaults(resume=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
