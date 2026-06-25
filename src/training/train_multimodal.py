"""
src/training/train_multimodal.py

Training script για multimodal Speech Emotion Recognition με
pre-extracted Whisper + RoBERTa features και KL Divergence loss.

Βασισμένο ακριβώς στο train_whisper.py — μόνο οι multimodal αλλαγές.

Χρήση στο Colab:
    !python src/training/train_multimodal.py \
        --hf_dataset_path  /content/drive/MyDrive/SLP/msp_podcast_hf \
        --shard_dir        /content/drive/MyDrive/SLP/features/whisper_shards \
        --roberta_dir      /content/drive/MyDrive/SLP/features/roberta-large \
        --output_dir       /content/drive/MyDrive/SLP/checkpoints/multimodal \
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

            whisper     = batch["whisper"].to(device)
            roberta     = batch["roberta"].unsqueeze(1).to(device)  # [B, 1024] -> [B, 1, 1024]
            soft_labels = batch["soft_labels"].to(device)
            hard_labels = batch["hard_labels"]
            lengths     = batch["whisper_lengths"]

            # speech_mask από lengths
            B, T, _ = whisper.shape
            speech_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)

            logits = model(
                whisper_hidden_states=whisper,
                roberta_hidden_states=roberta,
                speech_mask=speech_mask,
            )
            loss = kl_divergence_loss(logits, soft_labels)
            total_loss  += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(hard_labels.numpy().tolist())

    labels = list(range(len(emotion_cols)))
    macro_f1 = f1_score(all_targets, all_preds, labels=labels,
                        average="macro", zero_division=0)
    avg_loss  = total_loss / max(num_batches, 1)

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
    print(f"[train_multimodal] Device: {device}")

    # ── Datasets (streaming από τα shards + RoBERTa .npy) ────────────────────
    train_ds = MultimodalShardDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.shard_dir,
        roberta_dir=args.roberta_dir,
        split="train",
        split_mode=args.split_mode,
        seed=args.seed,
        annotation_dropout=args.annotation_dropout,
        n_annotators=args.n_annotators,
        drop_rate=args.drop_rate,
        drop_other=args.drop_other,
        audio_mixing=args.audio_mixing,
        audio_mix_prob=args.audio_mix_prob,
    )
    val_ds = MultimodalShardDataset(
        hf_dataset_path=args.hf_dataset_path,
        shard_dir=args.shard_dir,
        roberta_dir=args.roberta_dir,
        split="validation",
        split_mode=args.split_mode,
        seed=args.seed,
        drop_other=args.drop_other,
    )

    # ΠΡΟΣΟΧΗ: με IterableDataset ΔΕΝ βάζουμε shuffle=True (το shuffle γίνεται
    # μέσα στο dataset). num_workers=0 ασφαλές για streaming.
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

    # Πόσα batches περίπου ανά epoch (για το tqdm / μέσο loss)
    n_train_steps = math.ceil(len(train_ds) / args.batch_size)

    emotion_cols = train_ds.emotion_cols
    num_emotions = len(emotion_cols)
    print(f"[train_multimodal] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[train_multimodal] Emotion classes ({num_emotions}): {emotion_cols}")
    if args.audio_mixing:
        print(f"[train_multimodal] Audio mixing ON (p={args.audio_mix_prob})")

    # ── Distribution re-weighting (SAILER §2.4) ──────────────────────────────
    class_weights = None
    if args.reweight:
        q = np.zeros(num_emotions, dtype=np.float64)
        for uid in train_ds.split_ids:
            q += train_ds.soft_map[uid]
        q /= max(len(train_ds.split_ids), 1)
        w = 1.0 / (q + 1e-6)
        w = w / w.sum()
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[reweight] q (κατανομη): "
              f"{ {c: round(float(v), 4) for c, v in zip(emotion_cols, q)} }")
        print(f"[reweight] w (βαρη):     "
              f"{ {c: round(float(v), 4) for c, v in zip(emotion_cols, w)} }")

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

    # ── Resume από checkpoint (αν υπάρχει last_model.pt στο output_dir) ──────
    start_epoch   = 1
    best_macro_f1 = 0.0
    best_epoch    = 0
    last_ckpt = output_dir / "last_model.pt"
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
              f"(καλύτερο μέχρι τώρα: Macro-F1 {best_macro_f1:.4f} @ epoch {best_epoch})")

    # ── Training log CSV ─────────────────────────────────────────────────────
    log_path   = output_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "lr"]
    if start_epoch == 1:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # ── Training loop ────────────────────────────────────────────────────────
    if start_epoch > args.epochs:
        print(f"[train_multimodal] Είχε ήδη ολοκληρωθεί ({args.epochs} epochs). "
              f"Καλύτερο Macro-F1: {best_macro_f1:.4f} (epoch {best_epoch})")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss  = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                    total=n_train_steps, leave=False)
        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            whisper     = batch["whisper"].to(device)
            roberta     = batch["roberta"].unsqueeze(1).to(device)  # [B, 1024] -> [B, 1, 1024]
            soft_labels = batch["soft_labels"].to(device)
            lengths     = batch["whisper_lengths"]

            # speech_mask από lengths
            B, T, _ = whisper.shape
            speech_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)

            # Re-weighting — μόνο στο training
            if class_weights is not None:
                soft_labels = soft_labels * class_weights
                soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True).clamp_min(1e-8)

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
                    f"  Epoch {epoch} | Step {step+1}/{n_train_steps} "
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

        # ── Best checkpoint ───────────────────────────────────────────────────
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

        # ── Last checkpoint (για resume) ──────────────────────────────────────
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
                        help="Directory με τα shard_*.npz (pre-extracted Whisper features)")
    parser.add_argument("--roberta_dir",     type=str, required=True,
                        help="Directory με τα .npy RoBERTa features")
    parser.add_argument("--output_dir",      type=str, required=True,
                        help="Where to save checkpoints and training log")
    parser.add_argument("--split_mode",  type=str,   default="podcast",
                        choices=["podcast", "random"])
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=0,
                        help="0 = ασφαλές για streaming.")
    parser.add_argument("--log_every",   type=int,   default=50)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no_resume",   dest="resume", action="store_false",
                        help="Ξεκίνα από την αρχή, αγνοώντας τυχόν last_model.pt")
    parser.add_argument("--drop_other", action="store_true",
                        help="8-class mode: πέτα την κλάση 'Other', renormalize στις 8 primary")
    parser.add_argument("--reweight",    action="store_true",
                        help="Distribution re-weighting (SAILER §2.4) για το imbalance")
    parser.add_argument("--annotation_dropout", action="store_true",
                        help="Annotation dropout augmentation (SAILER §2.3, train-only)")
    parser.add_argument("--n_annotators", type=int, default=5,
                        help="Υποτιθέμενος αριθμός annotators για τα pseudo-counts")
    parser.add_argument("--drop_rate",   type=float, default=0.2,
                        help="Ποσοστό annotations που πέφτουν (από majority κλάσεις)")
    parser.add_argument("--audio_mixing", action="store_true",
                        help="Feature-space audio mixing (SAILER §2.3 adaptation, train-only)")
    parser.add_argument("--audio_mix_prob", type=float, default=0.5,
                        help="Πιθανότητα mixing ανά majority δείγμα")
    parser.set_defaults(resume=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
