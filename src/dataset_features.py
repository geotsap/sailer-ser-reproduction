"""
src/dataset_features.py

PyTorch Dataset που φορτώνει pre-extracted .npy features αντί για raw audio.

Αντί να τρέχει WavLM/Whisper κάθε φορά, φορτώνει τα αποθηκευμένα
hidden states από το disk — πολύ πιο γρήγορο για training.

Αναμενόμενη δομή φακέλων:
    SLP/
    ├── features/
    │   ├── whisper-large-v3/     ← ένα .npy ανά utterance [T, 1280]
    │   └── wavlm-large/          ← ένα .npy ανά utterance [T, 1024]  (προαιρετικό)
    └── msp_podcast_hf/           ← το HuggingFace dataset (για τα labels)

Τυπική χρήση:
    from src.dataset_features import FeatureDataset
    from torch.utils.data import DataLoader

    ds = FeatureDataset(
        hf_dataset_path = "/content/drive/MyDrive/SLP/msp_podcast_hf",
        feature_dirs     = {
            "whisper": "/content/drive/MyDrive/SLP/features/whisper-large-v3",
            "wavlm":   "/content/drive/MyDrive/SLP/features/wavlm-large",  # προαιρετικό
        },
        split            = "train",
    )

    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        collate_fn=FeatureDataset.collate_fn)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants — πρέπει να ταιριάζουν με τα extracted features
# ---------------------------------------------------------------------------

PRIMARY_LABELS: List[str] = [
    "Angry", "Sad", "Happy", "Surprise",
    "Fear", "Disgust", "Contempt", "Neutral", "Other",
]

# Τα emotion label columns στο HuggingFace dataset
EMOTION_COLS: List[str] = [
    "anger", "sadness", "happiness", "surprise",
    "fear", "disgust", "contempt", "neutral", "other",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    """
    Parameters
    ----------
    hf_dataset_path : str | Path
        Path to the HuggingFace dataset saved with save_to_disk().
    feature_dirs : dict[str, str | Path]
        Mapping από encoder name σε directory με .npy files.
        Π.χ. {"whisper": "/path/to/whisper-large-v3"}
        Υποστηριζόμενα keys: "whisper", "wavlm"
        Τουλάχιστον ένα key είναι υποχρεωτικό.
    split : str
        Ποιο split να φορτώσει: "train", "validation", "test".
        Αν το dataset έχει μόνο "train" split (όπως το HF dataset),
        κάνει manual split με fixed seed.
    train_ratio : float
        Ποσοστό για train όταν γίνεται manual split.
    val_ratio : float
        Ποσοστό για validation όταν γίνεται manual split.
    skip_missing : bool
        Αν True, παραλείπει utterances που δεν έχουν .npy file.
        Αν False, κάνει raise FileNotFoundError.
    """

    def __init__(
        self,
        hf_dataset_path: str | Path,
        feature_dirs: Dict[str, str | Path],
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        skip_missing: bool = True,
    ) -> None:
        if not feature_dirs:
            raise ValueError("Πρέπει να δώσεις τουλάχιστον ένα feature_dir.")

        self.feature_dirs = {k: Path(v) for k, v in feature_dirs.items()}
        self.split        = split

        # ── Φόρτωσε το HuggingFace dataset ──────────────────────────────────
        print(f"[FeatureDataset] Φόρτωση dataset από {hf_dataset_path} ...")
        ds = load_from_disk(str(hf_dataset_path))
        if hasattr(ds, "keys"):
            ds = ds["train"]   # το HF dataset έχει μόνο train split

        # ── Manual train/val/test split ──────────────────────────────────────
        total     = len(ds)
        indices   = list(range(total))

        # Reproducible shuffle με fixed seed
        rng = np.random.default_rng(seed=42)
        rng.shuffle(indices)

        n_train = int(total * train_ratio)
        n_val   = int(total * val_ratio)

        if split == "train":
            split_indices = indices[:n_train]
        elif split in ("validation", "val", "dev"):
            split_indices = indices[n_train : n_train + n_val]
        elif split == "test":
            split_indices = indices[n_train + n_val :]
        else:
            raise ValueError(f"Άγνωστο split: {split!r}. Επίλεξε train/validation/test.")

        ds = ds.select(split_indices)
        print(f"[FeatureDataset] Split '{split}': {len(ds)} utterances")

        # ── Φιλτράρισμα utterances χωρίς .npy ──────────────────────────────
        valid_indices = []
        missing       = 0

        for i in range(len(ds)):
            file_name = ds[i]["file"]
            utt_id    = Path(file_name).stem
            all_exist = all(
                (feat_dir / f"{utt_id}.npy").exists()
                for feat_dir in self.feature_dirs.values()
            )
            if all_exist:
                valid_indices.append(i)
            else:
                missing += 1
                if not skip_missing:
                    raise FileNotFoundError(
                        f"Missing .npy για utterance: {utt_id}"
                    )

        if missing > 0:
            print(f"[FeatureDataset] Παραλείφθηκαν {missing} utterances χωρίς features.")

        self.ds = ds.select(valid_indices)
        print(f"[FeatureDataset] Τελικό μέγεθος: {len(self.ds)} utterances")

        # ── Pre-compute soft labels ──────────────────────────────────────────
        # Ελέγχουμε ποιες emotion columns υπάρχουν στο dataset
        available_cols = [c for c in EMOTION_COLS if c in self.ds.features]
        if not available_cols:
            raise ValueError(
                f"Δεν βρέθηκαν emotion columns. "
                f"Διαθέσιμες columns: {list(self.ds.features.keys())}"
            )
        self.emotion_cols = available_cols
        print(f"[FeatureDataset] Emotion columns: {self.emotion_cols}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_features(self, utt_id: str) -> Dict[str, Tensor]:
        """Φόρτωσε τα .npy features για ένα utterance."""
        features = {}
        for encoder_name, feat_dir in self.feature_dirs.items():
            arr = np.load(str(feat_dir / f"{utt_id}.npy"))   # [T, D]
            features[encoder_name] = torch.from_numpy(arr)    # Tensor [T, D]
        return features

    def _get_soft_label(self, sample: dict) -> Tensor:
        """Εξάγαγε soft label από τις emotion columns."""
        values = [float(sample.get(col, 0.0) or 0.0) for col in self.emotion_cols]
        label  = torch.tensor(values, dtype=torch.float32)

        # Κανονικοποίηση σε probability distribution
        total = label.sum()
        if total > 0:
            label = label / total
        else:
            # Αν δεν υπάρχουν annotations, uniform distribution
            label = torch.ones(len(self.emotion_cols)) / len(self.emotion_cols)

        return label

    # ── Public API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        sample  = self.ds[idx]
        utt_id  = Path(sample["file"]).stem
        features = self._load_features(utt_id)
        label    = self._get_soft_label(sample)

        return {
            **features,              # "whisper": Tensor [T, 1280], "wavlm": Tensor [T, 1024]
            "soft_label": label,     # Tensor [num_emotions]
            "hard_label": int(label.argmax().item()),
            "utt_id":     utt_id,
        }

    # ── Collate ──────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Padding και stacking για το DataLoader.

        Returns
        -------
        dict με keys:
            <encoder_name>          : Tensor [B, T_max, D]  — zero-padded
            <encoder_name>_lengths  : Tensor [B]             — αρχικά μήκη
            soft_labels             : Tensor [B, num_emotions]
            hard_labels             : Tensor [B]
            utt_ids                 : List[str]
        """
        output: Dict = {}

        # Βρες ποιοι encoders υπάρχουν στο batch
        encoder_keys = [k for k in batch[0].keys()
                        if k not in ("soft_label", "hard_label", "utt_id")]

        for key in encoder_keys:
            tensors = [item[key] for item in batch]          # List of [T, D]
            lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
            max_len = lengths.max().item()
            D       = tensors[0].shape[1]

            padded = torch.zeros(len(tensors), max_len, D)
            for i, t in enumerate(tensors):
                padded[i, : t.shape[0], :] = t

            output[key]                  = padded            # [B, T_max, D]
            output[f"{key}_lengths"]     = lengths           # [B]

        output["soft_labels"] = torch.stack([item["soft_label"] for item in batch])
        output["hard_labels"] = torch.tensor([item["hard_label"] for item in batch])
        output["utt_ids"]     = [item["utt_id"] for item in batch]

        return output
