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
        },
        split = "train",
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
# Constants
# ---------------------------------------------------------------------------

# Τα 9 SAILER primary emotion classes (σταθερή σειρά — δεν αλλάζει)
PRIMARY_LABELS: List[str] = [
    "Angry", "Sad", "Happy", "Surprise",
    "Fear", "Disgust", "Contempt", "Neutral", "Other",
]

# Columns του HF dataset που αντιστοιχούν 1-1 στα primary labels
# "other" υπολογίζεται ως άθροισμα των υπόλοιπων secondary columns
PRIMARY_COLS: List[str] = [
    "angry", "sad", "happy", "surprise",
    "fear", "disgust", "contempt", "neutral",
]

# Secondary columns που αθροίζονται στο "Other" bucket
OTHER_COLS: List[str] = [
    "frustrated", "annoyed", "disappointed",
    "depressed", "confused", "concerned",
    "amused", "excited",
]

# Mapping από major_emotion string → index στο PRIMARY_LABELS
# Όλα τα secondary emotions που δεν είναι primary → Other (index 8)
MAJOR_EMOTION_TO_IDX: Dict[str, int] = {
    "angry":        0,
    "sad":          1,
    "happy":        2,
    "surprise":     3,
    "fear":         4,
    "disgust":      5,
    "contempt":     6,
    "neutral":      7,
    # secondary → Other
    "frustrated":   8,
    "annoyed":      8,
    "disappointed": 8,
    "depressed":    8,
    "confused":     8,
    "concerned":    8,
    "amused":       8,
    "excited":      8,
}

# Για συμβατότητα με train_whisper.py
EMOTION_COLS: List[str] = PRIMARY_LABELS


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
        Τουλάχιστον ένα key είναι υποχρεωτικό.
    split : str
        Ποιο split να φορτώσει: "train", "validation", "test".
        Το HF dataset έχει μόνο "train" split — γίνεται manual split
        με fixed seed για reproducibility.
    train_ratio : float
        Ποσοστό για train (default 0.8).
    val_ratio : float
        Ποσοστό για validation (default 0.1). Το test παίρνει τα υπόλοιπα.
    skip_missing : bool
        Αν True, παραλείπει utterances που δεν έχουν .npy file.
        Αν False, κάνει raise FileNotFoundError.
    seed : int
        Seed για το random split (default 42).
    """

    def __init__(
        self,
        hf_dataset_path: str | Path,
        feature_dirs: Dict[str, str | Path],
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        skip_missing: bool = True,
        seed: int = 42,
    ) -> None:
        if not feature_dirs:
            raise ValueError("Πρέπει να δώσεις τουλάχιστον ένα feature_dir.")

        self.feature_dirs = {k: Path(v) for k, v in feature_dirs.items()}
        self.split        = split
        self.emotion_cols = PRIMARY_LABELS   # για συμβατότητα με train_whisper.py

        # ── Φόρτωσε το HuggingFace dataset (χωρίς audio για ταχύτητα) ───────
        print(f"[FeatureDataset] Φόρτωση dataset από {hf_dataset_path} ...")
        ds = load_from_disk(str(hf_dataset_path))
        if hasattr(ds, "keys"):
            ds = ds["train"]

        # Αφαίρεσε το audio column — δεν το χρειαζόμαστε, εξοικονομεί RAM
        if "audio" in ds.column_names:
            ds = ds.remove_columns(["audio"])

        # ── Manual train/val/test split με fixed seed ────────────────────────
        total   = len(ds)
        indices = list(range(total))
        rng     = np.random.default_rng(seed=seed)
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
            utt_id    = Path(ds[i]["file"]).stem
            all_exist = all(
                (feat_dir / f"{utt_id}.npy").exists()
                for feat_dir in self.feature_dirs.values()
            )
            if all_exist:
                valid_indices.append(i)
            else:
                missing += 1
                if not skip_missing:
                    raise FileNotFoundError(f"Missing .npy για utterance: {utt_id}")

        if missing > 0:
            print(f"[FeatureDataset] Παραλείφθηκαν {missing} utterances χωρίς features.")

        self.ds = ds.select(valid_indices)
        print(f"[FeatureDataset] Τελικό μέγεθος: {len(self.ds)} utterances")

    # ── Soft label από τα emotion columns ────────────────────────────────────

    def _get_soft_label(self, sample: dict) -> Tensor:
        """
        Υπολογίζει 9-dim soft label distribution από τα HF dataset columns.

        Τα 8 primary columns διαβάζονται απευθείας.
        Το "Other" είναι το άθροισμα των secondary columns.
        Στο τέλος κανονικοποιείται σε probability distribution (sum=1).
        """
        # Primary values (8 classes)
        primary_vals = [float(sample.get(col, 0.0) or 0.0) for col in PRIMARY_COLS]

        # Other = άθροισμα secondary columns
        other_val = sum(float(sample.get(col, 0.0) or 0.0) for col in OTHER_COLS)

        values = primary_vals + [other_val]
        label  = torch.tensor(values, dtype=torch.float32)

        # Κανονικοποίηση
        total = label.sum()
        if total > 0:
            label = label / total
        else:
            label = torch.ones(len(PRIMARY_LABELS)) / len(PRIMARY_LABELS)

        return label  # [9]

    def _get_hard_label(self, sample: dict) -> int:
        """Hard label από το major_emotion column."""
        major = (sample.get("major_emotion") or "").strip().lower()
        return MAJOR_EMOTION_TO_IDX.get(major, 8)  # default → Other

    # ── Feature loading ───────────────────────────────────────────────────────

    def _load_features(self, utt_id: str) -> Dict[str, Tensor]:
        features = {}
        for encoder_name, feat_dir in self.feature_dirs.items():
            arr = np.load(str(feat_dir / f"{utt_id}.npy"))  # [T, D]
            features[encoder_name] = torch.from_numpy(arr)  # Tensor [T, D]
        return features

    # ── Public API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        sample   = self.ds[idx]
        utt_id   = Path(sample["file"]).stem
        features = self._load_features(utt_id)
        soft     = self._get_soft_label(sample)
        hard     = self._get_hard_label(sample)

        return {
            **features,           # "whisper": Tensor [T, 1280]
            "soft_label": soft,   # Tensor [9]
            "hard_label": hard,   # int
            "utt_id":     utt_id,
        }

    # ── Collate ──────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Padding και stacking για το DataLoader.

        Returns
        -------
        dict με keys:
            <encoder_name>         : Tensor [B, T_max, D]  — zero-padded
            <encoder_name>_lengths : Tensor [B]             — αρχικά μήκη σε frames
            soft_labels            : Tensor [B, 9]
            hard_labels            : Tensor [B]
            utt_ids                : List[str]
        """
        output: Dict = {}

        encoder_keys = [k for k in batch[0].keys()
                        if k not in ("soft_label", "hard_label", "utt_id")]

        for key in encoder_keys:
            tensors = [item[key] for item in batch]
            lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
            max_len = int(lengths.max().item())
            D       = tensors[0].shape[1]

            padded = torch.zeros(len(tensors), max_len, D)
            for i, t in enumerate(tensors):
                padded[i, : t.shape[0], :] = t

            output[key]              = padded   # [B, T_max, D]
            output[f"{key}_lengths"] = lengths  # [B]

        output["soft_labels"] = torch.stack([item["soft_label"] for item in batch])
        output["hard_labels"] = torch.tensor([item["hard_label"] for item in batch])
        output["utt_ids"]     = [item["utt_id"] for item in batch]

        return output
