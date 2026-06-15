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
"""

from __future__ import annotations

import time
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

PRIMARY_LABELS: List[str] = [
    "Angry", "Sad", "Happy", "Surprise",
    "Fear", "Disgust", "Contempt", "Neutral", "Other",
]

PRIMARY_COLS: List[str] = [
    "angry", "sad", "happy", "surprise",
    "fear", "disgust", "contempt", "neutral",
]

OTHER_COLS: List[str] = [
    "frustrated", "annoyed", "disappointed",
    "depressed", "confused", "concerned",
    "amused", "excited",
]

MAJOR_EMOTION_TO_IDX: Dict[str, int] = {
    "angry": 0, "sad": 1, "happy": 2, "surprise": 3,
    "fear": 4, "disgust": 5, "contempt": 6, "neutral": 7,
    "frustrated": 8, "annoyed": 8, "disappointed": 8,
    "depressed": 8, "confused": 8, "concerned": 8,
    "amused": 8, "excited": 8,
}

EMOTION_COLS: List[str] = PRIMARY_LABELS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    """
    Dataset που φορτώνει pre-extracted features χωρίς να ελέγχει
    ύπαρξη αρχείων στο __init__ — αποφεύγει Drive rate limits.
    """

    def __init__(
        self,
        hf_dataset_path: str | Path,
        feature_dirs: Dict[str, str | Path],
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not feature_dirs:
            raise ValueError("Πρέπει να δώσεις τουλάχιστον ένα feature_dir.")

        self.feature_dirs = {k: Path(v) for k, v in feature_dirs.items()}
        self.split        = split
        self.emotion_cols = PRIMARY_LABELS

        # ── Φόρτωσε το HuggingFace dataset (χωρίς audio) ────────────────────
        print(f"[FeatureDataset] Φόρτωση dataset από {hf_dataset_path} ...")
        ds = load_from_disk(str(hf_dataset_path))
        if hasattr(ds, "keys"):
            ds = ds["train"]

        if "audio" in ds.column_names:
            ds = ds.remove_columns(["audio"])

        # ── Manual train/val/test split ──────────────────────────────────────
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

        self.ds = ds.select(split_indices)
        print(f"[FeatureDataset] Split '{split}': {len(self.ds)} utterances")
        print(f"[FeatureDataset] ΔΕΝ ελέγχονται αρχεία κατά την εκκίνηση — lazy loading.")

    # ── Soft label ───────────────────────────────────────────────────────────

    def _get_soft_label(self, sample: dict) -> Tensor:
        primary_vals = [float(sample.get(col, 0.0) or 0.0) for col in PRIMARY_COLS]
        other_val = sum(float(sample.get(col, 0.0) or 0.0) for col in OTHER_COLS)
        values = primary_vals + [other_val]
        label  = torch.tensor(values, dtype=torch.float32)
        total = label.sum()
        if total > 0:
            label = label / total
        else:
            label = torch.ones(len(PRIMARY_LABELS)) / len(PRIMARY_LABELS)
        return label

    def _get_hard_label(self, sample: dict) -> int:
        major = (sample.get("major_emotion") or "").strip().lower()
        return MAJOR_EMOTION_TO_IDX.get(major, 8)

    # ── Feature loading με retry για Drive ───────────────────────────────────

    def _load_features(self, utt_id: str) -> Dict[str, Tensor]:
        """Φορτώνει features με retry logic για Google Drive I/O errors."""
        features = {}
        for encoder_name, feat_dir in self.feature_dirs.items():
            npy_path = str(feat_dir / f"{utt_id}.npy")
            arr = None
            for attempt in range(3):
                try:
                    arr = np.load(npy_path)
                    break
                except OSError:
                    if attempt < 2:
                        time.sleep(1.0 * (attempt + 1))  # 1s, 2s wait
                    else:
                        raise  # Αν 3 φορές αποτύχει, σηκώνει error
            features[encoder_name] = torch.from_numpy(arr)
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
            **features,
            "soft_label": soft,
            "hard_label": hard,
            "utt_id":     utt_id,
        }

    # ── Collate ──────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Padding και stacking — παραλείπει None items (αποτυχημένα loads)."""
        # Φιλτράρισμα τυχόν None entries
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

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

            output[key]              = padded
            output[f"{key}_lengths"] = lengths

        output["soft_labels"] = torch.stack([item["soft_label"] for item in batch])
        output["hard_labels"] = torch.tensor([item["hard_label"] for item in batch])
        output["utt_ids"]     = [item["utt_id"] for item in batch]

        return output
