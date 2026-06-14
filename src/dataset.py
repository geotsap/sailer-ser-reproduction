"""
src/dataset.py

PyTorch Dataset for the MSP-Podcast corpus.

Reads one of the manifest CSVs produced by scripts/create_msp_manifests.py and
returns raw waveform tensors together with soft primary-emotion labels and
utterance metadata.

Assumptions:
    - Audio files are 16 kHz mono .wav (or any format supported by torchaudio).
    - The manifest CSV contains columns:  audio_path, target_<emotion>, utt_id,
      speaker_id, consensus_label  (all produced by create_msp_manifests.py).
    - Missing audio is flagged at construction time so failures are loud and early.

Typical usage:
    from src.dataset import MspPodcastDataset
    from torch.utils.data import DataLoader

    train_ds = MspPodcastDataset("data/manifests/msp_podcast/train.csv")
    loader   = DataLoader(train_ds, batch_size=16, shuffle=True,
                          collate_fn=MspPodcastDataset.collate_fn)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants — must match create_msp_manifests.py
# ---------------------------------------------------------------------------

PRIMARY_LABELS: List[str] = [
    "Angry", "Sad", "Happy", "Surprise",
    "Fear", "Disgust", "Contempt", "Neutral", "Other",
]

TARGET_COLS: List[str] = [f"target_{label.lower()}" for label in PRIMARY_LABELS]

# Minority classes as defined in the SAILER paper — used by the AudioMixer
# to decide which samples are candidates for MixUp augmentation.
MINORITY_LABELS: List[str] = ["Surprise", "Fear", "Disgust", "Contempt"]
MINORITY_INDICES: List[int] = [PRIMARY_LABELS.index(l) for l in MINORITY_LABELS]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MspPodcastDataset(Dataset):
    """
    Parameters
    ----------
    manifest_path : str | Path
        Path to a manifest CSV (train.csv / validation.csv / test.csv).
    sample_rate : int
        Target sample rate.  Audio is resampled if the file differs.
    max_duration_sec : float
        Utterances longer than this are centre-cropped (not truncated from the
        end) so that the most expressive part of the clip is preserved.
        Set to 0 to disable cropping.
    min_duration_sec : float
        Utterances shorter than this are skipped at load time with a warning.
    require_audio : bool
        If True, raise FileNotFoundError for any row whose audio_path does not
        exist.  If False, those rows are silently dropped.
    label_schema_path : str | Path | None
        Optional path to label_schema.json.  Only used for validation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16_000,
        max_duration_sec: float = 15.0,
        min_duration_sec: float = 0.1,
        require_audio: bool = True,
        label_schema_path: Optional[str | Path] = None,
    ) -> None:
        self.manifest_path   = Path(manifest_path)
        self.sample_rate     = sample_rate
        self.max_samples     = int(max_duration_sec * sample_rate) if max_duration_sec > 0 else 0
        self.min_samples     = int(min_duration_sec * sample_rate)
        self.require_audio   = require_audio

        # ── Optional schema validation ───────────────────────────────────────
        if label_schema_path is not None:
            schema = json.loads(Path(label_schema_path).read_text(encoding="utf-8"))
            schema_targets = schema.get("target_columns", [])
            if schema_targets and schema_targets != TARGET_COLS:
                raise ValueError(
                    f"Schema target columns {schema_targets} do not match "
                    f"hardcoded TARGET_COLS {TARGET_COLS}."
                )

        # ── Load & validate manifest ─────────────────────────────────────────
        df = pd.read_csv(self.manifest_path, low_memory=False)
        self._validate_columns(df)

        # Drop rows where audio file is absent
        audio_exists = df["audio_path"].apply(lambda p: Path(p).exists())
        missing = (~audio_exists).sum()
        if missing:
            if require_audio:
                first_missing = df.loc[~audio_exists, "audio_path"].iloc[0]
                raise FileNotFoundError(
                    f"{missing} audio file(s) not found in manifest "
                    f"'{self.manifest_path.name}'. First missing: {first_missing}"
                )
            print(f"[MspPodcastDataset] Warning: dropping {missing} rows with missing audio.")
            df = df[audio_exists].reset_index(drop=True)

        self.df: pd.DataFrame = df.reset_index(drop=True)

        # Pre-build soft-label tensor  [N, num_classes]  (kept in RAM, cheap)
        self.soft_labels: Tensor = torch.tensor(
            self.df[TARGET_COLS].values, dtype=torch.float32
        )

        # Index of the argmax class for each sample — used by AudioMixer
        self.hard_labels: Tensor = self.soft_labels.argmax(dim=1)

        print(
            f"[MspPodcastDataset] Loaded '{self.manifest_path.name}': "
            f"{len(self.df)} utterances | "
            f"{self.soft_labels.shape[1]} emotion classes"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = {"audio_path", "utt_id", *TARGET_COLS}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )
        # Soft labels must be numeric and sum to ~1
        label_sums = df[TARGET_COLS].sum(axis=1)
        bad = (~label_sums.between(0.99, 1.01)).sum()
        if bad > 0:
            print(
                f"[MspPodcastDataset] Warning: {bad} rows have soft-label sums "
                "outside [0.99, 1.01] — check manifest generation."
            )

    def _load_waveform(self, audio_path: str) -> Tensor:
        """Load audio, resample if needed, convert to mono, return [T] tensor."""
        waveform, sr = torchaudio.load(audio_path)          # [C, T]

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]

        # Resample
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.squeeze(0)                       # [T]

        # Skip very short clips (corrupted / silence)
        if waveform.shape[0] < self.min_samples:
            raise RuntimeError(
                f"Audio too short ({waveform.shape[0]} samples < "
                f"{self.min_samples} min): {audio_path}"
            )

        # Centre-crop long clips
        if self.max_samples > 0 and waveform.shape[0] > self.max_samples:
            start = (waveform.shape[0] - self.max_samples) // 2
            waveform = waveform[start : start + self.max_samples]

        return waveform  # [T]

    # ── Public API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row      = self.df.iloc[idx]
        waveform = self._load_waveform(row["audio_path"])

        return {
            "waveform":        waveform,                      # Tensor [T]
            "soft_label":      self.soft_labels[idx],         # Tensor [9]
            "hard_label":      self.hard_labels[idx].item(),  # int
            "utt_id":          row["utt_id"],                 # str
            "speaker_id":      row.get("speaker_id", ""),     # str
            "consensus_label": row.get("consensus_label", ""),# str
            "audio_path":      row["audio_path"],             # str  (useful for debugging)
        }

    def get_class_indices(self, class_idx: int) -> List[int]:
        """Return dataset indices whose argmax label equals class_idx.
        Used by AudioMixer to sample from a specific emotion class."""
        return (self.hard_labels == class_idx).nonzero(as_tuple=True)[0].tolist()

    def get_minority_indices(self) -> List[int]:
        """Return dataset indices belonging to any SAILER minority class."""
        mask = torch.zeros(len(self), dtype=torch.bool)
        for idx in MINORITY_INDICES:
            mask |= (self.hard_labels == idx)
        return mask.nonzero(as_tuple=True)[0].tolist()

    # ── Collate ──────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """Pad waveforms to the longest in the batch and stack tensors.

        Returns
        -------
        dict with keys:
            waveforms   : Tensor [B, T_max]   — zero-padded
            lengths     : Tensor [B]           — original sample counts
            soft_labels : Tensor [B, 9]
            hard_labels : Tensor [B]
            utt_ids     : List[str]
            speaker_ids : List[str]
        """
        waveforms   = [item["waveform"]   for item in batch]
        soft_labels = [item["soft_label"] for item in batch]
        hard_labels = [item["hard_label"] for item in batch]
        utt_ids     = [item["utt_id"]     for item in batch]
        speaker_ids = [item["speaker_id"] for item in batch]

        lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        max_len = lengths.max().item()

        padded = torch.zeros(len(waveforms), max_len)
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w

        return {
            "waveforms":   padded,                              # [B, T_max]
            "lengths":     lengths,                             # [B]
            "soft_labels": torch.stack(soft_labels),           # [B, 9]
            "hard_labels": torch.tensor(hard_labels),          # [B]
            "utt_ids":     utt_ids,
            "speaker_ids": speaker_ids,
        }
