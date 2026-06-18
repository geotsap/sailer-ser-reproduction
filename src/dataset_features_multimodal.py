"""
src/dataset_features_multimodal.py

Shard-streaming Dataset για multimodal training με Whisper + RoBERTa features.

Επεκτείνει το ShardFeatureDataset ώστε να φορτώνει ταυτόχρονα:
    - Whisper hidden states από shard .npz αρχεία  [750, 1280]
    - RoBERTa pooled embeddings από individual .npy  [1, 1024]

ΔΟΜΗ ΦΑΚΕΛΩΝ:
    SLP/
    ├── features/
    │   ├── whisper_shards/   ← shard_0000.npz, shard_0001.npz, ...
    │   └── roberta-large/    ← MSP-PODCAST_XXXX_YYYY.npy  shape [1, 1024]
    └── msp_podcast_hf/       ← το HuggingFace dataset (για τα labels)

Χρήση:
    from src.dataset_features_multimodal import MultimodalShardDataset

    train_ds = MultimodalShardDataset(
        hf_dataset_path = "/content/drive/MyDrive/SLP/msp_podcast_hf",
        shard_dir       = "/content/drive/MyDrive/SLP/features/whisper_shards",
        roberta_dir     = "/content/drive/MyDrive/SLP/features/roberta-large",
        split           = "train",
    )

    loader = DataLoader(
        train_ds,
        batch_size  = 32,
        num_workers = 2,
        collate_fn  = MultimodalShardDataset.collate_fn,
    )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

# Επαναχρησιμοποιούμε το label index από το υπάρχον dataset_features.py
from src.dataset_features import (
    _build_label_index,
    PRIMARY_LABELS,
    KEEP_FRAMES,
    EMOTION_COLS,
)


class MultimodalShardDataset(IterableDataset):
    """
    Streaming dataset που φορτώνει Whisper shards + RoBERTa .npy features.

    Parameters
    ----------
    hf_dataset_path : str | Path
        Path to the HuggingFace dataset (για τα labels).
    shard_dir : str | Path
        Φάκελος με τα Whisper shard αρχεία (shard_*.npz).
    roberta_dir : str | Path
        Φάκελος με τα RoBERTa .npy αρχεία (ένα ανά utterance).
    split : str
        "train", "validation", ή "test".
    split_mode : str
        "podcast" (default) ή "random".
    shuffle : bool | None
        True για train, False για val/test (default: auto).
    buffer_size : int
        Μέγεθος buffer για ανάμειξη μεταξύ shards.
    skip_missing_roberta : bool
        Αν True, παραλείπει utterances χωρίς RoBERTa .npy.
    seed : int
        Seed για reproducibility.
    """

    def __init__(
        self,
        hf_dataset_path: str | Path,
        shard_dir: str | Path,
        roberta_dir: str | Path,
        split: str = "train",
        split_mode: str = "podcast",
        shuffle: Optional[bool] = None,
        buffer_size: int = 2000,
        skip_missing_roberta: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.shard_dir    = Path(shard_dir)
        self.roberta_dir  = Path(roberta_dir)
        self.split        = "validation" if split in ("val", "dev") else split
        self.shuffle      = (self.split == "train") if shuffle is None else shuffle
        self.buffer_size  = buffer_size
        self.skip_missing = skip_missing_roberta
        self.seed         = seed
        self.emotion_cols = PRIMARY_LABELS

        # Labels + split assignment (cached — κοινό με ShardFeatureDataset)
        idx = _build_label_index(hf_dataset_path, split_mode=split_mode, seed=seed)
        self.soft_map  = idx["soft"]
        self.hard_map  = idx["hard"]
        self.split_ids = idx["splits"][self.split]

        # Λίστα Whisper shard αρχείων
        self.shard_files = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shard_files:
            raise FileNotFoundError(
                f"Δεν βρέθηκαν shard_*.npz στο {self.shard_dir}"
            )

        # ── RoBERTa features ────────────────────────────────────────────────
        # Προτιμάμε ΕΝΑ packed αρχείο (roberta_all.npz με keys embeddings/utt_ids)
        # -> φορτώνεται μία φορά στη RAM, χωρίς Drive listing 149χιλ. αρχείων.
        # Το roberta_dir μπορεί να είναι: (α) path στο .npz, ή (β) φάκελος που το
        # περιέχει, ή (γ) legacy φάκελος με ξεχωριστά .npy (αργό/ασταθές σε Drive).
        rp = self.roberta_dir
        if rp.is_dir():
            cand = rp / "roberta_all.npz"
            npz_path = cand if cand.exists() else None
        elif rp.suffix == ".npz" and rp.exists():
            npz_path = rp
        else:
            npz_path = None

        self.roberta_map = None  # dict utt_id -> np.ndarray (όταn υπάρχει packed)
        if npz_path is not None:
            print(f"[MultimodalShardDataset] Φόρτωση RoBERTa από {npz_path.name} ...")
            data = np.load(str(npz_path))
            embs = data["embeddings"]            # [N, 1024]
            uids = data["utt_ids"]
            self.roberta_map = {str(u): embs[i] for i, u in enumerate(uids)}
            roberta_available = set(self.roberta_map.keys())
        else:
            if not self.roberta_dir.exists():
                raise FileNotFoundError(
                    "Δεν βρέθηκαν RoBERTa features: ούτε roberta_all.npz "
                    f"ούτε φάκελος {self.roberta_dir}"
                )
            print("[MultimodalShardDataset] ΠΡΟΣΟΧΗ: legacy per-file .npy mode "
                  "(αργό/ασταθές σε Drive). Προτιμήστε roberta_all.npz.")
            roberta_available = {p.stem for p in self.roberta_dir.glob("*.npy")}

        split_with_roberta = self.split_ids & roberta_available
        missing = len(self.split_ids) - len(split_with_roberta)
        if missing > 0:
            if not skip_missing_roberta:
                raise FileNotFoundError(
                    f"{missing} utterances δεν έχουν RoBERTa features."
                )
            print(f"[MultimodalShardDataset] Παραλείπονται {missing} utterances "
                  f"χωρίς RoBERTa features.")

        # Χρησιμοποιούμε μόνο utterances που έχουν ΚΑΙ τα δύο features
        self.split_ids = split_with_roberta

        print(f"[MultimodalShardDataset] split='{self.split}' | "
              f"{len(self.shard_files)} shards | "
              f"{len(self.split_ids)} utterances (Whisper + RoBERTa)")

    def __len__(self) -> int:
        return len(self.split_ids)

    def _load_roberta(self, utt_id: str) -> Tensor:
        """RoBERTa embedding [1024] για ένα utterance (από RAM ή legacy .npy)."""
        if self.roberta_map is not None:
            arr = self.roberta_map[utt_id]                 # [1024] (ή [1,1024])
        else:
            arr = np.load(str(self.roberta_dir / f"{utt_id}.npy"))
        arr = np.asarray(arr).reshape(-1)                  # -> [1024]
        return torch.from_numpy(arr.copy())

    def _iter_shards(self) -> List[Path]:
        """Διαμοιράζει τα shards στους workers και ανακατεύει αν χρειάζεται."""
        shards = list(self.shard_files)
        info   = get_worker_info()
        if info is not None and info.num_workers > 1:
            shards = shards[info.id :: info.num_workers]
        if self.shuffle:
            random.shuffle(shards)
        return shards

    def __iter__(self):
        shards = self._iter_shards()
        buffer: List[Dict] = []

        for sf in shards:
            data  = np.load(sf)
            feats = data["feats"]    # [K, 750, 1280] float16
            uids  = data["utt_ids"]
            lens  = data["lengths"]

            order = list(range(len(uids)))
            if self.shuffle:
                random.shuffle(order)

            for i in order:
                uid = str(uids[i])

                # Κρατάμε μόνο utterances του σωστού split που έχουν RoBERTa
                if uid not in self.split_ids:
                    continue

                # Φόρτωση RoBERTa embedding
                try:
                    roberta = self._load_roberta(uid)   # [1024]
                except Exception:
                    continue

                item = {
                    "whisper":    torch.from_numpy(feats[i].copy()),  # [750, 1280] fp16
                    "roberta":    roberta,                             # [1024] fp32
                    "length":     int(lens[i]),
                    "soft_label": torch.from_numpy(self.soft_map[uid].copy()),
                    "hard_label": self.hard_map[uid],
                    "utt_id":     uid,
                }
                buffer.append(item)

                if len(buffer) >= self.buffer_size:
                    if self.shuffle:
                        random.shuffle(buffer)
                    while buffer:
                        yield buffer.pop()

            del data, feats

        # Flush ό,τι έμεινε στο buffer
        if self.shuffle:
            random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Stacks και επιστρέφει batch με Whisper + RoBERTa features.

        Returns
        -------
        dict με keys:
            whisper         : Tensor [B, 750, 1280] float32
            whisper_lengths : Tensor [B]             int64
            roberta         : Tensor [B, 1024]       float32
            soft_labels     : Tensor [B, 9]          float32
            hard_labels     : Tensor [B]             int64
            utt_ids         : List[str]
        """
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        return {
            "whisper":         torch.stack([b["whisper"] for b in batch]).float(),
            "whisper_lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
            "roberta":         torch.stack([b["roberta"] for b in batch]).float(),
            "soft_labels":     torch.stack([b["soft_label"] for b in batch]).float(),
            "hard_labels":     torch.tensor([b["hard_label"] for b in batch], dtype=torch.long),
            "utt_ids":         [b["utt_id"] for b in batch],
        }
