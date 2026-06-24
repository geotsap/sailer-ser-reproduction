"""
src/dataset_features_multimodal.py

Shard-streaming Dataset for multimodal training with:
- Whisper-Large-v3 hidden-state shards: shard_*.npz, feats [K, 750, 1280]
- Packed RoBERTa embeddings: roberta_all.npz with embeddings/utt_ids

Supports:
- podcast/random split mode through src.dataset_features._build_label_index
- 8-class mode (--drop_other), where Other mass is removed and primary labels
  are renormalized
- annotation dropout on soft labels, train split only
- feature-space audio mixing, train split only
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset_features import _build_label_index, PRIMARY_LABELS
from src.augmentations_features import (
    MAJORITY_IDX_8,
    MAJORITY_IDX_9,
    annotation_dropout as apply_annotation_dropout,
    apply_feature_audio_mixing,
)


class MultimodalShardDataset(IterableDataset):
    """Streaming dataset that loads Whisper shards plus RoBERTa embeddings."""

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
        annotation_dropout: bool = False,
        n_annotators: int = 5,
        drop_rate: float = 0.2,
        drop_other: bool = False,
        audio_mixing: bool = False,
        audio_mix_prob: float = 0.5,
    ) -> None:
        super().__init__()

        self.shard_dir = Path(shard_dir)
        self.roberta_dir = Path(roberta_dir)
        self.split = "validation" if split in ("val", "dev") else split
        self.shuffle = (self.split == "train") if shuffle is None else shuffle
        self.buffer_size = int(buffer_size)
        self.skip_missing = bool(skip_missing_roberta)
        self.seed = int(seed)

        # 8-class setup used for the final paper-style augmentation runs.
        self.drop_other = bool(drop_other)
        if self.drop_other:
            self.emotion_cols = PRIMARY_LABELS[:8]
            self.majority_idx = MAJORITY_IDX_8                 # Angry/Sad/Happy/Neutral
            self.minority_idx = (3, 4, 5, 6)                   # Surprise/Fear/Disgust/Contempt
        else:
            self.emotion_cols = PRIMARY_LABELS
            self.majority_idx = MAJORITY_IDX_9                 # project convention for label dropout
            self.minority_idx = tuple(i for i in range(len(PRIMARY_LABELS)) if i not in self.majority_idx)

        # Augmentations are train-only.
        self.annotation_dropout = bool(annotation_dropout) and (self.split == "train")
        self.n_annotators = int(n_annotators)
        self.drop_rate = float(drop_rate)

        self.audio_mixing = bool(audio_mixing) and (self.split == "train")
        self.audio_mix_prob = float(audio_mix_prob)
        self.aug_rng = np.random.default_rng(self.seed + 777)

        if self.annotation_dropout:
            print(
                f"[MultimodalShardDataset] Annotation dropout ON "
                f"(N={self.n_annotators}, rate={self.drop_rate}, majority={self.majority_idx})"
            )
        if self.audio_mixing:
            print(
                f"[MultimodalShardDataset] Feature-space audio mixing ON "
                f"(p={self.audio_mix_prob}, majority={self.majority_idx}, minority={self.minority_idx})"
            )

        # Labels + split assignment.
        idx = _build_label_index(hf_dataset_path, split_mode=split_mode, seed=self.seed)
        self.hard_map = idx["hard"]
        self.split_ids = set(idx["splits"][self.split])

        if self.drop_other:
            # Remove Other mass and renormalize the 8 primary emotions.
            base = idx["soft"]
            self.soft_map: Dict[str, np.ndarray] = {}
            for uid in self.split_ids:
                v = np.asarray(base[uid][:8], dtype=np.float64)
                s = float(v.sum())
                self.soft_map[uid] = (v / s if s > 0 else np.full(8, 1.0 / 8)).astype(np.float32)
        else:
            self.soft_map = idx["soft"]

        # Whisper shard files.
        self.shard_files = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shard_files:
            raise FileNotFoundError(f"Δεν βρέθηκαν shard_*.npz στο {self.shard_dir}")

        # RoBERTa features. Prefer one packed .npz to avoid Drive listing/opening thousands of files.
        rp = self.roberta_dir
        if rp.is_dir():
            cand = rp / "roberta_all.npz"
            npz_path = cand if cand.exists() else None
        elif rp.suffix == ".npz" and rp.exists():
            npz_path = rp
        else:
            npz_path = None

        self.roberta_map = None
        if npz_path is not None:
            print(f"[MultimodalShardDataset] Φόρτωση RoBERTa από {npz_path} ...")
            data = np.load(str(npz_path))
            embs = data["embeddings"]
            uids = data["utt_ids"]
            self.roberta_map = {str(u): embs[i] for i, u in enumerate(uids)}
            roberta_available = set(self.roberta_map.keys())
        else:
            if not self.roberta_dir.exists():
                raise FileNotFoundError(
                    "Δεν βρέθηκαν RoBERTa features: ούτε roberta_all.npz "
                    f"ούτε φάκελος {self.roberta_dir}"
                )
            print(
                "[MultimodalShardDataset] ΠΡΟΣΟΧΗ: legacy per-file .npy mode "
                "(αργό/ασταθές σε Drive). Προτιμήστε roberta_all.npz."
            )
            roberta_available = {p.stem for p in self.roberta_dir.glob("*.npy")}

        split_with_roberta = self.split_ids & roberta_available
        missing = len(self.split_ids) - len(split_with_roberta)
        if missing > 0:
            if not self.skip_missing:
                raise FileNotFoundError(f"{missing} utterances δεν έχουν RoBERTa features.")
            print(f"[MultimodalShardDataset] Παραλείπονται {missing} utterances χωρίς RoBERTa features.")
        self.split_ids = split_with_roberta

        # Inverse-frequency sampling for the minority partner in audio mixing.
        self.minority_class_probs = self._build_minority_class_probs()

        print(
            f"[MultimodalShardDataset] split='{self.split}' | "
            f"{len(self.shard_files)} shards | {len(self.split_ids)} utterances (Whisper + RoBERTa)"
        )

    def __len__(self) -> int:
        return len(self.split_ids)

    def _build_minority_class_probs(self) -> Dict[int, float]:
        counts = {int(c): 0 for c in self.minority_idx}
        for uid in self.split_ids:
            h = int(self.hard_map.get(uid, -1))
            if h in counts:
                counts[h] += 1
        inv = {c: (1.0 / n if n > 0 else 0.0) for c, n in counts.items()}
        total = sum(inv.values())
        if total <= 0:
            return {c: 1.0 / max(len(counts), 1) for c in counts}
        probs = {c: v / total for c, v in inv.items()}
        if self.audio_mixing:
            print(f"[audio_mixing] minority counts: {counts}")
            print(f"[audio_mixing] inverse-frequency probs: { {k: round(v, 4) for k, v in probs.items()} }")
        return probs

    def _load_roberta(self, utt_id: str) -> Tensor:
        if self.roberta_map is not None:
            arr = self.roberta_map[utt_id]
        else:
            arr = np.load(str(self.roberta_dir / f"{utt_id}.npy"))
        arr = np.asarray(arr).reshape(-1)
        return torch.from_numpy(arr.copy())

    def _iter_shards(self) -> List[Path]:
        shards = list(self.shard_files)
        info = get_worker_info()
        if info is not None and info.num_workers > 1:
            shards = shards[info.id :: info.num_workers]
        if self.shuffle:
            random.shuffle(shards)
        return shards

    def _yield_buffer(self, buffer: List[Dict]):
        if not buffer:
            return
        if self.shuffle:
            random.shuffle(buffer)
        if self.audio_mixing:
            buffer = apply_feature_audio_mixing(
                buffer,
                rng=self.aug_rng,
                mix_prob=self.audio_mix_prob,
                majority_idx=self.majority_idx,
                minority_idx=self.minority_idx,
                minority_class_probs=self.minority_class_probs,
            )
            if self.shuffle:
                random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __iter__(self):
        shards = self._iter_shards()
        buffer: List[Dict] = []

        for sf in shards:
            data = np.load(sf)
            feats = data["feats"]       # [K, 750, 1280] float16
            uids = data["utt_ids"]
            lens = data["lengths"]

            order = list(range(len(uids)))
            if self.shuffle:
                random.shuffle(order)

            for i in order:
                uid = str(uids[i])
                if uid not in self.split_ids:
                    continue

                try:
                    roberta = self._load_roberta(uid)
                except Exception:
                    continue

                if self.annotation_dropout:
                    soft = apply_annotation_dropout(
                        self.soft_map[uid],
                        self.aug_rng,
                        n_annotators=self.n_annotators,
                        drop_rate=self.drop_rate,
                        majority_idx=self.majority_idx,
                    )
                else:
                    soft = self.soft_map[uid].copy()
                soft = np.asarray(soft, dtype=np.float32)

                item = {
                    "whisper": torch.from_numpy(feats[i].copy()),
                    "roberta": roberta,
                    "length": int(lens[i]),
                    "soft_label": torch.from_numpy(soft),
                    "hard_label": int(self.hard_map[uid]),
                    "utt_id": uid,
                }
                buffer.append(item)

                if len(buffer) >= self.buffer_size:
                    yield from self._yield_buffer(buffer)
                    buffer = []

            del data, feats

        yield from self._yield_buffer(buffer)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {
            "whisper": torch.stack([b["whisper"] for b in batch]).float(),
            "whisper_lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
            "roberta": torch.stack([b["roberta"] for b in batch]).float(),
            "soft_labels": torch.stack([b["soft_label"] for b in batch]).float(),
            "hard_labels": torch.tensor([b["hard_label"] for b in batch], dtype=torch.long),
            "utt_ids": [b["utt_id"] for b in batch],
        }
