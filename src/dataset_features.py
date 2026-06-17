"""
src/dataset_features.py

Shard-streaming Dataset για τα pre-extracted Whisper features.

ΓΙΑΤΙ STREAMING;
    Τα features είναι ~287 GB αποθηκευμένα σε ~150 shard αρχεία (.npz),
    το καθένα με ~1000 utterances. Δεν χωράνε στον τοπικό δίσκο του Colab,
    και δεν μπορούμε να διαβάζουμε τυχαίες γραμμές από κάθε shard (θα έπρεπε
    να φορτώνουμε ολόκληρο το shard κάθε φορά). Οπότε χρησιμοποιούμε
    IterableDataset: διαβάζουμε τα shards ΣΕΙΡΙΑΚΑ, ένα-ένα ολόκληρο στη RAM,
    και ανακατεύουμε (α) τη σειρά των shards, (β) τις γραμμές μέσα στο shard,
    (γ) ένα buffer για ανάμειξη μεταξύ διαφορετικών shards.

ΜΟΡΦΗ SHARD (.npz):
    feats   : float16  [K, 750, 1280]   ← Whisper-large-v3 last hidden state, 15s
    utt_ids : <U...     [K]              ← το stem κάθε αρχείου (π.χ. MSP-PODCAST_0001_0008)
    lengths : int32     [K]              ← πραγματικό μήκος σε frames (για σωστό masking)

ΔΟΜΗ ΦΑΚΕΛΩΝ:
    SLP/
    ├── features/
    │   └── whisper_shards/      ← shard_0000.npz, shard_0001.npz, ...
    └── msp_podcast_hf/          ← το HuggingFace dataset (για τα labels)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


# ---------------------------------------------------------------------------
# Constants  (ίδια με πριν — μην τα αλλάξεις, οι δείκτες πρέπει να ταιριάζουν)
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

# Πόσα frames κρατάμε (15s → 750 frames). Πρέπει να ταιριάζει με το extraction.
KEEP_FRAMES: int = 750


# ---------------------------------------------------------------------------
# Label index — υπολογίζεται ΜΙΑ φορά και μοιράζεται μεταξύ των splits
# ---------------------------------------------------------------------------

# Module-level cache ώστε το train_ds και το val_ds να μην ξαναφορτώνουν
# το HF dataset δύο φορές.
_LABEL_CACHE: Dict[str, dict] = {}


def _build_label_index(
    hf_dataset_path: str | Path,
    split_mode: str = "podcast",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Φορτώνει το HF dataset (χωρίς audio), υπολογίζει soft & hard labels για
    κάθε utterance, και κάνει 80/10/10 split.

    split_mode:
        "podcast" → ομαδοποιημένο ανά podcast id (το ΙΔΙΟ podcast δεν εμφανίζεται
                    σε δύο splits). Αποφεύγει speaker leakage — αξιόπιστο νούμερο.
        "random"  → καθαρά τυχαίο 80/10/10 (έχει leakage, αισιόδοξο — μόνο για
                    σύγκριση/αναφορά).

    Επιστρέφει dict με:
        soft   : {utt_id -> np.ndarray[9] float32}
        hard   : {utt_id -> int}
        splits : {"train"/"validation"/"test" -> set(utt_id)}
    """
    # Το cache key περιλαμβάνει mode+seed ώστε podcast/random να μη μπερδεύονται
    key = f"{hf_dataset_path}::{split_mode}::{seed}"
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]

    print(f"[labels] Φόρτωση labels από {hf_dataset_path} ...")
    ds = load_from_disk(str(hf_dataset_path))
    if hasattr(ds, "keys"):
        ds = ds["train"]
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])

    # Κρατάμε μόνο τις στήλες που χρειαζόμαστε -> γρήγορο to_pandas
    needed = ["file", "major_emotion"] + PRIMARY_COLS + \
             [c for c in OTHER_COLS if c in ds.column_names]
    needed = [c for c in needed if c in ds.column_names]
    df = ds.select_columns(needed).to_pandas()

    total = len(df)

    # ── soft labels (vectorized) ─────────────────────────────────────────────
    primary = df[PRIMARY_COLS].fillna(0.0).to_numpy(dtype=np.float32)        # [N, 8]
    other_present = [c for c in OTHER_COLS if c in df.columns]
    if other_present:
        other = df[other_present].fillna(0.0).to_numpy(dtype=np.float32).sum(axis=1, keepdims=True)
    else:
        other = np.zeros((total, 1), dtype=np.float32)
    soft = np.concatenate([primary, other], axis=1)                          # [N, 9]
    row_sum = soft.sum(axis=1, keepdims=True)
    soft = np.where(row_sum > 0, soft / np.clip(row_sum, 1e-8, None),
                    np.full_like(soft, 1.0 / len(PRIMARY_LABELS)))

    # ── hard labels ──────────────────────────────────────────────────────────
    hard = (
        df["major_emotion"].fillna("").str.strip().str.lower()
        .map(MAJOR_EMOTION_TO_IDX).fillna(8).astype(int).to_numpy()
    )

    # ── utt_ids ──────────────────────────────────────────────────────────────
    utt_ids = df["file"].map(lambda f: Path(str(f)).stem).to_numpy()

    soft_map = {uid: soft[i] for i, uid in enumerate(utt_ids)}
    hard_map = {uid: int(hard[i]) for i, uid in enumerate(utt_ids)}

    # ── 80/10/10 split ───────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=seed)

    if split_mode == "random":
        # Καθαρά τυχαίο (έχει speaker leakage — μόνο για σύγκριση)
        indices = list(range(total))
        rng.shuffle(indices)
        n_train = int(total * train_ratio)
        n_val   = int(total * val_ratio)
        train_ids = set(utt_ids[i] for i in indices[:n_train])
        val_ids   = set(utt_ids[i] for i in indices[n_train : n_train + n_val])
        test_ids  = set(utt_ids[i] for i in indices[n_train + n_val :])

    elif split_mode == "podcast":
        # Ομαδοποίηση ανά podcast id (το μεσαίο πεδίο του ονόματος:
        # MSP-PODCAST_<podcast>_<segment>). Ολόκληρο το podcast πάει σε ΕΝΑ split,
        # ώστε ίδιοι ομιλητές να μην μοιράζονται μεταξύ train/val/test.
        from collections import defaultdict
        groups: Dict[str, List[str]] = defaultdict(list)
        for uid in utt_ids:
            parts = uid.split("_")
            pid = parts[-2] if len(parts) >= 2 else uid   # podcast id
            groups[pid].append(uid)

        pids = list(groups.keys())
        rng.shuffle(pids)

        n_train = total * train_ratio
        n_val   = total * val_ratio
        train_ids, val_ids, test_ids = set(), set(), set()
        seen = 0
        for pid in pids:
            members = groups[pid]
            if seen < n_train:                       # γέμισε πρώτα το train
                train_ids.update(members)
            elif seen < n_train + n_val:             # μετά το val
                val_ids.update(members)
            else:                                    # ό,τι μένει -> test
                test_ids.update(members)
            seen += len(members)
        print(f"[labels] split_mode='podcast' | {len(pids)} μοναδικά podcasts")

    else:
        raise ValueError(f"Άγνωστο split_mode: {split_mode!r} (podcast/random)")

    splits = {"train": train_ids, "validation": val_ids, "test": test_ids}
    print(f"[labels] total={total} | train={len(splits['train'])} "
          f"| val={len(splits['validation'])} | test={len(splits['test'])}")

    out = {"soft": soft_map, "hard": hard_map, "splits": splits}
    _LABEL_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------------

class ShardFeatureDataset(IterableDataset):
    """
    Streaming dataset πάνω από τα shard αρχεία.

    Σε κάθε epoch:
        - (αν shuffle) ανακατεύει τη σειρά των shards
        - για κάθε shard: το φορτώνει ολόκληρο, κρατάει μόνο τις γραμμές που
          ανήκουν στο ζητούμενο split, (αν shuffle) ανακατεύει, και τις βάζει
          σε ένα buffer
        - όταν το buffer γεμίσει, το ανακατεύει και το αδειάζει (ανάμειξη
          μεταξύ shards)
    """

    def __init__(
        self,
        hf_dataset_path: str | Path,
        shard_dir: str | Path,
        split: str = "train",
        split_mode: str = "podcast",
        shuffle: Optional[bool] = None,
        buffer_size: int = 2000,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.shard_dir   = Path(shard_dir)
        self.split       = "validation" if split in ("val", "dev") else split
        self.shuffle     = (self.split == "train") if shuffle is None else shuffle
        self.buffer_size = buffer_size
        self.seed        = seed
        self.emotion_cols = PRIMARY_LABELS

        # Labels + split assignment (cached)
        idx = _build_label_index(hf_dataset_path, split_mode=split_mode, seed=seed)
        self.soft_map  = idx["soft"]
        self.hard_map  = idx["hard"]
        self.split_ids = idx["splits"][self.split]

        # Λίστα shard αρχείων (τα ανοίγουμε σειριακά -> φιλικό προς το Drive)
        self.shard_files = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shard_files:
            raise FileNotFoundError(
                f"Δεν βρέθηκαν shard_*.npz στο {self.shard_dir}"
            )
        print(f"[ShardFeatureDataset] split='{self.split}' | "
              f"{len(self.shard_files)} shards | {len(self.split_ids)} utterances")

    # Το __len__ βοηθάει το tqdm να ξέρει πόσα samples υπάρχουν
    def __len__(self) -> int:
        return len(self.split_ids)

    def _iter_shards(self) -> List[Path]:
        """Επιστρέφει τα shards για ΤΟΝ ΤΡΕΧΟΝ worker, σε σωστή σειρά.

        ΣΗΜΑΝΤΙΚΟ: ο διαμοιρασμός στους workers γίνεται ΠΡΩΤΑ (ντετερμινιστικά
        πάνω στη sorted λίστα) ώστε κάθε worker να πάρει ΞΕΧΩΡΙΣΤΑ shards — αλλιώς
        με num_workers>1 κάποια shards θα διαβάζονταν διπλά και άλλα καθόλου.
        Το shuffle γίνεται ΜΕΤΑ, με το global RNG (seeded από set_seed), οπότε η
        σειρά αλλάζει ανά epoch αλλά παραμένει reproducible."""
        # 1) ντετερμινιστικός διαμοιρασμός στους workers
        shards = list(self.shard_files)
        info = get_worker_info()
        if info is not None and info.num_workers > 1:
            shards = shards[info.id :: info.num_workers]

        # 2) shuffle σειράς shards (διαφορετική ανά epoch)
        if self.shuffle:
            random.shuffle(shards)
        return shards

    def __iter__(self):
        shards = self._iter_shards()
        buffer: List[Dict] = []

        for sf in shards:
            data = np.load(sf)                  # σειριακή ανάγνωση ολόκληρου shard
            feats = data["feats"]               # [K, 750, 1280] float16
            uids  = data["utt_ids"]
            lens  = data["lengths"]

            order = list(range(len(uids)))
            if self.shuffle:
                random.shuffle(order)

            for i in order:
                uid = str(uids[i])
                if uid not in self.split_ids:
                    continue
                item = {
                    # .copy() ΣΗΜΑΝΤΙΚΟ: αλλιώς κρατάμε view σε όλο το shard (1.9GB)
                    "whisper":    torch.from_numpy(feats[i].copy()),   # fp16
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

            del data, feats  # ελευθέρωσε τη RAM πριν το επόμενο shard

        # flush ό,τι έμεινε
        if self.shuffle:
            random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    # ── Collate ──────────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        # Όλα τα features είναι ήδη [750, 1280] -> απλό stack, χωρίς padding.
        # Μετατροπή σε float32 εδώ (τα conv weights είναι float32).
        whisper = torch.stack([b["whisper"] for b in batch]).float()        # [B,750,1280]
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        soft    = torch.stack([b["soft_label"] for b in batch]).float()      # [B,9]
        hard    = torch.tensor([b["hard_label"] for b in batch], dtype=torch.long)

        return {
            "whisper":         whisper,
            "whisper_lengths": lengths,
            "soft_labels":     soft,
            "hard_labels":     hard,
            "utt_ids":         [b["utt_id"] for b in batch],
        }
