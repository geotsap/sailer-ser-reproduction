"""
src/augmentations_features.py

Feature-space data augmentation for SAILER-style SER experiments.

Implemented augmentations:
1. Annotation dropout on soft labels.
2. Feature-space audio mixing for pre-extracted Whisper features.

Important:
- Annotation dropout changes only the target distribution.
- Feature-space audio mixing changes only the audio/Whisper representation and
  the target distribution. RoBERTa/text embeddings should remain unchanged in
  the multimodal dataset, because this augmentation is the feature-space
  analogue of audio-only mixing.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

# 8-class order: ['Angry','Sad','Happy','Surprise','Fear','Disgust','Contempt','Neutral']
# SAILER majority classes: neutral, happy, sad, angry.
MAJORITY_IDX_8 = (0, 1, 2, 7)

# 9-class order: 8 primary + Other. For annotation dropout in this project we
# previously also allowed Other to be treated as de-facto majority because it is
# very frequent in the public HF-derived labels. For audio mixing, use the
# paper-faithful 8-class setting whenever --drop_other is active.
MAJORITY_IDX_9 = (0, 1, 2, 7, 8)


def annotation_dropout(
    soft: np.ndarray,
    rng: np.random.Generator,
    n_annotators: int = 5,
    drop_rate: float = 0.2,
    majority_idx: Sequence[int] = MAJORITY_IDX_9,
) -> np.ndarray:
    """
    Apply annotation dropout to one soft-label distribution.

    This follows the SAILER idea using pseudo-counts, because our stored labels
    are normalized distributions rather than integer annotation counts.

    Steps:
    1. Convert distribution d to pseudo-counts c = d * n_annotators.
    2. Remove round(drop_rate * n_annotators) pseudo-votes from majority
       classes, sampled proportionally to their current mass.
    3. Re-normalize to sum 1.

    Parameters
    ----------
    soft:
        Soft label distribution [C].
    rng:
        NumPy random generator.
    n_annotators:
        Pseudo number of annotators.
    drop_rate:
        Fraction of pseudo-votes to drop.
    majority_idx:
        Class indices considered majority.

    Returns
    -------
    np.ndarray float32 [C], sum approximately 1.
    """
    d = np.asarray(soft, dtype=np.float64).copy()
    s = float(d.sum())
    if s <= 0:
        return np.asarray(soft, dtype=np.float32)
    d /= s

    c = d * float(n_annotators)
    n_drop = int(round(drop_rate * n_annotators))
    maj = [int(i) for i in majority_idx if int(i) < len(c)]

    for _ in range(n_drop):
        avail = np.array([max(c[i], 0.0) for i in maj], dtype=np.float64)
        if avail.sum() <= 0:
            break
        probs = avail / avail.sum()
        j = maj[int(rng.choice(len(maj), p=probs))]
        c[j] = max(c[j] - 1.0, 0.0)

    total = float(c.sum())
    if total <= 0:
        return d.astype(np.float32)
    return (c / total).astype(np.float32)


def _normalize_probs(weights: np.ndarray) -> Optional[np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return None
    return weights / total


def apply_feature_audio_mixing(
    buffer: List[Dict],
    rng: np.random.Generator,
    mix_prob: float = 0.5,
    majority_idx: Sequence[int] = MAJORITY_IDX_8,
    minority_idx: Sequence[int] = (3, 4, 5, 6),
    minority_class_probs: Optional[Dict[int, float]] = None,
) -> List[Dict]:
    """
    Apply SAILER-style audio mixing in Whisper feature space.

    Paper-faithful intent:
    - For each majority-class sample, with probability p_a, sample one minority
      sample using inverse-frequency sampling.
    - Mix the audio signal and average the two distributions.

    Our adaptation for pre-extracted Whisper features:
    - Mix only the Whisper hidden states, not the RoBERTa/text embedding.
    - Use 50/50 averaging, matching the paper's target rule:
      d_mix = (d_majority + d_minority) / 2.
    - Mix only up to the common valid length L = min(length_majority, length_minority).
      The returned sample length becomes L, so masked pooling ignores the rest.

    This returns the same number of samples as the input buffer: majority items
    are stochastically replaced by mixed versions; minority items remain as-is.
    """
    if not buffer or mix_prob <= 0:
        return buffer

    majority = {int(i) for i in majority_idx}
    minority = {int(i) for i in minority_idx}

    minority_positions = [
        pos for pos, item in enumerate(buffer)
        if int(item.get("hard_label", -1)) in minority
    ]
    if not minority_positions:
        return buffer

    if minority_class_probs is not None:
        weights = np.array(
            [minority_class_probs.get(int(buffer[pos]["hard_label"]), 0.0) for pos in minority_positions],
            dtype=np.float64,
        )
        partner_probs = _normalize_probs(weights)
    else:
        partner_probs = None

    out: List[Dict] = []
    for item in buffer:
        hard = int(item.get("hard_label", -1))
        should_mix = hard in majority and rng.random() < mix_prob
        if not should_mix:
            out.append(item)
            continue

        partner_pos = int(rng.choice(len(minority_positions), p=partner_probs))
        partner = buffer[minority_positions[partner_pos]]

        len_a = int(item["length"])
        len_b = int(partner["length"])
        L = max(1, min(len_a, len_b))

        # Keep original shape [750, 1280]. Only the first L valid frames are mixed.
        wa = item["whisper"]
        wb = partner["whisper"]
        mixed_w = wa.clone()
        mixed_w[:L] = (0.5 * wa[:L].float() + 0.5 * wb[:L].float()).to(dtype=wa.dtype)

        sa = item["soft_label"].float()
        sb = partner["soft_label"].float()
        mixed_soft = 0.5 * sa + 0.5 * sb
        mixed_soft = mixed_soft / mixed_soft.sum().clamp_min(1e-8)

        mixed_item = dict(item)
        mixed_item["whisper"] = mixed_w
        mixed_item["length"] = L
        mixed_item["soft_label"] = mixed_soft
        mixed_item["hard_label"] = int(torch.argmax(mixed_soft).item())
        mixed_item["utt_id"] = f"{item.get('utt_id', 'maj')}+mix+{partner.get('utt_id', 'min')}"

        # Deliberately leave mixed_item['roberta'] unchanged: audio mixing only.
        out.append(mixed_item)

    return out
