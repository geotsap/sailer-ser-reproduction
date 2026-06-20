"""
src/augmentations_features.py

Data augmentation για το feature-based pipeline (SAILER §2.3 / Πίνακας 3).

Επειδή δουλεύουμε με pre-extracted features (όχι raw audio), οι τεχνικές
υλοποιούνται στον χώρο που έχουμε διαθέσιμο:

ANNOTATION DROPOUT (SAILER §2.3)
    Το paper πετάει τυχαία 20% των annotations (ψήφων) κάθε δείγματος, και
    ΜΟΝΟ από τις κυρίαρχες κλάσεις, ώστε να (α) εισάγει στοχαστικότητα στο soft
    label ανά epoch και (β) να ανεβάζει σχετικά τη μάζα των μειονοτήτων.

    ΠΡΟΣΟΧΗ (απόκλιση από το paper): το dataset μας δεν κρατά τους αρχικούς
    ακέραιους ψήφους — μόνο την κανονικοποιημένη κατανομή (sum=1, με ένα μικρό
    label-smoothing πάτωμα). Οπότε ανακατασκευάζουμε "pseudo-counts":
        counts ≈ round(soft * N),   N = υποτιθέμενος αριθμός annotators (paper: ≥5)
    πετάμε ~drop_rate των ψευδο-ψήφων από τις κυρίαρχες κλάσεις, και
    ξανακανονικοποιούμε σε άθροισμα 1. Ίδια λογική με το paper, προσεγγιστική
    υλοποίηση λόγω της μορφής των δεδομένων.
"""

from __future__ import annotations

import numpy as np

# Σειρά κλάσεων (PRIMARY_LABELS):
#   0 Angry | 1 Sad | 2 Happy | 3 Surprise | 4 Fear | 5 Disgust | 6 Contempt
#   7 Neutral | 8 Other
# Κυρίαρχες κλάσεις κατά SAILER: Neutral, Happy, Sad, Angry
MAJORITY_IDX = (0, 1, 2, 7)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / len(v))
    return v / s


def annotation_dropout(
    soft: np.ndarray,
    rng: np.random.Generator,
    n_annotators: int = 5,
    drop_rate: float = 0.2,
    majority_idx: tuple = MAJORITY_IDX,
) -> np.ndarray:
    """
    Επιστρέφει ΝΕΟ soft label [K] (sum=1) αφού "πετάξει" ~drop_rate των
    (pseudo-)ψήφων από τις κυρίαρχες κλάσεις.

    Δεν τροποποιεί το input. Ντετερμινιστικό δοθέντος του rng.
    """
    soft = np.asarray(soft, dtype=np.float64)

    # 1) pseudo-counts (το label-smoothing πάτωμα ~0.003 -> 0 μετά το round)
    counts = np.rint(soft * n_annotators).astype(int)
    counts = np.clip(counts, 0, None)
    total = int(counts.sum())
    if total <= 0:
        return _normalize(soft)

    # 2) πόσους ψήφους πετάμε
    n_drop = int(round(drop_rate * total))
    if n_drop <= 0:
        return _normalize(counts.astype(float))

    # 3) pool με τους ψήφους ΜΟΝΟ των κυρίαρχων κλάσεων
    pool = []
    for k in majority_idx:
        pool.extend([k] * int(counts[k]))
    if not pool:
        # καθαρά μειονοτικό δείγμα -> δεν πετάμε τίποτα
        return _normalize(counts.astype(float))

    # 4) πέτα n_drop ψήφους (τυχαία, χωρίς επανατοποθέτηση)
    n_drop = min(n_drop, len(pool))
    dropped = rng.choice(np.array(pool), size=n_drop, replace=False)
    for k in dropped:
        counts[int(k)] -= 1
    counts = np.clip(counts, 0, None)

    if counts.sum() <= 0:
        return _normalize(soft)          # ασφάλεια
    return _normalize(counts.astype(float))
