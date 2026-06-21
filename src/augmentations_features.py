"""
Feature-space data augmentation για SER (SAILER §4.2 / Πίνακας 3).

Υλοποιεί το **annotation dropout**. Επειδή το HF dataset αποθηκεύει τα labels
ως ΚΑΝΟΝΙΚΟΠΟΙΗΜΕΝΕΣ κατανομές (όχι ακέραιους ψήφους), αναπαριστούμε τους
ψήφους με "pseudo-counts": c = d * N_annotators.

Annotation dropout (πιστό στο paper):
  1. pseudo-counts c = soft * N            (sum = N)
  2. αφαιρούμε round(drop_rate * N) ψευδο-ψήφους, ΜΟΝΟ από κυρίαρχες κλάσεις
     (Angry/Sad/Happy/Neutral), στοχαστικά ανάλογα με τη μάζα τους
  3. ξανακανονικοποιούμε σε άθροισμα 1

Εφαρμόζεται ΜΟΝΟ στο training.
"""
from __future__ import annotations
import numpy as np

# PRIMARY_LABELS = ['Angry','Sad','Happy','Surprise','Fear','Disgust',
#                   'Contempt','Neutral','Other']
# Κυρίαρχες κλάσεις: Angry(0), Sad(1), Happy(2), Neutral(7) + Other(8).
# ΣΗΜΕΙΩΣΗ: το paper ορίζει μόνο Angry/Sad/Happy/Neutral, αλλά στα ΔΙΚΑ ΜΑΣ
# δεδομένα το 'Other' είναι ~37% (de-facto κυρίαρχο), οπότε το συμπεριλαμβάνουμε
# ώστε το dropout να μειώνει ΚΑΙ αυτό και να ενισχύει τις πραγματικές μειονότητες.
MAJORITY_IDX = (0, 1, 2, 7, 8)


def annotation_dropout(soft, rng, n_annotators=5, drop_rate=0.2,
                       majority_idx=MAJORITY_IDX):
    """Νέο soft label [C] (float32, sum=1) μετά annotation dropout. Δεν αλλάζει το input."""
    d = np.asarray(soft, dtype=np.float64).copy()
    s = d.sum()
    if s <= 0:
        return np.asarray(soft, dtype=np.float32)
    d /= s

    c = d * n_annotators                       # pseudo-counts, sum = N
    n_drop = int(round(drop_rate * n_annotators))
    maj = list(majority_idx)

    for _ in range(n_drop):
        avail = np.array([max(c[i], 0.0) for i in maj], dtype=np.float64)
        if avail.sum() <= 0:                   # καμία κυρίαρχη μάζα -> stop
            break
        probs = avail / avail.sum()
        j = maj[int(rng.choice(len(maj), p=probs))]
        c[j] = max(c[j] - 1.0, 0.0)            # αφαίρεσε 1 ψευδο-ψήφο

    total = c.sum()
    if total <= 0:
        return d.astype(np.float32)
    return (c / total).astype(np.float32)
