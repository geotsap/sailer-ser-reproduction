"""
src/augmentation.py

Online audio augmentation for Speech Emotion Recognition training.

Two augmentation strategies (both from the SAILER paper):
    1. Utterance MixUp  — linearly blend two waveforms and their soft labels.
       By default applied only to minority-class samples to combat class
       imbalance, but configurable to apply to all classes.

    2. Noise Mixing     — add background noise at a random SNR.
       Supports two noise modes:
           - "synthetic" : white / pink noise generated on-the-fly (no extra data needed)
           - "musan"     : real noise clips loaded from a MUSAN noise directory
           - "both"      : randomly picks between synthetic and MUSAN each call

Usage:
    from src.dataset import MspPodcastDataset
    from src.augmentation import AudioMixer
    from torch.utils.data import DataLoader

    dataset = MspPodcastDataset("data/manifests/msp_podcast/train.csv")
    mixer   = AudioMixer(
        dataset,
        mixup_alpha=0.4,
        mixup_prob=0.5,
        mixup_minority_only=True,
        noise_prob=0.5,
        noise_mode="synthetic",        # or "musan" or "both"
        noise_dir=None,                # path to MUSAN noise/ folder if mode needs it
        snr_db_range=(5.0, 20.0),
    )

    # Wrap your DataLoader __getitem__ call:
    sample  = dataset[idx]
    sample  = mixer(sample)            # apply augmentation
    # ... then pass to collate_fn as usual
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms(waveform: Tensor) -> Tensor:
    """Root-mean-square energy of a 1-D waveform tensor."""
    return waveform.pow(2).mean().sqrt().clamp_min(1e-9)


def _mix_at_snr(signal: Tensor, noise: Tensor, snr_db: float) -> Tensor:
    """Add noise to signal at the requested SNR (dB).

    The noise is scaled so that:
        20 * log10(RMS_signal / RMS_noise_scaled) = snr_db
    """
    if noise.shape[0] == 0:
        return signal

    # Tile or crop noise to match signal length
    if noise.shape[0] < signal.shape[0]:
        repeats = math.ceil(signal.shape[0] / noise.shape[0])
        noise = noise.repeat(repeats)
    noise = noise[: signal.shape[0]]

    snr_linear = 10 ** (snr_db / 20.0)
    noise_scaled = noise * (_rms(signal) / (_rms(noise) * snr_linear))
    return (signal + noise_scaled).clamp(-1.0, 1.0)


def _white_noise(length: int) -> Tensor:
    """Unit-variance white noise."""
    return torch.randn(length)


def _pink_noise(length: int) -> Tensor:
    """Approximate pink noise via 1/f shaping in the frequency domain."""
    fft = torch.fft.rfft(torch.randn(length))
    freqs = torch.arange(1, fft.shape[0] + 1, dtype=torch.float32)
    fft = fft / freqs.sqrt()
    pink = torch.fft.irfft(fft, n=length)
    # Normalise to unit variance
    std = pink.std().clamp_min(1e-9)
    return pink / std


# ---------------------------------------------------------------------------
# AudioMixer
# ---------------------------------------------------------------------------

class AudioMixer:
    """Applies online MixUp and/or noise augmentation to a single dataset sample.

    Parameters
    ----------
    dataset : MspPodcastDataset
        The training dataset.  The mixer keeps a reference so it can sample
        a second utterance for MixUp without loading the whole dataset again.
    mixup_alpha : float
        Beta distribution concentration parameter.  Higher → mixes closer to
        0.5/0.5; lower → one utterance dominates.  SAILER uses ~0.4.
    mixup_prob : float
        Probability of applying MixUp to any given sample.
    mixup_minority_only : bool
        If True (default, faithful to SAILER), MixUp is only applied when the
        primary sample belongs to a minority class.  Set False to apply to all.
    noise_prob : float
        Probability of adding background noise to any given sample.
    noise_mode : str
        One of "synthetic", "musan", or "both".
    noise_dir : str | Path | None
        Path to the MUSAN noise/ directory (required when noise_mode is
        "musan" or "both").
    snr_db_range : tuple[float, float]
        (min_snr, max_snr) in dB.  SNR is sampled uniformly in this range.
    sample_rate : int
        Expected sample rate of all waveforms (must match dataset).
    """

    def __init__(
        self,
        dataset,                              # MspPodcastDataset (avoid circular import)
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        mixup_minority_only: bool = True,
        noise_prob: float = 0.5,
        noise_mode: str = "synthetic",        # "synthetic" | "musan" | "both"
        noise_dir: Optional[str | Path] = None,
        snr_db_range: Tuple[float, float] = (5.0, 20.0),
        sample_rate: int = 16_000,
    ) -> None:
        self.dataset             = dataset
        self.mixup_alpha         = mixup_alpha
        self.mixup_prob          = mixup_prob
        self.mixup_minority_only = mixup_minority_only
        self.noise_prob          = noise_prob
        self.noise_mode          = noise_mode
        self.snr_db_range        = snr_db_range
        self.sample_rate         = sample_rate

        # Pre-build index lists used for MixUp partner sampling
        self._minority_indices: List[int] = dataset.get_minority_indices()
        self._all_indices: List[int]       = list(range(len(dataset)))

        # MUSAN noise file list (lazy — only populated when needed)
        self._musan_files: Optional[List[Path]] = None
        if noise_mode in ("musan", "both"):
            if noise_dir is None:
                raise ValueError(
                    "noise_dir must be provided when noise_mode is 'musan' or 'both'."
                )
            self._musan_files = self._index_musan(Path(noise_dir))
            if not self._musan_files:
                raise FileNotFoundError(
                    f"No .wav files found under noise_dir: {noise_dir}"
                )
            print(f"[AudioMixer] Found {len(self._musan_files)} MUSAN noise files.")

        print(
            f"[AudioMixer] mixup_prob={mixup_prob} | minority_only={mixup_minority_only} "
            f"| noise_prob={noise_prob} | noise_mode={noise_mode} "
            f"| SNR={snr_db_range[0]}–{snr_db_range[1]} dB"
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def __call__(self, sample: Dict) -> Dict:
        """Apply augmentation to a single sample dict from MspPodcastDataset.

        The sample dict is modified in-place and returned.
        Keys modified: "waveform", "soft_label".
        A new key "augmented" (bool) is added for debugging / logging.
        """
        augmented = False

        # 1. Utterance MixUp
        if self._should_mixup(sample):
            sample = self._apply_mixup(sample)
            augmented = True

        # 2. Noise mixing
        if random.random() < self.noise_prob:
            sample["waveform"] = self._apply_noise(sample["waveform"])
            augmented = True

        sample["augmented"] = augmented
        return sample

    # ── MixUp ────────────────────────────────────────────────────────────────

    def _should_mixup(self, sample: Dict) -> bool:
        if random.random() >= self.mixup_prob:
            return False
        if self.mixup_minority_only:
            # Only proceed if this sample is a minority-class utterance
            from src.dataset import MINORITY_INDICES
            return int(sample["hard_label"]) in MINORITY_INDICES
        return True

    def _apply_mixup(self, sample: Dict) -> Dict:
        """Sample a second utterance and blend waveforms + soft labels."""
        lam = self._sample_lambda()

        # Pick a random partner from the appropriate pool
        pool = self._minority_indices if self.mixup_minority_only else self._all_indices
        partner_idx = random.choice(pool)
        partner     = self.dataset[partner_idx]

        wave_a: Tensor = sample["waveform"]
        wave_b: Tensor = partner["waveform"]

        # Match lengths by padding the shorter one
        wave_a, wave_b = self._match_lengths(wave_a, wave_b)

        # Blend waveforms and soft labels
        mixed_wave  = lam * wave_a + (1.0 - lam) * wave_b
        mixed_label = lam * sample["soft_label"] + (1.0 - lam) * partner["soft_label"]

        sample["waveform"]   = mixed_wave.clamp(-1.0, 1.0)
        sample["soft_label"] = mixed_label
        # hard_label: keep the dominant class after mixing
        sample["hard_label"] = int(mixed_label.argmax().item())
        return sample

    def _sample_lambda(self) -> float:
        """Sample λ from Beta(α, α), biased toward the primary utterance (λ ≥ 0.5)."""
        if self.mixup_alpha <= 0:
            return 1.0
        lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
        return max(lam, 1.0 - lam)   # ensure primary utterance dominates

    @staticmethod
    def _match_lengths(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        """Zero-pad the shorter waveform so both have the same length."""
        la, lb = a.shape[0], b.shape[0]
        if la == lb:
            return a, b
        if la < lb:
            a = torch.nn.functional.pad(a, (0, lb - la))
        else:
            b = torch.nn.functional.pad(b, (0, la - lb))
        return a, b

    # ── Noise mixing ─────────────────────────────────────────────────────────

    def _apply_noise(self, waveform: Tensor) -> Tensor:
        snr_db = random.uniform(*self.snr_db_range)
        noise  = self._sample_noise(len(waveform))
        return _mix_at_snr(waveform, noise, snr_db)

    def _sample_noise(self, length: int) -> Tensor:
        """Return a noise tensor of the requested length."""
        mode = self.noise_mode
        if mode == "both":
            mode = random.choice(["synthetic", "musan"])

        if mode == "synthetic":
            return self._synthetic_noise(length)
        elif mode == "musan":
            return self._load_musan_noise(length)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}")

    def _synthetic_noise(self, length: int) -> Tensor:
        """Randomly pick white or pink noise."""
        if random.random() < 0.5:
            return _white_noise(length)
        return _pink_noise(length)

    def _load_musan_noise(self, length: int) -> Tensor:
        """Load a random MUSAN noise clip, resampling if necessary."""
        assert self._musan_files is not None
        path = random.choice(self._musan_files)
        noise, sr = torchaudio.load(str(path))
        noise = noise.mean(dim=0)                         # mono [T]
        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(
                noise.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)
        return noise

    # ── MUSAN indexing ───────────────────────────────────────────────────────

    @staticmethod
    def _index_musan(noise_dir: Path) -> List[Path]:
        """Recursively find all .wav files under noise_dir."""
        return sorted(noise_dir.rglob("*.wav"))
