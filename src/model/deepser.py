"""
src/model/deepser.py

DeepSER: Deep hierarchical fusion architecture for multimodal SER.

Βασισμένο στο Algorithm 1 & 2 του MEDUSA paper (Chatzichristodoulou et al., 2025).

Inputs:
    whisper: [B, T, 1280]  — Whisper-Large-v3 last hidden state (sequence)
    roberta: [B, 1024]     — RoBERTa-Large pooled embedding (single vector)

Architecture:
    1. Project both modalities to hidden_dim (1024)
    2. RoBERTa επεκτείνεται σε sequence μήκους 1: [B, 1, hidden_dim]
    3. Unimodal encoders (2-layer transformer) → h1,h2,h3 και g1,g2,g3
    4. Fusion stage 1: ENC_f1(h1 || g1) → f1_h1, f1_h2, f1_h3
    5. Fusion stage 2: ENC_f2(h2 || g2 || f1_h2) → f2_h1, f2_h2, f2_h3
    6. Pooled fusion: W(h3 || g3 || f2_h3) → x
    7. Classification head: MLP → logits
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Base Transformer Encoder (Algorithm 1)
# ---------------------------------------------------------------------------

class BaseTransformerEncoder(nn.Module):
    """
    Algorithm 1 από το MEDUSA paper.

    Input:  x [B, L, D]
    Output: [h1, h2, h3] όπου:
        h1 = output of TransformerLayer 1  [B, L, hidden_dim]
        h2 = output of TransformerLayer 2  [B, L, hidden_dim]
        h3 = POOL(h2)                       [B, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        nhead: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # W2(σ(W1(x))) — projection to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2-layer transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.layer1 = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.layer2 = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Parameters
        ----------
        x : [B, L, input_dim]
        src_key_padding_mask : [B, L] bool, True = padding (ignored)

        Returns
        -------
        h1 : [B, L, hidden_dim]
        h2 : [B, L, hidden_dim]
        h3 : [B, hidden_dim]   (mean pooled)
        """
        x = self.input_proj(x)                                    # [B, L, hidden_dim]
        h1 = self.layer1(x, src_key_padding_mask=src_key_padding_mask)   # [B, L, hidden_dim]
        h2 = self.layer2(h1, src_key_padding_mask=src_key_padding_mask)  # [B, L, hidden_dim]

        # Mean pooling (masked)
        if src_key_padding_mask is not None:
            # padding mask: True = ignore → valid = ~mask
            valid = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, L, 1]
            h3 = (h2 * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            h3 = h2.mean(dim=1)                                    # [B, hidden_dim]

        return h1, h2, h3


# ---------------------------------------------------------------------------
# DeepSER Model (Algorithm 2)
# ---------------------------------------------------------------------------

class DeepSERModel(nn.Module):
    """
    DeepSER: Hierarchical deep fusion για Whisper + RoBERTa.

    Υλοποιεί το Algorithm 2 του MEDUSA paper για 2 modalities.

    Parameters
    ----------
    num_emotions : int
        Αριθμός κλάσεων (8 ή 9).
    whisper_dim : int
        Διάσταση Whisper features (1280 για Large-v3).
    roberta_dim : int
        Διάσταση RoBERTa features (1024 για Large).
    hidden_dim : int
        Hidden dimension για όλους τους transformer encoders (1024, όπως paper).
    nhead : int
        Αριθμός attention heads (8 για hidden_dim=1024).
    dropout : float
        Dropout rate.
    classifier_hidden_dim : int
        Hidden dim του classification MLP.
    """

    def __init__(
        self,
        num_emotions: int = 9,
        whisper_dim: int = 1280,
        roberta_dim: int = 1024,
        hidden_dim: int = 1024,
        nhead: int = 8,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.num_emotions = num_emotions
        self.hidden_dim   = hidden_dim

        # ── Unimodal encoders ────────────────────────────────────────────────
        # ENC_h: Whisper encoder
        self.enc_h = BaseTransformerEncoder(
            input_dim=whisper_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )

        # ENC_g: RoBERTa encoder (input_dim=1024, sequence length=1)
        self.enc_g = BaseTransformerEncoder(
            input_dim=roberta_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )

        # ── Fusion encoders ──────────────────────────────────────────────────
        # ENC_f1: 1ο fusion — παίρνει h1 || g1 (concatenation στη dim L)
        # Input dim = hidden_dim (γιατί ενώνουμε sequences, όχι features)
        self.enc_f1 = BaseTransformerEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )

        # ENC_f2: 2ο fusion — παίρνει h2 || g2 || f1_h2
        self.enc_f2 = BaseTransformerEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dropout=dropout,
        )

        # ── Pooled representation fusion ─────────────────────────────────────
        # W(h3 || g3 || f2_h3): 3 * hidden_dim → hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_emotions),
        )

    def forward(
        self,
        whisper: Tensor,
        roberta: Tensor,
        whisper_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        whisper : [B, T, 1280]
            Whisper hidden states sequence.
        roberta : [B, 1024]
            RoBERTa pooled embedding.
        whisper_lengths : [B] int64, optional
            Πραγματικό μήκος κάθε Whisper sequence σε frames.

        Returns
        -------
        logits : [B, num_emotions]
        """
        B, T, _ = whisper.shape

        # Whisper padding mask [B, T]: True = padding position
        if whisper_lengths is not None:
            whisper_mask = torch.arange(T, device=whisper.device).unsqueeze(0) >= \
                           whisper_lengths.unsqueeze(1)  # True = ignore
        else:
            whisper_mask = None

        # RoBERTa: [B, 1024] → [B, 1, 1024] (sequence μήκους 1)
        roberta_seq = roberta.unsqueeze(1)  # [B, 1, 1024]

        # ── Unimodal encoders (Algorithm 1) ──────────────────────────────────
        h1, h2, h3 = self.enc_h(whisper, src_key_padding_mask=whisper_mask)
        # h1, h2: [B, T, hidden_dim], h3: [B, hidden_dim]

        g1, g2, g3 = self.enc_g(roberta_seq, src_key_padding_mask=None)
        # g1, g2: [B, 1, hidden_dim], g3: [B, hidden_dim]

        # ── Fusion stage 1: ENC_f1(h1 || g1) ────────────────────────────────
        # Concatenate sequences along time dim: [B, T+1, hidden_dim]
        f1_input = torch.cat([h1, g1], dim=1)

        # Padding mask για το fusion (T positions από whisper + 1 από roberta)
        if whisper_mask is not None:
            roberta_valid = torch.zeros(B, 1, dtype=torch.bool, device=whisper.device)
            f1_mask = torch.cat([whisper_mask, roberta_valid], dim=1)
        else:
            f1_mask = None

        f1_h1, f1_h2, f1_h3 = self.enc_f1(f1_input, src_key_padding_mask=f1_mask)
        # f1_h2: [B, T+1, hidden_dim], f1_h3: [B, hidden_dim]

        # ── Fusion stage 2: ENC_f2(h2 || g2 || f1_h2) ───────────────────────
        # Concatenate: [B, T+1+T+1, hidden_dim]
        f2_input = torch.cat([h2, g2, f1_h2], dim=1)

        if whisper_mask is not None:
            f2_mask = torch.cat([whisper_mask, roberta_valid, f1_mask], dim=1)
        else:
            f2_mask = None

        f2_h1, f2_h2, f2_h3 = self.enc_f2(f2_input, src_key_padding_mask=f2_mask)
        # f2_h3: [B, hidden_dim]

        # ── Pooled representation fusion: W(h3 || g3 || f2_h3) ──────────────
        x = torch.cat([h3, g3, f2_h3], dim=-1)  # [B, 3 * hidden_dim]
        x = self.fusion_proj(x)                  # [B, hidden_dim]

        # ── Classification ────────────────────────────────────────────────────
        logits = self.classifier(x)              # [B, num_emotions]
        return logits
