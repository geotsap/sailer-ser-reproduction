"""Compact SER models that consume cached Whisper and RoBERTa hidden states.

Place this file at:
    src/model/emotion_models.py

Design assumptions:
    - Whisper and RoBERTa are frozen and are NOT run inside these models.
    - Feature extraction is done offline and all hidden states are saved to disk.
    - Whisper starts with last-hidden-state usage only.
    - RoBERTa uses a learnable weighted average over all saved hidden states.

Expected tensor shapes:
    whisper_hidden_states: [B, Lw, T, Dw] or [B, T, Dw]
    roberta_hidden_states: [B, Lr, S, Dr] or [B, S, Dr]
    speech_mask:           [B, T] optional; 1/True means valid frame
    text_mask:             [B, S] optional; 1/True means valid token
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class WhisperEmotionModel(nn.Module):
    """Speech-only emotion model over cached Whisper hidden states.

    The model selects the last Whisper hidden state, applies the downstream
    3-layer pointwise Conv1D module, mean-pools over time, and predicts emotion
    logits with a 2-layer MLP.
    """

    def __init__(
        self,
        num_emotions: int = 9,
        whisper_dim: int = 1280,
        conv_dim: int = 256,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.1,
        auxiliary_output_dims: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()

        self.num_emotions = num_emotions
        self.whisper_dim = whisper_dim
        self.conv_dim = conv_dim

        self.speech_conv = nn.Sequential(
            nn.Conv1d(whisper_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(conv_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_emotions),
        )

        self.auxiliary_heads = nn.ModuleDict()
        if auxiliary_output_dims is not None:
            for name, output_dim in auxiliary_output_dims.items():
                self.auxiliary_heads[name] = nn.Sequential(
                    nn.Linear(conv_dim, classifier_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden_dim, output_dim),
                )

    def forward(
        self,
        whisper_hidden_states: Tensor,
        speech_mask: Optional[Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if whisper_hidden_states.ndim == 4:
            # Cached all Whisper hidden states: use only the last one initially.
            speech = whisper_hidden_states[:, -1, :, :]
        elif whisper_hidden_states.ndim == 3:
            # Already a single selected hidden-state tensor.
            speech = whisper_hidden_states
        else:
            raise ValueError(
                "whisper_hidden_states must have shape [B, L, T, D] or [B, T, D]; "
                f"received {tuple(whisper_hidden_states.shape)}."
            )

        if speech.shape[-1] != self.whisper_dim:
            raise ValueError(
                f"Expected Whisper hidden dim {self.whisper_dim}, received {speech.shape[-1]}."
            )

        # [B, T, D] -> [B, D, T] for Conv1D -> [B, T, conv_dim]
        speech = self.speech_conv(speech.transpose(1, 2)).transpose(1, 2)
        speech_embedding = self._masked_mean_pool(speech, speech_mask)
        logits = self.emotion_classifier(speech_embedding)

        if not return_embeddings and len(self.auxiliary_heads) == 0:
            return logits

        output: Dict[str, Tensor] = {
            "logits": logits,
            "speech_embedding": speech_embedding,
        }
        for name, head in self.auxiliary_heads.items():
            output[name] = head(speech_embedding)
        return output

    @staticmethod
    def _masked_mean_pool(x: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return x.mean(dim=1)

        if mask.ndim != 2:
            raise ValueError(f"mask must have shape [B, T]; received {tuple(mask.shape)}.")
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} is incompatible with tensor shape {tuple(x.shape)}."
            )

        mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class WhisperRobertaEmotionModel(nn.Module):
    """Multimodal emotion model over cached Whisper and RoBERTa hidden states.

    Speech branch:
        last Whisper hidden state -> 3-layer pointwise Conv1D -> mean pooling

    Text branch:
        weighted average over all RoBERTa hidden states -> mean pooling

    Fusion:
        concatenate speech and text embeddings -> 2-layer MLP
    """

    def __init__(
        self,
        num_emotions: int = 9,
        whisper_dim: int = 1280,
        roberta_dim: int = 1024,
        num_roberta_layers: int = 25,
        conv_dim: int = 256,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.1,
        auxiliary_output_dims: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()

        self.num_emotions = num_emotions
        self.whisper_dim = whisper_dim
        self.roberta_dim = roberta_dim
        self.num_roberta_layers = num_roberta_layers
        self.conv_dim = conv_dim

        self.speech_conv = nn.Sequential(
            nn.Conv1d(whisper_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
        )

        # RoBERTa-large with output_hidden_states=True usually gives 25 tensors:
        # embedding output + 24 transformer layer outputs.
        self.roberta_layer_logits = nn.Parameter(torch.zeros(num_roberta_layers))

        fusion_dim = conv_dim + roberta_dim
        self.emotion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_emotions),
        )

        self.auxiliary_heads = nn.ModuleDict()
        if auxiliary_output_dims is not None:
            for name, output_dim in auxiliary_output_dims.items():
                self.auxiliary_heads[name] = nn.Sequential(
                    nn.Linear(fusion_dim, classifier_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden_dim, output_dim),
                )

    def forward(
        self,
        whisper_hidden_states: Tensor,
        roberta_hidden_states: Tensor,
        speech_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if whisper_hidden_states.ndim == 4:
            speech = whisper_hidden_states[:, -1, :, :]
        elif whisper_hidden_states.ndim == 3:
            speech = whisper_hidden_states
        else:
            raise ValueError(
                "whisper_hidden_states must have shape [B, L, T, D] or [B, T, D]; "
                f"received {tuple(whisper_hidden_states.shape)}."
            )

        if roberta_hidden_states.ndim == 4:
            if roberta_hidden_states.shape[1] != self.num_roberta_layers:
                raise ValueError(
                    f"Expected {self.num_roberta_layers} RoBERTa hidden-state tensors, "
                    f"received {roberta_hidden_states.shape[1]}."
                )
            weights = F.softmax(self.roberta_layer_logits, dim=0).to(
                device=roberta_hidden_states.device,
                dtype=roberta_hidden_states.dtype,
            )
            text = torch.einsum("blsd,l->bsd", roberta_hidden_states, weights)
        elif roberta_hidden_states.ndim == 3:
            text = roberta_hidden_states
            weights = None
        else:
            raise ValueError(
                "roberta_hidden_states must have shape [B, L, S, D] or [B, S, D]; "
                f"received {tuple(roberta_hidden_states.shape)}."
            )

        if speech.shape[-1] != self.whisper_dim:
            raise ValueError(
                f"Expected Whisper hidden dim {self.whisper_dim}, received {speech.shape[-1]}."
            )
        if text.shape[-1] != self.roberta_dim:
            raise ValueError(
                f"Expected RoBERTa hidden dim {self.roberta_dim}, received {text.shape[-1]}."
            )

        # Speech branch: [B, T, D] -> [B, D, T] -> [B, T, conv_dim]
        speech = self.speech_conv(speech.transpose(1, 2)).transpose(1, 2)
        speech_embedding = self._masked_mean_pool(speech, speech_mask)

        # Text branch.
        text_embedding = self._masked_mean_pool(text, text_mask)

        # Fusion/classification.
        fused_embedding = torch.cat([speech_embedding, text_embedding], dim=-1)
        logits = self.emotion_classifier(fused_embedding)

        if not return_embeddings and len(self.auxiliary_heads) == 0:
            return logits

        output: Dict[str, Tensor] = {
            "logits": logits,
            "speech_embedding": speech_embedding,
            "text_embedding": text_embedding,
            "fused_embedding": fused_embedding,
        }
        if weights is not None:
            output["roberta_layer_weights"] = weights
        for name, head in self.auxiliary_heads.items():
            output[name] = head(fused_embedding)
        return output

    @staticmethod
    def _masked_mean_pool(x: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return x.mean(dim=1)

        if mask.ndim != 2:
            raise ValueError(f"mask must have shape [B, T/S]; received {tuple(mask.shape)}.")
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} is incompatible with tensor shape {tuple(x.shape)}."
            )

        mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
