from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerStateEncoderV3(nn.Module):
    """
    State-focused Transformer encoder for FD00x-style turbine data.

    - Input:  x_seq: [B, T, F] with ms+DT feature vectors (same as EOL encoder / world model)
              cond_seq: [B, T, C] optional continuous condition vector (Cond_* features)
    - Output:
        - rul_raw: [B, 1] unnormalized RUL logit (mapped to [0,1] outside via sigmoid if needed)
        - hi_raw:  [B, 1] health logit (sigmoid -> [0,1])
        - enc_seq: [B, T, d_model] latent sequence

    This encoder is trained with simple MSE losses on HI and normalized RUL and is intended
    as a smooth state encoder for downstream world models.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_cond_encoder: bool = True,
        cond_in_dim: int = 0,  # number of continuous condition features (Cond_*)
        cond_emb_dim: Optional[int] = None,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.use_cond_encoder = use_cond_encoder
        self.cond_in_dim = cond_in_dim
        self.cond_emb_dim = cond_emb_dim or d_model
        self.max_len = max_len

        # Project sensor + engineered features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Optional continuous condition encoder (Cond_* features)
        if self.use_cond_encoder and self.cond_in_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_in_dim, self.cond_emb_dim),
                nn.ReLU(),
                nn.Linear(self.cond_emb_dim, d_model),
            )
        else:
            self.cond_proj = None

        # Simple learnable positional embeddings
        self.pos_embedding = nn.Embedding(self.max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Heads for HI and RUL
        self.fc_hi = nn.Linear(d_model, 1)
        self.fc_rul = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Standard Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        if self.cond_proj is not None:
            for m in self.cond_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.fc_hi.weight)
        if self.fc_hi.bias is not None:
            nn.init.zeros_(self.fc_hi.bias)
        nn.init.xavier_uniform_(self.fc_rul.weight)
        if self.fc_rul.bias is not None:
            nn.init.zeros_(self.fc_rul.bias)

        # Positional embeddings: small random init
        nn.init.uniform_(self.pos_embedding.weight, -0.02, 0.02)

    def forward(
        self,
        x_seq: torch.Tensor,  # (B, T, input_dim)
        cond_seq: Optional[torch.Tensor] = None,  # (B, T, cond_in_dim) or None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: [B, T, F] input sequence of ms+DT features
            cond_seq: [B, T, C] continuous condition vector (Cond_*), if available

        Returns:
            rul_raw: [B, 1] RUL logit (normalized to [0,1] outside if needed)
            hi_raw:  [B, 1] HI logit (sigmoid -> [0,1])
            enc_seq: [B, T, d_model] latent sequence
        """
        if x_seq.dim() != 3:
            raise ValueError(f"x_seq must be 3D [B,T,F], got shape {tuple(x_seq.shape)}")

        B, T, _ = x_seq.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}")

        x_proj = self.input_proj(x_seq)  # [B, T, d_model]

        if self.use_cond_encoder and self.cond_proj is not None and cond_seq is not None:
            if cond_seq.shape[0] != B or cond_seq.shape[1] != T or cond_seq.shape[2] != self.cond_in_dim:
                raise ValueError(
                    f"[TransformerStateEncoderV3] cond_seq shape mismatch: "
                    f"expected ({B}, {T}, {self.cond_in_dim}), got {tuple(cond_seq.shape)}"
                )
            cond_emb = self.cond_proj(cond_seq)  # [B, T, d_model]
            tokens = x_proj + cond_emb
        else:
            tokens = x_proj

        # Positional embeddings
        positions = torch.arange(T, device=x_seq.device).unsqueeze(0)  # [1, T]
        pos_emb = self.pos_embedding(positions)  # [1, T, d_model]
        tokens = tokens + pos_emb

        enc_seq = self.transformer_encoder(tokens)  # [B, T, d_model]
        z_t = enc_seq[:, -1, :]  # [B, d_model] last time step as state

        hi_raw = self.fc_hi(z_t)   # [B, 1] logit
        rul_raw = self.fc_rul(z_t) # [B, 1] logit

        return rul_raw, hi_raw, enc_seq


