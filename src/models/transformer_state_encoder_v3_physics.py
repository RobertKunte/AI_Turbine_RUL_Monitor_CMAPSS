from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.damage_head import CumulativeDamageHead


class TransformerStateEncoderV3_Physics(nn.Module):
    """
    Physics-informed state encoder for FD00x:

    - Input:
        x_seq:   [B, T, F]  full ms+DT feature sequence (raw + ms + residuals + Twin_* + Cond_*)
        cond_seq: [B, T, C] continuous condition vector (Cond_*), optional

    - Output:
        hi_raw:  [B, 1]  health logit (sigmoid -> [0,1])
        rul_raw: [B, 1]  auxiliary RUL_norm logit (sigmoid -> [0,1])
        enc_seq: [B, T, d_model] latent sequence
        z_t:     [B, d_model]    last-step latent state
    """

    def __init__(
        self,
        input_dim: int,
        cond_in_dim: int,
        d_model: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        # ------------------------------------------------------------------
        # Optional cumulative damage head (HI_phys trajectory)
        # ------------------------------------------------------------------
        use_damage_head: bool = False,
        L_ref: float = 300.0,
        alpha_base: float = 0.1,
        damage_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.max_len = max_len
        self.cond_in_dim = cond_in_dim
        self.use_cond = cond_in_dim > 0

        # Damage-head configuration
        self.use_damage_head = bool(use_damage_head)

        # Project full feature vector (e.g. 349 dims for FD004 ms+DT) to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Continuous condition encoder (Cond_* features)
        if self.use_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_in_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.cond_proj = None

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Heads: small MLPs for HI and RUL_norm (scalar heads)
        self.fc_hi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.fc_rul = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Optional cumulative damage head for HI_phys trajectory
        if self.use_damage_head:
            cond_dim_for_damage = cond_in_dim if cond_in_dim and cond_in_dim > 0 else None
            self.damage_head = CumulativeDamageHead(
                d_model=d_model,
                cond_dim=cond_dim_for_damage,
                L_ref=L_ref,
                alpha_base=alpha_base,
                hidden_dim=damage_hidden_dim,
            )
        else:
            self.damage_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        if self.use_cond and self.cond_proj is not None:
            for m in self.cond_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for head in (self.fc_hi, self.fc_rul):
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.uniform_(self.pos_embedding.weight, -0.02, 0.02)

    def forward(
        self,
        x_seq: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        """
        Args:
            x_seq:   [B, T, F]
            cond_seq: [B, T, C] (optional continuous condition features Cond_*)

        Returns:
            If return_dict:
                {"hi_raw": hi_raw, "rul_raw": rul_raw, "enc_seq": enc_seq, "z_t": z_t}
            else:
                hi_raw, rul_raw, enc_seq
        """
        if x_seq.dim() != 3:
            raise ValueError(f"x_seq must be [B,T,F], got shape {tuple(x_seq.shape)}")

        B, T, _ = x_seq.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}")

        tokens = self.input_proj(x_seq)  # [B,T,d_model]

        if self.use_cond and self.cond_proj is not None and cond_seq is not None:
            if cond_seq.shape[0] != B or cond_seq.shape[1] != T or cond_seq.shape[2] != self.cond_in_dim:
                raise ValueError(
                    f"[TransformerStateEncoderV3_Physics] cond_seq shape mismatch: "
                    f"expected ({B}, {T}, {self.cond_in_dim}), got {tuple(cond_seq.shape)}"
                )
            cond_emb = self.cond_proj(cond_seq)  # [B,T,d_model]
            tokens = tokens + cond_emb

        # Positional encoding
        positions = torch.arange(T, device=x_seq.device).unsqueeze(0)  # [1,T]
        pos_emb = self.pos_embedding(positions)  # [1,T,d_model]
        tokens = tokens + pos_emb

        # Transformer encoding
        enc_seq = self.encoder(tokens)  # [B,T,d_model]
        z_t = enc_seq[:, -1, :]         # [B,d_model]

        hi_raw = self.fc_hi(z_t)        # [B,1]
        rul_raw = self.fc_rul(z_t)      # [B,1]

        # Optional cumulative damage-based HI trajectory
        hi_seq_phys = None
        hi_last_phys = None
        damage_seq = None
        delta_damage_seq = None
        if self.damage_head is not None:
            # Pass original Cond_* sequence into the damage head if available
            damage_cond_seq = cond_seq if (self.use_cond and cond_seq is not None) else None
            hi_seq_phys, hi_last_phys, damage_seq, delta_damage_seq = self.damage_head(
                enc_seq, cond_seq=damage_cond_seq
            )

        if return_dict:
            return {
                "hi_raw": hi_raw,
                "rul_raw": rul_raw,
                "enc_seq": enc_seq,
                "z_t": z_t,
                "hi_seq_phys": hi_seq_phys,
                "hi_last_phys": hi_last_phys,
                "damage_seq": damage_seq,
                "delta_damage_seq": delta_damage_seq,
            }
        return hi_raw, rul_raw, enc_seq


