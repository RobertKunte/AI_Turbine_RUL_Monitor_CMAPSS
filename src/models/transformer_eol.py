from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.universal_encoder_v1 import PositionalEncoding
from src.models.damage_head import CumulativeDamageHead



class PiecewiseRULMapper(nn.Module):
    """
    Map a scalar degradation / health score to RUL in cycles using a simple
    piecewise-linear (actually piecewise power-law) mapping.

    Design:
        - Input: unconstrained score s -> mapped via sigmoid to p in (0, 1)
        - p represents a normalized "remaining life" / health measure
        - A knee at RUL = tau splits the range into two segments:
            * Early life:    RUL in [tau, max_rul]
            * Near failure:  RUL in [0, tau]
        - In the near-failure segment we use a slightly concave mapping
          (gamma_low < 1) to create a steeper decay towards EOL.

    This mapping is monotone in p and controlled by max_rul and tau.
    """

    def __init__(
        self,
        max_rul: float = 125.0,
        tau: float = 45.0,
        gamma_low: float = 0.5,
        gamma_high: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_rul = float(max_rul)
        self.tau = float(tau)
        self.gamma_low = float(gamma_low)
        self.gamma_high = float(gamma_high)

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score: [B] or [B, 1] unconstrained scalar per sample

        Returns:
            rul: [B] RUL in cycles, clipped to [0, max_rul]
        """
        if score.dim() > 1:
            score = score.squeeze(-1)

        # Normalized "health" / remaining-life score in (0, 1)
        p = torch.sigmoid(score)

        # Normalized knee location in health space corresponding to RUL ~ tau
        r_mid = self.tau / max(self.max_rul, 1e-6)
        # Clamp to avoid numerical issues
        r_mid = max(min(r_mid, 0.999), 0.001)

        r_mid_tensor = torch.tensor(
            r_mid, dtype=p.dtype, device=p.device
        )

        # Near-failure segment: map p in [0, r_mid] -> [0, tau]
        # Use a concave mapping (gamma_low < 1) for steeper decay near EOL.
        tail_ratio = torch.clamp(p / r_mid_tensor, min=0.0, max=1.0)
        rul_tail = self.tau * torch.pow(tail_ratio, self.gamma_low)

        # Early-life / plateau segment: map p in [r_mid, 1] -> [tau, max_rul]
        high_ratio = torch.clamp(
            (p - r_mid_tensor) / (1.0 - r_mid_tensor),
            min=0.0,
            max=1.0,
        )
        rul_high = self.tau + (self.max_rul - self.tau) * torch.pow(
            high_ratio, self.gamma_high
        )

        rul = torch.where(p < r_mid_tensor, rul_tail, rul_high)
        # Final safety clamp
        rul = torch.clamp(rul, min=0.0, max=self.max_rul)
        return rul


class ImprovedRULHead(nn.Module):
    """
    Deeper RUL head with optional HI-fusion, residual skip, and piecewise mapping.

    The head operates on a latent representation (typically the pooled sequence
    embedding) and can optionally fuse a compact HI latent (e.g. health_last).

    Depending on configuration, it either:
        - returns a direct RUL prediction in cycles, or
        - produces a scalar score that is mapped to RUL via PiecewiseRULMapper.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        dropout: float = 0.1,
        use_skip: bool = True,
        use_hi_fusion: bool = True,
        hi_dim: Optional[int] = None,
        use_piecewise_mapping: bool = True,
        max_rul: float = 125.0,
        tau: float = 45.0,
    ) -> None:
        super().__init__()

        self.use_skip = use_skip
        self.use_hi_fusion = use_hi_fusion and (hi_dim is not None and hi_dim > 0)
        self.use_piecewise_mapping = use_piecewise_mapping

        in_dim = input_dim + (hi_dim if self.use_hi_fusion else 0)

        layers = []
        last_dim = in_dim
        for _ in range(max(num_hidden_layers, 1)):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        # Final projection to scalar score/logit
        layers.append(nn.Linear(last_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.skip = nn.Linear(input_dim, 1) if self.use_skip else None

        self.mapper = PiecewiseRULMapper(
            max_rul=max_rul,
            tau=tau,
        ) if self.use_piecewise_mapping else None

    def forward(
        self,
        latent: torch.Tensor,
        hi_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent:   [B, D] pooled sequence representation
            hi_latent:[B, H] optional HI latent/feature vector

        Returns:
            rul_pred: [B] predicted RUL in cycles
        """
        if self.use_hi_fusion and hi_latent is not None:
            x = torch.cat([latent, hi_latent], dim=-1)
        else:
            x = latent

        score = self.mlp(x).squeeze(-1)  # [B]

        if self.skip is not None:
            score = score + self.skip(latent).squeeze(-1)

        if self.mapper is not None:
            rul = self.mapper(score)
        else:
            rul = score

        return rul


class RULHeadV4(nn.Module):
    """
    Compact, residual-enhanced RUL head used in Transformer V4 experiments.

    Uses a small MLP transform + gated residual path and clamps the output
    to [0, max_rul] for numerical stability.
    """

    def __init__(self, d_model: int, max_rul: float = 125.0) -> None:
        super().__init__()
        self.max_rul = float(max_rul)

        self.fc_h = nn.Linear(d_model, d_model)
        self.fc_gate = nn.Linear(d_model, d_model)
        self.fc_res = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, 1)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, d_model] pooled sequence representation

        Returns:
            rul: [B] RUL prediction in cycles, clamped to [0, max_rul]
        """
        h = self.act(self.fc_h(z))             # [B, d_model]
        gate = torch.sigmoid(self.fc_gate(z))  # [B, d_model]
        res = self.fc_res(z)                   # [B, d_model]

        h_final = gate * h + (1.0 - gate) * res
        rul = self.fc_out(h_final).squeeze(-1)  # [B]
        rul = torch.clamp(rul, min=0.0, max=self.max_rul)
        return rul


class EOLFullTransformerEncoder(nn.Module):
    """
    Transformer-basierter EOL-Regressor mit Health-Index Kopf (RUL + HI).

    Designziele:
    - Gleiche Eingaben wie EOLFullLSTMWithHealth:
        * x: [B, T, F] mit F≈244 Features, T=seq_len (z.B. 30)
        * cond_ids: [B] mit ConditionIDs (z.B. 7 Bedingungen)
    - Gleiche Schnittstelle wie EOLFullLSTMWithHealth:
        forward(x, cond_ids) -> (rul_pred, health_last, health_seq)
      sodass train_eol_full_lstm und evaluate_eol_full_lstm unverändert
      weiterverwendet werden können.
    - Multi-Task Heads:
        * RUL-Head (Skalar)
        * HI-Head (Skalar + Sequenz, optional damage-basiert)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_layers: int = 3,
        n_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        # Condition embeddings (multi-condition FD002/FD004)
        use_condition_embedding: bool = False,
        num_conditions: int = 1,
        cond_emb_dim: int = 4,
        max_seq_len: int = 300,
        # ------------------------------------------------------------------
        # NEW: continuous condition encoder (Cond_* vector per timestep)
        # ------------------------------------------------------------------
        use_cond_encoder: bool = False,
        cond_in_dim: int = 0,
        cond_encoder_dim: Optional[int] = None,
        use_cond_recon_head: bool = False,
        # ------------------------------------------------------------------
        # Optional cumulative damage head (separate from legacy damage HI)
        # ------------------------------------------------------------------
        use_damage_head: bool = False,
        damage_L_ref: float = 300.0,
        damage_alpha_base: float = 0.1,
        damage_hidden_dim: int = 64,
        # NEW (v3c): optional MLP-based damage head configuration
        damage_use_mlp: bool = False,
        damage_mlp_hidden_factor: int = 2,
        damage_mlp_num_layers: int = 2,
        damage_mlp_dropout: float = 0.1,
        # NEW: v3d delta cumsum parameters
        damage_use_delta_cumsum: bool = False,
        damage_delta_alpha: float = 1.0,
        # NEW (v3e): temporal smoothing
        damage_use_temporal_conv: bool = False,
        damage_temporal_conv_kernel_size: int = 3,
        damage_temporal_conv_num_layers: int = 1,
        # NEW (v4): optional calibrated HI head (HI_cal_v2 supervision)
        use_hi_cal_head: bool = False,
    ) -> None:
        super().__init__()

        from src.config import (
            USE_DAMAGE_HEALTH_HEAD,
            DAMAGE_ALPHA_INIT,
            DAMAGE_SOFTPLUS_BETA,
        )

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # ------------------------------------------------------------------
        # Continuous condition encoder configuration
        # ------------------------------------------------------------------
        self.use_cond_encoder: bool = use_cond_encoder
        self.cond_in_dim: int = cond_in_dim
        self.cond_encoder_dim: int = cond_encoder_dim or d_model
        self.use_cond_recon_head: bool = use_cond_recon_head
        # Indices of Cond_* features inside the full feature vector.
        # These will be set from run_experiments.py for ms+DT configs.
        self.cond_feature_indices: Optional[list[int]] = None

        # ------------------------------------------------------------------
        # Condition embeddings (Flag-API kompatibel zu EOLFullLSTMWithHealth)
        # ------------------------------------------------------------------
        self.use_condition_embedding = use_condition_embedding
        self.num_conditions = num_conditions
        self.cond_emb_dim = cond_emb_dim if use_condition_embedding else 0

        if use_condition_embedding:
            if num_conditions is None or num_conditions < 1:
                raise ValueError("num_conditions must be >= 1 when use_condition_embedding=True")
            self.condition_embedding = nn.Embedding(num_conditions, cond_emb_dim)
            # Projektion auf d_model, damit wir Condition-Info additiv einfügen können
            self.cond_proj = nn.Linear(cond_emb_dim, d_model)
        else:
            self.condition_embedding = None
            self.cond_proj = None

        # ------------------------------------------------------------------
        # Optional: continuous condition encoder MLP (Cond_* per timestep)
        # ------------------------------------------------------------------
        if self.use_cond_encoder and self.cond_in_dim > 0:
            self.cond_encoder = nn.Sequential(
                nn.Linear(self.cond_in_dim, self.cond_encoder_dim),
                nn.ReLU(),
                nn.Linear(self.cond_encoder_dim, d_model),
            )
        else:
            self.cond_encoder = None

        # ------------------------------------------------------------------
        # 1. Feature-Embedding (Linear → d_model)
        # ------------------------------------------------------------------
        self.input_proj = nn.Linear(input_dim, d_model)

        # ------------------------------------------------------------------
        # 2. Positions-Encoding
        # ------------------------------------------------------------------
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # ------------------------------------------------------------------
        # 3. Transformer Encoder
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ------------------------------------------------------------------
        # 4. Temporale Attention-Pooling für globale Repräsentation
        # ------------------------------------------------------------------
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),  # Score pro Zeitschritt
        )

        # ------------------------------------------------------------------
        # 5. Geteilte Projektion + RUL / HI Heads
        # ------------------------------------------------------------------
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Default (v1) RUL/HI heads – kept for backwards compatibility.
        self.fc_rul = nn.Linear(d_model, 1)
        self.fc_health = nn.Linear(d_model, 1)

        # Damage-basierte HI-Variante (identische API wie EOLFullLSTMWithHealth)
        self.use_damage_health_head = USE_DAMAGE_HEALTH_HEAD
        self.fc_damage = nn.Linear(d_model, 1)

        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(DAMAGE_ALPHA_INIT), dtype=torch.float32)
        )
        self.damage_softplus_beta = DAMAGE_SOFTPLUS_BETA

        # ------------------------------------------------------------------
        # Optional: calibrated HI head (v4, predicts HI_cal_v2 sequence)
        # ------------------------------------------------------------------
        self.use_hi_cal_head: bool = bool(use_hi_cal_head)
        if self.use_hi_cal_head:
            self.hi_cal_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.hi_cal_head = None

        # Placeholder for optional improved RUL head (v3).
        # These attributes may be set from the outside (e.g. in run_experiments)
        # without breaking older experiments.
        self.rul_head: Optional[ImprovedRULHead] = None
        self.rul_head_type: str = "linear"
        self.max_rul: float = 125.0  # will typically be overridden from config
        self.tau: float = 40.0       # will typically be overridden from config

        # ------------------------------------------------------------------
        # Optional: condition reconstruction head (auxiliary task)
        # ------------------------------------------------------------------
        if self.use_cond_recon_head and self.use_cond_encoder and self.cond_in_dim > 0:
            self.fc_cond_recon = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.cond_in_dim),
            )
        else:
            self.fc_cond_recon = None

        # ------------------------------------------------------------------
        # Optional: cumulative damage head for explicit damage-based HI
        # ------------------------------------------------------------------
        self.use_cum_damage_head: bool = bool(use_damage_head)
        if self.use_cum_damage_head:
            cond_dim_for_damage = self.cond_in_dim if self.use_cond_encoder and self.cond_in_dim > 0 else None
            self.damage_head = CumulativeDamageHead(
                d_model=d_model,
                cond_dim=cond_dim_for_damage,
                L_ref=damage_L_ref,
                alpha_base=damage_alpha_base,
                hidden_dim=damage_hidden_dim,
                use_mlp=damage_use_mlp,
                mlp_hidden_factor=damage_mlp_hidden_factor,
                mlp_num_layers=damage_mlp_num_layers,
                mlp_dropout=damage_mlp_dropout,
                use_delta_cumsum=damage_use_delta_cumsum,
                delta_alpha=damage_delta_alpha,
                damage_use_temporal_conv=damage_use_temporal_conv,
                damage_temporal_conv_kernel_size=damage_temporal_conv_kernel_size,
                damage_temporal_conv_num_layers=damage_temporal_conv_num_layers,
            )
        else:
            self.damage_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming/He-Initialisierung analog zu den bestehenden Modellen."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Schaden-Head klein initialisieren (wie bei EOLFullLSTMWithHealth)
        if self.use_damage_health_head:
            nn.init.xavier_normal_(self.fc_damage.weight, gain=0.1)
            if self.fc_damage.bias is not None:
                nn.init.zeros_(self.fc_damage.bias)

        # Small init for HI_cal head to avoid large initial deviations
        if getattr(self, "hi_cal_head", None) is not None:
            for m in self.hi_cal_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
        cond_seq: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Forward-Pass.

        Args:
            x: [B, T, F] – Eingangssequenzen
            cond_ids: [B] – optionale ConditionIDs; erforderlich, wenn
                use_condition_embedding=True

        Returns:
            If return_aux is False (default, V1 behaviour):
                rul_pred:    [B]       – RUL-Vorhersagen (Zyklen)
                health_last: [B]       – HI am letzten Zeitschritt
                health_seq:  [B, T, 1] – HI über die gesamte Sequenz

            If return_aux is True (V2, with condition reconstruction):
                rul_pred, health_last, health_seq, cond_seq_avg, cond_recon
                where:
                    cond_seq_avg: [B, cond_in_dim] or None
                    cond_recon:   [B, cond_in_dim] or None
        """
        B, T, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got F={feat_dim}"
            )

        # ------------------------------------------------------------------
        # 1. Feature-Embedding
        # ------------------------------------------------------------------
        x_proj = self.input_proj(x)  # [B, T, d_model]

        # Optional continuous condition sequence (Cond_* per timestep)
        # If not provided explicitly but indices are known, derive from x.
        if cond_seq is None and self.use_cond_encoder and self.cond_encoder is not None:
            if self.cond_feature_indices is not None and len(self.cond_feature_indices) == self.cond_in_dim:
                cond_seq = x[:, :, self.cond_feature_indices]  # [B, T, cond_in_dim]

        # ------------------------------------------------------------------
        # 2. Condition-Fusion (additiv auf d_model)
        # ------------------------------------------------------------------
        if self.use_condition_embedding:
            if cond_ids is None:
                raise ValueError("cond_ids required when use_condition_embedding=True")
            cond_emb = self.condition_embedding(cond_ids)  # [B, cond_emb_dim]
            cond_up = self.cond_proj(cond_emb)  # [B, d_model]
            cond_up = cond_up.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model]
            x_seq = x_proj + cond_up
        else:
            x_seq = x_proj

        # Add continuous condition embedding (if enabled and available)
        if self.use_cond_encoder and self.cond_encoder is not None and cond_seq is not None:
            if cond_seq.shape[0] != B or cond_seq.shape[1] != T or cond_seq.shape[2] != self.cond_in_dim:
                raise ValueError(
                    f"[EOLFullTransformerEncoder] cond_seq shape mismatch: "
                    f"expected ({B}, {T}, {self.cond_in_dim}), got {tuple(cond_seq.shape)}"
                )
            cond_emb_seq = self.cond_encoder(cond_seq)  # [B, T, d_model]
            x_seq = x_seq + cond_emb_seq

        # ------------------------------------------------------------------
        # 3. Positions-Encoding + Transformer Encoder
        # ------------------------------------------------------------------
        x_pos = self.pos_encoding(x_seq)  # [B, T, d_model]
        enc_out = self.transformer(x_pos)  # [B, T, d_model]

        # ------------------------------------------------------------------
        # 4. Attention-Pooling über die Zeit -> globale Repräsentation
        # ------------------------------------------------------------------
        scores = self.attn_pool(enc_out)  # [B, T, 1]
        attn_weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        pooled = torch.sum(attn_weights * enc_out, dim=1)  # [B, d_model]

        # Geteilte Features
        shared = self.shared_head(pooled)  # [B, d_model]
        
        # ------------------------------------------------------------------
        # 5a. HI-Head (damage-basiert oder einfacher Sigmoid-Fallback)
        # ------------------------------------------------------------------
        if self.use_damage_health_head:
            # Schaden-Head auf alle Zeitschritte anwenden
            B, T, H = enc_out.shape
            enc_flat = enc_out.reshape(B * T, H)  # [B*T, d_model]
            shared_seq = self.shared_head(enc_flat)  # [B*T, d_model]
            shared_seq = shared_seq.reshape(B, T, -1)  # [B, T, d_model]

            raw_rates = self.fc_damage(shared_seq)  # [B, T, 1]
            if self.damage_softplus_beta != 1.0:
                rates = F.softplus(raw_rates, beta=self.damage_softplus_beta)
            else:
                rates = F.softplus(raw_rates)  # >= 0

            damage = torch.cumsum(rates, dim=1)  # [B, T, 1]
            alpha = torch.exp(self.log_alpha)  # > 0
            health_seq = torch.exp(-alpha * damage)  # [B, T, 1], monoton fallend
            health_last = health_seq[:, -1, :].squeeze(-1)  # [B]
        else:
            # Einfacher Sigmoid-Head auf globalem Vektor
            health_logit = self.fc_health(shared)  # [B, 1]
            health_last = torch.sigmoid(health_logit).squeeze(-1)  # [B]
            # Für Kompatibilität broadcasten wir über die Sequenz
            health_seq = health_last.unsqueeze(1).unsqueeze(2).repeat(1, T, 1)  # [B, T, 1]
        
        # ------------------------------------------------------------------
        # 5b. RUL-Head (wahlweise klassisch oder ImprovedRULHead)
        # ------------------------------------------------------------------
        if self.rul_head is not None and self.rul_head_type == "improved":
            hi_latent = health_last.unsqueeze(-1) if self.rul_head.use_hi_fusion else None
            rul_pred = self.rul_head(shared, hi_latent=hi_latent)
        else:
            rul_logit = self.fc_rul(shared)  # [B, 1]
            rul_pred = rul_logit.squeeze(-1)  # [B]

        # ------------------------------------------------------------------
        # 5c. Optional: condition reconstruction head (auxiliary output)
        # ------------------------------------------------------------------
        cond_seq_avg: Optional[torch.Tensor] = None
        cond_recon: Optional[torch.Tensor] = None
        if (
            return_aux
            and self.fc_cond_recon is not None
            and self.use_cond_encoder
            and self.cond_in_dim > 0
            and cond_seq is not None
        ):
            cond_seq_avg = cond_seq.mean(dim=1)  # [B, cond_in_dim]
            cond_recon = self.fc_cond_recon(shared)  # [B, cond_in_dim]

        if return_aux:
            return rul_pred, health_last, health_seq, cond_seq_avg, cond_recon
        return rul_pred, health_last, health_seq

    # ------------------------------------------------------------------
    # Encoder-side helper for world models
    # ------------------------------------------------------------------
    def encode(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
        return_seq: bool = False,
        cond_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Run only the encoder stack (feature embedding, condition fusion,
        positional encoding, Transformer encoder + attention pooling).

        This is used by TransformerWorldModelV1 to reuse the same encoder
        backbone without going through the RUL/HI heads.

        Args:
            x: [B, T, F] input sequences (same as in forward).
            cond_ids: Optional condition IDs if condition embeddings are enabled.
            return_seq: If True, return (enc_seq, pooled); otherwise only pooled.

        Returns:
            If return_seq:
                enc_seq:  [B, T, d_model] – encoded sequence
                pooled:   [B, d_model]    – attention pooled representation
            Else:
                pooled:   [B, d_model]
        """
        B, T, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got F={feat_dim}"
            )

        # Feature embedding
        x_proj = self.input_proj(x)  # [B, T, d_model]

        # Optional continuous condition sequence (Cond_* per timestep)
        if cond_seq is None and self.use_cond_encoder and self.cond_encoder is not None:
            if self.cond_feature_indices is not None and len(self.cond_feature_indices) == self.cond_in_dim:
                cond_seq = x[:, :, self.cond_feature_indices]  # [B, T, cond_in_dim]

        # Condition fusion (if enabled)
        if self.use_condition_embedding:
            if cond_ids is None:
                raise ValueError("cond_ids required when use_condition_embedding=True")
            cond_emb = self.condition_embedding(cond_ids)  # [B, cond_emb_dim]
            cond_up = self.cond_proj(cond_emb)  # [B, d_model]
            cond_up = cond_up.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model]
            x_seq = x_proj + cond_up
        else:
            x_seq = x_proj

        # Add continuous condition embedding (if enabled and available)
        if self.use_cond_encoder and self.cond_encoder is not None and cond_seq is not None:
            if cond_seq.shape[0] != B or cond_seq.shape[1] != T or cond_seq.shape[2] != self.cond_in_dim:
                raise ValueError(
                    f"[EOLFullTransformerEncoder.encode] cond_seq shape mismatch: "
                    f"expected ({B}, {T}, {self.cond_in_dim}), got {tuple(cond_seq.shape)}"
                )
            cond_emb_seq = self.cond_encoder(cond_seq)  # [B, T, d_model]
            x_seq = x_seq + cond_emb_seq

        # Positional encoding + Transformer encoder
        x_pos = self.pos_encoding(x_seq)  # [B, T, d_model]
        enc_out = self.transformer(x_pos)  # [B, T, d_model]

        # Attention pooling
        scores = self.attn_pool(enc_out)  # [B, T, 1]
        attn_weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        pooled = torch.sum(attn_weights * enc_out, dim=1)  # [B, d_model]

        if return_seq:
            return enc_out, pooled
        return pooled

    # ------------------------------------------------------------------
    # Helper: HI_cal_v2 prediction from encoder sequence (v4)
    # ------------------------------------------------------------------
    def predict_hi_cal_seq(self, enc_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict calibrated HI_cal_v2 sequence from encoder latent sequence.

        Args:
            enc_seq: [B, T, d_model] latent sequence from the Transformer encoder.

        Returns:
            hi_cal_seq_pred: [B, T] predicted HI_cal_v2 trajectory.
        """
        if not (self.use_hi_cal_head and self.hi_cal_head is not None):
            raise RuntimeError(
                "predict_hi_cal_seq called but use_hi_cal_head=False or hi_cal_head is None."
            )

        B, T, H = enc_seq.shape
        flat = enc_seq.reshape(B * T, H)  # [B*T, d_model]
        hi_flat = self.hi_cal_head(flat).squeeze(-1)  # [B*T]
        hi_seq = hi_flat.view(B, T)  # [B, T]
        # Optionally clamp to [0, 1] as HI_cal_v2 is a bounded health index
        return torch.clamp(hi_seq, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Encoder + HI helper for downstream decoders (e.g. RUL trajectory)
    # ------------------------------------------------------------------
    def encode_with_hi(
        self,
        x: torch.Tensor,
        cond_ids: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run the encoder and (optionally) the cumulative damage head to obtain:

          - z_seq:        [B, T, d_model] latent encoder sequence
          - hi_phys_seq:  None for now (physics HI is passed separately if available)
          - hi_damage_seq:[B, T] learned damage HI sequence from damage_head, if present

        This helper is intentionally lightweight and does not alter gradients
        or training behaviour. It is primarily used by external decoders
        (e.g. RUL trajectory decoders) on a frozen encoder.
        """
        # Reuse encoder stack; treat cond_vec as explicit cond_seq if provided.
        enc_outputs = self.encode(
            x,
            cond_ids=cond_ids,
            return_seq=True,
            cond_seq=cond_vec,
        )

        if isinstance(enc_outputs, tuple):
            z_seq, _ = enc_outputs
        else:
            # encode(return_seq=True) always returns a tuple, but keep this
            # for defensive programming.
            z_seq = enc_outputs

        hi_phys_seq: Optional[torch.Tensor] = None
        hi_damage_seq: Optional[torch.Tensor] = None

        # If a cumulative damage head is available, run it to obtain the
        # damage-based HI sequence. We do not pass an explicit cond_seq here;
        # the head itself does not require it (cond-sequence effects are
        # already embedded via the encoder if use_cond_encoder=True).
        if hasattr(self, "damage_head") and getattr(self, "damage_head", None) is not None:
            hi_damage_seq, _, _, _ = self.damage_head(
                z_seq,
                cond_seq=None,
            )

        return z_seq, hi_phys_seq, hi_damage_seq



