from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerWorldModelV1(nn.Module):
    """
    Transformer-based World Model V1.

    - Encoder: reuses an existing Transformer encoder (e.g. EOLFullTransformerEncoder)
      that operates on full ms+DT feature vectors (z.B. 349D für FD004 phys/ms_dt).
    - Decoder: GRU-based autoregressive head for future sensor trajectories.
    - Optional auxiliary heads for future HI and future RUL.

    This module is intentionally kept generic: it expects that the caller
    constructs an appropriate encoder instance and passes it in.
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        num_sensors_out: int,
        cond_dim: int,
        future_horizon: int = 20,
        decoder_hidden_dim: int = 256,
        num_layers_decoder: int = 1,
        dropout: float = 0.1,
        predict_hi: bool = True,
        predict_rul: bool = True,
        target_mode: str = "sensors",      # "sensors", "residuals", or "latent_hi_rul"
        init_from_rul_hi: bool = False,    # decoder init from current RUL/HI
        # Dynamic latent world-model flags (Branch A+)
        use_latent_history: bool = False,
        use_hi_anchor: bool = False,
        use_future_conds: bool = False,
        # Dynamic latent WM + EOL fusion (A+) – backward-compatible defaults
        use_future_conditions: bool = False,
        use_eol_fusion: bool = False,
        eol_fusion_mode: str = "token",   # "token" | "feature"
        predict_latent: bool = False,     # if True: decoder produces z_future_seq first
        latent_decoder_type: str = "gru", # "gru" | "transformer"
        latent_decoder_num_layers: int = 2,
        latent_decoder_nhead: int = 4,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.input_dim = int(input_dim)
        self.num_sensors_out = int(num_sensors_out)
        self.cond_dim = int(cond_dim)
        self.d_dec = int(decoder_hidden_dim)
        self.future_horizon = int(future_horizon)
        self.predict_hi = bool(predict_hi)
        self.predict_rul = bool(predict_rul)
        self.num_layers_decoder = int(num_layers_decoder)
        self.target_mode = str(target_mode)
        self.init_from_rul_hi = bool(init_from_rul_hi)
        # Dynamic latent flags
        self.use_latent_history = bool(use_latent_history)
        self.use_hi_anchor = bool(use_hi_anchor)
        # Back-compat alias: use_future_conditions mirrors use_future_conds
        self.use_future_conds = bool(use_future_conds or use_future_conditions)

        # A+ extensions
        self.use_eol_fusion = bool(use_eol_fusion)
        self.eol_fusion_mode = str(eol_fusion_mode).lower()
        self.predict_latent = bool(predict_latent)
        self.latent_decoder_type = str(latent_decoder_type).lower()
        self.latent_decoder_num_layers = int(latent_decoder_num_layers)
        self.latent_decoder_nhead = int(latent_decoder_nhead)

        # Try to infer encoder hidden size (d_model) from common attributes.
        if hasattr(encoder, "d_model"):
            self.d_model = int(encoder.d_model)
        else:
            # Fallback: use a generic hidden_dim attribute if present
            self.d_model = int(getattr(encoder, "hidden_dim", 256))

        # Simple temporal pooling over encoder sequence output.
        # The encoder is expected to provide either:
        #   - an `encode(x, cond_ids, return_seq=True)` method returning (enc_seq, pooled)
        #   - or a standard forward returning sequence features where the caller pools.
        # Here we standardise on pooling inside this module for the simple case.
        self.pool = lambda x: x.mean(dim=1)

        # Project pooled encoder state + continuous condition vector into
        # the initial decoder hidden state. When init_from_rul_hi=True we
        # augment the latent with (current_rul, current_hi).
        self.decoder_init_from_latent = nn.Linear(self.d_model + self.cond_dim, self.d_dec)
        self.decoder_init_from_latent_rul_hi = nn.Linear(self.d_model + self.cond_dim + 2, self.d_dec)

        # Decoder GRU (sensor-autoregressive path) takes as input: [prev_sensors, cond_vector].
        # This path is used for the original V1 world model ("sensors"/"residuals" modes).
        self.dec_input_dim = self.num_sensors_out + self.cond_dim
        self.decoder_rnn = nn.GRU(
            input_size=self.dec_input_dim,
            hidden_size=self.d_dec,
            num_layers=self.num_layers_decoder,
            batch_first=True,
        )

        # Output head for future sensors
        self.sensor_head = nn.Linear(self.d_dec, self.num_sensors_out)

        # Optional heads for future HI and RUL
        if self.predict_hi:
            self.hi_head = nn.Linear(self.d_dec, 1)
        else:
            self.hi_head = None

        if self.predict_rul:
            self.rul_head = nn.Linear(self.d_dec, 1)
        else:
            self.rul_head = None

        self.dropout = nn.Dropout(dropout)

        # Optional dynamic-latent decoder (Branch A+). We keep this separate from the
        # sensor-autoregressive decoder to avoid changing behaviour of existing runs.
        if self.use_latent_history or self.use_future_conds or self.use_hi_anchor or self.target_mode in ["latent_hi_rul", "latent_hi_rul_dynamic_delta_v2"]:
            # Latent context: either last encoder state or a short history.
            self.latent_history_steps = 3
            if self.use_latent_history:
                latent_ctx_dim = self.latent_history_steps * self.d_model
            else:
                latent_ctx_dim = self.d_model

            latent_input_dim = latent_ctx_dim
            if self.use_future_conds:
                latent_input_dim += self.cond_dim
            if self.use_hi_anchor:
                # In the simple latent mode we append only the HI anchor (1 dim).
                # In the dynamic delta v2 mode we append both HI and RUL anchors (2 dims).
                if self.target_mode == "latent_hi_rul_dynamic_delta_v2":
                    latent_input_dim += 2  # HI + RUL anchor
                else:
                    latent_input_dim += 1  # scalar HI anchor

            self.latent_dec_input_dim = latent_input_dim
            self.latent_decoder_rnn = nn.GRU(
                input_size=self.latent_dec_input_dim,
                hidden_size=self.d_dec,
                num_layers=self.num_layers_decoder,
                batch_first=True,
            )
        else:
            self.latent_history_steps = 3
            self.latent_dec_input_dim = None
            self.latent_decoder_rnn = None

        # ------------------------------------------------------------------
        # Transformer latent decoder (A+): Cross-attn to latent history z_seq,
        # query tokens built from future conditions (+ pos emb) and optional EOL fusion.
        # This stays OFF unless latent_decoder_type="transformer" or predict_latent=True.
        # ------------------------------------------------------------------
        self.future_cond_proj: Optional[nn.Module] = None
        self.future_pos_emb: Optional[nn.Embedding] = None
        self.eol_ctx_to_token: Optional[nn.Module] = None
        self.eol_ctx_to_feature: Optional[nn.Module] = None
        self.latent_transformer_decoder: Optional[nn.Module] = None
        self.latent_to_delta_hi: Optional[nn.Module] = None
        self.latent_to_delta_rul: Optional[nn.Module] = None
        self.eol_scalar_head: Optional[nn.Module] = None
        # Latent-token heads (d_model -> 1) for the transformer-latent path.
        # Needed because the classic heads (hi_head/rul_head) operate on d_dec (GRU hidden size).
        self.hi_head_latent: Optional[nn.Module] = None
        self.rul_head_latent: Optional[nn.Module] = None

        use_transformer_latent = (
            self.latent_decoder_type == "transformer"
            or self.predict_latent
            or self.use_eol_fusion
        ) and (self.target_mode in ["latent_hi_rul", "latent_hi_rul_dynamic_delta_v2"])

        if use_transformer_latent:
            # Project future conditions -> d_model query tokens
            self.future_cond_proj = nn.Linear(self.cond_dim, self.d_model)
            # Horizon positional embedding (supports up to future_horizon tokens)
            self.future_pos_emb = nn.Embedding(self.future_horizon + 2, self.d_model)

            # EOL context derived from past (encoder output): predicted scalar + embedding for fusion.
            self.eol_scalar_head = nn.Linear(self.d_model, 1)
            self.eol_ctx_to_token = nn.Linear(1, self.d_model)
            self.eol_ctx_to_feature = nn.Linear(1, self.d_model)

            dec_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=max(1, self.latent_decoder_nhead),
                dim_feedforward=4 * self.d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.latent_transformer_decoder = nn.TransformerDecoder(
                dec_layer, num_layers=max(1, self.latent_decoder_num_layers)
            )

            # Delta heads from latent tokens (dynamic delta semantics)
            self.latent_to_delta_hi = nn.Linear(self.d_model, 1)
            self.latent_to_delta_rul = nn.Linear(self.d_model, 1)
            # Direct HI/RUL heads from latent tokens (simple latent mode)
            self.hi_head_latent = nn.Linear(self.d_model, 1)
            self.rul_head_latent = nn.Linear(self.d_model, 1)

    def forward(
        self,
        past_seq: torch.Tensor,
        cond_vec: torch.Tensor,
        *,
        cond_ids: Optional[torch.Tensor] = None,
        future_horizon: Optional[int] = None,
        teacher_forcing_targets: Optional[torch.Tensor] = None,
        current_rul: Optional[torch.Tensor] = None,
        current_hi: Optional[torch.Tensor] = None,
        future_conds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            past_seq: (B, T_in, input_dim)   – past window (same features as encoder).
            cond_vec: (B, cond_dim)          – continuous condition vector per sequence
                                               (e.g. Cond_* features).
            cond_ids: (B,) optional integer condition IDs if the encoder expects them.
            future_horizon: number of forecast steps (default: self.future_horizon).
            teacher_forcing_targets:
                Optional tensor of shape (B, T_out, num_sensors_out).
                If provided, will be used for teacher forcing (using ground truth
                as decoder input at the next step).

        Returns:
            pred_sensors: (B, T_out, num_sensors_out)
            pred_hi:      (B, T_out, 1) or None
            pred_rul:     (B, T_out, 1) or None
            pred_eol:     (B, 1) normalized RUL proxy (optional, for A+)
        """
        B, T_in, D = past_seq.shape
        if D != self.input_dim:
            raise ValueError(f"[TransformerWorldModelV1] Expected input_dim={self.input_dim}, got {D}")

        if future_horizon is None:
            future_horizon = self.future_horizon

        # ------------------------------------------------------------------
        # Dynamic latent world model path ("latent_hi_rul" or delta v2 mode)
        # ------------------------------------------------------------------
        if self.target_mode in ["latent_hi_rul", "latent_hi_rul_dynamic_delta_v2"]:
            # We require the encoder sequence output to build a short latent history.
            if hasattr(self.encoder, "encode"):
                if cond_ids is not None:
                    enc_seq, _ = self.encoder.encode(past_seq, cond_ids=cond_ids, return_seq=True)
                else:
                    enc_seq, _ = self.encoder.encode(past_seq, cond_ids=None, return_seq=True)
            else:
                # Fallback: treat encoder output as sequence features if possible.
                if cond_ids is not None:
                    enc_out = self.encoder(past_seq, cond_ids=cond_ids)
                else:
                    enc_out = self.encoder(past_seq)

                if isinstance(enc_out, tuple):
                    # enc_out[2] may be HI sequence [B, T, 1] – use it as proxy.
                    _, _, hi_seq = enc_out
                    if hi_seq.dim() == 3:
                        enc_seq = hi_seq  # [B, T, 1]
                    else:
                        enc_seq = hi_seq.unsqueeze(-1)
                else:
                    enc_seq = enc_out  # [B, T, d_model or similar]

            B_enc, T_enc, D_enc = enc_seq.shape

            # ------------------------------------------------------------------
            # 2a) Latent context for simple latent mode ("latent_hi_rul")
            # ------------------------------------------------------------------
            if self.target_mode == "latent_hi_rul":
                # If the transformer latent decoder is enabled, prefer it (A+).
                use_tf_latent = self.latent_transformer_decoder is not None and self.future_cond_proj is not None
                if use_tf_latent and self.use_future_conds:
                    H = int(future_horizon)
                    if future_conds is None:
                        future_conds = past_seq.new_zeros(B, H, self.cond_dim)
                    # Build query tokens from future conditions + pos emb
                    q = self.future_cond_proj(future_conds)
                    pos = self.future_pos_emb(
                        torch.arange(H, device=past_seq.device, dtype=torch.long)
                    ).unsqueeze(0)
                    q = q + pos

                    # EOL context (predicted scalar from past)
                    z0 = enc_seq[:, -1, :]  # (B, d_model)
                    eol_scalar = torch.sigmoid(self.eol_scalar_head(z0))  # (B,1) in [0,1]
                    if self.use_eol_fusion:
                        if self.eol_fusion_mode == "token":
                            eol_tok = self.eol_ctx_to_token(eol_scalar).unsqueeze(1)  # (B,1,d_model)
                            q_full = torch.cat([eol_tok, q], dim=1)  # (B,1+H,d_model)
                            dec = self.latent_transformer_decoder(tgt=q_full, memory=enc_seq)  # (B,1+H,d_model)
                            z_future = dec[:, 1:, :]  # (B,H,d_model)
                        else:
                            # feature fusion
                            eol_feat = self.eol_ctx_to_feature(eol_scalar).unsqueeze(1)  # (B,1,d_model)
                            q = q + eol_feat
                            z_future = self.latent_transformer_decoder(tgt=q, memory=enc_seq)  # (B,H,d_model)
                    else:
                        z_future = self.latent_transformer_decoder(tgt=q, memory=enc_seq)  # (B,H,d_model)

                    # Predict HI/RUL directly from latent tokens (no sensor forecasts)
                    pred_sensors = past_seq.new_zeros(B, H, self.num_sensors_out)
                    pred_hi = None
                    pred_rul = None
                    if self.predict_hi:
                        head = self.hi_head_latent if self.hi_head_latent is not None else self.hi_head
                        if head is not None:
                            pred_hi = torch.sigmoid(head(z_future)).view(B, H, 1)
                    if self.predict_rul:
                        head = self.rul_head_latent if self.rul_head_latent is not None else self.rul_head
                        if head is not None:
                            pred_rul = torch.sigmoid(head(z_future)).view(B, H, 1)
                    return pred_sensors, pred_hi, pred_rul, eol_scalar

                if self.use_latent_history:
                    steps = min(self.latent_history_steps, T_enc)
                    z_hist = enc_seq[:, -steps:, :]  # [B, steps, D]
                    if steps < self.latent_history_steps:
                        # Pad on the left with the first frame to keep dimensionality stable.
                        pad_frames = [enc_seq[:, :1, :]] * (self.latent_history_steps - steps)
                        z_hist = torch.cat(pad_frames + [z_hist], dim=1)
                    z_ctx = z_hist.reshape(B_enc, self.latent_history_steps * D_enc)  # [B, 3*D]
                else:
                    # Only use the last encoder state as context.
                    z_ctx = enc_seq[:, -1, :]  # [B, D]

                # Optional HI anchor: use the provided current_hi (physics-informed HI).
                # If not provided but the flag is enabled, fall back to zeros to keep the
                # decoder input dimensionality consistent between training and inference.
                hi_anchor = None
                if self.use_hi_anchor:
                    if current_hi is not None:
                        hi_anchor = current_hi
                        if hi_anchor.dim() == 1:
                            hi_anchor = hi_anchor.unsqueeze(-1)  # [B, 1]
                    else:
                        hi_anchor = past_seq.new_zeros(B, 1)

                # Optional future condition vectors: [B, H, cond_dim]
                if self.use_future_conds and future_conds is not None:
                    if future_conds.dim() != 3:
                        raise ValueError(
                            f"[TransformerWorldModelV1] future_conds must be [B, H, cond_dim], "
                            f"got shape {tuple(future_conds.shape)}"
                        )
                    if future_conds.size(0) != B or future_conds.size(1) != future_horizon:
                        raise ValueError(
                            f"[TransformerWorldModelV1] future_conds batch/horizon mismatch: "
                            f"got (B={future_conds.size(0)}, H={future_conds.size(1)}), "
                            f"expected (B={B}, H={future_horizon})"
                        )
                # If use_future_conds is True but future_conds is None, we silently fall back to zeros.

                if self.latent_decoder_rnn is None:
                    raise RuntimeError(
                        "[TransformerWorldModelV1] latent_decoder_rnn is not initialised "
                        "but target_mode='latent_hi_rul' was requested."
                    )

                preds_hi = []
                preds_rul = []

                h_t: Optional[torch.Tensor] = None
                for t in range(future_horizon):
                    parts = [z_ctx]

                    if self.use_future_conds:
                        if future_conds is not None:
                            cond_t = future_conds[:, t, :]  # [B, cond_dim]
                        else:
                            cond_t = past_seq.new_zeros(B, self.cond_dim)
                        parts.append(cond_t)

                    if self.use_hi_anchor and (hi_anchor is not None):
                        parts.append(hi_anchor)  # [B, 1]

                    dec_input_t = torch.cat(parts, dim=-1).unsqueeze(1)  # [B, 1, latent_dec_input_dim]

                    dec_out_t, h_t = self.latent_decoder_rnn(dec_input_t, h_t)  # [B, 1, d_dec]
                    dec_out_t = self.dropout(dec_out_t)
                    hid_t = dec_out_t[:, -1, :]  # [B, d_dec]

                    if self.predict_hi and self.hi_head is not None:
                        hi_t = self.hi_head(hid_t)  # [B, 1]
                        preds_hi.append(hi_t.unsqueeze(1))  # [B, 1, 1]
                    if self.predict_rul and self.rul_head is not None:
                        rul_t = self.rul_head(hid_t)  # [B, 1]
                        preds_rul.append(rul_t.unsqueeze(1))  # [B, 1, 1]

                # In latent mode we don't use sensor forecasts – return zeros for compatibility.
                pred_sensors = past_seq.new_zeros(B, future_horizon, self.num_sensors_out)

                pred_hi = None
                if self.predict_hi and len(preds_hi) > 0:
                    pred_hi = torch.cat(preds_hi, dim=1)  # [B, H, 1]

                pred_rul = None
                if self.predict_rul and len(preds_rul) > 0:
                    pred_rul = torch.cat(preds_rul, dim=1)  # [B, H, 1]

                return pred_sensors, pred_hi, pred_rul, None

            # ------------------------------------------------------------------
            # 2b) Dynamic delta world model v2 ("latent_hi_rul_dynamic_delta_v2")
            # ------------------------------------------------------------------
            if self.target_mode == "latent_hi_rul_dynamic_delta_v2":
                # If transformer latent decoder is enabled, decode z_future_seq using cross-attn:
                use_tf_latent = self.latent_transformer_decoder is not None and self.future_cond_proj is not None
                # Latent history: z_t, v1 = z_t - z_{t-1}, v2 = z_t - z_{t-2}
                if T_enc >= 3:
                    z_t = enc_seq[:, -1, :]
                    z_tm1 = enc_seq[:, -2, :]
                    z_tm2 = enc_seq[:, -3, :]
                elif T_enc == 2:
                    z_t = enc_seq[:, -1, :]
                    z_tm1 = enc_seq[:, -2, :]
                    z_tm2 = z_tm1
                else:  # T_enc == 1
                    z_t = enc_seq[:, -1, :]
                    z_tm1 = z_t
                    z_tm2 = z_t

                v1 = z_t - z_tm1
                v2 = z_t - z_tm2
                z_ctx = torch.cat([z_t, v1, v2], dim=-1)  # [B, 3*D]

                # Anchors from encoder heads (HI in [0,1], RUL normalized to [0,1])
                hi_anchor_vec = None
                rul_anchor_vec = None
                if hasattr(self.encoder, "fc_health") and hasattr(self.encoder, "fc_rul"):
                    # Use full forward pass to respect internal shared_head etc.
                    if cond_ids is not None:
                        rul_anchor_raw, hi_anchor_raw, _ = self.encoder(past_seq, cond_ids)
                    else:
                        rul_anchor_raw, hi_anchor_raw, _ = self.encoder(past_seq, None)

                    hi_anchor_vec = torch.sigmoid(hi_anchor_raw).unsqueeze(1)  # [B,1]

                    # Normalize RUL anchor to [0,1] using encoder.max_rul if available
                    max_rul = float(getattr(self.encoder, "max_rul", 125.0))
                    rul_clamped = torch.clamp(rul_anchor_raw, 0.0, max_rul)
                    rul_anchor_norm = (rul_clamped / max_rul).unsqueeze(1)  # [B,1]
                    rul_anchor_vec = rul_anchor_norm
                else:
                    # Fallback: approximate anchors from provided current targets if available.
                    if current_hi is not None:
                        hi_anchor_vec = current_hi.view(B_enc, 1)  # [B,1]
                    else:
                        hi_anchor_vec = past_seq.new_full((B_enc, 1), 1.0)

                    if current_rul is not None:
                        rul_anchor_vec = current_rul.view(B_enc, 1)  # [B,1]
                    else:
                        rul_anchor_vec = past_seq.new_full((B_enc, 1), float(self.future_horizon))

                # Ensure shapes [B,1]
                hi_anchor_vec = hi_anchor_vec.view(B_enc, 1)  # [B,1]
                rul_anchor_vec = rul_anchor_vec.view(B_enc, 1)  # [B,1]

                # Future conditions
                H = future_horizon
                if self.use_future_conds:
                    if future_conds is None:
                        # Fallback: zeros if not provided
                        future_conds = past_seq.new_zeros(B, H, self.cond_dim)
                    else:
                        if future_conds.dim() != 3 or future_conds.size(1) != H:
                            raise ValueError(
                                f"[TransformerWorldModelV1] future_conds must be [B, H, cond_dim] "
                                f"with H={H}, got {tuple(future_conds.shape)}"
                            )
                else:
                    future_conds = None

                if self.latent_decoder_rnn is None:
                    raise RuntimeError(
                        "[TransformerWorldModelV1] latent_decoder_rnn is not initialised "
                        "but target_mode='latent_hi_rul_dynamic_delta_v2' was requested."
                    )

                if use_tf_latent:
                    # Query tokens: future conditions (+ pos emb)
                    q = self.future_cond_proj(future_conds)  # (B,H,d_model)
                    pos = self.future_pos_emb(
                        torch.arange(int(H), device=past_seq.device, dtype=torch.long)
                    ).unsqueeze(0)
                    q = q + pos

                    # Predicted EOL scalar from past (for fusion + optional supervision)
                    z0 = enc_seq[:, -1, :]  # (B,d_model)
                    eol_scalar = torch.sigmoid(self.eol_scalar_head(z0)) if self.eol_scalar_head is not None else None

                    if self.use_eol_fusion and eol_scalar is not None:
                        if self.eol_fusion_mode == "token":
                            eol_tok = self.eol_ctx_to_token(eol_scalar).unsqueeze(1)  # (B,1,d_model)
                            q_full = torch.cat([eol_tok, q], dim=1)
                            dec = self.latent_transformer_decoder(tgt=q_full, memory=enc_seq)  # (B,1+H,d_model)
                            z_future_seq = dec[:, 1:, :]  # (B,H,d_model)
                        else:
                            eol_feat = self.eol_ctx_to_feature(eol_scalar).unsqueeze(1)  # (B,1,d_model)
                            q = q + eol_feat
                            z_future_seq = self.latent_transformer_decoder(tgt=q, memory=enc_seq)  # (B,H,d_model)
                    else:
                        z_future_seq = self.latent_transformer_decoder(tgt=q, memory=enc_seq)  # (B,H,d_model)

                    raw_delta_hi = self.latent_to_delta_hi(z_future_seq).squeeze(-1) if self.latent_to_delta_hi is not None else self.hi_head(z_future_seq).squeeze(-1)
                    raw_delta_rul = self.latent_to_delta_rul(z_future_seq).squeeze(-1) if self.latent_to_delta_rul is not None else self.rul_head(z_future_seq).squeeze(-1)
                else:
                    # Build full GRU decoder input sequence [B, H, latent_dec_input_dim]
                    z_ctx_rep = z_ctx.unsqueeze(1).repeat(1, H, 1)  # [B,H,3D]
                    inputs = [z_ctx_rep]

                    if self.use_future_conds:
                        inputs.append(future_conds)  # [B,H,cond_dim]

                    if self.use_hi_anchor and hi_anchor_vec is not None:
                        hi_rep = hi_anchor_vec.unsqueeze(1).expand(B_enc, H, 1)  # [B,H,1]
                        inputs.append(hi_rep)

                    if self.use_hi_anchor and rul_anchor_vec is not None:
                        # We overload the same flag to include RUL anchor as well (delta mode only).
                        rul_rep = rul_anchor_vec.unsqueeze(1).expand(B_enc, H, 1)  # [B,H,1]
                        inputs.append(rul_rep)

                    decoder_in = torch.cat(inputs, dim=-1)  # [B,H,input_dim]

                    # GRU decoding with zero initial hidden state
                    dec_out, _ = self.latent_decoder_rnn(decoder_in)  # [B,H,d_dec]
                    dec_out = self.dropout(dec_out)

                    # Delta predictions (GRU hidden -> delta)
                    raw_delta_hi = self.hi_head(dec_out).squeeze(-1)   # [B,H]
                    raw_delta_rul = self.rul_head(dec_out).squeeze(-1) # [B,H]

                delta_hi = -F.softplus(raw_delta_hi)   # (B,H) <= 0
                delta_rul = -F.softplus(raw_delta_rul) # (B,H) <= 0

                # Accumulate from anchors
                hi_future = hi_anchor_vec + torch.cumsum(delta_hi, dim=1)  # (B,H)
                rul_future = rul_anchor_vec + torch.cumsum(delta_rul, dim=1)  # (B,H)

                # Clamp to valid ranges: HI ∈ [0,1], normalized RUL ∈ [0,1]
                hi_future = torch.clamp(hi_future, 0.0, 1.0)
                rul_future = torch.clamp(rul_future, 0.0, 1.0)

                pred_sensors = past_seq.new_zeros(B, H, self.num_sensors_out)
                pred_hi = hi_future.unsqueeze(-1)   # [B,H,1]
                pred_rul = rul_future.unsqueeze(-1) # [B,H,1]

                # Optional EOL scalar prediction: use first future step as EOL proxy.
                pred_eol = None
                if "eol_scalar" in locals() and eol_scalar is not None:
                    pred_eol = eol_scalar  # (B,1) in [0,1]
                return pred_sensors, pred_hi, pred_rul, pred_eol

        # ------------------------------------------------------------------
        # Original sensor-autoregressive world model path
        # ------------------------------------------------------------------
        # 1) Encode past sequence
        # ------------------------------------------------------------------
        # Preferred path: encoder exposes an `encode` method returning
        # (enc_seq, pooled). This keeps us independent of the encoder's
        # EOL/HI heads.
        if hasattr(self.encoder, "encode"):
            # Some encoders may require cond_ids for condition embeddings.
            if cond_ids is not None:
                enc_seq, enc_pooled = self.encoder.encode(past_seq, cond_ids=cond_ids, return_seq=True)
            else:
                enc_seq, enc_pooled = self.encoder.encode(past_seq, cond_ids=None, return_seq=True)
        else:
            # Fallback: call encoder and assume it returns sequence features
            # (B, T_in, d_model). Pool in this module.
            if cond_ids is not None:
                enc_out = self.encoder(past_seq, cond_ids=cond_ids)
            else:
                enc_out = self.encoder(past_seq)

            # If encoder returns a tuple (rul, hi_last, hi_seq), we can't extract
            # internal states; in that case, we treat the EOL features as latent.
            if isinstance(enc_out, tuple):
                # enc_out[2] may be HI sequence [B, T, 1] – use it as proxy.
                _, _, hi_seq = enc_out
                if hi_seq.dim() == 3:
                    enc_seq = hi_seq  # [B, T, 1]
                else:
                    enc_seq = hi_seq.unsqueeze(-1)
            else:
                enc_seq = enc_out  # [B, T, d_model]

            enc_pooled = self.pool(enc_seq)

        # ------------------------------------------------------------------
        # 2) Initialize decoder hidden state from latent + condition (+ RUL/HI)
        # ------------------------------------------------------------------
        cond_vec_flat = cond_vec.view(B, -1)
        if cond_vec_flat.shape[1] != self.cond_dim:
            raise ValueError(
                f"[TransformerWorldModelV1] Expected cond_dim={self.cond_dim}, "
                f"got {cond_vec_flat.shape[1]}"
            )

        base_init_input = torch.cat([enc_pooled, cond_vec_flat], dim=-1)

        if self.init_from_rul_hi and (current_rul is not None) and (current_hi is not None):
            # Ensure shape (B, 1) for both scalars
            if current_rul.dim() == 1:
                current_rul = current_rul.unsqueeze(-1)
            if current_hi.dim() == 1:
                current_hi = current_hi.unsqueeze(-1)
            init_input = torch.cat([base_init_input, current_rul, current_hi], dim=-1)
            h0_flat = torch.tanh(self.decoder_init_from_latent_rul_hi(init_input))  # (B, d_dec)
        else:
            h0_flat = torch.tanh(self.decoder_init_from_latent(base_init_input))  # (B, d_dec)

        # GRU expects (num_layers, B, hidden_dim)
        h0 = h0_flat.unsqueeze(0).repeat(self.num_layers_decoder, 1, 1)  # (num_layers, B, d_dec)

        # ------------------------------------------------------------------
        # 3) Autoregressive decoding
        # ------------------------------------------------------------------
        if teacher_forcing_targets is not None:
            T_out = teacher_forcing_targets.size(1)
            if T_out != future_horizon:
                raise ValueError(
                    "[TransformerWorldModelV1] teacher_forcing_targets length "
                    f"{T_out} must match future_horizon={future_horizon}"
                )
        else:
            T_out = future_horizon

        # Start token: use last observed sensors as initial decoder input.
        # Convention: the first `num_sensors_out` features in past_seq correspond
        # to sensor outputs we want to forecast.
        prev_sensors = past_seq[:, -1, : self.num_sensors_out]  # (B, num_sensors_out)

        preds_sensors = []
        preds_hi = []
        preds_rul = []

        cond_expanded = cond_vec_flat  # (B, cond_dim)
        h_t = h0  # (1, B, d_dec)

        for t in range(T_out):
            # Build decoder input at this step
            dec_input_t = torch.cat([prev_sensors, cond_expanded], dim=-1).unsqueeze(1)  # (B, 1, dec_input_dim)

            dec_out_t, h_t = self.decoder_rnn(dec_input_t, h_t)  # dec_out_t: (B, 1, d_dec)
            dec_out_t = self.dropout(dec_out_t)
            hid_t = dec_out_t[:, -1, :]  # (B, d_dec)

            # Predict sensors
            sensors_t = self.sensor_head(hid_t)  # (B, num_sensors_out)
            preds_sensors.append(sensors_t.unsqueeze(1))

            # Predict HI / RUL if enabled
            if self.predict_hi and self.hi_head is not None:
                hi_t = self.hi_head(hid_t)
                preds_hi.append(hi_t.unsqueeze(1))
            if self.predict_rul and self.rul_head is not None:
                rul_t = self.rul_head(hid_t)
                preds_rul.append(rul_t.unsqueeze(1))

            # Teacher forcing or autoregressive update
            if teacher_forcing_targets is not None:
                prev_sensors = teacher_forcing_targets[:, t, :]
            else:
                prev_sensors = sensors_t

        pred_sensors = torch.cat(preds_sensors, dim=1)  # (B, T_out, num_sensors_out)

        pred_hi = None
        if self.predict_hi and len(preds_hi) > 0:
            pred_hi = torch.cat(preds_hi, dim=1)  # (B, T_out, 1)

        pred_rul = None
        if self.predict_rul and len(preds_rul) > 0:
            pred_rul = torch.cat(preds_rul, dim=1)  # (B, T_out, 1)

        return pred_sensors, pred_hi, pred_rul, None


