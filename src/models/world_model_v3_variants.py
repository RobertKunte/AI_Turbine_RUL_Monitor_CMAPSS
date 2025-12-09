"""
World Model v3 Architecture Variants for Ablation Study.

This module implements six architecture variants (A1-A6) for systematic ablation:
- A1: Full Model (Traj + EOL + HI + Monotonicity)
- A2: No Trajectory Head
- A3: No HI Head
- A4: No Monotonicity
- A5: EOL-Only
- A6: Traj-Only
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from .universal_encoder_v1 import UniversalEncoderV2


class WorldModelV3A1Full(nn.Module):
    """
    A1 - Full Model: Traj + EOL + HI + Monotonicity
    This is the baseline complete architecture.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 96,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        num_conditions: Optional[int] = None,
        cond_emb_dim: int = 4,
        kernel_sizes: List[int] = None,
        seq_encoder_type: str = "transformer",
        use_layer_norm: bool = True,
        max_seq_len: int = 300,
        decoder_hidden_size: Optional[int] = None,
        decoder_num_layers: int = 2,
        horizon: int = 40,
        fusion_mode: str = "late",  # "early" or "late"
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9]
        
        if decoder_hidden_size is None:
            decoder_hidden_size = d_model
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_conditions = num_conditions
        self.decoder_num_layers = decoder_num_layers
        self.horizon = horizon
        self.fusion_mode = fusion_mode
        
        # Encoder: UniversalEncoderV2 (with optional early fusion)
        if fusion_mode == "early":
            # For early fusion, we need to modify input_dim to include cond_emb_dim
            # But we'll handle this in forward pass
            encoder_input_dim = input_size
        else:
            encoder_input_dim = input_size
        
        self.encoder = UniversalEncoderV2(
            input_dim=encoder_input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions if fusion_mode == "late" else None,  # Only late fusion uses encoder's cond fusion
            cond_emb_dim=cond_emb_dim,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=use_layer_norm,
            max_seq_len=max_seq_len,
        )
        
        # For early fusion, we need our own condition embedding
        if fusion_mode == "early" and num_conditions and num_conditions > 1:
            self.cond_emb = nn.Embedding(num_conditions, cond_emb_dim)
            # Project concatenated features to input_dim
            self.early_fusion_proj = nn.Linear(input_size + cond_emb_dim, input_size)
        else:
            self.cond_emb = None
            self.early_fusion_proj = None
        
        # Decoder: LSTM for trajectory prediction
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            batch_first=True,
            dropout=dropout if decoder_num_layers > 1 else 0.0,
        )
        
        # Project encoder output to decoder initial hidden state
        self.encoder_to_decoder_h = nn.Linear(d_model, decoder_hidden_size * decoder_num_layers)
        self.encoder_to_decoder_c = nn.Linear(d_model, decoder_hidden_size * decoder_num_layers)
        
        # Shared head for EOL and HI
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Trajectory-Head
        self.traj_head = nn.Linear(decoder_hidden_size, 1)
        
        # EOL-Head
        self.fc_rul = nn.Linear(d_model, 1)
        
        # HI-Head
        self.fc_health = nn.Linear(d_model, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for A1 (Full Model)."""
        B = encoder_inputs.size(0)
        
        # Early fusion: concatenate condition embedding with features before CNN
        if self.fusion_mode == "early" and self.cond_emb is not None and cond_ids is not None:
            c = self.cond_emb(cond_ids)  # [B, cond_emb_dim]
            c_expanded = c.unsqueeze(1).expand(-1, encoder_inputs.size(1), -1)  # [B, T, cond_emb_dim]
            encoder_inputs = torch.cat([encoder_inputs, c_expanded], dim=-1)  # [B, T, F + cond_emb_dim]
            encoder_inputs = self.early_fusion_proj(encoder_inputs)  # [B, T, F]
            cond_ids = None  # Don't pass to encoder (already fused)
        
        # Encoder
        enc_emb = self.encoder(encoder_inputs, cond_ids=cond_ids)  # [B, d_model]
        
        # Shared features
        h_shared = self.shared_head(enc_emb)  # [B, d_model]
        
        # EOL prediction
        eol_pred = self.fc_rul(h_shared)  # [B, 1]
        
        # HI prediction
        hi_logit = self.fc_health(h_shared)  # [B, 1]
        hi_pred = torch.sigmoid(hi_logit)  # [B, 1]
        
        # Decoder setup
        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        elif horizon is None:
            horizon = self.horizon
        
        h_0 = self.encoder_to_decoder_h(enc_emb)
        c_0 = self.encoder_to_decoder_c(enc_emb)
        h_0 = h_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        c_0 = c_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        dec_hidden = (h_0, c_0)
        
        # Initial decoder input
        if decoder_targets is not None:
            dec_input = decoder_targets[:, 0:1, :]
        else:
            dec_input = hi_pred.unsqueeze(1)
        
        # Decoder loop
        traj_outputs = []
        for t in range(horizon):
            dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
            step_pred = self.traj_head(dec_out)
            traj_outputs.append(step_pred)
            
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                if t + 1 < decoder_targets.size(1):
                    dec_input = decoder_targets[:, t + 1 : t + 2, :]
                else:
                    dec_input = step_pred
            else:
                dec_input = step_pred
        
        traj_outputs = torch.cat(traj_outputs, dim=1)  # [B, H, 1]
        
        return {
            "traj": traj_outputs,
            "eol": eol_pred,
            "hi": hi_pred,
        }


class WorldModelV3A2NoTraj(WorldModelV3A1Full):
    """
    A2 - No Trajectory Head: Remove decoder + traj loss
    Only EOL + HI heads remain.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove decoder components
        self.decoder = None
        self.encoder_to_decoder_h = None
        self.encoder_to_decoder_c = None
        self.traj_head = None
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for A2 (No Trajectory)."""
        B = encoder_inputs.size(0)
        
        # Early fusion
        if self.fusion_mode == "early" and self.cond_emb is not None and cond_ids is not None:
            c = self.cond_emb(cond_ids)
            c_expanded = c.unsqueeze(1).expand(-1, encoder_inputs.size(1), -1)
            encoder_inputs = torch.cat([encoder_inputs, c_expanded], dim=-1)
            encoder_inputs = self.early_fusion_proj(encoder_inputs)
            cond_ids = None
        
        # Encoder
        enc_emb = self.encoder(encoder_inputs, cond_ids=cond_ids)
        h_shared = self.shared_head(enc_emb)
        
        # EOL and HI only
        eol_pred = self.fc_rul(h_shared)
        hi_logit = self.fc_health(h_shared)
        hi_pred = torch.sigmoid(hi_logit)
        
        # Return empty trajectory (for compatibility)
        traj_outputs = torch.zeros(B, self.horizon, 1, device=encoder_inputs.device)
        
        return {
            "traj": traj_outputs,
            "eol": eol_pred,
            "hi": hi_pred,
        }


class WorldModelV3A3NoHI(WorldModelV3A1Full):
    """
    A3 - No HI Head: Remove HI head + HI loss
    Only Traj + EOL remain.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc_health = None
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for A3 (No HI)."""
        B = encoder_inputs.size(0)
        
        # Early fusion
        if self.fusion_mode == "early" and self.cond_emb is not None and cond_ids is not None:
            c = self.cond_emb(cond_ids)
            c_expanded = c.unsqueeze(1).expand(-1, encoder_inputs.size(1), -1)
            encoder_inputs = torch.cat([encoder_inputs, c_expanded], dim=-1)
            encoder_inputs = self.early_fusion_proj(encoder_inputs)
            cond_ids = None
        
        # Encoder
        enc_emb = self.encoder(encoder_inputs, cond_ids=cond_ids)
        h_shared = self.shared_head(enc_emb)
        
        # EOL prediction
        eol_pred = self.fc_rul(h_shared)
        
        # Trajectory (use EOL as proxy for initial input)
        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        elif horizon is None:
            horizon = self.horizon
        
        h_0 = self.encoder_to_decoder_h(enc_emb)
        c_0 = self.encoder_to_decoder_c(enc_emb)
        h_0 = h_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        c_0 = c_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        dec_hidden = (h_0, c_0)
        
        if decoder_targets is not None:
            dec_input = decoder_targets[:, 0:1, :]
        else:
            dec_input = eol_pred.unsqueeze(1)
        
        traj_outputs = []
        for t in range(horizon):
            dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
            step_pred = self.traj_head(dec_out)
            traj_outputs.append(step_pred)
            
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                if t + 1 < decoder_targets.size(1):
                    dec_input = decoder_targets[:, t + 1 : t + 2, :]
                else:
                    dec_input = step_pred
            else:
                dec_input = step_pred
        
        traj_outputs = torch.cat(traj_outputs, dim=1)
        
        # Return zero HI (for compatibility)
        hi_pred = torch.zeros(B, 1, device=encoder_inputs.device)
        
        return {
            "traj": traj_outputs,
            "eol": eol_pred,
            "hi": hi_pred,
        }


class WorldModelV3A4NoMono(WorldModelV3A1Full):
    """
    A4 - No Monotonicity: Set mono weights=0
    Architecture unchanged, but monotonicity loss will be disabled in training.
    """
    pass  # Same as A1, but training will set mono weights to 0


class WorldModelV3A5EOLOnly(nn.Module):
    """
    A5 - EOL-Only: No decoder, pure regressor
    Only EOL head, no trajectory, no HI.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 96,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        num_conditions: Optional[int] = None,
        cond_emb_dim: int = 4,
        kernel_sizes: List[int] = None,
        seq_encoder_type: str = "transformer",
        use_layer_norm: bool = True,
        max_seq_len: int = 300,
        horizon: int = 40,
        fusion_mode: str = "late",
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9]
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_conditions = num_conditions
        self.horizon = horizon
        self.fusion_mode = fusion_mode
        
        # Encoder
        encoder_input_dim = input_size
        self.encoder = UniversalEncoderV2(
            input_dim=encoder_input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions if fusion_mode == "late" else None,
            cond_emb_dim=cond_emb_dim,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=use_layer_norm,
            max_seq_len=max_seq_len,
        )
        
        # Early fusion
        if fusion_mode == "early" and num_conditions and num_conditions > 1:
            self.cond_emb = nn.Embedding(num_conditions, cond_emb_dim)
            self.early_fusion_proj = nn.Linear(input_size + cond_emb_dim, input_size)
        else:
            self.cond_emb = None
            self.early_fusion_proj = None
        
        # EOL head only
        self.fc_rul = nn.Linear(d_model, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for A5 (EOL-Only)."""
        B = encoder_inputs.size(0)
        
        # Early fusion
        if self.fusion_mode == "early" and self.cond_emb is not None and cond_ids is not None:
            c = self.cond_emb(cond_ids)
            c_expanded = c.unsqueeze(1).expand(-1, encoder_inputs.size(1), -1)
            encoder_inputs = torch.cat([encoder_inputs, c_expanded], dim=-1)
            encoder_inputs = self.early_fusion_proj(encoder_inputs)
            cond_ids = None
        
        # Encoder
        enc_emb = self.encoder(encoder_inputs, cond_ids=cond_ids)
        
        # EOL only
        eol_pred = self.fc_rul(enc_emb)
        
        # Return zeros for traj and hi (for compatibility)
        traj_outputs = torch.zeros(B, self.horizon, 1, device=encoder_inputs.device)
        hi_pred = torch.zeros(B, 1, device=encoder_inputs.device)
        
        return {
            "traj": traj_outputs,
            "eol": eol_pred,
            "hi": hi_pred,
        }


class WorldModelV3A6TrajOnly(WorldModelV3A1Full):
    """
    A6 - Traj-Only: No EOL, no HI, only trajectory
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc_rul = None
        self.fc_health = None
        self.shared_head = None
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
        cond_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for A6 (Traj-Only)."""
        B = encoder_inputs.size(0)
        
        # Early fusion
        if self.fusion_mode == "early" and self.cond_emb is not None and cond_ids is not None:
            c = self.cond_emb(cond_ids)
            c_expanded = c.unsqueeze(1).expand(-1, encoder_inputs.size(1), -1)
            encoder_inputs = torch.cat([encoder_inputs, c_expanded], dim=-1)
            encoder_inputs = self.early_fusion_proj(encoder_inputs)
            cond_ids = None
        
        # Encoder
        enc_emb = self.encoder(encoder_inputs, cond_ids=cond_ids)
        
        # Trajectory only
        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        elif horizon is None:
            horizon = self.horizon
        
        h_0 = self.encoder_to_decoder_h(enc_emb)
        c_0 = self.encoder_to_decoder_c(enc_emb)
        h_0 = h_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        c_0 = c_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        dec_hidden = (h_0, c_0)
        
        # Use zero as initial input
        if decoder_targets is not None:
            dec_input = decoder_targets[:, 0:1, :]
        else:
            dec_input = torch.zeros(B, 1, 1, device=encoder_inputs.device)
        
        traj_outputs = []
        for t in range(horizon):
            dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
            step_pred = self.traj_head(dec_out)
            traj_outputs.append(step_pred)
            
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                if t + 1 < decoder_targets.size(1):
                    dec_input = decoder_targets[:, t + 1 : t + 2, :]
                else:
                    dec_input = step_pred
            else:
                dec_input = step_pred
        
        traj_outputs = torch.cat(traj_outputs, dim=1)
        
        # Return zeros for eol and hi (for compatibility)
        eol_pred = torch.zeros(B, 1, device=encoder_inputs.device)
        hi_pred = torch.zeros(B, 1, device=encoder_inputs.device)
        
        return {
            "traj": traj_outputs,
            "eol": eol_pred,
            "hi": hi_pred,
        }


# Variant registry
VARIANT_REGISTRY = {
    "A1": WorldModelV3A1Full,
    "A2": WorldModelV3A2NoTraj,
    "A3": WorldModelV3A3NoHI,
    "A4": WorldModelV3A4NoMono,
    "A5": WorldModelV3A5EOLOnly,
    "A6": WorldModelV3A6TrajOnly,
}


def create_world_model_v3_variant(
    variant: str,
    input_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a World Model v3 variant.
    
    Args:
        variant: Variant name ("A1", "A2", ..., "A6")
        input_size: Input feature dimension
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Model instance
    """
    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(VARIANT_REGISTRY.keys())}")
    
    model_class = VARIANT_REGISTRY[variant]
    return model_class(input_size=input_size, **kwargs)

