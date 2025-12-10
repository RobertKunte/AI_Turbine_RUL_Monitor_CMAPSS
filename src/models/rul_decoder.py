import torch
import torch.nn as nn

class RULTrajectoryDecoderV1(nn.Module):
    """
    Simple RUL trajectory decoder on top of frozen encoder + HI sequences.

    Inputs:
      - z_seq:        [B, T, D]  latent encoder sequence from EOLFullTransformerEncoder
      - hi_phys_seq:  [B, T]     physics HI_phys_v3 (continuous, monotone decreasing)
      - hi_damage_seq:[B, T]     learned damage HI (v3d) from CumulativeDamageHead
    Output:
      - rul_seq_pred: [B, T]     predicted RUL trajectory over the same T timesteps
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # project HIs to small embedding
        hi_input_dim = 2  # [hi_phys, hi_damage]
        self.hi_proj = nn.Linear(hi_input_dim, latent_dim)

        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, z_seq, hi_phys_seq, hi_damage_seq):
        # z_seq: [B, T, D]
        # hi_*:  [B, T]
        
        # Handle missing hi_phys_seq by using zeros if None
        if hi_phys_seq is None:
            hi_phys_seq = torch.zeros_like(hi_damage_seq)
            
        hi = torch.stack([hi_phys_seq, hi_damage_seq], dim=-1)  # [B, T, 2]
        hi_emb = torch.relu(self.hi_proj(hi))                   # [B, T, D]
        x = torch.cat([z_seq, hi_emb], dim=-1)                  # [B, T, 2D]
        x = torch.relu(self.input_proj(x))                      # [B, T, H]
        out_seq, _ = self.gru(x)                                # [B, T, H]
        rul_seq = self.out(out_seq).squeeze(-1)                 # [B, T]
        return rul_seq


class DecoderV1Wrapper(nn.Module):
    """
    Wrapper that combines a frozen EOLFullTransformerEncoder and a RULTrajectoryDecoderV1.
    Provides a standard forward(x, cond_ids=...) interface for evaluation/diagnostics.
    """
    def __init__(self, encoder: nn.Module, decoder: RULTrajectoryDecoderV1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Expose attributes expected by diagnostics/evaluation tools
        self.d_model = getattr(encoder, "d_model", 64)
        
    def forward(self, x, cond_ids=None, cond_vec=None):
        """
        Forward pass for inference/evaluation.
        Returns:
            rul_pred: [B, 1] predicted RUL at the last timestep (EOL prediction)
            hi_last:  Dummy (or last HI if available) [B, 1]
            hi_seq:   Dummy (or predicted HI seq) [B, T]
        
        Note: The standard evaluation interface expects (rul_pred, hi_last, hi_seq).
        """
        # Encoder forward (frozen)
        # We assume self.encoder has encode_with_hi (added in previous step)
        if hasattr(self.encoder, "encode_with_hi"):
            z_seq, hi_phys_seq, hi_damage_seq = self.encoder.encode_with_hi(x, cond_ids=cond_ids, cond_vec=cond_vec)
        else:
            # Fallback if method missing (should not happen if encoder updated)
            z_seq = self.encoder(x, cond_ids=cond_ids)
            hi_phys_seq = None
            hi_damage_seq = None
            if hasattr(self.encoder, "damage_head") and self.encoder.damage_head is not None:
                 # Try to extract damage HI manually
                 pass # Simplified for now
        
        # Decoder forward
        rul_seq = self.decoder(z_seq, hi_phys_seq, hi_damage_seq) # [B, T]
        
        # For compatibility with evaluate_on_test_data which expects:
        # rul_pred (B, 1), hi_last (B, 1), hi_seq (B, T)
        rul_pred = rul_seq[:, -1].unsqueeze(-1)
        
        # We can use hi_damage_seq as the "Health Index" for plotting if available
        if hi_damage_seq is not None:
            hi_seq = hi_damage_seq
            hi_last = hi_seq[:, -1].unsqueeze(-1)
        else:
            hi_seq = torch.zeros_like(rul_seq)
            hi_last = torch.zeros_like(rul_pred)
            
        return rul_pred, hi_last, hi_seq


