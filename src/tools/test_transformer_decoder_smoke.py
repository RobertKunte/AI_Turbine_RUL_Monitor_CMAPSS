"""
Smoke tests for Transformer AR Decoder.

Lightweight tests to verify:
- Model instantiation
- Forward pass shapes
- Causal masking
- No NaNs
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.decoders.transformer_ar_decoder import TransformerARDecoder


def test_transformer_decoder_self_attn():
    """Test self-attention only variant."""
    print("Testing TransformerARDecoder (self-attention only)...")
    
    B = 4
    H = 30
    d_model = 128
    
    # Dummy encoder token
    enc_token = torch.randn(B, d_model)
    
    # Dummy teacher (training)
    y_teacher = torch.randn(B, H, 1)
    
    # Instantiate decoder
    decoder = TransformerARDecoder(
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.1,
        horizon=H,
        use_cross_attention=False,
    )
    
    # Forward pass: teacher forcing
    y_hat_train = decoder(
        enc_token=enc_token,
        y_teacher=y_teacher,
        mode="train",
    )
    
    assert y_hat_train.shape == (B, H, 1), f"Expected (B, H, 1), got {y_hat_train.shape}"
    assert torch.isfinite(y_hat_train).all(), "Output contains NaN or Inf"
    print(f"  [OK] Teacher forcing: output shape {y_hat_train.shape}, all finite")
    
    # Forward pass: inference
    y_hat_inf = decoder(
        enc_token=enc_token,
        y_teacher=None,
        mode="inference",
    )
    
    assert y_hat_inf.shape == (B, H, 1), f"Expected (B, H, 1), got {y_hat_inf.shape}"
    assert torch.isfinite(y_hat_inf).all(), "Output contains NaN or Inf"
    print(f"  [OK] Inference: output shape {y_hat_inf.shape}, all finite")
    
    print("  [OK] Self-attention variant passed\n")


def test_transformer_decoder_cross_attn():
    """Test cross-attention variant."""
    print("Testing TransformerARDecoder (cross-attention)...")
    
    B = 4
    H = 30
    T_past = 30
    d_model = 128
    
    # Dummy encoder token
    enc_token = torch.randn(B, d_model)
    
    # Dummy encoder sequence (for cross-attention)
    enc_seq = torch.randn(B, T_past, d_model)
    
    # Dummy teacher (training)
    y_teacher = torch.randn(B, H, 1)
    
    # Instantiate decoder
    decoder = TransformerARDecoder(
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.1,
        horizon=H,
        use_cross_attention=True,
    )
    
    # Forward pass: teacher forcing with encoder sequence
    y_hat_train = decoder(
        enc_token=enc_token,
        y_teacher=y_teacher,
        enc_seq=enc_seq,
        mode="train",
    )
    
    assert y_hat_train.shape == (B, H, 1), f"Expected (B, H, 1), got {y_hat_train.shape}"
    assert torch.isfinite(y_hat_train).all(), "Output contains NaN or Inf"
    print(f"  [OK] Teacher forcing (with enc_seq): output shape {y_hat_train.shape}, all finite")
    
    # Forward pass: inference (fallback to enc_token)
    y_hat_inf = decoder(
        enc_token=enc_token,
        y_teacher=None,
        enc_seq=None,  # Will fallback to enc_token
        mode="inference",
    )
    
    assert y_hat_inf.shape == (B, H, 1), f"Expected (B, H, 1), got {y_hat_inf.shape}"
    assert torch.isfinite(y_hat_inf).all(), "Output contains NaN or Inf"
    print(f"  [OK] Inference (enc_token fallback): output shape {y_hat_inf.shape}, all finite")
    
    print("  [OK] Cross-attention variant passed\n")


def test_causal_mask():
    """Test that causal mask is correctly applied."""
    print("Testing causal mask creation...")
    
    decoder = TransformerARDecoder(
        d_model=128,
        nhead=4,
        num_layers=1,
        horizon=30,
        use_cross_attention=False,
    )
    
    seq_len = 10
    mask = decoder._create_causal_mask(seq_len, torch.device("cpu"))
    
    # Check upper-triangular structure
    assert mask.shape == (seq_len, seq_len), f"Expected ({seq_len}, {seq_len}), got {mask.shape}"
    
    # Lower triangle (including diagonal) should be 0
    for i in range(seq_len):
        for j in range(i + 1):
            assert mask[i, j] == 0.0, f"Lower triangle should be 0, got {mask[i, j]} at ({i}, {j})"
    
    # Upper triangle should be -inf
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert mask[i, j] == float('-inf'), f"Upper triangle should be -inf, got {mask[i, j]} at ({i}, {j})"
    
    print(f"  [OK] Causal mask shape: {mask.shape}")
    print(f"  [OK] Lower triangle (including diagonal): all zeros")
    print(f"  [OK] Upper triangle: all -inf")
    print("  [OK] Causal mask test passed\n")


def test_world_model_integration():
    """Test integration with WorldModelUniversalV3 (minimal, no dataset)."""
    print("Testing WorldModelUniversalV3 integration...")
    
    try:
        from src.models.world_model import WorldModelUniversalV3
        
        B = 2
        input_dim = 659
        past_len = 30
        horizon = 30
        
        # Instantiate with LSTM decoder (default)
        model_lstm = WorldModelUniversalV3(
            input_size=input_dim,
            d_model=96,
            num_layers=3,
            nhead=4,
            decoder_num_layers=1,
            horizon=horizon,
            decoder_type="lstm",
        )
        
        # Dummy inputs
        encoder_inputs = torch.randn(B, past_len, input_dim)
        decoder_targets = torch.randn(B, horizon, 1)
        
        # Forward pass
        out_lstm = model_lstm(
            encoder_inputs=encoder_inputs,
            decoder_targets=decoder_targets,
            teacher_forcing_ratio=0.5,
        )
        
        assert "traj" in out_lstm, "Missing 'traj' key"
        assert "eol" in out_lstm, "Missing 'eol' key"
        assert "hi" in out_lstm, "Missing 'hi' key"
        assert out_lstm["traj"].shape == (B, horizon, 1), f"Expected traj shape (B, H, 1), got {out_lstm['traj'].shape}"
        assert torch.isfinite(out_lstm["traj"]).all(), "LSTM decoder output contains NaN/Inf"
        print(f"  [OK] LSTM decoder: traj shape {out_lstm['traj'].shape}, all finite")
        
        # Instantiate with Transformer AR decoder (self-attention)
        model_tf = WorldModelUniversalV3(
            input_size=input_dim,
            d_model=96,
            num_layers=3,
            nhead=4,
            decoder_num_layers=1,
            horizon=horizon,
            decoder_type="tf_ar",
        )
        
        # Forward pass
        out_tf = model_tf(
            encoder_inputs=encoder_inputs,
            decoder_targets=decoder_targets,
            teacher_forcing_ratio=0.5,
        )
        
        assert "traj" in out_tf, "Missing 'traj' key"
        assert out_tf["traj"].shape == (B, horizon, 1), f"Expected traj shape (B, H, 1), got {out_tf['traj'].shape}"
        assert torch.isfinite(out_tf["traj"]).all(), "Transformer decoder output contains NaN/Inf"
        print(f"  [OK] Transformer AR decoder: traj shape {out_tf['traj'].shape}, all finite")
        
        # Instantiate with Transformer AR decoder (cross-attention)
        model_tf_xattn = WorldModelUniversalV3(
            input_size=input_dim,
            d_model=96,
            num_layers=3,
            nhead=4,
            decoder_num_layers=1,
            horizon=horizon,
            decoder_type="tf_ar_xattn",
        )
        
        # Forward pass
        out_tf_xattn = model_tf_xattn(
            encoder_inputs=encoder_inputs,
            decoder_targets=decoder_targets,
            teacher_forcing_ratio=0.5,
        )
        
        assert "traj" in out_tf_xattn, "Missing 'traj' key"
        assert out_tf_xattn["traj"].shape == (B, horizon, 1), f"Expected traj shape (B, H, 1), got {out_tf_xattn['traj'].shape}"
        assert torch.isfinite(out_tf_xattn["traj"]).all(), "Transformer decoder (cross-attn) output contains NaN/Inf"
        print(f"  [OK] Transformer AR decoder (cross-attn): traj shape {out_tf_xattn['traj'].shape}, all finite")
        
        print("  [OK] World Model integration passed\n")
        
    except Exception as e:
        print(f"  [FAIL] World Model integration failed: {e}")
        raise


if __name__ == "__main__":
    print("=" * 80)
    print("Transformer AR Decoder Smoke Tests")
    print("=" * 80)
    print()
    
    try:
        test_transformer_decoder_self_attn()
        test_transformer_decoder_cross_attn()
        test_causal_mask()
        test_world_model_integration()
        
        print("=" * 80)
        print("All smoke tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

