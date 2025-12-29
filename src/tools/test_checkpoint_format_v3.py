"""
Regression test for World Model v3 checkpoint format.

Verifies that checkpoints contain full model state_dict including heads.
"""

import tempfile
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.world_model import WorldModelUniversalV3


def test_checkpoint_format_v3():
    """Test that WM v3 checkpoint contains heads."""
    print("=" * 80)
    print("World Model v3 Checkpoint Format Test")
    print("=" * 80)
    print()
    
    # Create a minimal WM v3 model
    print("Creating minimal WorldModelUniversalV3...")
    model = WorldModelUniversalV3(
        input_size=10,  # Small input dim for test
        d_model=32,     # Small d_model for test
        num_layers=1,
        nhead=2,
        dim_feedforward=64,
        dropout=0.0,
        decoder_num_layers=1,
        horizon=5,
        decoder_type="lstm",  # Use LSTM for simplicity
    )
    
    # Get state dict
    state_dict = model.state_dict()
    num_keys = len(state_dict)
    print(f"Model state_dict keys: {num_keys}")
    
    # Check for head keys
    head_patterns = ["traj_head", "fc_rul", "fc_eol", "fc_health", "hi_head", "eol_head"]
    head_keys = [k for k in state_dict.keys() if any(p in k.lower() for p in head_patterns)]
    
    print(f"Head keys found: {len(head_keys)}")
    for key in head_keys:
        print(f"  - {key}")
    
    if len(head_keys) == 0:
        raise AssertionError("No head keys found in model state_dict!")
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_wm_v3_best.pt"
        
        checkpoint = {
            "model_state_dict": state_dict,
            "epoch": 0,
            "val_loss": 0.0,
        }
        
        torch.save(checkpoint, ckpt_path)
        print(f"\nSaved checkpoint to {ckpt_path}")
        
        # Load checkpoint
        print("Loading checkpoint...")
        loaded = torch.load(ckpt_path, map_location="cpu")
        
        # Verify format
        assert isinstance(loaded, dict), "Checkpoint should be a dict"
        assert "model_state_dict" in loaded, "Checkpoint should contain 'model_state_dict'"
        
        loaded_state_dict = loaded["model_state_dict"]
        loaded_head_keys = [k for k in loaded_state_dict.keys() if any(p in k.lower() for p in head_patterns)]
        
        print(f"Loaded checkpoint head keys: {len(loaded_head_keys)}")
        
        assert len(loaded_head_keys) > 0, "Loaded checkpoint missing head keys!"
        assert len(loaded_head_keys) == len(head_keys), f"Head key count mismatch: {len(loaded_head_keys)} != {len(head_keys)}"
        
        # Verify keys match
        for key in head_keys:
            assert key in loaded_state_dict, f"Missing key in loaded checkpoint: {key}"
        
        print("\n[OK] Checkpoint format test passed!")
        print(f"   - Saved {num_keys} keys")
        print(f"   - Saved {len(head_keys)} head keys")
        print(f"   - Loaded {len(loaded_state_dict)} keys")
        print(f"   - Loaded {len(loaded_head_keys)} head keys")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_checkpoint_format_v3()

