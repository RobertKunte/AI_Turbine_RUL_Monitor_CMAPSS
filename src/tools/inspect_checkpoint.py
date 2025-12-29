"""
Checkpoint inspection tool for debugging checkpoint format issues.

Usage:
    python -m src.tools.inspect_checkpoint --ckpt results/fd004/<run>/world_model_v3_best.pt --topk 40
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Any


def inspect_checkpoint(ckpt_path: Path, topk: int = 40) -> None:
    """
    Inspect checkpoint file and print diagnostic information.
    
    Args:
        ckpt_path: Path to checkpoint file
        topk: Number of keys to show in sample
    """
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint file not found: {ckpt_path}")
        return
    
    print("=" * 80)
    print(f"Checkpoint Inspection: {ckpt_path.name}")
    print("=" * 80)
    print()
    
    # Load checkpoint (force CPU to avoid GPU issues)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return
    
    # Inspect top-level structure
    print(f"Top-level type: {type(ckpt).__name__}")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        print()
        
        # Check if it's a wrapped format
        if "state_dict" in ckpt:
            print("Format: Wrapped dict with 'state_dict' key")
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            print("Format: Wrapped dict with 'model_state_dict' key")
            state_dict = ckpt["model_state_dict"]
        else:
            print("Format: Raw state_dict (or custom dict)")
            state_dict = ckpt
    else:
        print("Format: Raw state_dict (not a dict)")
        state_dict = ckpt
    
    # Inspect state_dict
    if not isinstance(state_dict, dict):
        print(f"ERROR: Expected dict, got {type(state_dict).__name__}")
        return
    
    num_keys = len(state_dict)
    print(f"State dict keys: {num_keys}")
    print()
    
    # Sample keys
    all_keys = list(state_dict.keys())
    print(f"First {min(topk, num_keys)} keys:")
    for i, key in enumerate(all_keys[:topk]):
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
        print(f"  {i+1:3d}. {key:<60} shape={shape}")
    if num_keys > topk:
        print(f"  ... ({num_keys - topk} more keys)")
    print()
    
    # Filter keys by head-related patterns
    head_patterns = ["traj_head", "eol_head", "hi_head", "fc_rul", "fc_health", "fc_eol"]
    head_keys = []
    for key in all_keys:
        for pattern in head_patterns:
            if pattern in key.lower():
                head_keys.append(key)
                break
    
    if head_keys:
        print(f"Head-related keys ({len(head_keys)}):")
        for key in head_keys:
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  - {key:<60} shape={shape}")
    else:
        print("Head-related keys: NONE FOUND")
    print()
    
    # Check for V3 heads
    has_traj_head = any("traj_head" in k.lower() for k in all_keys)
    has_fc_rul = any("fc_rul" in k.lower() for k in all_keys)
    has_fc_eol = any("fc_eol" in k.lower() for k in all_keys)
    has_fc_health = any("fc_health" in k.lower() for k in all_keys)
    has_hi_head = any("hi_head" in k.lower() for k in all_keys)
    has_eol_head = any("eol_head" in k.lower() for k in all_keys)
    
    has_v3_heads = (
        (has_traj_head or has_hi_head) and
        (has_fc_rul or has_fc_eol or has_eol_head) and
        (has_fc_health or has_hi_head)
    )
    
    print("V3 Head Detection:")
    print(f"  traj_head: {has_traj_head}")
    print(f"  fc_rul: {has_fc_rul}")
    print(f"  fc_eol: {has_fc_eol}")
    print(f"  eol_head: {has_eol_head}")
    print(f"  fc_health: {has_fc_health}")
    print(f"  hi_head: {has_hi_head}")
    print(f"  → has_v3_heads: {has_v3_heads}")
    print()
    
    # Check for encoder-only patterns
    encoder_only_patterns = ["encoder.", "universal_encoder"]
    encoder_keys = [k for k in all_keys if any(p in k.lower() for p in encoder_only_patterns)]
    decoder_keys = [k for k in all_keys if "decoder" in k.lower()]
    
    print("Component Detection:")
    print(f"  Encoder keys: {len(encoder_keys)}")
    print(f"  Decoder keys: {len(decoder_keys)}")
    if encoder_keys and not decoder_keys and not head_keys:
        print("  ⚠️  WARNING: Looks like encoder-only checkpoint!")
    print()
    
    # Show metadata if available
    if isinstance(ckpt, dict):
        meta_keys = [k for k in ckpt.keys() if k not in ["state_dict", "model_state_dict"]]
        if meta_keys:
            print("Metadata keys:")
            for key in meta_keys:
                value = ckpt[key]
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")
            print()
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect checkpoint file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--topk", type=int, default=40, help="Number of keys to show in sample")
    
    args = parser.parse_args()
    ckpt_path = Path(args.ckpt)
    inspect_checkpoint(ckpt_path, args.topk)

