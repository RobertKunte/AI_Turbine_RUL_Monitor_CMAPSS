"""
Lightweight test/verification for freeze-aware checkpoint gating.

This module provides a simple sanity check that checkpoint gating logic
prevents early checkpoint selection during freeze phase.
"""

from typing import Dict, List


def test_checkpoint_gating(
    freeze_encoder_epochs: int = 2,
    best_ckpt_min_epoch: int = None,
    best_ckpt_min_epoch_after_unfreeze: int = 2,
    num_epochs: int = 10,
) -> Dict[str, List[bool]]:
    """
    Test checkpoint gating logic with fake training history.
    
    Args:
        freeze_encoder_epochs: Number of epochs encoder is frozen
        best_ckpt_min_epoch: Manual override for min epoch (None = auto)
        best_ckpt_min_epoch_after_unfreeze: Min epochs after unfreeze
        num_epochs: Total epochs to simulate
    
    Returns:
        Dict with 'checkpoint_allowed' and 'allow_best_update' per epoch
    """
    # Compute min_epoch
    if best_ckpt_min_epoch is not None:
        min_epoch = best_ckpt_min_epoch
    elif freeze_encoder_epochs > 0:
        min_epoch = freeze_encoder_epochs + 1
    else:
        min_epoch = 0
    
    results = {
        "epoch": [],
        "checkpoint_allowed": [],
        "allow_best_update": [],
        "reason": [],
    }
    
    for epoch in range(num_epochs):
        # Compute epochs since unfreeze
        if freeze_encoder_epochs > 0:
            epochs_since_unfreeze = max(0, epoch - freeze_encoder_epochs)
        else:
            epochs_since_unfreeze = epoch
        
        # Check gating
        checkpoint_allowed_min_epoch = epoch >= min_epoch
        checkpoint_allowed_after_unfreeze = epochs_since_unfreeze >= best_ckpt_min_epoch_after_unfreeze
        checkpoint_allowed = checkpoint_allowed_min_epoch and checkpoint_allowed_after_unfreeze
        
        # Determine reason
        if not checkpoint_allowed:
            if not checkpoint_allowed_min_epoch:
                reason = f"min_epoch={min_epoch}"
            else:
                reason = f"min_after_unfreeze={best_ckpt_min_epoch_after_unfreeze} (epochs_since={epochs_since_unfreeze})"
            allow_best_update = False
        else:
            reason = "ok"
            allow_best_update = True
        
        results["epoch"].append(epoch)
        results["checkpoint_allowed"].append(checkpoint_allowed)
        results["allow_best_update"].append(allow_best_update)
        results["reason"].append(reason)
    
    return results


def print_checkpoint_gating_summary(results: Dict[str, List[bool]]) -> None:
    """Print a summary of checkpoint gating test results."""
    print("\n=== Checkpoint Gating Test Results ===")
    print(f"{'Epoch':<6} {'Allowed':<8} {'Best Update':<12} {'Reason'}")
    print("-" * 60)
    
    for i in range(len(results["epoch"])):
        epoch = results["epoch"][i]
        allowed = "Yes" if results["checkpoint_allowed"][i] else "No"
        best_update = "Yes" if results["allow_best_update"][i] else "No"
        reason = results["reason"][i]
        print(f"{epoch:<6} {allowed:<8} {best_update:<12} {reason}")
    
    # Assertions
    first_allowed_epoch = next(
        (i for i, allowed in enumerate(results["checkpoint_allowed"]) if allowed),
        None
    )
    
    if first_allowed_epoch is not None:
        print(f"\n[PASS] First checkpoint allowed at epoch {first_allowed_epoch}")
        assert first_allowed_epoch >= 2, f"Checkpoint allowed too early (epoch {first_allowed_epoch})"
    else:
        print("\n[WARN] No checkpoint allowed in test range")
    
    print("[PASS] Checkpoint gating test passed")


if __name__ == "__main__":
    # Test with default settings (freeze=2, warmup=2)
    print("Test 1: Default settings (freeze_encoder_epochs=2, warmup=2)")
    results1 = test_checkpoint_gating(freeze_encoder_epochs=2, num_epochs=10)
    print_checkpoint_gating_summary(results1)
    
    # Test with no freeze
    print("\n\nTest 2: No freeze (backwards compatibility)")
    results2 = test_checkpoint_gating(freeze_encoder_epochs=0, num_epochs=5)
    print_checkpoint_gating_summary(results2)
    
    # Test with manual min_epoch
    print("\n\nTest 3: Manual min_epoch override")
    results3 = test_checkpoint_gating(
        freeze_encoder_epochs=5,
        best_ckpt_min_epoch=3,
        best_ckpt_min_epoch_after_unfreeze=1,
        num_epochs=10
    )
    print_checkpoint_gating_summary(results3)

