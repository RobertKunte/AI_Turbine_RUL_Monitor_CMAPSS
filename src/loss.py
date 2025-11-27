try:
    import torch  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the custom loss. Please install torch."
    ) from exc

def rul_asymmetric_weighted_loss(pred, target,
                                 over_factor=2.0,
                                 min_rul_weight=1.0,
                                 max_rul_weight=0.3):
    """
    Custom loss for RUL:
    - Overestimation (pred > target) is penalized stronger than underestimation.
    - Low RUL values are weighted higher than large RUL values.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    error = pred - target
    over  = torch.clamp(error, min=0.0)      # overestimation
    under = torch.clamp(-error, min=0.0)     # underestimation
    
    # Asymmetric penalty (like MSE but scaled for overestimation)
    base_loss = over_factor * over**2 + under**2
    
    # RUL-based weights: higher weight for small RUL
    t_norm = target / (target.max() + 1e-6)  # normalize to [0,1]
    # Map to [max_rul_weight, min_rul_weight]
    weights = max_rul_weight + (min_rul_weight - max_rul_weight) * (1.0 - t_norm)
    
    weighted_loss = weights * base_loss
    return weighted_loss.mean()