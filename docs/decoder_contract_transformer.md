# Decoder Contract: Transformer Autoregressive Decoder

**Context**: World Model v3 outputs trajectory, EOL, and HI predictions. The decoder is responsible for generating the trajectory component.

## Inputs

**Required**:
- `enc_token`: `(B, d_model)` - Encoder output (single embedding vector from UniversalEncoderV2)
  - In this codebase, UniversalEncoderV2 returns `(B, d_model)`, not a sequence
  - This is the primary encoder context

**Optional**:
- `y_teacher`: `(B, H, 1)` - Ground truth trajectory for teacher forcing (training only)
- `enc_seq`: `(B, T_past, d_model)` or `None` - Encoder sequence (for cross-attention)
- `cond_ctx`: `(B, cond_emb_dim)` or `None` - Condition context (optional, reserved for future use)
- `mode`: `"train"` or `"inference"` - Operation mode

**Constants**:
- `past_len = 30` (encoder input sequence length)
- `horizon = 30` (decoder output sequence length)
- `d_model`: Encoder embedding dimension (typically 96)

## Outputs

- `y_hat_traj`: `(B, H, 1)` - Predicted trajectory (RUL sequence)

## Masking Rules

**Causal mask in decoder self-attention**:
- Upper-triangular mask: `mask[i, j] = -inf if j > i, else 0`
- Prevents future leakage: position `i` cannot attend to positions `j > i`
- Applied to decoder self-attention layers

**Horizon mask** (handled by loss, not decoder):
- Padded timesteps are masked out in loss computation (existing logic)
- Decoder produces full `(B, H, 1)` output; masking happens downstream

## Teacher Forcing

**Training** (`mode="train"`, `y_teacher` provided):
- Shift-right strategy: `y_teacher[:, :-1, :]` → prepend start token → `(B, H, d_model)`
- Start token: Learned parameter `(1, d_model)` expanded to batch
- Projection: `Linear(1, d_model)` maps RUL values to token embeddings
- Causal masking ensures no future leakage during training

**Inference** (`mode="inference"`, `y_teacher=None`):
- Autoregressive rollout: Start with `[start_token]`
- Iteratively predict next token, append to sequence, repeat for H steps
- Each step uses only previously predicted tokens (causal)

## Start Token Strategy

- Learned start token: `nn.Parameter(torch.randn(1, d_model))`
- Expanded to batch: `(B, d_model)`
- Optionally concatenated with encoder context token if needed

## Positional Encoding

- Learnable positional embeddings for max length `horizon + 2` (start + context + H steps)
- Applied to token sequence before transformer layers

---

## Integration Audit Results

**Last updated**: 2025-12-28  
**Status**: ✅ All smoke tests pass

### Smoke Test Results

All tests in `src/tools/test_transformer_decoder_smoke.py` pass:

- ✅ **Self-attention variant**: Teacher forcing and inference paths work correctly
- ✅ **Cross-attention variant**: Teacher forcing and inference paths work correctly
- ✅ **Causal mask**: Correctly creates upper-triangular mask (lower triangle + diagonal = 0, upper triangle = -inf)
- ✅ **World Model integration**: All three decoder types (`lstm`, `tf_ar`, `tf_ar_xattn`) instantiate and forward pass correctly

### Training Code Path Audit

**File**: `src/world_model_training_v3.py` (lines 430-460)

✅ **decoder_type reading**:
- Line 432: `decoder_type = str(getattr(world_model_config, "decoder_type", "lstm"))`
- Reads from `world_model_config.decoder_type` with default `"lstm"`
- Correctly handles missing config (defaults to LSTM)

✅ **Model instantiation**:
- Line 451: `decoder_type=decoder_type` passed to `WorldModelUniversalV3` constructor
- Parameter correctly forwarded

✅ **Logging**:
- Lines 454-460: Conditional logging based on `decoder_type`
- Shows correct decoder type in training header

### World Model Forward Pass Audit

**File**: `src/models/world_model.py` (lines 886-947)

✅ **LSTM path** (lines 886-928):
- Unchanged from original implementation
- Uses `self.decoder` (LSTM), `self.encoder_to_decoder_h/c`, `self.traj_head`
- Teacher forcing logic preserved

✅ **Transformer path** (lines 930-947):
- Correctly routes to `self.tf_decoder` when `decoder_type != "lstm"`
- Mode determination: `mode = "train" if decoder_targets is not None else "inference"`
- Returns `y_hat_traj` with shape `(B, H, 1)` ✅
- No double-processing: Single call to transformer decoder

✅ **Teacher forcing verification**:
- Shift-right strategy: `[start, y_0, y_1, ..., y_{H-2}]` (H tokens)
- Causal mask prevents future leakage
- Output predicts `[y_0, y_1, ..., y_{H-1}]` (H predictions) ✅

### Decoder Implementation Audit

**File**: `src/models/decoders/transformer_ar_decoder.py`

✅ **Teacher forcing path** (lines 140-151):
- Shift-right: Prepends start token, removes last teacher step
- Creates sequence: `[start, y_0, ..., y_{H-2}]` (H tokens)
- Applies transformer with causal mask
- Output head produces `(B, H, 1)` predictions

✅ **Inference path** (lines 153-209):
- Autoregressive loop: Predicts H steps iteratively
- Each step: Apply transformer → predict next RUL → project to embedding → append
- No double-processing: Returns concatenated predictions directly
- Layer norm applied inside loop (per step) ✅

✅ **Cross-attention handling**:
- PyTorch `TransformerDecoder` requires `batch_first=False`
- Correctly transposes: `(B, S, E)` → `(S, B, E)` for transformer → `(B, S, E)` back
- Memory handling: Uses `enc_seq` if provided, falls back to `enc_token` ✅

### Issues Found and Fixed

1. **Cross-attention batch dimension**: Fixed PyTorch `TransformerDecoder` API usage
   - Issue: `batch_first=True` not supported for `TransformerDecoder`
   - Fix: Set `batch_first=False` and manually transpose inputs/outputs
   - Status: ✅ Fixed

2. **Unicode encoding in tests**: Fixed Windows console encoding issues
   - Issue: Unicode checkmarks (`✓`, `✗`) not supported in Windows cp1252
   - Fix: Replaced with `[OK]` and `[FAIL]` markers
   - Status: ✅ Fixed

### How to Run Smoke Tests

```bash
# From project root
python src/tools/test_transformer_decoder_smoke.py
```

**Expected output**:
- All tests pass with `[OK]` markers
- No errors or warnings
- Final message: "All smoke tests passed!"

### Enabling Transformer Decoder in Experiments

**Via experiment config**:

```python
# In experiment config world_model_params:
wmp["decoder_type"] = "tf_ar"  # Self-attention only
# OR
wmp["decoder_type"] = "tf_ar_xattn"  # Cross-attention variant
```

**Default**: `decoder_type="lstm"` (LSTM decoder, unchanged)

**Example**:
```python
def get_fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_config():
    cfg = copy.deepcopy(get_fd004_wm_v1_p0_softcap_k3_hm_pad_config())
    cfg["experiment_name"] = "fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar"
    wmp = cfg.setdefault("world_model_params", {})
    wmp["decoder_type"] = "tf_ar"  # Enable Transformer AR decoder
    return cfg
```

---

**Document version**: 1.1  
**Last audit**: 2025-12-28  
**Auditor**: Implementer
