# Decoder Contract: Transformer Autoregressive Decoder

**Context**: World Model v3 outputs trajectory, EOL, and HI predictions. The decoder is responsible for generating the trajectory component.

## Inputs

**Required**:
- `enc_token`: `(B, d_model)` - Encoder output (single embedding vector from UniversalEncoderV2)
  - In this codebase, UniversalEncoderV2 returns `(B, d_model)`, not a sequence
  - This is the primary encoder context

**Optional**:
- `y_teacher`: `(B, H, 1)` - Ground truth trajectory for teacher forcing (training only)
- `cond_ctx`: `(B, cond_emb_dim)` or `None` - Condition embedding context (if available)
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
- Autoregressive rollout: Start with `[start_token, enc_context]`
- Iteratively predict next token, append to sequence, repeat for H steps
- Each step uses only previously predicted tokens (causal)

## Start Token Strategy

- Learned start token: `nn.Parameter(torch.randn(1, d_model))`
- Expanded to batch: `(B, d_model)`
- Optionally concatenated with encoder context token if needed

## Positional Encoding

- Learnable positional embeddings for max length `horizon + 2` (start + context + H steps)
- Applied to token sequence before transformer layers

