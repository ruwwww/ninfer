# Laguna XS 2.1 — Model Mathematics Reference

## Overview

**Laguna XS 2.1** (Poolside, 33B total / ~3B activated per token) is a decoder-only
Mixture-of-Experts language model. This document describes the exact model mathematics,
dimensions, and state semantics for the NInfer implementation.

| Property | Value |
|---|---|
| `model_type` | `laguna` |
| Total parameters | ~33 billion |
| Activated per token | ~3 billion |
| `hidden_size` | 2048 |
| `intermediate_size` (dense MLP) | 8192 |
| `moe_intermediate_size` (per expert) | 512 |
| `shared_expert_intermediate_size` | 512 |
| `num_hidden_layers` | 40 |
| `vocab_size` | 100 352 |
| `head_dim` | 128 |
| `max_position_embeddings` | 262 144 |
| `rms_norm_eps` | 1e-6 |
| `torch_dtype` | BF16 |

## Layer Types

The 40 layers follow a strict **1:3 alternating pattern** of global (full) attention and
sliding window attention (SWA):

| Layer | Type | Query Heads | KV Heads | Group Size | MLP |
|---|---|---|---|---|---|
| 0 | `full_attention` | 48 | 8 | 6 | Dense (no MoE) |
| 1–3 | `sliding_attention` | 64 | 8 | 8 | MoE (256 experts, top-8) |
| 4 | `full_attention` | 48 | 8 | 6 | MoE |
| 5–7 | `sliding_attention` | 64 | 8 | 8 | MoE |
| … | … | … | … | … | … |
| 36 | `full_attention` | 48 | 8 | 6 | MoE |
| 37–39 | `sliding_attention` | 64 | 8 | 8 | MoE |

- **10 global/full-attention layers**: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36
- **30 sliding-window layers**: all others
- Sliding window size: **512 tokens**

```python
# layer_types[i] == "full_attention"  =>  num_attention_heads_per_layer[i] == 48
# layer_types[i] == "sliding_attention"  =>  num_attention_heads_per_layer[i] == 64
```

Layer-type classification function:

```cpp
// Returns true for global/full-attention layers (0, 4, 8, ..., 36).
constexpr bool is_full_attention(int layer) noexcept {
    return (layer % 4) == 0;
}

// Returns true for sliding-window attention layers.
constexpr bool is_swa(int layer) noexcept {
    return (layer % 4) != 0;
}
```

## Attention

All layers use **Grouped-Query Attention (GQA)** with the following per-head structure:

```
QK Norm ── RoPE ── Matmul(Q, K^T) ── Mask ── Softmax ── Scale ── Matmul(Attn, V) ── Gating ── O Proj
                                    ^^^^^^^^
                              Sliding window for SWA layers, full causal for global
```

### QK Normalization (before RoPE)

Each layer applies per-head RMSNorm to Q and K vectors **before** rotary embedding:

```
q_norm[i] = q[i] / sqrt(mean(q[i]^2) + eps)   for i in 0..head_dim-1
k_norm[i] = k[i] / sqrt(mean(k[i]^2) + eps)
```

where `eps = 1e-6` and the mean is over the `head_dim` elements of each head. The normalized
Q/K are then passed through RoPE.

Weights: `q_norm.weight` and `k_norm.weight`, each of shape `(head_dim,)` = `(128,)`.

### Rotary Position Embeddings (RoPE)

Two different RoPE configurations are used depending on layer type:

**Global attention layers** (0, 4, 8, …, 36) use **YARN-extrapolated** RoPE:

| Parameter | Value |
|---|---|
| `rope_type` | `yarn` |
| `rope_theta` | 500 000 |
| `factor` | 32.0 |
| `original_max_position_embeddings` | 8 192 |
| `beta_slow` | 1.0 |
| `beta_fast` | 64.0 |
| `attention_factor` | 1.3465735902799727 |
| `partial_rotary_factor` | 0.5 |

Only 50% of the rotary dimensions are applied (`partial_rotary_factor = 0.5`), meaning
`rotary_dim = 0.5 * head_dim = 64`.

**Sliding window layers** use **standard** RoPE:

| Parameter | Value |
|---|---|
| `rope_type` | `default` |
| `rope_theta` | 10 000 |
| `partial_rotary_factor` | 1.0 |

All rotary dimensions are applied (`partial_rotary_factor = 1.0`), meaning
`rotary_dim = head_dim = 128`.

### Sliding Window Causal Mask

For SWA layers, the attention mask restricts each query position `i` to attend only to
key positions `j` where `i - window_size < j <= i`:

```
mask(i, j) = -inf  if j > i or j <= i - window_size
             0     otherwise
```

where `window_size = 512`. For global layers, the standard causal mask applies:

```
mask(i, j) = -inf  if j > i
             0     otherwise
```

### Per-Head Attention Gating

After the attention output is computed (softmax × V), a learnable gating projection is
applied before the final `o_proj`:

```
g = softplus(g_proj.weight @ h)   // shape: (num_heads,)
attn_out = g[:, None] * o_proj @ h
```

where `softplus(x) = ln(exp(x) + 1)`. The gate is per-head, broadcast across `head_dim`.
Weight shape: `g_proj.weight` is `(num_heads,)`.

### Projection Dimensions

| Layer type | Q heads | K heads | V heads | Q rows | KV rows | O output |
|---|---|---|---|---|---|---|
| Full attention | 48 | 8 | 8 | 6144 (48×128) | 1024 (8×128) each | 2048 |
| SWA | 64 | 8 | 8 | 8192 (64×128) | 1024 (8×128) each | 2048 |

QKV projections are concatenated into a single weight of shape `(q_rows + 2*kv_rows, hidden)`:
- Full: `(6144 + 2×1024, 2048) = (8192, 2048)`
- SWA: `(8192 + 2×1024, 2048) = (10240, 2048)`

## Mixture of Experts (MoE)

Layers 1–39 use MoE with 256 experts + 1 shared expert. Layer 0 uses a standard dense SwiGLU
MLP.

### Router

The router computes a dot-product score for each expert:

```
score[e] = sigmoid(w_router[e] @ h + b_router[e])   for e in 0..255
```

The weights `w_router` have shape `(256, hidden_size)` = `(256, 2048)`. There is no bias in
the reference implementation (bias is absorbed or zero).

Scores are passed through an auxiliary load-balancing bias `e_score_correction_bias` of shape
`(256,)`, then top-8 experts are selected. The selection uses a tie-breaking rule: when scores
are equal (after bias), lower expert ID wins.

Routing weights are L1-normalized across the 8 selected experts:

```
alpha[e] = score[e] / sum(score[selected])
```

### Routed Expert Output Scaling

The routed expert output is scaled by `moe_routed_scaling_factor = 2.5` before adding the
shared expert output:

```
expert_output = moe_routed_scaling_factor * sum(alpha[e] * expert_e(h)) + shared_expert(h)
```

### Expert Structure

Each of the 256 experts implements a SwiGLU feed-forward network with intermediate size 512:

```
gate_val = silu(gate_proj @ h)
up_val   = up_proj @ h
expert_out = down_proj @ (gate_val * up_val)
```

Weights per expert:
- `gate_proj`: `(512, 2048)`
- `up_proj`: `(512, 2048)`
- `down_proj`: `(2048, 512)`

These are stored as fused tensors in the checkpoint:
- `experts.{e}.gate_up_proj`: `(1024, 2048)` — concatenation of gate and up
- `experts.{e}.down_proj`: `(2048, 512)`

### Shared Expert

A shared expert runs on every token alongside the routed experts:

```
shared_gate_val = silu(shared_gate_proj @ h)
shared_up_val   = shared_up_proj @ h
shared_out      = shared_down_proj @ (shared_gate_val * shared_up_val)
```

Weights:
- `shared_expert.gate_proj`: `(512, 2048)`
- `shared_expert.up_proj`: `(512, 2048)`
- `shared_expert.down_proj`: `(2048, 512)`

### Layer 0 Dense MLP

Layer 0 uses a standard dense SwiGLU MLP (no MoE):

- `mlp.gate_proj`: `(8192, 2048)`
- `mlp.up_proj`: `(8192, 2048)`
- `mlp.down_proj`: `(2048, 8192)`

## Layer Normalization

Each layer has two RMSNorm layers (eps = 1e-6):

- `input_layernorm.weight`: `(2048,)` — applied before attention
- `post_attention_layernorm.weight`: `(2048,)` — applied after attention/MoE

The model has a final normalization:

- `norm.weight`: `(2048,)` — applied after all layers, before lm_head

## Output Head

- `lm_head.weight`: `(100352, 2048)` — maps hidden state to logits

## KV Cache

### Global Attention Layers

Standard cyclic KV cache with BF16 storage:

```
K[layer][pos % capacity][kv_head][d]   for d in 0..head_dim-1
V[layer][pos % capacity][kv_head][d]
```

Capacity = `max_context = 262144`.

### Sliding Window Layers

Circular buffer with window-sized capacity:

```
K[layer][pos % window_size][kv_head][d]
V[layer][pos % window_size][kv_head][d]
```

Window size = 512 tokens. This ensures constant memory usage regardless of sequence length.

### FP8 KV Cache

The ticket specifies native FP8 KV cache quantization. FP8 (E4M3 format) stores each K/V
element as an 8-bit value with an associated scale factor. The scale is computed per-token
per-head (or per-group) and stored separately. During attention compute, FP8 values are
dequantized to FP32/BF16 on-the-fly.

FP8 storage reduces KV cache memory by 50% compared to BF16 (1 byte vs 2 bytes per element).

## Weight Tensor Inventory (HuggingFace Checkpoint)

The checkpoint consists of 14 sharded `.safetensors` files (~62.3 GB total). Key tensor
naming patterns:

```
model.embed_tokens.weight                    # (100352, 2048)
model.norm.weight                            # (2048,)
lm_head.weight                               # (100352, 2048)

# Layer 0 (dense MLP)
model.layers.0.input_layernorm.weight        # (2048,)
model.layers.0.self_attn.q_proj.weight       # (6144, 2048)
model.layers.0.self_attn.k_proj.weight       # (1024, 2048)
model.layers.0.self_attn.v_proj.weight       # (1024, 2048)
model.layers.0.self_attn.q_norm.weight       # (128,)
model.layers.0.self_attn.k_norm.weight       # (128,)
model.layers.0.self_attn.g_proj.weight       # (48,)
model.layers.0.self_attn.o_proj.weight       # (2048, 6144)
model.layers.0.post_attention_layernorm.weight  # (2048,)
model.layers.0.mlp.gate_proj.weight          # (8192, 2048)
model.layers.0.mlp.up_proj.weight            # (8192, 2048)
model.layers.0.mlp.down_proj.weight          # (2048, 8192)

# Layers 1-39 (MoE)
model.layers.{n}.input_layernorm.weight      # (2048,)
model.layers.{n}.self_attn.q_proj.weight     # (Q_rows, 2048)  where Q_rows = 6144 or 8192
model.layers.{n}.self_attn.k_proj.weight     # (1024, 2048)
model.layers.{n}.self_attn.v_proj.weight     # (1024, 2048)
model.layers.{n}.self_attn.q_norm.weight     # (128,)
model.layers.{n}.self_attn.k_norm.weight     # (128,)
model.layers.{n}.self_attn.g_proj.weight     # (num_heads,)   — 48 or 64
model.layers.{n}.self_attn.o_proj.weight     # (2048, Q_rows)
model.layers.{n}.post_attention_layernorm.weight # (2048,)
model.layers.{n}.mlp.gate.weight             # (256, 2048)    — router
model.layers.{n}.mlp.experts.{e}.gate_up_proj  # (1024, 2048) — fused gate+up for expert e
model.layers.{n}.mlp.experts.{e}.down_proj     # (2048, 512)  — expert e
model.layers.{n}.mlp.shared_expert.gate_proj # (512, 2048)
model.layers.{n}.mlp.shared_expert.up_proj   # (512, 2048)
model.layers.{n}.mlp.shared_expert.down_proj # (2048, 512)
model.layers.{n}.mlp.e_score_correction_bias # (256,)

# Final
model.norm.weight                            # (2048,)
lm_head.weight                               # (100352, 2048)
```

## Activation Functions

- **SwiGLU**: `silu(x) * y` where `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
- **Softplus**: `softplus(x) = log(1 + exp(x))` — used in attention gating
- **Sigmoid**: `sigmoid(x) = 1 / (1 + exp(-x))` — used in MoE routing probabilities

## Complete Forward Pass (Layer n)

### Layer 0 (Dense):

```python
x = input
h = x + attention(input_layernorm(x))
h = h + mlp(post_attention_layernorm(h))
output = h
```

### Layers 1-39 (MoE):

```python
x = input
h = x + attention(input_layernorm(x))
h = h + moe(post_attention_layernorm(h))
output = h
```

where `attention()` includes QK norm → RoPE → GQA → mask → softmax → gating → o_proj,
and `moe()` includes router → top-8 selection → routed SwiGLU experts → shared SwiGLU →
scale + merge.