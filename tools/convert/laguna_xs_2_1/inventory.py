"""Persistent-object contract for the Laguna XS 2.1 target.

This target module owns the complete ordered inventory. Source-checkpoint
names and transformations live in :mod:`recipe`.
"""

from __future__ import annotations

from tools.convert.laguna_xs_2_1.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    tensor_spec,
)


MODEL_ID = "laguna-xs-2.1"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "laguna_xs_2_1"

TEXT_LAYERS = tuple(range(40))
FULL_ATTENTION_LAYERS = tuple(layer for layer in TEXT_LAYERS if (layer % 4) == 0)  # 0, 4, 8, ..., 36
SWA_LAYERS = tuple(layer for layer in TEXT_LAYERS if (layer % 4) != 0)             # 1, 2, 3, 5, ..., 39
DENSE_MLP_LAYERS = (0,)
MOE_LAYERS = tuple(layer for layer in TEXT_LAYERS if layer not in DENSE_MLP_LAYERS)

# Per-layer head counts
FULL_Q_HEADS = 48
FULL_KV_HEADS = 8
SWA_Q_HEADS = 64
SWA_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE = 512
DENSE_INTERMEDIATE = 8192
NUM_EXPERTS = 256
SLIDING_WINDOW = 512
MAX_CONTEXT = 262144
VOCAB_SIZE = 100352


def _q_rows(layer: int) -> int:
    """Return Q projection output rows for a given layer."""
    if layer in FULL_ATTENTION_LAYERS:
        return FULL_Q_HEADS * HEAD_DIM  # 48 * 128 = 6144
    return SWA_Q_HEADS * HEAD_DIM  # 64 * 128 = 8192


def _kv_rows(layer: int) -> int:
    """Return K/V projection output rows for a given layer."""
    # Same for both layer types
    return SWA_KV_HEADS * HEAD_DIM  # 8 * 128 = 1024


def _q_heads(layer: int) -> int:
    if layer in FULL_ATTENTION_LAYERS:
        return FULL_Q_HEADS
    return SWA_Q_HEADS


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (VOCAB_SIZE, HIDDEN_SIZE), W8),
    ]

    for layer in TEXT_LAYERS:
        prefix = f"text/layers/{layer}/"
        q_rows = _q_rows(layer)
        kv_rows = _kv_rows(layer)
        q_heads = _q_heads(layer)

        specs.append(tensor_spec(prefix + "input_norm", (HIDDEN_SIZE,), BF16))

        # Attention projections — stored as separate Q, K, V weights (no concatenation needed)
        specs.extend(
            (
                tensor_spec(prefix + "attention/q_proj", (q_rows, HIDDEN_SIZE), W8),
                tensor_spec(prefix + "attention/k_proj", (kv_rows, HIDDEN_SIZE), W8),
                tensor_spec(prefix + "attention/v_proj", (kv_rows, HIDDEN_SIZE), W8),
                tensor_spec(prefix + "attention/q_norm", (HEAD_DIM,), BF16),
                tensor_spec(prefix + "attention/k_norm", (HEAD_DIM,), BF16),
                tensor_spec(prefix + "attention/g_proj", (q_heads,), BF16),
                tensor_spec(prefix + "attention/o_proj", (HIDDEN_SIZE, q_rows), W8),
            )
        )

        specs.append(
            tensor_spec(prefix + "post_attention_norm", (HIDDEN_SIZE,), BF16)
        )

        if layer in DENSE_MLP_LAYERS:
            # Layer 0: dense SwiGLU MLP
            specs.extend(
                (
                    tensor_spec(prefix + "mlp/gate_proj", (DENSE_INTERMEDIATE, HIDDEN_SIZE), W8),
                    tensor_spec(prefix + "mlp/up_proj", (DENSE_INTERMEDIATE, HIDDEN_SIZE), W8),
                    tensor_spec(prefix + "mlp/down_proj", (HIDDEN_SIZE, DENSE_INTERMEDIATE), W8),
                )
            )
        else:
            # Layers 1-39: MoE
            specs.extend(
                (
                    tensor_spec(prefix + "moe/router_gate", (NUM_EXPERTS, HIDDEN_SIZE), BF16),
                    tensor_spec(prefix + "moe/routed_gate_up", (NUM_EXPERTS * MOE_INTERMEDIATE, HIDDEN_SIZE), Q4),
                    tensor_spec(
                        prefix + "moe/routed_down",
                        (NUM_EXPERTS * HIDDEN_SIZE, MOE_INTERMEDIATE),
                        Q5,
                    ),
                    tensor_spec(prefix + "moe/shared_gate_up", (MOE_INTERMEDIATE * 2, HIDDEN_SIZE), W8),
                    tensor_spec(prefix + "moe/shared_down", (HIDDEN_SIZE, MOE_INTERMEDIATE), W8),
                )
            )

    specs.extend(
        (
            tensor_spec("text/final_norm", (HIDDEN_SIZE,), BF16),
            tensor_spec("text/output_head", (VOCAB_SIZE, HIDDEN_SIZE), Q6),
        )
    )
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()

TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}


__all__ = [
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DENSE_INTERMEDIATE",
    "DENSE_MLP_LAYERS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "FULL_KV_HEADS",
    "FULL_Q_HEADS",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "I32",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "MAX_CONTEXT",
    "MODEL_ID",
    "MOE_INTERMEDIATE",
    "MOE_LAYERS",
    "NUM_EXPERTS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "SLIDING_WINDOW",
    "SWA_KV_HEADS",
    "SWA_LAYERS",
    "SWA_Q_HEADS",
    "TARGET_KEY",
    "TEXT_CORE_TENSOR_SPECS",
    "TEXT_LAYERS",
    "TensorSpec",
    "VOCAB_SIZE",
    "WEIGHTS_ID",
    "W8",
]