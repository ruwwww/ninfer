"""Hugging Face source recipe for Laguna XS 2.1."""

from __future__ import annotations

from pathlib import Path

from tools.convert.common.safetensors import ShardReader
from tools.convert.laguna_xs_2_1.common.recipe import (
    Concat,
    Expression,
    Reshape,
    SOURCE_DTYPE,
    SourcePreflight,
    SourceTensor,
    StackExpertDown,
    StackExpertGateUp,
    TensorRecipe,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_source_reader,
    source,
    source_requirements as _common_source_requirements,
    validate_recipe_coverage as _common_validate_recipe_coverage,
)

from . import inventory


def _moe_recipes(source_prefix: str, object_prefix: str) -> tuple[TensorRecipe, ...]:
    """Build MoE expert source transforms for layers 1-39.

    The checkpoint stores per-expert tensors separately:
      experts.{e}.gate_proj.weight   (512, 2048)
      experts.{e}.up_proj.weight     (512, 2048)
      experts.{e}.down_proj.weight   (2048, 512)

    The artifact expects:
      router_gate                   (256, 2048)
      routed_gate_up                (262144, 2048)  — 256 experts * 1024 fused rows
      routed_down                   (524288, 512)   — 256 experts * 2048 rows
      shared_gate_up                (1024, 2048)    — concatenated gate+up
      shared_down                   (2048, 512)
    """

    return (
        # Router: direct map
        TensorRecipe(
            object_prefix + "router_gate",
            source(source_prefix + "gate.weight", (256, 2048)),
        ),
        # Router correction bias: direct map (exactly zero for this checkpoint)
        TensorRecipe(
            object_prefix + "router_bias",
            source(source_prefix + "experts.e_score_correction_bias", (256,)),
        ),
        # Routed gate/up: stack all 256 experts' gate+up, concat along intermediate dim, flatten
        TensorRecipe(
            object_prefix + "routed_gate_up",
            StackExpertGateUp(source_prefix + "experts."),
        ),
        # Routed down: stack all 256 experts' down_proj, flatten to (524288, 512)
        TensorRecipe(
            object_prefix + "routed_down",
            StackExpertDown(source_prefix + "experts."),
        ),
        # Shared gate/up: concatenate shared gate and up along row axis
        TensorRecipe(
            object_prefix + "shared_gate_up",
            Concat(
                (
                    source(source_prefix + "shared_expert.gate_proj.weight", (512, 2048)),
                    source(source_prefix + "shared_expert.up_proj.weight", (512, 2048)),
                ),
                0,
            ),
        ),
        # Shared down: direct map
        TensorRecipe(
            object_prefix + "shared_down",
            source(source_prefix + "shared_expert.down_proj.weight", (2048, 512)),
        ),
    )


def _dense_mlp_recipes(source_prefix: str, object_prefix: str) -> tuple[TensorRecipe, ...]:
    """Build dense MLP recipes for layer 0."""

    return (
        TensorRecipe(
            object_prefix + "mlp/gate_proj",
            source(source_prefix + "gate_proj.weight", (8192, 2048)),
        ),
        TensorRecipe(
            object_prefix + "mlp/up_proj",
            source(source_prefix + "up_proj.weight", (8192, 2048)),
        ),
        TensorRecipe(
            object_prefix + "mlp/down_proj",
            source(source_prefix + "down_proj.weight", (2048, 8192)),
        ),
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            source("model.embed_tokens.weight", (100352, 2048)),
        )
    ]

    for layer in inventory.TEXT_LAYERS:
        source_prefix = f"model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"

        q_heads = inventory._q_heads(layer)
        q_rows = inventory._q_rows(layer)
        kv_rows = inventory._kv_rows(layer)

        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                source(source_prefix + "input_layernorm.weight", (2048,)),
            )
        )

        # Attention weights
        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "attention/q_proj",
                    source(source_prefix + "self_attn.q_proj.weight", (q_rows, 2048)),
                ),
                TensorRecipe(
                    object_prefix + "attention/k_proj",
                    source(source_prefix + "self_attn.k_proj.weight", (kv_rows, 2048)),
                ),
                TensorRecipe(
                    object_prefix + "attention/v_proj",
                    source(source_prefix + "self_attn.v_proj.weight", (kv_rows, 2048)),
                ),
                TensorRecipe(
                    object_prefix + "attention/q_norm",
                    source(source_prefix + "self_attn.q_norm.weight", (128,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/k_norm",
                    source(source_prefix + "self_attn.k_norm.weight", (128,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/g_proj",
                    source(source_prefix + "self_attn.g_proj.weight", (q_heads, 2048)),
                ),
                TensorRecipe(
                    object_prefix + "attention/o_proj",
                    source(source_prefix + "self_attn.o_proj.weight", (2048, q_rows)),
                ),
            )
        )

        recipes.append(
            TensorRecipe(
                object_prefix + "post_attention_norm",
                source(source_prefix + "post_attention_layernorm.weight", (2048,)),
            )
        )

        if layer in inventory.DENSE_MLP_LAYERS:
            recipes.extend(_dense_mlp_recipes(source_prefix + "mlp.", object_prefix))
        else:
            recipes.extend(_moe_recipes(source_prefix + "mlp.", object_prefix + "moe/"))

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                source("model.norm.weight", (2048,)),
            ),
            TensorRecipe(
                "text/output_head",
                source("lm_head.weight", (100352, 2048)),
            ),
        )
    )
    return tuple(recipes)


TEXT_CORE_RECIPE_SPECS = _build_text_recipes()
RECIPE_SPECS = TEXT_CORE_RECIPE_SPECS
RECIPES_BY_NAME = {item.object_name: item for item in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    """Validate exact output pairing and source-checkpoint inventories."""

    _common_validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)
    if len(RECIPE_SPECS) != len(RECIPES_BY_NAME):
        raise ValueError(
            f"Laguna recipe has {len(RECIPE_SPECS)} entries but {len(RECIPES_BY_NAME)} unique names"
        )


def source_requirements() -> dict[str, SourceTensor]:
    return _common_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    model = Path(model_dir)
    return _preflight_exact_source(
        ShardReader.from_index(model / "model.safetensors.index.json"),
        RECIPE_SPECS,
        source_requirements(),
        "Laguna XS 2.1 checkpoint",
    )


def _preflight_exact_source(
    reader: ShardReader,
    recipes: tuple[TensorRecipe, ...],
    requirements: dict[str, SourceTensor],
    label: str,
) -> SourcePreflight:
    with reader:
        actual_names = set(reader.names)
    required_names = set(requirements)
    if actual_names != required_names:
        missing = sorted(required_names - actual_names)
        extra = sorted(actual_names - required_names)
        details = []
        if missing:
            details.append(f"missing={missing[:8]!r}")
        if extra:
            details.append(f"extra={extra[:8]!r}")
        raise ValueError(
            f"{label} source inventory differs from its tensor contract"
            + (": " + ", ".join(details) if details else "")
        )
    with reader:
        return preflight_source_reader(reader, recipes)


__all__ = [
    "Concat",
    "Expression",
    "RECIPE_SPECS",
    "RECIPES_BY_NAME",
    "Reshape",
    "SOURCE_DTYPE",
    "ShardReader",
    "SourcePreflight",
    "SourceTensor",
    "TensorRecipe",
    "expression_shape",
    "expression_sources",
    "materialize_expression",
    "materialize_recipe",
    "preflight_sources",
    "source_requirements",
    "validate_recipe_coverage",
]