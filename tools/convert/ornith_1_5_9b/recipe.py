"""Ornith source recipes over the shared Qwen3.5-9B object geometry."""

from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6.common.recipe import (
    Cast,
    Concat,
    DraftHeadTokenIds,
    GatherRows,
    Reshape,
    Slice,
    SourceTensor,
    TensorRecipe,
    Transpose,
    preflight_source_reader as _preflight_source_reader,
    source_requirements as _source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)
from tools.convert.qwen3_5_9b.recipe import *  # noqa: F401,F403
from tools.convert.qwen3_5_9b import recipe as _base

from . import inventory
from .source import OrnithShardReader


def _rewrite_source(expression):
    if isinstance(expression, SourceTensor):
        if (
            expression.dtype == "F32"
            and expression.name.endswith((".A_log", ".dt_bias", ".norm.weight"))
        ):
            return SourceTensor(expression.name, expression.shape, "BF16")
        return expression
    if isinstance(expression, Cast):
        return Cast(_rewrite_source(expression.source), expression.dtype)
    if isinstance(expression, Slice):
        return Slice(_rewrite_source(expression.source), expression.axis, expression.begin, expression.end)
    if isinstance(expression, Reshape):
        return Reshape(_rewrite_source(expression.source), expression.shape)
    if isinstance(expression, Transpose):
        return Transpose(_rewrite_source(expression.source), expression.axes)
    if isinstance(expression, Concat):
        return Concat(tuple(_rewrite_source(part) for part in expression.sources), expression.axis)
    if isinstance(expression, (DraftHeadTokenIds, GatherRows)):
        return expression
    raise TypeError(f"unknown recipe expression {type(expression)!r}")


RECIPE_SPECS = tuple(
    TensorRecipe(recipe.object_name, _rewrite_source(recipe.expression))
    for recipe in _base.RECIPE_SPECS
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements():
    return _source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    with OrnithShardReader(model_dir) as reader:
        return _preflight_source_reader(reader, RECIPE_SPECS)


validate_recipe_coverage()
