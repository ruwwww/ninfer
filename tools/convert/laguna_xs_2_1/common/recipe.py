"""Leaf source-recipe primitives shared by Laguna XS 2.1 artifact targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Mapping, Sequence

import torch

from tools.convert.common.safetensors import ShardReader

from .inventory import TensorSpec


SOURCE_DTYPE = "BF16"


@dataclass(frozen=True, slots=True)
class SourceTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = SOURCE_DTYPE


@dataclass(frozen=True, slots=True)
class Slice:
    source: Expression
    axis: int
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class Reshape:
    source: Expression
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Transpose:
    source: Expression
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Concat:
    sources: tuple[Expression, ...]
    axis: int


@dataclass(frozen=True, slots=True)
class Cast:
    source: Expression
    dtype: str


Expression = Slice | Reshape | Transpose | Concat | Cast | SourceTensor


@dataclass(frozen=True, slots=True)
class TensorRecipe:
    object_name: str
    expression: Expression


@dataclass(frozen=True, slots=True)
class SourcePreflight:
    recipe_count: int
    source_tensor_count: int
    source_shard_count: int
    source_dtype_counts: dict[str, int]


def source(name: str, shape: tuple[int, ...]) -> SourceTensor:
    return SourceTensor(name=name, shape=shape)


def expression_shape(expression: Expression) -> tuple[int, ...]:
    if isinstance(expression, SourceTensor):
        return expression.shape

    if isinstance(expression, Slice):
        shape = list(expression_shape(expression.source))
        if not 0 <= expression.axis < len(shape):
            raise ValueError(f"slice axis {expression.axis} is outside rank {len(shape)}")
        if not 0 <= expression.begin < expression.end <= shape[expression.axis]:
            raise ValueError(
                f"invalid slice [{expression.begin},{expression.end}) for axis size "
                f"{shape[expression.axis]}"
            )
        shape[expression.axis] = expression.end - expression.begin
        return tuple(shape)

    if isinstance(expression, Reshape):
        source_shape = expression_shape(expression.source)
        if prod(source_shape) != prod(expression.shape):
            raise ValueError(f"cannot reshape {source_shape} to {expression.shape}")
        return expression.shape

    if isinstance(expression, Transpose):
        source_shape = expression_shape(expression.source)
        if tuple(sorted(expression.axes)) != tuple(range(len(source_shape))):
            raise ValueError(f"invalid transpose axes {expression.axes} for {source_shape}")
        return tuple(source_shape[axis] for axis in expression.axes)

    if isinstance(expression, Concat):
        if not expression.sources:
            raise ValueError("concat requires at least one source")
        shapes = [expression_shape(part) for part in expression.sources]
        rank = len(shapes[0])
        if not 0 <= expression.axis < rank:
            raise ValueError(f"concat axis {expression.axis} is outside rank {rank}")
        output = list(shapes[0])
        output[expression.axis] = 0
        for shape in shapes:
            if len(shape) != rank:
                raise ValueError("concat sources have different ranks")
            for axis, (got, expected) in enumerate(zip(shape, shapes[0])):
                if axis != expression.axis and got != expected:
                    raise ValueError("concat sources have incompatible shapes")
            output[expression.axis] += shape[axis]
        return tuple(output)

    if isinstance(expression, Cast):
        return expression_shape(expression.source)

    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def expression_sources(expression: Expression) -> tuple[SourceTensor, ...]:
    if isinstance(expression, SourceTensor):
        return (expression,)
    if isinstance(expression, (Slice, Reshape, Transpose, Cast)):
        return expression_sources(expression.source)
    if isinstance(expression, Concat):
        return tuple(item for part in expression.sources for item in expression_sources(part))
    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def validate_recipe_coverage(
    recipes: Sequence[TensorRecipe],
    tensor_specs: Sequence[TensorSpec],
) -> None:
    inventory_names = tuple(spec.name for spec in tensor_specs)
    recipe_names = tuple(recipe.object_name for recipe in recipes)
    if recipe_names != inventory_names:
        raise ValueError("recipe order or coverage does not match the tensor inventory")
    if len({recipe.object_name for recipe in recipes}) != len(recipes):
        raise ValueError("more than one recipe targets the same artifact object")
    inventory_by_name = {spec.name: spec for spec in tensor_specs}
    for recipe in recipes:
        expected = inventory_by_name[recipe.object_name].shape
        actual = expression_shape(recipe.expression)
        if actual != expected:
            raise ValueError(
                f"{recipe.object_name}: recipe shape {actual} != inventory {expected}"
            )


def source_requirements(
    recipes: Sequence[TensorRecipe],
) -> dict[str, SourceTensor]:
    requirements: dict[str, SourceTensor] = {}
    for recipe in recipes:
        for requirement in expression_sources(recipe.expression):
            previous = requirements.setdefault(requirement.name, requirement)
            if previous != requirement:
                raise ValueError(f"inconsistent source declaration for {requirement.name}")
    return requirements


def preflight_source_reader(
    reader: ShardReader,
    recipes: Sequence[TensorRecipe],
) -> SourcePreflight:
    requirements = source_requirements(recipes)
    metadata = reader.metadata(requirements)

    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, requirement in requirements.items():
        actual = metadata[name]
        if actual.shape != requirement.shape:
            raise ValueError(f"{name}: source shape {actual.shape} != required {requirement.shape}")
        if actual.dtype != requirement.dtype:
            raise ValueError(f"{name}: source dtype {actual.dtype} != required {requirement.dtype}")
        dtype_counts[actual.dtype] = dtype_counts.get(actual.dtype, 0) + 1
        shards.add(actual.shard)

    return SourcePreflight(
        recipe_count=len(recipes),
        source_tensor_count=len(requirements),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def materialize_expression(
    expression: Expression,
    reader: ShardReader,
    derived_tensors: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if isinstance(expression, SourceTensor):
        tensor = reader.get(expression.name)
        if tuple(tensor.shape) != expression.shape:
            raise ValueError(
                f"{expression.name}: source shape {tuple(tensor.shape)} != {expression.shape}"
            )
        return tensor

    if isinstance(expression, Slice):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.narrow(expression.axis, expression.begin, expression.end - expression.begin)

    if isinstance(expression, Reshape):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.reshape(expression.shape)

    if isinstance(expression, Transpose):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.permute(expression.axes).contiguous()

    if isinstance(expression, Concat):
        tensors = [
            materialize_expression(part, reader, derived_tensors)
            for part in expression.sources
        ]
        return torch.cat(tensors, dim=expression.axis)

    if isinstance(expression, Cast):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.to(torch.float32)

    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def materialize_recipe(
    recipe: TensorRecipe,
    reader: ShardReader,
    derived_tensors: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    tensor = materialize_expression(recipe.expression, reader, derived_tensors)
    expected = expression_shape(recipe.expression)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{recipe.object_name}: materialized shape {tuple(tensor.shape)} != {expected}"
        )
    return tensor


__all__ = [
    "Cast",
    "Concat",
    "Expression",
    "Reshape",
    "SOURCE_DTYPE",
    "ShardReader",
    "Slice",
    "SourcePreflight",
    "SourceTensor",
    "TensorRecipe",
    "Transpose",
    "expression_shape",
    "expression_sources",
    "materialize_expression",
    "materialize_recipe",
    "preflight_source_reader",
    "source",
    "source_requirements",
    "validate_recipe_coverage",
]