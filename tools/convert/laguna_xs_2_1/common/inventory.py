"""Leaf storage contracts shared by Laguna XS 2.1 artifact targets.

This module owns format/layout vocabulary, resource specs, and helper functions.
"""

from __future__ import annotations

from dataclasses import dataclass


CONTIGUOUS_LAYOUT = "contiguous-le-v1"
ROW_SPLIT_LAYOUT = "row-split-k128-v1"
RESOURCE_ENCODING = "raw-bytes-v1"

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
Q4 = "Q4G64_F16S"
Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"
W8 = "W8G32_F16S"

DIRECT_FORMATS = frozenset((BF16, FP32, I32))
FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str

    @property
    def kind(self) -> str:
        return "tensor"


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    encoding: str = RESOURCE_ENCODING

    @property
    def kind(self) -> str:
        return "resource"


StoredObjectSpec = TensorSpec | ResourceSpec


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    """Build a tensor spec with the canonical layout for its numeric format."""

    layout = CONTIGUOUS_LAYOUT if numeric_format in DIRECT_FORMATS else ROW_SPLIT_LAYOUT
    return TensorSpec(name=name, shape=shape, format=numeric_format, layout=layout)


RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)


__all__ = [
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "FORMAT_NAMES",
    "FP32",
    "I32",
    "LAYOUT_NAMES",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_ENCODING",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TensorSpec",
    "tensor_spec",
]