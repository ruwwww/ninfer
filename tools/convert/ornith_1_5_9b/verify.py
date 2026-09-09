"""Verify the structure and representative payloads of an Ornith artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from tools.artifact.container import Artifact, ResourceObject, TensorObject
from tools.artifact.layouts import decode_direct, decode_row_split_codes

from . import inventory, recipe
from .source import OrnithShardReader


DIRECT_PROBE_OBJECTS = (
    "text/layers/0/input_norm",
    "text/layers/0/gdn/a_log",
    "text/layers/0/gdn/convolution",
)
QUANT_PROBE_OBJECTS = (
    "text/layers/3/attention/query_key",
    "text/layers/0/gdn/value_z",
    "vision/patch_embedding",
    "mtp/layer/attention/query_key_gate_value",
    "text/draft_head",
)


class VerificationError(ValueError):
    """The artifact does not satisfy the registered Ornith contract."""


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    payload_bytes: int


def verify_structure(artifact_path: str | Path) -> tuple[Artifact, StructureSummary]:
    artifact = Artifact(artifact_path)
    if artifact.identity.model_id != inventory.MODEL_ID:
        raise VerificationError(
            f"expected model_id={inventory.MODEL_ID}, got {artifact.identity.model_id}"
        )
    if artifact.identity.weights_id != inventory.WEIGHTS_ID:
        raise VerificationError(
            f"expected weights_id={inventory.WEIGHTS_ID}, got {artifact.identity.weights_id}"
        )

    tensors = [obj for obj in artifact.objects if isinstance(obj, TensorObject)]
    resources = [obj for obj in artifact.objects if isinstance(obj, ResourceObject)]
    if len(tensors) != len(inventory.TENSOR_SPECS):
        raise VerificationError(f"expected {len(inventory.TENSOR_SPECS)} tensors, got {len(tensors)}")
    if len(resources) != len(inventory.RESOURCE_SPECS):
        raise VerificationError(
            f"expected {len(inventory.RESOURCE_SPECS)} resources, got {len(resources)}"
        )
    if len(artifact.objects) != len(inventory.OBJECT_SPECS):
        raise VerificationError(
            f"expected {len(inventory.OBJECT_SPECS)} objects, got {len(artifact.objects)}"
        )

    actual_names = tuple(obj.name for obj in artifact.objects)
    expected_names = tuple(spec.name for spec in inventory.OBJECT_SPECS)
    if actual_names != expected_names:
        raise VerificationError("artifact object order does not match the Ornith inventory")

    return artifact, StructureSummary(
        objects=len(artifact.objects),
        tensors=len(tensors),
        resources=len(resources),
        payload_bytes=artifact.file_bytes - artifact.payload_offset,
    )


def _three_indices(count: int) -> torch.Tensor:
    return torch.tensor(tuple(dict.fromkeys((0, count // 2, count - 1))), dtype=torch.long)


def _logical_words(tensor: torch.Tensor, format_name: str) -> torch.Tensor:
    contiguous = tensor.detach().contiguous().cpu()
    if format_name == inventory.BF16:
        return contiguous.view(torch.int16)
    if format_name in (inventory.FP32, inventory.I32):
        return contiguous.view(torch.int32)
    raise TypeError(f"{format_name} is not a direct format")


def verify_direct_tensors(artifact: Artifact, model_dir: str | Path) -> int:
    count = 0
    with OrnithShardReader(model_dir) as reader:
        for object_name in DIRECT_PROBE_OBJECTS:
            obj = artifact.find(object_name)
            if not isinstance(obj, TensorObject):
                raise VerificationError(f"{object_name} is not a tensor")
            expected = recipe.materialize_recipe(recipe.RECIPES_BY_NAME[object_name], reader)
            stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
            expected_words = _logical_words(expected, obj.format).reshape(-1)
            stored_words = _logical_words(stored, obj.format).reshape(-1)
            indices = _three_indices(stored_words.numel())
            if not torch.equal(
                stored_words.index_select(0, indices),
                expected_words.index_select(0, indices),
            ):
                raise VerificationError(f"{object_name}: representative direct words differ")
            count += 1
    return count


def verify_quantized_tensors(artifact: Artifact) -> tuple[int, int]:
    specs = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    quantized = 0
    rows = 0
    for object_name in QUANT_PROBE_OBJECTS:
        obj = artifact.find(object_name)
        if not isinstance(obj, TensorObject):
            raise VerificationError(f"{object_name} is not a tensor")
        spec = specs[obj.name]
        if spec.format in (inventory.BF16, inventory.FP32, inventory.I32):
            raise VerificationError(f"{object_name} is not quantized")
        decode_row_split_codes(artifact.payload(obj), obj.format, obj.shape)
        rows += obj.shape[0]
        quantized += 1
    return quantized, rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args(argv)

    artifact, structure = verify_structure(args.artifact)
    direct_count = verify_direct_tensors(artifact, args.model)
    quant_probe_count, quant_rows = verify_quantized_tensors(artifact)
    print(
        f"structure: {structure.objects} objects ({structure.tensors} tensors, "
        f"{structure.resources} resources), {structure.payload_bytes} payload bytes",
        flush=True,
    )
    print(
        f"payloads: {direct_count} direct probes, {quant_probe_count} quantized probes, "
        f"{quant_rows} quantized rows",
        flush=True,
    )
    print("verification complete", flush=True)


if __name__ == "__main__":
    main()
