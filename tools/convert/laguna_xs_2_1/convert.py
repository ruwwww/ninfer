"""Convert the registered Laguna XS 2.1 checkpoint into one complete artifact.

Canonical invocation::

    python -m tools.convert.laguna_xs_2_1.convert \
      --model /path/to/Laguna-XS-2.1 \
      --out out/laguna_xs_2_1.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.laguna_xs_2_1.common import conversion as family_conversion
from tools.convert.laguna_xs_2_1 import inventory, recipe


RECIPE_ID = "laguna_xs_2_1-v1"

_ROOT_CONFIG = {
    "architectures": ["LagunaForCausalLM"],
    "model_type": "laguna",
    "tie_word_embeddings": False,
}
_TEXT_CONFIG = {
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "vocab_size": 100352,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "sliding_window": 512,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
    "mlp_only_layers": [0],
    "moe_routed_scaling_factor": 2.5,
    "gating": "per-head",
    "norm_topk_prob": True,
    "router_aux_loss_coef": 0.0,
}


ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config(model_dir: Path) -> dict[str, object]:
    return family_conversion.load_json(model_dir / "config.json")


def _check_members(
    scope: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    family_conversion.check_members(scope, actual, expected)


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact registered checkpoint dimensions and summarize them."""

    _check_members("config", config, _ROOT_CONFIG)
    _check_members("text_config (embedded)", config, _TEXT_CONFIG)

    # Validate layer_types match expected 1:3 pattern
    layer_types = config.get("layer_types", [])
    if len(layer_types) != 40:
        raise ValueError(
            f"config layer_types has {len(layer_types)} entries, expected 40"
        )
    for i, lt in enumerate(layer_types):
        expected_type = "full_attention" if (i % 4) == 0 else "sliding_attention"
        if lt != expected_type:
            raise ValueError(
                f"layer {i} type = {lt!r}, expected {expected_type!r}"
            )

    # Validate mlp_layer_types: layer 0 = dense, rest = sparse
    mlp_types = config.get("mlp_layer_types", [])
    if mlp_types[0] != "dense":
        raise ValueError(f"layer 0 mlp type = {mlp_types[0]!r}, expected 'dense'")
    for i in range(1, 40):
        if mlp_types[i] != "sparse":
            raise ValueError(
                f"layer {i} mlp type = {mlp_types[i]!r}, expected 'sparse'"
            )

    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "layers": len(layer_types),
        "full_attention_layers": sum(1 for lt in layer_types if lt == "full_attention"),
        "sliding_attention_layers": sum(1 for lt in layer_types if lt == "sliding_attention"),
        "dense_mlp_layers": 1,
        "moe_layers": 39,
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "head_dim": config["head_dim"],
        "num_experts": config["num_experts"],
        "sliding_window": config["sliding_window"],
        "max_position_embeddings": config["max_position_embeddings"],
    }


def preflight_inventory() -> None:
    """Establish the one complete target inventory and recipe pairing."""

    if len(inventory.OBJECT_SPECS) != 578:
        raise ValueError(
            f"registered inventory has {len(inventory.OBJECT_SPECS)} objects, expected 578"
        )
    recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    return family_conversion.load_resources(
        model_dir, inventory.RESOURCE_SPECS
    )


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    """Compute every payload-relative object offset for the full inventory."""

    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    object_plan: ObjectPlan


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    """Finish all checkpoint, inventory, and offset work before writing."""

    model = Path(model_dir)
    config_summary = validate_config(_load_config(model))
    preflight_inventory()
    source = recipe.preflight_sources(model)
    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        object_plan=object_plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> torch.Tensor:
    tensor = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME[spec.name],
        reader,
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    """Encode one materialized tensor according to its registered signature."""

    return family_conversion.encode_tensor_payload(tensor, spec, device)


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the complete registered conversion and return the report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with ShardReader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, reader)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
    }

    # Build minimal conversion report
    report = {
        "recipe_id": RECIPE_ID,
        "identity": {
            "model_id": inventory.MODEL_ID,
            "weights_id": inventory.WEIGHTS_ID,
        },
        "target_key": inventory.TARGET_KEY,
        "config_summary": preflight.config_summary,
        "source_preflight": {
            "recipe_count": preflight.source.recipe_count,
            "source_tensor_count": preflight.source.source_tensor_count,
            "source_shard_count": preflight.source.source_shard_count,
            "source_dtype_counts": preflight.source.source_dtype_counts,
        },
        "objects_written": len(inventory.OBJECT_SPECS),
        "format_counts": inventory.FORMAT_COUNTS,
        "layout_counts": inventory.LAYOUT_COUNTS,
        "elapsed_seconds": round(elapsed, 1),
        "final_bytes": final_bytes,
        "device": str(resolved_device),
        "arguments": arguments,
    }
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()