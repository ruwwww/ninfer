"""Convert Ornith-1.5-9B-NVFP4 into a groupwise-int NInfer artifact.

Canonical invocation::

    python -m tools.convert.ornith_1_5_9b.convert \
      --model /path/to/Ornith-1.5-9B-NVFP4 \
      --out models/ornith_1_5_9b.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import draft_head, inventory, official_resources, recipe
from .source import OrnithShardReader


RECIPE_ID = "ornith_1_5_9b-v1"

_ROOT_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_TEXT_CONFIG = {
    "num_hidden_layers": 32,
    "full_attention_interval": 4,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
}
_ROPE_CONFIG = {"rope_theta": 10000000, "mrope_section": [11, 11, 10]}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 4096,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config(model_dir: Path) -> dict[str, object]:
    return family_conversion.load_json(model_dir / "config.json")


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    family_conversion.check_members("config", config, _ROOT_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    family_conversion.check_members("text_config", text, _TEXT_CONFIG)
    expected_layer_types = tuple(
        "full_attention"
        if layer in inventory.FULL_ATTENTION_LAYERS
        else "linear_attention"
        for layer in range(32)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError("text_config.layer_types does not match the registered 32-layer schedule")
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    family_conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("vision_config", vision, _VISION_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(inventory.FULL_ATTENTION_LAYERS),
            "linear_attention": 32 - len(inventory.FULL_ATTENTION_LAYERS),
            "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
        "mtp_num_hidden_layers": text["mtp_num_hidden_layers"],
        "vision_token_ids": {
            name: config[name]
            for name in ("vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id")
        },
    }


def preflight_inventory() -> None:
    if (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    ) != (6, 387, 2, 12, 333, 734, 740):
        raise ValueError("registered inventory is incomplete")
    recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    return official_resources.load_official_resources(model_dir, inventory.RESOURCE_SPECS)


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    config_summary = validate_config(_load_config(model))
    preflight_inventory()
    source = recipe.preflight_sources(model)
    resources = load_resources(model)
    object_plan = build_object_plan({resource.name: resource.data for resource in resources})
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(model, config_summary, source, resources, draft, object_plan)


def materialize_tensor(spec, reader: OrnithShardReader, draft: draft_head.DraftHeadContext) -> torch.Tensor:
    derived = None
    if spec.name in (draft_head.DRAFT_HEAD_OBJECT, draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT):
        derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_head.materialize_draft_head_token_ids(draft)}
    tensor = recipe.materialize_recipe(recipe.RECIPES_BY_NAME[spec.name], reader, derived)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}")
    return tensor


def encode_tensor_payload(tensor: torch.Tensor, spec, device: str | torch.device) -> bytes:
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *, model_dir: str | Path, out_path: str | Path, arguments: Mapping[str, object],
    config_summary: Mapping[str, object], source_preflight: recipe.SourcePreflight,
    objects: Sequence[ArtifactObject], elapsed_seconds: float, final_bytes: int,
    device: torch.device, ranking_path: str | Path, revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY, recipe_id=RECIPE_ID, repo_root=_repo_root(),
        model_dir=model_dir, out_path=out_path, arguments=arguments,
        config_summary=config_summary, source_preflight=source_preflight, objects=objects,
        elapsed_seconds=elapsed_seconds, final_bytes=final_bytes, device=device,
        ranking_path=ranking_path, revision=revision, environment_summary=environment,
    )


def convert(model_dir: str | Path, out_path: str | Path, *, device: str | torch.device = "cuda") -> Path:
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
    with OrnithShardReader(model) as reader:
        with ArtifactWriter(output, ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID), preflight.object_plan.specs) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, reader, preflight.draft)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)
    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = build_conversion_report(
        model_dir=model, out_path=output,
        arguments={"model": str(model_dir), "out": str(out_path), "device": requested_device},
        config_summary=preflight.config_summary, source_preflight=preflight.source,
        objects=preflight.object_plan.objects, elapsed_seconds=elapsed,
        final_bytes=final_bytes, device=resolved_device, ranking_path=ranking,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(arguments.model, arguments.out, device=arguments.device)


if __name__ == "__main__":
    main()
