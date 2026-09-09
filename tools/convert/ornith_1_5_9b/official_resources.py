"""Pinned frontend resources for Ornith-1.5-9B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from tools.convert.qwen3_6.common.conversion import ResourcePayload, load_resources
from tools.convert.qwen3_6.common.inventory import ResourceSpec


OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "frontend/tokenizer_config.json": (
        "def4680dd6b26fd704414f1e868c48973d34990f55560abe88f8d3bee5d2be9c"
    ),
    "frontend/chat_template.jinja": (
        "9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235"
    ),
    "frontend/generation_config.json": (
        "f155b072c600dc4a39b17102a94850113a510d6f6b4aef3c893c82ead226b9b4"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}

# The release's tokenizer_config.json embeds the Qwen chat template, while the
# adjacent Ornith chat_template.jinja is the authoritative resource selected by
# this profile.  Keep the source hash pinned before normalizing that stale field
# into the self-consistent resource stored in the artifact.
SOURCE_TOKENIZER_CONFIG_SHA256 = (
    "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
)


def validate_official_resource_hashes(actual_hashes: Mapping[str, str]) -> None:
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    if tuple(actual_hashes) != expected_names:
        raise ValueError(
            "Ornith frontend resource set mismatch: "
            f"expected {expected_names!r}, got {tuple(actual_hashes)!r}"
        )
    for name, expected in OFFICIAL_RESOURCE_SHA256.items():
        actual = actual_hashes[name]
        if actual != expected:
            raise ValueError(
                f"official Ornith resource hash mismatch for {name.removeprefix('frontend/')}: "
                f"expected {expected}, got {actual}"
            )


def validate_official_resources(resources: Sequence[ResourcePayload]) -> None:
    hashes = {
        resource.name: hashlib.sha256(resource.data).hexdigest()
        for resource in resources
    }
    if len(hashes) != len(resources):
        raise ValueError("Ornith frontend resource set contains duplicate names")
    validate_official_resource_hashes(hashes)


def load_official_resources(
    model_dir: str | Path, resource_specs: Sequence[ResourceSpec]
) -> tuple[ResourcePayload, ...]:
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    actual_names = tuple(spec.name for spec in resource_specs)
    if actual_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official Ornith profile: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    resources = list(load_resources(model_dir, resource_specs))
    source_tokenizer_config = next(
        resource.data
        for resource in resources
        if resource.name == "frontend/tokenizer_config.json"
    )
    source_hash = hashlib.sha256(source_tokenizer_config).hexdigest()
    if source_hash != SOURCE_TOKENIZER_CONFIG_SHA256:
        raise ValueError(
            "official Ornith source tokenizer_config.json hash mismatch: "
            f"expected {SOURCE_TOKENIZER_CONFIG_SHA256}, got {source_hash}"
        )

    by_name = {resource.name: resource for resource in resources}
    chat_template = by_name["frontend/chat_template.jinja"].data.decode("utf-8")
    tokenizer_config = json.loads(source_tokenizer_config.decode("utf-8"))
    if not isinstance(tokenizer_config, dict):
        raise ValueError("Ornith tokenizer_config.json root must be an object")
    if tokenizer_config.get("chat_template") != chat_template:
        tokenizer_config["chat_template"] = chat_template
        normalized = json.dumps(tokenizer_config, indent=4).encode("utf-8")
        if len(normalized) > len(source_tokenizer_config):
            raise ValueError(
                "normalized Ornith tokenizer_config.json exceeds its source size"
            )
        # Preserve the registered resource allocation and all following object
        # offsets.  JSON permits trailing whitespace, and the runtime compares
        # the parsed chat_template value rather than raw resource bytes.
        normalized += b" " * (len(source_tokenizer_config) - len(normalized))
        by_name["frontend/tokenizer_config.json"] = ResourcePayload(
            "frontend/tokenizer_config.json", normalized
        )
    resources = [by_name[spec.name] for spec in resource_specs]
    validate_official_resources(resources)
    return tuple(resources)


__all__ = [
    "OFFICIAL_RESOURCE_SHA256",
    "SOURCE_TOKENIZER_CONFIG_SHA256",
    "load_official_resources",
    "validate_official_resource_hashes",
    "validate_official_resources",
]
