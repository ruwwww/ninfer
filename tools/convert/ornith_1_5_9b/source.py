"""Lazy decoding of the Ornith NVFP4/FP8 safetensors source."""

from __future__ import annotations

from pathlib import Path

import torch

from tools.convert.common.safetensors import ShardReader, TensorMetadata


_E2M1_VALUES = torch.tensor(
    (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
    dtype=torch.float32,
)


def _weight_scale_name(name: str) -> str:
    return name.removesuffix(".weight") + ".weight_scale"


def _global_scale_name(name: str) -> str:
    return name.removesuffix(".weight") + ".weight_scale_2"


def _decode_nvfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError("NVFP4 packed weights must be a rank-two U8 tensor")
    rows, packed_width = packed.shape
    width = packed_width * 2
    if width % 16 != 0:
        raise ValueError("NVFP4 logical width must be divisible by 16")
    if block_scales.dtype != torch.float8_e4m3fn or tuple(block_scales.shape) != (
        rows,
        width // 16,
    ):
        raise ValueError("NVFP4 block scale shape or dtype is invalid")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise ValueError("NVFP4 global scale must be one F32 value")

    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=2).reshape(rows, width)
    values = _E2M1_VALUES[codes.to(torch.long)]
    scales = block_scales.float().repeat_interleave(16, dim=1)
    return (values * scales * global_scale.reshape(()).float()).to(torch.bfloat16)


def _decode_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        raise TypeError("FP8 weights must be a rank-two F8_E4M3 tensor")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("FP8 weight scale must be one F32 value")
    return (weight.float() * scale.reshape(()).float()).to(torch.bfloat16)


def _logical_metadata(metadata: TensorMetadata) -> TensorMetadata:
    if metadata.name.endswith(".weight") and metadata.dtype == "U8":
        if len(metadata.shape) != 2:
            raise ValueError(f"{metadata.name}: NVFP4 weight must be rank two")
        return TensorMetadata(
            metadata.name,
            metadata.shard,
            (metadata.shape[0], metadata.shape[1] * 2),
            "BF16",
        )
    if metadata.name.endswith(".weight") and metadata.dtype == "F8_E4M3":
        return TensorMetadata(metadata.name, metadata.shard, metadata.shape, "BF16")
    return metadata


class OrnithShardReader(ShardReader):
    """Read the single Ornith safetensors file as logical BF16 source tensors."""

    def __init__(self, model_dir: str | Path) -> None:
        source = ShardReader.from_file(Path(model_dir) / "model.safetensors")
        self.model_dir = source.model_dir
        self.weight_map = source.weight_map
        self._reset_handle()

    def get(self, name: str) -> torch.Tensor:
        tensor = super().get(name)
        if not name.endswith(".weight"):
            return tensor
        if tensor.dtype == torch.uint8:
            return _decode_nvfp4(
                tensor,
                super().get(_weight_scale_name(name)),
                super().get(_global_scale_name(name)),
            )
        if tensor.dtype == torch.float8_e4m3fn:
            return _decode_fp8(tensor, super().get(_weight_scale_name(name)))
        return tensor

    def metadata(self, names):
        return {
            name: _logical_metadata(metadata)
            for name, metadata in super().metadata(names).items()
        }


__all__ = ["OrnithShardReader"]
