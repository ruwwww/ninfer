from __future__ import annotations

from safetensors.torch import save_file
import torch

from tools.convert.ornith_1_5_9b.source import OrnithShardReader


def test_reader_decodes_nvfp4_blocks_and_fp8_scalar_scales(tmp_path) -> None:
    nvfp4_codes = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x8F, 0xED, 0xCB, 0xA9]], dtype=torch.uint8)
    nvfp4_scales = torch.tensor([[0x38]], dtype=torch.uint8).view(torch.float8_e4m3fn)
    fp8_codes = torch.tensor(
        [[0x38, 0xB8, 0x40, 0xC0] * 4], dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    tensors = {
        "nv.weight": nvfp4_codes,
        "nv.weight_scale": nvfp4_scales,
        "nv.weight_scale_2": torch.tensor(2.0, dtype=torch.float32),
        "fp.weight": fp8_codes,
        "fp.weight_scale": torch.tensor(0.5, dtype=torch.float32),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    with OrnithShardReader(tmp_path) as reader:
        nvfp4 = reader.get("nv.weight")
        fp8 = reader.get("fp.weight")

    assert nvfp4.dtype == torch.bfloat16
    assert tuple(nvfp4.shape) == (1, 16)
    torch.testing.assert_close(
        nvfp4.float(),
        torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
              -12.0, -0.0, -6.0, -8.0, -3.0, -4.0, -1.0, -2.0]],
            dtype=torch.float32,
        ),
        atol=0.0,
        rtol=0.0,
    )
    assert fp8.dtype == torch.bfloat16
    torch.testing.assert_close(
        fp8.float(),
        torch.tensor([[0.5, -0.5, 1.0, -1.0] * 4], dtype=torch.float32),
        atol=0.0,
        rtol=0.0,
    )


def test_reader_metadata_exposes_logical_bf16_shapes(tmp_path) -> None:
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    save_file(
        {
            name: torch.zeros((2, 8), dtype=torch.uint8),
            name.removesuffix(".weight") + ".weight_scale": torch.zeros(
                (2, 1), dtype=torch.float8_e4m3fn
            ),
            name.removesuffix(".weight") + ".weight_scale_2": torch.tensor(
                1.0, dtype=torch.float32
            ),
        },
        tmp_path / "model.safetensors",
    )
    with OrnithShardReader(tmp_path) as reader:
        metadata = reader.metadata([name])[name]

    assert metadata.shape == (2, 16)
    assert metadata.dtype == "BF16"
