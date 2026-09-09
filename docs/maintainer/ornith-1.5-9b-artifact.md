# Ornith-1.5-9B Artifact Reference

The registered Ornith identity is `ornith-1.5-9b/groupwise-int`, with target key
`ornith_1_5_9b`. Its runtime uses the Qwen3.5-9B-shaped family execution package: 32 layers,
24 Gated DeltaNet layers, 8 full-attention layers, and one MTP layer. The public artifact keeps
the complete Text, Vision, MTP, and frontend resource inventory.

## Source and conversion

The source is the official
[`ornith-ai/Ornith-1.5-9B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-9B-NVFP4)
checkpoint. The converter is [`tools/convert/ornith_1_5_9b/`](../../tools/convert/ornith_1_5_9b/):

```bash
python3 -m tools.convert.ornith_1_5_9b.convert \
  --model /path/to/Ornith-1.5-9B-NVFP4 \
  --out models/ornith_1_5_9b.ninfer
```

The checkpoint contains one safetensors file with NVFP4 U8 MLP/output-head weights and FP8
linear-attention weights. `source.py` decodes those representations on demand to logical BF16;
the normal groupwise-int converter then emits NInfer Q4/Q5/Q6/W8 tensors. The MTP tensors are
already BF16 in the source checkpoint and use the registered W8 MTP profile.

The six frontend resources are pinned independently from Qwen3.5-9B. The Ornith chat template
digest is registered as `ThinkingToggle`, preserving `<think>` reasoning and final-answer
separation for CLI and HTTP serving. The source tokenizer config's stale embedded Qwen template is
validated by source hash and normalized to the adjacent Ornith template before artifact writing.

## Verification

```bash
python3 -m tools.convert.ornith_1_5_9b.verify \
  --artifact models/ornith_1_5_9b.ninfer \
  --model /path/to/Ornith-1.5-9B-NVFP4
```

The verifier checks artifact identity, complete object order/counts, representative direct
payloads against the dequantizing source reader, and representative row-split quantized payloads.
