#include "targets/laguna_xs_2_1/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include "targets/laguna_xs_2_1/impl/config.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::laguna_xs_2_1::detail {
namespace {

using artifact::NumericFormat;

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::kGroupwiseInt:
        return NumericFormat::Q4G64_F16S;
    }
    throw std::invalid_argument("laguna_xs_2_1: invalid weights profile");
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{
        .object = artifact::bind_device_tensor(binder, name, format, shape),
        .format = format,
    };
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

void bind_layer(artifact::Binder& binder, int layer, BindingPlan& out) {
    auto& lp = out.layers[layer];
    lp.layer_index = layer;
    lp.is_full_attention = (layer % 4) == 0;
    lp.is_swa = (layer % 4) != 0;
    lp.is_dense_mlp = (layer == 0);
    lp.is_moe = (layer > 0);
    lp.q_heads = lp.is_full_attention ? TextConfig::full_q_heads : TextConfig::swa_q_heads;
    lp.q_rows = lp.is_full_attention ? TextConfig::full_q_rows : TextConfig::swa_q_rows;

    const std::string prefix = "text/layers/" + std::to_string(layer) + "/";

    // Attention weights
    lp.q_proj = bind_weight(binder, prefix + "attention/q_proj", NumericFormat::Q4G64_F16S,
                            {static_cast<std::uint64_t>(lp.q_rows), TextConfig::hidden});
    lp.k_proj = bind_weight(binder, prefix + "attention/k_proj", NumericFormat::Q4G64_F16S,
                            {static_cast<std::uint64_t>(TextConfig::kv_rows), TextConfig::hidden});
    lp.v_proj = bind_weight(binder, prefix + "attention/v_proj", NumericFormat::Q4G64_F16S,
                            {static_cast<std::uint64_t>(TextConfig::kv_rows), TextConfig::hidden});
    lp.q_norm = artifact::bind_device_tensor(
        binder, prefix + "attention/q_norm", NumericFormat::BF16,
        {static_cast<std::uint64_t>(TextConfig::head_dim)});
    lp.k_norm = artifact::bind_device_tensor(
        binder, prefix + "attention/k_norm", NumericFormat::BF16,
        {static_cast<std::uint64_t>(TextConfig::head_dim)});
    lp.g_proj = artifact::bind_device_tensor(
        binder, prefix + "attention/g_proj", NumericFormat::BF16,
        {static_cast<std::uint64_t>(lp.q_heads)});
    lp.o_proj = bind_weight(binder, prefix + "attention/o_proj", NumericFormat::Q5G64_F16S,
                            {static_cast<std::uint64_t>(TextConfig::hidden),
                             static_cast<std::uint64_t>(lp.q_rows)});

    // MLP/MoE weights
    if (lp.is_dense_mlp) {
        // Layer 0: Dense SwiGLU MLP, intermediate = 8192
        lp.mlp.gate_proj = bind_weight(binder, prefix + "mlp/gate_proj", NumericFormat::Q4G64_F16S,
                                       {8192ULL, static_cast<std::uint64_t>(TextConfig::hidden)});
        lp.mlp.up_proj = bind_weight(binder, prefix + "mlp/up_proj", NumericFormat::Q4G64_F16S,
                                     {8192ULL, static_cast<std::uint64_t>(TextConfig::hidden)});
        lp.mlp.down_proj =
            bind_weight(binder, prefix + "mlp/down_proj", NumericFormat::Q5G64_F16S,
                        {static_cast<std::uint64_t>(TextConfig::hidden), 8192ULL});
    } else {
        // Layers 1-39: MoE
        lp.mlp.router_gate = bind_weight(
            binder, prefix + "moe/router_gate", NumericFormat::Q4G64_F16S,
            {256ULL, static_cast<std::uint64_t>(TextConfig::hidden)});
        lp.mlp.routed_gate_up = bind_weight(
            binder, prefix + "moe/routed_gate_up", NumericFormat::Q4G64_F16S,
            {256ULL * 512ULL, static_cast<std::uint64_t>(TextConfig::hidden)});
        lp.mlp.routed_down = bind_weight(
            binder, prefix + "moe/routed_down", NumericFormat::Q5G64_F16S,
            {256ULL * 2048ULL, 512ULL});
        lp.mlp.shared_gate_up = bind_weight(
            binder, prefix + "moe/shared_gate_up", NumericFormat::W8G32_F16S,
            {512ULL, static_cast<std::uint64_t>(TextConfig::hidden)});
        lp.mlp.shared_down =
            bind_weight(binder, prefix + "moe/shared_down", NumericFormat::W8G32_F16S,
                        {static_cast<std::uint64_t>(TextConfig::hidden), 512ULL});
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                                qwen3_6::StartupFeatures /*features*/) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;

    // Frontend resources
    out.frontend = qwen3_6::bind_frontend_resources(binder);

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);

    // Token embedding
    out.token_embedding =
        bind_weight(binder, "text/token_embedding", vocabulary_format,
                    {100352ULL, static_cast<std::uint64_t>(TextConfig::hidden)});

    // Per-layer bindings
    for (int layer = 0; layer < 40; ++layer) {
        bind_layer(binder, layer, out);
    }

    // Final norm and output head
    out.final_norm = artifact::bind_device_tensor(
        binder, "text/final_norm", NumericFormat::BF16,
        {static_cast<std::uint64_t>(TextConfig::hidden)});
    out.output_head =
        bind_weight(binder, "text/output_head", vocabulary_format, {100352ULL, 2048ULL});

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData materialize_weights(BindingPlan&& plan,
                                     artifact::MaterializedArtifact&& materialized) {
    LoadedModelData data;
    data.features = plan.features;

    // Frontend resources
    data.frontend = qwen3_6::take_frontend_resources(materialized, plan.frontend);

    // Token embedding
    data.token_embedding = materialized_weight(materialized, plan.token_embedding, 100352,
                                                TextConfig::hidden);

    // Per-layer weights
    for (int layer = 0; layer < 40; ++layer) {
        auto& lp = plan.layers[layer];

        data.layers[layer].attn.q_proj = materialized_weight(materialized, lp.q_proj, lp.q_rows,
                                                              TextConfig::hidden);
        data.layers[layer].attn.k_proj = materialized_weight(materialized, lp.k_proj,
                                                              TextConfig::kv_rows, TextConfig::hidden);
        data.layers[layer].attn.v_proj = materialized_weight(materialized, lp.v_proj,
                                                              TextConfig::kv_rows, TextConfig::hidden);
        data.layers[layer].attn.q_norm = artifact::materialized_tensor(
            materialized, lp.q_norm, NumericFormat::BF16, {TextConfig::head_dim});
        data.layers[layer].attn.k_norm = artifact::materialized_tensor(
            materialized, lp.k_norm, NumericFormat::BF16, {TextConfig::head_dim});
        data.layers[layer].attn.g_proj = artifact::materialized_tensor(
            materialized, lp.g_proj, NumericFormat::BF16, {lp.q_heads});
        data.layers[layer].attn.o_proj = materialized_weight(materialized, lp.o_proj,
                                                              TextConfig::hidden, lp.q_rows);

        if (lp.is_dense_mlp) {
            data.layers[layer].mixer = DenseMlpPayload{
                .gate_proj = materialized_weight(materialized, lp.mlp.gate_proj, 8192,
                                                  TextConfig::hidden),
                .up_proj = materialized_weight(materialized, lp.mlp.up_proj, 8192,
                                                TextConfig::hidden),
                .down_proj = materialized_weight(materialized, lp.mlp.down_proj, TextConfig::hidden,
                                                  8192),
            };
        } else {
            ops::SparseMoeWeights moe_weights{};
            moe_weights.router_shared_gate = materialized_weight(materialized, lp.mlp.router_gate,
                                                                  256, TextConfig::hidden);
            moe_weights.routed_gate_up = materialized_weight(materialized, lp.mlp.routed_gate_up,
                                                              256 * 512, TextConfig::hidden);
            moe_weights.routed_down = materialized_weight(materialized, lp.mlp.routed_down,
                                                           256 * 2048, 512);
            moe_weights.shared_gate_up = materialized_weight(materialized, lp.mlp.shared_gate_up,
                                                              512, TextConfig::hidden);
            moe_weights.shared_down = materialized_weight(materialized, lp.mlp.shared_down,
                                                           TextConfig::hidden, 512);
            data.layers[layer].mixer = SparseMoePayload{.op = moe_weights};
        }
    }

    data.final_norm = artifact::materialized_tensor(
        materialized, plan.final_norm, NumericFormat::BF16, {TextConfig::hidden});
    data.output_head = materialized_weight(materialized, plan.output_head, 100352, 2048);

    return data;
}

} // namespace ninfer::targets::laguna_xs_2_1::detail