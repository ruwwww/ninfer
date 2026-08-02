#include "targets/laguna_xs_2_1/impl/load/bindings.h"

#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "targets/laguna_xs_2_1/impl/config.h"

#include <stdexcept>
#include <string>

namespace ninfer::targets::laguna_xs_2_1::detail {
namespace {

constexpr int kLayers = 40;
constexpr int kHidden = TextConfig::hidden;

void bind_weight(WeightPlan& plan, artifact::Reader& reader, std::string_view path) {
    auto obj = reader.resolve(std::string(path));
    if (!obj.has_value()) {
        throw std::runtime_error("missing artifact object: " + std::string(path));
    }
    plan.object = *obj;
}

ArtifactLoadPlan do_bind(artifact::Binder& binder, WeightsProfile /*profile*/) {
    artifact::Reader& reader = binder.reader();
    qwen3_6::StartupFeatures features = qwen3_6::detect_startup_features(reader);

    BindingPlan plan;
    plan.frontend = qwen3_6::plan_frontend_resources(reader);
    plan.features = features;

    // Token embedding
    bind_weight(plan.token_embedding, reader, "text/token_embedding");

    // Per-layer bindings
    for (int layer = 0; layer < kLayers; ++layer) {
        auto& lp = plan.layers[layer];
        lp.layer_index = layer;
        lp.is_full_attention = (layer % 4) == 0;
        lp.is_swa = (layer % 4) != 0;
        lp.is_dense_mlp = (layer == 0);
        lp.is_moe = (layer > 0);
        lp.q_heads = lp.is_full_attention ? TextConfig::full_q_heads : TextConfig::swa_q_heads;
        lp.q_rows = lp.is_full_attention ? TextConfig::full_q_rows : TextConfig::swa_q_rows;

        std::string prefix = "text/layers/" + std::to_string(layer) + "/";

        bind_weight(lp.q_proj, reader, prefix + "attention/q_proj");
        bind_weight(lp.k_proj, reader, prefix + "attention/k_proj");
        bind_weight(lp.v_proj, reader, prefix + "attention/v_proj");
        bind_weight(lp.q_norm, reader, prefix + "attention/q_norm");
        bind_weight(lp.k_norm, reader, prefix + "attention/k_norm");
        bind_weight(lp.g_proj, reader, prefix + "attention/g_proj");
        bind_weight(lp.o_proj, reader, prefix + "attention/o_proj");

        if (lp.is_dense_mlp) {
            bind_weight(lp.mlp.gate_proj, reader, prefix + "mlp/gate_proj");
            bind_weight(lp.mlp.up_proj, reader, prefix + "mlp/up_proj");
            bind_weight(lp.mlp.down_proj, reader, prefix + "mlp/down_proj");
        } else {
            bind_weight(lp.mlp.router_gate, reader, prefix + "moe/router_gate");
            bind_weight(lp.mlp.routed_gate_up, reader, prefix + "moe/routed_gate_up");
            bind_weight(lp.mlp.routed_down, reader, prefix + "moe/routed_down");
            bind_weight(lp.mlp.shared_gate_up, reader, prefix + "moe/shared_gate_up");
            bind_weight(lp.mlp.shared_down, reader, prefix + "moe/shared_down");
        }
    }

    bind_weight(plan.final_norm, reader, "text/final_norm");
    bind_weight(plan.output_head, reader, "text/output_head");

    // Compute materialization plan from object handles
    artifact::MaterializationPlan mat_plan;
    // Add token embedding
    mat_plan.objects.push_back(plan.token_embedding.object);
    // Add all layer objects
    for (int layer = 0; layer < kLayers; ++layer) {
        auto& lp = plan.layers[layer];
        mat_plan.objects.push_back(lp.q_proj.object);
        mat_plan.objects.push_back(lp.k_proj.object);
        mat_plan.objects.push_back(lp.v_proj.object);
        mat_plan.objects.push_back(lp.q_norm.object);
        mat_plan.objects.push_back(lp.k_norm.object);
        mat_plan.objects.push_back(lp.g_proj.object);
        mat_plan.objects.push_back(lp.o_proj.object);
        if (lp.is_dense_mlp) {
            mat_plan.objects.push_back(lp.mlp.gate_proj.object);
            mat_plan.objects.push_back(lp.mlp.up_proj.object);
            mat_plan.objects.push_back(lp.mlp.down_proj.object);
        } else {
            mat_plan.objects.push_back(lp.mlp.router_gate.object);
            mat_plan.objects.push_back(lp.mlp.routed_gate_up.object);
            mat_plan.objects.push_back(lp.mlp.routed_down.object);
            mat_plan.objects.push_back(lp.mlp.shared_gate_up.object);
            mat_plan.objects.push_back(lp.mlp.shared_down.object);
        }
    }
    mat_plan.objects.push_back(plan.final_norm.object);
    mat_plan.objects.push_back(plan.output_head.object);

    return ArtifactLoadPlan{.materialization = std::move(mat_plan), .bindings = std::move(plan)};
}

}  // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder,
                                WeightsProfile profile,
                                qwen3_6::StartupFeatures features) {
    (void)features;
    return do_bind(binder, profile);
}

LoadedModelData materialize_weights(BindingPlan&& plan,
                                     artifact::MaterializedArtifact&& artifact) {
    LoadedModelData data;

    // Frontend resources are handled by the shared frontend machinery
    data.features = plan.features;

    // Token embedding
    data.token_embedding = artifact.bind_weight(plan.token_embedding);

    // Per-layer weights
    for (int layer = 0; layer < kLayers; ++layer) {
        auto& lp = plan.layers[layer];

        data.layers[layer].attn.q_proj = artifact.bind_weight(lp.q_proj);
        data.layers[layer].attn.k_proj = artifact.bind_weight(lp.k_proj);
        data.layers[layer].attn.v_proj = artifact.bind_weight(lp.v_proj);
        data.layers[layer].attn.q_norm = artifact.bind_weight(lp.q_norm);
        data.layers[layer].attn.k_norm = artifact.bind_weight(lp.k_norm);
        data.layers[layer].attn.g_proj = artifact.bind_weight(lp.g_proj);
        data.layers[layer].attn.o_proj = artifact.bind_weight(lp.o_proj);

        if (lp.is_dense_mlp) {
            data.layers[layer].mixer = DenseMlpPayload{
                .gate_proj = artifact.bind_weight(lp.mlp.gate_proj),
                .up_proj = artifact.bind_weight(lp.mlp.up_proj),
                .down_proj = artifact.bind_weight(lp.mlp.down_proj),
            };
        } else {
            ops::SparseMoeWeights moe_weights{};
            moe_weights.router_shared_gate = artifact.bind_weight(lp.mlp.router_gate);
            moe_weights.routed_gate_up = artifact.bind_weight(lp.mlp.routed_gate_up);
            moe_weights.routed_down = artifact.bind_weight(lp.mlp.routed_down);
            moe_weights.shared_gate_up = artifact.bind_weight(lp.mlp.shared_gate_up);
            moe_weights.shared_down = artifact.bind_weight(lp.mlp.shared_down);
            data.layers[layer].mixer = SparseMoePayload{.op = moe_weights};
        }
    }

    data.final_norm = artifact.bind_weight(plan.final_norm);
    data.output_head = artifact.bind_weight(plan.output_head);

    return data;
}

}  // namespace ninfer::targets::laguna_xs_2_1::detail