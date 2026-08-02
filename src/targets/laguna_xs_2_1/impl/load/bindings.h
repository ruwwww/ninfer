#pragma once

// Binding plans and payload types for Laguna XS 2.1.
//
// Maps .ninfer artifact objects to runtime weight views used by the Variant's
// execution leaves.

#include <ninfer/targets/laguna_xs_2_1/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <array>
#include <cstdint>
#include <memory>
#include <variant>

namespace ninfer::targets::laguna_xs_2_1::detail {

// ---- Weight plan ----

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

// ---- Payload types (runtime weight views per layer) ----

struct AttentionProjectionPayload {
    Weight q_proj;
    Weight k_proj;
    Weight v_proj;
    Tensor q_norm;
    Tensor k_norm;
    Tensor g_proj;
    Weight o_proj;
};

struct GdnProjectionPayload {};

struct SparseMoePayload {
    ops::SparseMoeWeights op;
    Weight e_score_correction_bias;  // [num_experts] load balancing bias
    float moe_routed_scaling_factor = 2.5f;
};

struct DenseMlpPayload {
    Weight gate_proj;
    Weight up_proj;
    Weight down_proj;
};

using PostMixerPayload = std::variant<DenseMlpPayload, SparseMoePayload>;

// ---- Layer binding plan ----

struct LayerBindingPlan {
    int layer_index = 0;
    bool is_full_attention = false;
    bool is_swa = false;
    bool is_dense_mlp = false;
    bool is_moe = false;
    int q_heads = 0;
    int q_rows = 0;

    WeightPlan q_proj;
    WeightPlan k_proj;
    WeightPlan v_proj;
    artifact::ObjectHandle q_norm;
    artifact::ObjectHandle k_norm;
    artifact::ObjectHandle g_proj;
    WeightPlan o_proj;

    struct {
        WeightPlan router_gate;
        WeightPlan routed_gate_up;
        WeightPlan routed_down;
        WeightPlan shared_gate_up;
        WeightPlan shared_down;
        // Dense MLP (layer 0 only)
        WeightPlan gate_proj;
        WeightPlan up_proj;
        WeightPlan down_proj;
    } mlp;
};

// ---- Full binding plan ----

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    WeightPlan token_embedding;
    std::array<LayerBindingPlan, 40> layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
};

// ---- Loaded model data ----

struct LoadedModelData {
    qwen3_6::FrontendResources frontend;
    qwen3_6::StartupFeatures features;
    Weight token_embedding;
    struct {
        AttentionProjectionPayload attn;
        PostMixerPayload mixer;
    } layers[40];
    Tensor final_norm;
    Weight output_head;
};

// ---- Runtime model view alias ----

template <class...>
using RuntimeModelView = LoadedModelData;

// ---- ArtifactLoadPlan (internal, passed through LoadPlan) ----

struct ArtifactLoadPlan {
    artifact::MaterializationPlan materialization;
    BindingPlan bindings;
};

// ---- bind_artifact: main entry point ----

/**
 * Bind artifact objects to the target-specific binding plan.
 *
 * Takes artifact::Binder& by reference (as per the standard pattern) and
 * returns an ArtifactLoadPlan containing both the materialization plan
 * and the bound weights.
 */
ArtifactLoadPlan bind_artifact(artifact::Binder& binder,
                                WeightsProfile profile,
                                qwen3_6::StartupFeatures features);

// ---- materialize_weights: convert bound plans to runtime weights ----

/**
 * Materialize bound weight plans into runtime Weight/Tensor objects
 * using the materialized artifact data.
 */
LoadedModelData materialize_weights(BindingPlan&& plan,
                                     artifact::MaterializedArtifact&& materialized);

// ---- PIMPL implementations ----

class LoadPlan::Impl {
public:
    Impl(WeightsProfile profile_in, ArtifactLoadPlan plan_in)
        : profile(profile_in), plan(std::move(plan_in)) {}

    WeightsProfile profile;
    ArtifactLoadPlan plan;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile profile_in, BindingPlan&& bindings,
         LoadedModelData data_in)
        : profile(profile_in),
          bindings(std::move(bindings)),
          data(std::move(data_in)) {}

    WeightsProfile profile;
    BindingPlan bindings;
    LoadedModelData data;
};

}  // namespace ninfer::targets::laguna_xs_2_1::detail