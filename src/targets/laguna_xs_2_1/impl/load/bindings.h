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
#include "ops/sparse_moe.h"

#include <array>
#include <cstdint>
#include <memory>
#include <variant>

namespace ninfer::targets::laguna_xs_2_1::detail {

// ---- Weight plan ----

struct WeightPlan {
    artifact::ObjectHandle object;
};

// ---- WeightsProfile ----

enum class WeightsProfile : std::uint8_t {
    kGroupwiseInt = 0,
};

// ---- Payload types (runtime weight views per layer) ----

struct AttentionProjectionPayload {
    Weight q_proj;
    Weight k_proj;
    Weight v_proj;
    Weight q_norm;
    Weight k_norm;
    Weight g_proj;
    Weight o_proj;
};

struct SparseMoePayload {
    ops::SparseMoeWeights op;
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
    WeightPlan q_norm;
    WeightPlan k_norm;
    WeightPlan g_proj;
    WeightPlan o_proj;

    struct {
        WeightPlan gate_proj;
        WeightPlan up_proj;
        WeightPlan down_proj;
        WeightPlan router_gate;
        WeightPlan routed_gate_up;
        WeightPlan routed_down;
        WeightPlan shared_gate_up;
        WeightPlan shared_down;
    } mlp;
};

// ---- Full binding plan ----

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    WeightPlan token_embedding;
    std::array<LayerBindingPlan, 40> layers;
    WeightPlan final_norm;
    WeightPlan output_head;
};

// ---- Loaded model data ----

struct LoadedModelData {
    qwen3_6::FrontendResourceStore frontend;
    qwen3_6::StartupFeatures features;
    Weight token_embedding;
    struct {
        AttentionProjectionPayload attn;
        PostMixerPayload mixer;
    } layers[40];
    Weight final_norm;
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