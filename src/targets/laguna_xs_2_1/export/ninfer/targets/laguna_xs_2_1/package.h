#pragma once

// Public contract for the Laguna XS 2.1 target package.

#include "artifact/identity.h"
#include "core/device.h"
#include "infra/string_view.h"
#include "ninfer/targets/qwen3_6/prepared_prompt.h"
#include "ninfer/targets/qwen3_6/sequence_plan.h"
#include "ninfer/targets/qwen3_6/program.h"
#include "runtime/engine_options.h"

#include <memory>
#include <string_view>

namespace ninfer::targets::laguna_xs_2_1 {

namespace detail {
class LoadPlan;
class LoadedModel;
struct SequencePlan;
struct Variant;
enum class WeightsProfile : std::uint8_t { kGroupwiseInt = 0 };
}  // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "laguna-xs-2.1";
    static constexpr std::string_view target_key = "laguna_xs_2_1";

    using WeightsProfile = detail::WeightsProfile;
    using LoadPlan       = detail::LoadPlan;
    using LoadedModel    = detail::LoadedModel;
    using SequencePlan   = detail::SequencePlan;
    using Frontend       = qwen3_6::Frontend;
    using Program        = qwen3_6::Program<detail::Variant>;

    static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    static LoadPlan plan_load(artifact::Binder& binder,
                               const EngineOptions& options,
                               WeightsProfile weights_profile);
    static std::unique_ptr<LoadedModel> construct_loaded_model(LoadPlan&& plan,
                                                                artifact::MaterializedArtifact&& materialized);
    static Frontend make_frontend(const LoadedModel& model);
    static SequencePlan plan_sequence(DeviceContext& device,
                                       const EngineOptions& options,
                                       WeightsProfile weights_profile);
    static std::unique_ptr<Program> create_program(const LoadedModel& model,
                                                    SequencePlan&& plan,
                                                    DeviceContext& device);
};

}  // namespace ninfer::targets::laguna_xs_2_1