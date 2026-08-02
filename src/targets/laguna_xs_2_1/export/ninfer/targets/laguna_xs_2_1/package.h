#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::laguna_xs_2_1 {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    kGroupwiseInt = 0,
};

using Frontend       = qwen3_6::Frontend;
using PreparedPrompt = qwen3_6::PreparedPrompt;
using OutputSession  = qwen3_6::OutputSession;
using SequencePlan   = qwen3_6::SequencePlan<detail::Variant>;
using RequestPlan    = qwen3_6::RequestPlan<detail::Variant>;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct laguna_xs_2_1::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct laguna_xs_2_1::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "laguna-xs-2.1";
    static constexpr std::string_view target_key = "laguna_xs_2_1";

    using WeightsProfile = detail::WeightsProfile;
    using LoadPlan       = detail::LoadPlan;
    using LoadedModel    = detail::LoadedModel;
    using Frontend       = detail::Frontend;
    using PreparedPrompt = detail::PreparedPrompt;
    using OutputSession  = detail::OutputSession;
    using SequencePlan   = detail::SequencePlan;
    using RequestPlan    = detail::RequestPlan;
    using Program        = qwen3_6::Program<detail::Variant>;

    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                             WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model);
[[nodiscard]] static SequencePlan plan_sequence(DeviceContext& /*device*/,
                                                      const EngineOptions& options,
                                                      WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::laguna_xs_2_1
} // namespace ninfer