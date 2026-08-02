#include <ninfer/targets/laguna_xs_2_1/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/laguna_xs_2_1/impl/load/bindings.h"
#include "targets/laguna_xs_2_1/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::laguna_xs_2_1::detail {

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::laguna_xs_2_1::detail

namespace ninfer::targets::laguna_xs_2_1 {
namespace {

qwen3_6::FrontendProfile laguna_frontend_profile() {
    // Registered laguna special tokens pinned from the checkpoint tokenizer:
    // "〈|EOS|〉" is both bos and eos (template head + stop id 2), "〈|PAD|〉" is
    // the registered pad token id 9.
    constexpr std::array<std::pair<std::string_view, ninfer::TokenId>, 2>
        kLagunaSpecialTokens = {{
            {"\xE3\x80\x88|EOS|\xE3\x80\x89", 2},
            {"\xE3\x80\x88|PAD|\xE3\x80\x89", 9},
        }};
    qwen3_6::FrontendProfile profile;
    profile.token_domain                = 100352;
    profile.pad_token                   = "\xE3\x80\x88|PAD|\xE3\x80\x89";
    profile.bos_absent_default          = false;
    profile.prefix_space_absent_default = false;
    profile.bos_token                   = "\xE3\x80\x88|EOS|\xE3\x80\x89";
    profile.vision_special_tokens       = {};
    profile.config_only_tokens          = kLagunaSpecialTokens;
    return profile;
}

} // namespace

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::kGroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                      WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options))));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->profile, std::move(plan.impl_->plan.bindings),
        detail::materialize_weights(std::move(plan.impl_->plan.bindings), std::move(materialized)));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                   model.impl_->data.features.vision,
                                   laguna_frontend_profile());
}

Package::SequencePlan Package::plan_sequence(DeviceContext& device, const EngineOptions& options,
                                                WeightsProfile weights_profile) {
    return qwen3_6::plan_sequence<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(
        model.impl_->data, model.impl_->profile, std::move(plan), device);
}

} // namespace ninfer::targets::laguna_xs_2_1