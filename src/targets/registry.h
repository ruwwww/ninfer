#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/laguna_xs_2_1/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
using LagunaXS21     = laguna_xs_2_1::Package;

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    explicit LoadedQwen3_6_27B(
        std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                               Qwen3_6_27B::SequencePlan sequence_plan,
                               DeviceContext& device);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    explicit LoadedQwen3_6_35BA3B(
        std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                  Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                  DeviceContext& device);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};

struct LoadedLagunaXS21 {
    std::unique_ptr<LagunaXS21::LoadedModel> model;
    LagunaXS21::Frontend frontend;

    explicit LoadedLagunaXS21(
        std::unique_ptr<LagunaXS21::LoadedModel> stable_model);
    ~LoadedLagunaXS21();

    LoadedLagunaXS21(const LoadedLagunaXS21&)            = delete;
    LoadedLagunaXS21& operator=(const LoadedLagunaXS21&) = delete;
};

struct LagunaXS21Instance {
    using Package = LagunaXS21;

    std::unique_ptr<LoadedLagunaXS21> loaded;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<LagunaXS21::Program> program;

    LagunaXS21Instance(std::unique_ptr<LoadedLagunaXS21> stable_loaded,
                              LagunaXS21::SequencePlan sequence_plan,
                              DeviceContext& device);
    ~LagunaXS21Instance();

    LagunaXS21Instance(const LagunaXS21Instance&)            = delete;
    LagunaXS21Instance& operator=(const LagunaXS21Instance&) = delete;
};

using ActiveTarget = std::variant<std::unique_ptr<Qwen3_6_27BInstance>,
                                   std::unique_ptr<Qwen3_6_35BA3BInstance>,
                                   std::unique_ptr<LagunaXS21Instance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
