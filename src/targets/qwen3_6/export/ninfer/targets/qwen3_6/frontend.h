#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::size_t kTokenDomain = 248077;

inline constexpr std::array<std::pair<std::string_view, TokenId>, 4> kVisionSpecialTokens = {{
    {"<|vision_start|>", 248053},
    {"<|vision_end|>", 248054},
    {"<|image_pad|>", 248056},
    {"<|video_pad|>", 248057},
}};

inline constexpr std::array<std::pair<std::string_view, TokenId>, 7> kConfigOnlyTokens = {{
    {"<|audio_start|>", 248070},
    {"<|audio_end|>", 248071},
    {"<tts_pad>", 248072},
    {"<tts_text_bos>", 248073},
    {"<tts_text_eod>", 248074},
    {"<tts_text_bos_single>", 248075},
    {"<|audio_pad|>", 248076},
}};

struct FrontendProfile {
    std::size_t token_domain          = kTokenDomain;
    std::string pad_token             = "<|endoftext|>";
    bool require_bos                  = false;
    bool bos_absent_default           = true;
    bool require_prefix_space         = false;
    bool prefix_space_absent_default  = true;
    std::string bos_token;
    std::span<const std::pair<std::string_view, TokenId>> vision_special_tokens =
        kVisionSpecialTokens;
    std::span<const std::pair<std::string_view, TokenId>> config_only_tokens =
        kConfigOnlyTokens;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] double prepare_seconds() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

using PublishedOutput = std::vector<OutputDelta>;

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview(std::span<const TokenId> tokens,
                                                  std::uint32_t budget_remaining,
                                                  FinishReason limit_reason);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview() noexcept;

private:
    class Impl;
    explicit OutputSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
                                                    const StopPolicy& caller_stop,
                                                    const OutputOptions& output = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, bool vision_enabled,
                                  FrontendProfile profile);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, bool vision_enabled,
                                     FrontendProfile profile = {});

} // namespace ninfer::targets::qwen3_6
