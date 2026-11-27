#pragma once

#include "ISamplerBackend.h"
#include "TextTokenizer.h"

#include <atomic>
#include <string>
#include <unordered_map>

namespace netbender::sampler {

using PromptEmbeddingCache = std::unordered_map<std::string, PromptEmbedding>;

/**
 * Headless text-prompt -> stereo audio + latents generation pipeline for Stable Audio 3.
 * Orchestrates tokenization, prompt/duration conditioning, CFG/APG diffusion sampling,
 * cancellation checks, SAME-S decoding, and padding-mask trimming.
 */
class SamplerPrompter {
public:
    SamplerPrompter(ISamplerBackend& backend, const TextTokenizer& tokenizer);

    auto generate(const GenerationRequest& req, GeneratedSample& out,
                  const std::atomic<bool>* cancel_flag = nullptr,
                  ProgressCallback progress_cb = nullptr,
                  PromptEmbeddingCache* prompt_cache = nullptr) -> bool;

    [[nodiscard]] auto get_last_error() const -> std::string { return m_last_error; }

private:
    ISamplerBackend& m_backend;
    const TextTokenizer& m_tokenizer;
    std::string m_last_error;
};

}  // namespace netbender::sampler
