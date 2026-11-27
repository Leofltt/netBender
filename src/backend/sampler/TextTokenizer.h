#pragma once

#include "SamplerTypes.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace netbender::sampler {

/**
 * Zero-dependency C++20 tokenizer reading HuggingFace `tokenizer.json`.
 * Supports SentencePiece Unigram (Viterbi DP + Metaspace + byte fallback)
 * and BPE merges, special tokens, truncation, and attention mask generation.
 */
class TextTokenizer {
public:
    enum class ModelType {
        Unigram,
        BPE,
    };

    auto load(const std::string& tokenizer_json_path, int max_length = 128, int pad_token_id = 0,
              int eos_token_id = 1, bool add_eos = true) -> bool;

    auto load_from_json_string(const std::string& json_text, int max_length = 128,
                               int pad_token_id = 0, int eos_token_id = 1, bool add_eos = true)
        -> bool;

    [[nodiscard]] auto is_loaded() const noexcept -> bool { return m_loaded; }
    [[nodiscard]] auto get_max_length() const noexcept -> int { return m_max_length; }
    [[nodiscard]] auto get_model_type() const noexcept -> ModelType { return m_model_type; }
    [[nodiscard]] auto get_last_error() const -> std::string { return m_last_error; }

    [[nodiscard]] auto encode(const std::string& text) const -> TokenizedPrompt;

private:
    struct SpecialToken {
        std::string content;
        int id{0};
    };

    struct NormalizerStep {
        enum class Kind { Replace, Lowercase, Strip, Prepend };
        Kind kind{Kind::Replace};
        std::string pattern;
        std::string replacement;
    };

    bool m_loaded{false};
    ModelType m_model_type{ModelType::Unigram};
    int m_max_length{128};
    int m_pad_id{0};
    int m_eos_id{1};
    int m_bos_id{-1};
    int m_unk_id{2};
    bool m_add_eos{true};
    bool m_add_bos{false};
    std::string m_last_error;

    // Pre-tokenization config
    bool m_use_metaspace{false};
    bool m_metaspace_add_prefix{true};
    std::string m_metaspace_replacement{"\xE2\x96\x81"};  // U+2581
    bool m_split_whitespace{false};
    std::vector<NormalizerStep> m_normalizer_steps;

    // Vocabulary
    std::vector<std::string> m_id_to_token;
    std::unordered_map<std::string, int> m_token_to_id;
    std::vector<float> m_unigram_scores;
    std::vector<SpecialToken> m_special_tokens;  // sorted longest content first
    int m_byte_token_ids[256]{};

    // BPE merges: packed ("left\0right" -> rank)
    std::unordered_map<std::string, int> m_bpe_ranks;

    auto reset_state() -> void;
    [[nodiscard]] auto normalize_text(const std::string& text) const -> std::string;
    auto encode_segment(const std::string& segment, std::vector<int64_t>& ids_out,
                        std::vector<std::string>& pieces_out) const -> void;
    auto encode_unigram(const std::string& piece, std::vector<int64_t>& ids_out,
                        std::vector<std::string>& pieces_out) const -> void;
    auto encode_bpe(const std::string& piece, std::vector<int64_t>& ids_out,
                    std::vector<std::string>& pieces_out) const -> void;
    auto append_byte_fallback_or_unk(const std::string& bytes, std::vector<int64_t>& ids_out,
                                     std::vector<std::string>& pieces_out) const -> void;
};

}  // namespace netbender::sampler
