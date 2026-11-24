#include "TextTokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

namespace netbender::sampler {

namespace {

auto utf8_char_length(unsigned char lead) noexcept -> size_t {
    if ((lead & 0x80U) == 0U)
        return 1;
    if ((lead & 0xE0U) == 0xC0U)
        return 2;
    if ((lead & 0xF0U) == 0xE0U)
        return 3;
    if ((lead & 0xF8U) == 0xF0U)
        return 4;
    return 1;
}

auto split_utf8_boundaries(const std::string& s) -> std::vector<size_t> {
    std::vector<size_t> offsets;
    offsets.reserve(s.size() + 1);
    size_t i = 0;
    while (i < s.size()) {
        offsets.push_back(i);
        size_t len = utf8_char_length(static_cast<unsigned char>(s[i]));
        if (i + len > s.size()) {
            len = 1;
        }
        i += len;
    }
    offsets.push_back(s.size());
    return offsets;
}

auto replace_all_str(std::string s, const std::string& from, const std::string& to) -> std::string {
    if (from.empty()) {
        return s;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

auto starts_with(const std::string& s, const std::string& prefix) -> bool {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

auto is_ascii_space(char c) noexcept -> bool {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

}  // namespace

auto TextTokenizer::reset_state() -> void {
    m_loaded = false;
    m_model_type = ModelType::Unigram;
    m_max_length = 128;
    m_pad_id = 0;
    m_eos_id = 1;
    m_bos_id = -1;
    m_unk_id = 2;
    m_add_eos = true;
    m_add_bos = false;
    m_last_error.clear();

    m_use_metaspace = false;
    m_metaspace_add_prefix = true;
    m_metaspace_replacement = "\xE2\x96\x81";
    m_split_whitespace = false;
    m_normalizer_steps.clear();

    m_id_to_token.clear();
    m_token_to_id.clear();
    m_unigram_scores.clear();
    m_special_tokens.clear();
    m_bpe_ranks.clear();
    for (int i = 0; i < 256; ++i) {
        m_byte_token_ids[i] = -1;
    }
}

auto TextTokenizer::load(const std::string& tokenizer_json_path, int max_length, int pad_token_id,
                         int eos_token_id, bool add_eos) -> bool {
    std::ifstream ifs(tokenizer_json_path);
    if (!ifs.is_open()) {
        reset_state();
        m_last_error = "Cannot open tokenizer.json: " + tokenizer_json_path;
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return load_from_json_string(oss.str(), max_length, pad_token_id, eos_token_id, add_eos);
}

auto TextTokenizer::load_from_json_string(const std::string& json_text, int max_length,
                                          int pad_token_id, int eos_token_id, bool add_eos)
    -> bool {
    reset_state();
    m_max_length = std::max(1, max_length);
    m_pad_id = pad_token_id;
    m_eos_id = eos_token_id;
    m_add_eos = add_eos;

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_text);
    } catch (const std::exception& e) {
        m_last_error = std::string("Invalid tokenizer JSON: ") + e.what();
        return false;
    }

    if (!j.is_object() || !j.contains("model") || !j["model"].is_object()) {
        m_last_error = "tokenizer.json missing 'model' object";
        return false;
    }
    const auto& model = j["model"];
    const std::string type_str = model.value("type", "Unigram");

    if (type_str == "Unigram") {
        m_model_type = ModelType::Unigram;
        if (!model.contains("vocab") || !model["vocab"].is_array()) {
            m_last_error = "Unigram model missing 'vocab' array";
            return false;
        }
        const auto& vocab = model["vocab"];
        m_id_to_token.reserve(vocab.size());
        m_unigram_scores.reserve(vocab.size());
        for (size_t idx = 0; idx < vocab.size(); ++idx) {
            const auto& entry = vocab[idx];
            if (!entry.is_array() || entry.size() < 1 || !entry[0].is_string()) {
                continue;
            }
            const std::string piece = entry[0].get<std::string>();
            float score = 0.0f;
            if (entry.size() >= 2 && entry[1].is_number()) {
                score = entry[1].get<float>();
            }
            const int tok_id = static_cast<int>(m_id_to_token.size());
            m_id_to_token.push_back(piece);
            m_unigram_scores.push_back(score);
            m_token_to_id[piece] = tok_id;
        }
        m_unk_id = model.value("unk_id", 2);
    } else if (type_str == "BPE") {
        m_model_type = ModelType::BPE;
        if (!model.contains("vocab") || !model["vocab"].is_object()) {
            m_last_error = "BPE model missing 'vocab' object";
            return false;
        }
        int max_id = -1;
        for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
            if (it.value().is_number_integer()) {
                max_id = std::max(max_id, it.value().get<int>());
            }
        }
        if (max_id >= 0) {
            m_id_to_token.resize(static_cast<size_t>(max_id + 1));
        }
        m_token_to_id.reserve(model["vocab"].size());
        for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
            if (!it.value().is_number_integer()) {
                continue;
            }
            const int id = it.value().get<int>();
            const std::string tok = it.key();
            m_token_to_id[tok] = id;
            if (id >= 0 && id < static_cast<int>(m_id_to_token.size())) {
                m_id_to_token[static_cast<size_t>(id)] = tok;
            }
        }
        if (model.contains("unk_token") && model["unk_token"].is_string()) {
            auto it = m_token_to_id.find(model["unk_token"].get<std::string>());
            if (it != m_token_to_id.end()) {
                m_unk_id = it->second;
            }
        }
        if (model.contains("merges") && model["merges"].is_array()) {
            m_bpe_ranks.reserve(model["merges"].size());
            int rank = 0;
            for (const auto& m : model["merges"]) {
                if (m.is_string()) {
                    const std::string s = m.get<std::string>();
                    const size_t sp = s.find(' ');
                    if (sp != std::string::npos) {
                        std::string key = s.substr(0, sp);
                        key.push_back('\0');
                        key += s.substr(sp + 1);
                        m_bpe_ranks[key] = rank++;
                    }
                } else if (m.is_array() && m.size() >= 2 && m[0].is_string() && m[1].is_string()) {
                    std::string key = m[0].get<std::string>();
                    key.push_back('\0');
                    key += m[1].get<std::string>();
                    m_bpe_ranks[key] = rank++;
                }
            }
        }
    } else {
        m_last_error = "Unsupported tokenizer model type: " + type_str;
        return false;
    }

    // Parse added_tokens (special tokens such as <pad>, <eos>, <bos>, <unk>)
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& at : j["added_tokens"]) {
            if (!at.is_object() || !at.contains("content") || !at.contains("id")) {
                continue;
            }
            const std::string content = at["content"].get<std::string>();
            const int id = at["id"].get<int>();
            if (at.value("special", false)) {
                m_special_tokens.push_back({content, id});
            }
            if (id >= static_cast<int>(m_id_to_token.size())) {
                m_id_to_token.resize(static_cast<size_t>(id + 1));
            }
            if (id >= 0) {
                m_id_to_token[static_cast<size_t>(id)] = content;
            }
            if (m_token_to_id.find(content) == m_token_to_id.end()) {
                m_token_to_id[content] = id;
            }
            if (content == "<bos>" || content == "<s>") {
                m_bos_id = id;
            } else if (content == "<unk>") {
                m_unk_id = id;
            }
        }
        std::sort(m_special_tokens.begin(), m_special_tokens.end(),
                  [](const SpecialToken& a, const SpecialToken& b) {
                      return a.content.size() > b.content.size();
                  });
    }

    // Index SentencePiece byte-fallback tokens <0x00> .. <0xFF>
    char byte_buf[16];
    for (int b = 0; b < 256; ++b) {
        std::snprintf(byte_buf, sizeof(byte_buf), "<0x%02X>", b);
        auto it = m_token_to_id.find(byte_buf);
        if (it != m_token_to_id.end()) {
            m_byte_token_ids[b] = it->second;
        }
    }

    // Inspect normalizer
    auto parse_norm = [&](const auto& self, const nlohmann::json& norm) -> void {
        if (!norm.is_object()) {
            return;
        }
        const std::string ntype = norm.value("type", "");
        if (ntype == "Sequence" && norm.contains("normalizers") && norm["normalizers"].is_array()) {
            for (const auto& sub : norm["normalizers"]) {
                self(self, sub);
            }
        } else if (ntype == "Replace" && norm.contains("pattern") && norm.contains("content")) {
            std::string pat;
            if (norm["pattern"].is_object()) {
                if (norm["pattern"].contains("String")) {
                    pat = norm["pattern"]["String"].get<std::string>();
                } else if (norm["pattern"].contains("Regex")) {
                    const std::string rx = norm["pattern"]["Regex"].get<std::string>();
                    if (rx == " " || rx == "\\s+") {
                        pat = " ";
                    }
                }
            }
            if (!pat.empty() && norm["content"].is_string()) {
                m_normalizer_steps.push_back(
                    {NormalizerStep::Kind::Replace, pat, norm["content"].get<std::string>()});
            }
        } else if (ntype == "Lowercase") {
            m_normalizer_steps.push_back({NormalizerStep::Kind::Lowercase, "", ""});
        } else if (ntype == "Strip") {
            m_normalizer_steps.push_back({NormalizerStep::Kind::Strip, "", ""});
        } else if (ntype == "Prepend" && norm.contains("prepend") && norm["prepend"].is_string()) {
            m_normalizer_steps.push_back(
                {NormalizerStep::Kind::Prepend, "", norm["prepend"].get<std::string>()});
        }
    };
    if (j.contains("normalizer") && !j["normalizer"].is_null()) {
        parse_norm(parse_norm, j["normalizer"]);
    }

    // Inspect pre_tokenizer
    bool has_explicit_pretok = false;
    auto parse_pretok = [&](const auto& self, const nlohmann::json& pt) -> void {
        if (!pt.is_object()) {
            return;
        }
        has_explicit_pretok = true;
        const std::string ptype = pt.value("type", "");
        if (ptype == "Sequence" && pt.contains("pretokenizers") && pt["pretokenizers"].is_array()) {
            for (const auto& sub : pt["pretokenizers"]) {
                self(self, sub);
            }
        } else if (ptype == "Metaspace") {
            m_use_metaspace = true;
            if (pt.contains("replacement") && pt["replacement"].is_string()) {
                m_metaspace_replacement = pt["replacement"].get<std::string>();
            }
            if (pt.contains("prepend_scheme") && pt["prepend_scheme"].is_string()) {
                m_metaspace_add_prefix = (pt["prepend_scheme"].get<std::string>() != "never");
            } else if (pt.contains("add_prefix_space") && pt["add_prefix_space"].is_boolean()) {
                m_metaspace_add_prefix = pt["add_prefix_space"].get<bool>();
            }
        } else if (ptype == "Whitespace" || ptype == "WhitespaceSplit") {
            m_split_whitespace = true;
        } else if (ptype == "ByteLevel") {
            m_split_whitespace = true;
        }
    };
    if (j.contains("pre_tokenizer") && !j["pre_tokenizer"].is_null()) {
        parse_pretok(parse_pretok, j["pre_tokenizer"]);
    }

    // Default SentencePiece Unigram behavior when pre_tokenizer is omitted and no metaspace
    // prepended
    if (!has_explicit_pretok && m_model_type == ModelType::Unigram && m_normalizer_steps.empty()) {
        m_use_metaspace = true;
        m_metaspace_add_prefix = true;
    }

    // Inspect post_processor for optional BOS / EOS insertion
    if (j.contains("post_processor") && j["post_processor"].is_object()) {
        const auto& pp = j["post_processor"];
        if (pp.value("type", "") == "TemplateProcessing" && pp.contains("single") &&
            pp["single"].is_array() && !pp["single"].empty()) {
            const auto& first = pp["single"].front();
            if (first.is_object() && first.contains("SpecialToken")) {
                const std::string sp_id = first["SpecialToken"].value("id", "");
                if ((sp_id == "<bos>" || sp_id == "<s>") && m_bos_id >= 0) {
                    m_add_bos = true;
                }
            }
            const auto& last = pp["single"].back();
            bool ends_with_eos = false;
            if (last.is_object() && last.contains("SpecialToken")) {
                const std::string sp_id = last["SpecialToken"].value("id", "");
                if (sp_id == "<eos>" || sp_id == "</s>") {
                    ends_with_eos = true;
                }
            }
            m_add_eos = ends_with_eos;
        }
    }

    m_loaded = true;
    return true;
}

auto TextTokenizer::normalize_text(const std::string& text) const -> std::string {
    std::string out = text;
    for (const auto& step : m_normalizer_steps) {
        switch (step.kind) {
            case NormalizerStep::Kind::Replace:
                out = replace_all_str(std::move(out), step.pattern, step.replacement);
                break;
            case NormalizerStep::Kind::Lowercase:
                std::transform(out.begin(), out.end(), out.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                break;
            case NormalizerStep::Kind::Strip: {
                size_t first = 0;
                while (first < out.size() && is_ascii_space(out[first]))
                    ++first;
                size_t last = out.size();
                while (last > first && is_ascii_space(out[last - 1]))
                    --last;
                out = out.substr(first, last - first);
                break;
            }
            case NormalizerStep::Kind::Prepend:
                if (!out.empty() && !starts_with(out, step.replacement)) {
                    out = step.replacement + out;
                }
                break;
        }
    }
    return out;
}

auto TextTokenizer::append_byte_fallback_or_unk(const std::string& bytes,
                                                std::vector<int64_t>& ids_out,
                                                std::vector<std::string>& pieces_out) const
    -> void {
    bool all_bytes_present = !bytes.empty();
    for (unsigned char b : bytes) {
        if (m_byte_token_ids[b] < 0) {
            all_bytes_present = false;
            break;
        }
    }
    if (all_bytes_present) {
        for (unsigned char b : bytes) {
            const int bid = m_byte_token_ids[b];
            ids_out.push_back(bid);
            pieces_out.push_back(m_id_to_token[static_cast<size_t>(bid)]);
        }
    } else {
        ids_out.push_back(m_unk_id);
        if (m_unk_id >= 0 && m_unk_id < static_cast<int>(m_id_to_token.size()) &&
            !m_id_to_token[static_cast<size_t>(m_unk_id)].empty()) {
            pieces_out.push_back(m_id_to_token[static_cast<size_t>(m_unk_id)]);
        } else {
            pieces_out.push_back("<unk>");
        }
    }
}

auto TextTokenizer::encode_unigram(const std::string& piece, std::vector<int64_t>& ids_out,
                                   std::vector<std::string>& pieces_out) const -> void {
    if (piece.empty()) {
        return;
    }

    const auto bounds = split_utf8_boundaries(piece);
    const size_t n_cp = bounds.size() - 1;
    if (n_cp == 0) {
        return;
    }

    const float kNegInf = -std::numeric_limits<float>::infinity();
    const float kUnkPenalty = -100.0f;
    std::vector<float> best_score(n_cp + 1, kNegInf);
    std::vector<int> best_next(n_cp, -1);
    std::vector<int> best_tok(n_cp, -1);
    best_score[n_cp] = 0.0f;

    for (size_t i = n_cp; i-- > 0;) {
        // Fallback single codepoint option
        {
            const std::string ch = piece.substr(bounds[i], bounds[i + 1] - bounds[i]);
            auto it = m_token_to_id.find(ch);
            if (it != m_token_to_id.end()) {
                const int tid = it->second;
                const float sc = (tid >= 0 && tid < static_cast<int>(m_unigram_scores.size()))
                                     ? m_unigram_scores[static_cast<size_t>(tid)]
                                     : 0.0f;
                best_score[i] = sc + best_score[i + 1];
                best_next[i] = static_cast<int>(i + 1);
                best_tok[i] = tid;
            } else {
                best_score[i] = kUnkPenalty + best_score[i + 1];
                best_next[i] = static_cast<int>(i + 1);
                best_tok[i] = -1;  // mark missing so byte-fallback/unk can run
            }
        }

        // Check longer multi-codepoint candidates (limit subword search to <= 24 codepoints)
        const size_t max_j = std::min(n_cp, i + size_t{24});
        for (size_t j = i + 1; j <= max_j; ++j) {
            const std::string sub = piece.substr(bounds[i], bounds[j] - bounds[i]);
            auto it = m_token_to_id.find(sub);
            if (it == m_token_to_id.end()) {
                continue;
            }
            const int tid = it->second;
            const float sc = (tid >= 0 && tid < static_cast<int>(m_unigram_scores.size()))
                                 ? m_unigram_scores[static_cast<size_t>(tid)]
                                 : 0.0f;
            const float cand = sc + best_score[j];
            if (cand > best_score[i]) {
                best_score[i] = cand;
                best_next[i] = static_cast<int>(j);
                best_tok[i] = tid;
            }
        }
    }

    size_t pos = 0;
    while (pos < n_cp) {
        const size_t nxt = static_cast<size_t>(best_next[pos]);
        const int tid = best_tok[pos];
        if (tid >= 0) {
            ids_out.push_back(tid);
            pieces_out.push_back(m_id_to_token[static_cast<size_t>(tid)]);
        } else {
            append_byte_fallback_or_unk(piece.substr(bounds[pos], bounds[nxt] - bounds[pos]),
                                        ids_out, pieces_out);
        }
        pos = nxt;
    }
}

auto TextTokenizer::encode_bpe(const std::string& piece, std::vector<int64_t>& ids_out,
                               std::vector<std::string>& pieces_out) const -> void {
    if (piece.empty()) {
        return;
    }

    // If entire piece matches vocab directly as a single token, return immediately
    if (auto whole_it = m_token_to_id.find(piece); whole_it != m_token_to_id.end()) {
        ids_out.push_back(whole_it->second);
        pieces_out.push_back(piece);
        return;
    }

    const auto bounds = split_utf8_boundaries(piece);
    std::vector<std::string> symbols;
    symbols.reserve(bounds.size());
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        symbols.push_back(piece.substr(bounds[i], bounds[i + 1] - bounds[i]));
    }

    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string key = symbols[i];
            key.push_back('\0');
            key += symbols[i + 1];
            auto it = m_bpe_ranks.find(key);
            if (it != m_bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == symbols.size()) {
            break;
        }
        symbols[best_idx] += symbols[best_idx + 1];
        symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_idx + 1));
    }

    for (const auto& sym : symbols) {
        auto it = m_token_to_id.find(sym);
        if (it != m_token_to_id.end()) {
            ids_out.push_back(it->second);
            pieces_out.push_back(sym);
        } else {
            append_byte_fallback_or_unk(sym, ids_out, pieces_out);
        }
    }
}

auto TextTokenizer::encode_segment(const std::string& segment, std::vector<int64_t>& ids_out,
                                   std::vector<std::string>& pieces_out) const -> void {
    if (segment.empty()) {
        return;
    }

    std::string norm = normalize_text(segment);
    if (norm.empty()) {
        return;
    }

    auto dispatch = [&](const std::string& unit) {
        if (m_model_type == ModelType::Unigram) {
            encode_unigram(unit, ids_out, pieces_out);
        } else {
            encode_bpe(unit, ids_out, pieces_out);
        }
    };

    if (m_split_whitespace) {
        size_t i = 0;
        while (i < norm.size()) {
            while (i < norm.size() && is_ascii_space(norm[i]))
                ++i;
            if (i >= norm.size())
                break;
            size_t j = i;
            while (j < norm.size() && !is_ascii_space(norm[j]))
                ++j;
            dispatch(norm.substr(i, j - i));
            i = j;
        }
        return;
    }

    if (m_use_metaspace) {
        norm = replace_all_str(std::move(norm), " ", m_metaspace_replacement);
        if (m_metaspace_add_prefix && !norm.empty() &&
            !starts_with(norm, m_metaspace_replacement)) {
            norm = m_metaspace_replacement + norm;
        }
    }

    dispatch(norm);
}

auto TextTokenizer::encode(const std::string& text) const -> TokenizedPrompt {
    TokenizedPrompt out;
    out.input_ids.assign(static_cast<size_t>(m_max_length), m_pad_id);
    out.attention_mask.assign(static_cast<size_t>(m_max_length), 0);

    if (!m_loaded) {
        return out;
    }

    std::vector<int64_t> raw_ids;
    std::vector<std::string> raw_pieces;

    if (m_add_bos && m_bos_id >= 0) {
        raw_ids.push_back(m_bos_id);
        raw_pieces.push_back(m_id_to_token[static_cast<size_t>(m_bos_id)]);
    }

    // Scan left-to-right extracting exact special tokens (like <eos>, <bos>, etc.) verbatim
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_match_pos = text.size();
        const SpecialToken* best_sp = nullptr;
        for (const auto& sp : m_special_tokens) {
            if (sp.content.empty()) {
                continue;
            }
            const size_t found = text.find(sp.content, pos);
            if (found != std::string::npos &&
                (found < best_match_pos || (found == best_match_pos && best_sp &&
                                            sp.content.size() > best_sp->content.size()))) {
                best_match_pos = found;
                best_sp = &sp;
            }
        }

        if (best_sp == nullptr) {
            encode_segment(text.substr(pos), raw_ids, raw_pieces);
            break;
        }

        if (best_match_pos > pos) {
            encode_segment(text.substr(pos, best_match_pos - pos), raw_ids, raw_pieces);
        }
        raw_ids.push_back(best_sp->id);
        raw_pieces.push_back(best_sp->content);
        pos = best_match_pos + best_sp->content.size();
    }

    const int reserve_eos = m_add_eos ? 1 : 0;
    const int max_content = std::max(0, m_max_length - reserve_eos);
    int n_copy = std::min(static_cast<int>(raw_ids.size()), max_content);

    for (int i = 0; i < n_copy; ++i) {
        out.input_ids[static_cast<size_t>(i)] = raw_ids[static_cast<size_t>(i)];
        out.attention_mask[static_cast<size_t>(i)] = 1;
        out.token_pieces.push_back(raw_pieces[static_cast<size_t>(i)]);
    }

    if (m_add_eos && n_copy < m_max_length) {
        out.input_ids[static_cast<size_t>(n_copy)] = m_eos_id;
        out.attention_mask[static_cast<size_t>(n_copy)] = 1;
        std::string eos_piece = "</s>";
        if (m_eos_id >= 0 && m_eos_id < static_cast<int>(m_id_to_token.size()) &&
            !m_id_to_token[static_cast<size_t>(m_eos_id)].empty()) {
            eos_piece = m_id_to_token[static_cast<size_t>(m_eos_id)];
        }
        out.token_pieces.push_back(std::move(eos_piece));
        ++n_copy;
    }

    out.valid_token_count = n_copy;
    return out;
}

}  // namespace netbender::sampler
