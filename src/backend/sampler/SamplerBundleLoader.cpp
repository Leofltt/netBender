#include "SamplerBundleLoader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace netbender::sampler {

namespace {

auto to_lower_ascii(std::string value) -> std::string {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

auto resolve_path(const std::string& bundle_dir, const std::string& rel_or_abs) -> std::string {
    if (rel_or_abs.empty()) {
        return "";
    }
    fs::path p(rel_or_abs);
    if (p.is_absolute() || bundle_dir.empty()) {
        return p.lexically_normal().string();
    }
    return (fs::path(bundle_dir) / p).lexically_normal().string();
}

auto file_exists_regular(const std::string& path) -> bool {
    if (path.empty()) {
        return false;
    }
    std::error_code ec;
    return fs::is_regular_file(fs::path(path), ec);
}

auto parse_dist_shift(const nlohmann::json& ds_json, DistShiftConfig& out_ds) -> void {
    if (!ds_json.is_object()) {
        return;
    }
    const std::string type_str = to_lower_ascii(ds_json.value("type", "logsnr"));
    if (type_str == "identity" || type_str == "none") {
        out_ds.type = DistShiftType::Identity;
    } else if (type_str == "flux") {
        out_ds.type = DistShiftType::Flux;
    } else if (type_str == "full") {
        out_ds.type = DistShiftType::Full;
    } else {
        out_ds.type = DistShiftType::LogSNR;
    }

    if (ds_json.contains("anchor_length") && ds_json["anchor_length"].is_number()) {
        out_ds.anchor_length = ds_json["anchor_length"].get<float>();
    }
    if (ds_json.contains("anchor_logsnr") && ds_json["anchor_logsnr"].is_number()) {
        out_ds.anchor_logsnr = ds_json["anchor_logsnr"].get<float>();
    }
    if (ds_json.contains("rate") && ds_json["rate"].is_number()) {
        out_ds.rate = ds_json["rate"].get<float>();
    }
    if (ds_json.contains("logsnr_end") && ds_json["logsnr_end"].is_number()) {
        out_ds.logsnr_end = ds_json["logsnr_end"].get<float>();
    }
    if (ds_json.contains("min_length") && ds_json["min_length"].is_number()) {
        out_ds.min_length = ds_json["min_length"].get<float>();
    }
    if (ds_json.contains("max_length") && ds_json["max_length"].is_number()) {
        out_ds.max_length = ds_json["max_length"].get<float>();
    }
    if (ds_json.contains("alpha_min") && ds_json["alpha_min"].is_number()) {
        out_ds.alpha_min = ds_json["alpha_min"].get<float>();
    }
    if (ds_json.contains("alpha_max") && ds_json["alpha_max"].is_number()) {
        out_ds.alpha_max = ds_json["alpha_max"].get<float>();
    }
    if (ds_json.contains("base_shift") && ds_json["base_shift"].is_number()) {
        out_ds.base_shift = ds_json["base_shift"].get<float>();
    }
    if (ds_json.contains("max_shift") && ds_json["max_shift"].is_number()) {
        out_ds.max_shift = ds_json["max_shift"].get<float>();
    }
    if (ds_json.contains("use_sine") && ds_json["use_sine"].is_boolean()) {
        out_ds.use_sine = ds_json["use_sine"].get<bool>();
    }
}

auto read_string_array(const nlohmann::json& obj, const char* key,
                       std::vector<std::string>& out_vec) -> void {
    if (!obj.contains(key) || !obj[key].is_array()) {
        return;
    }
    out_vec.clear();
    for (const auto& elem : obj[key]) {
        if (elem.is_string()) {
            out_vec.push_back(elem.get<std::string>());
        }
    }
}

}  // namespace

auto sampler_algorithm_to_string(SamplerAlgorithm algo) -> std::string {
    switch (algo) {
        case SamplerAlgorithm::PingPong:
            return "pingpong";
        case SamplerAlgorithm::Euler:
            return "euler";
        case SamplerAlgorithm::DpmPP:
            return "dpmpp";
    }
    return "pingpong";
}

auto parse_sampler_algorithm(const std::string& name, SamplerAlgorithm& out_algo) -> bool {
    const std::string lower = to_lower_ascii(name);
    if (lower == "pingpong") {
        out_algo = SamplerAlgorithm::PingPong;
        return true;
    }
    if (lower == "euler") {
        out_algo = SamplerAlgorithm::Euler;
        return true;
    }
    if (lower == "dpmpp") {
        out_algo = SamplerAlgorithm::DpmPP;
        return true;
    }
    return false;
}

auto parse_sampler_bundle_json(const std::string& json_text, const std::string& bundle_dir,
                               SamplerBundleMetadata& out_meta, std::string& out_error,
                               bool check_files) -> bool {
    out_meta = SamplerBundleMetadata{};
    out_meta.bundle_dir = bundle_dir;
    out_error.clear();

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_text);
    } catch (const std::exception& e) {
        out_error = std::string("Failed to parse sampler manifest JSON: ") + e.what();
        return false;
    }

    if (!j.is_object()) {
        out_error = "Sampler manifest root must be a JSON object";
        return false;
    }

    out_meta.format = j.value("format", "");
    if (out_meta.format != "netbender.sampler") {
        out_error =
            "Invalid manifest format: expected 'netbender.sampler', got '" + out_meta.format + "'";
        return false;
    }

    if (!j.contains("version") || !j["version"].is_number_integer()) {
        out_error = "Missing or non-integer manifest 'version'";
        return false;
    }
    out_meta.version = j["version"].get<int>();
    if (out_meta.version != 1) {
        out_error = "Unsupported manifest version: " + std::to_string(out_meta.version);
        return false;
    }

    out_meta.architecture = j.value("architecture", "stable_audio_3");
    out_meta.variant = j.value("variant", "");
    out_meta.runtime = to_lower_ascii(j.value("runtime", ""));
    if (out_meta.runtime != "executorch" && out_meta.runtime != "onnx" &&
        out_meta.runtime != "torchscript") {
        out_error = "Unsupported runtime '" + out_meta.runtime +
                    "'; expected executorch, onnx, or torchscript";
        return false;
    }

    out_meta.delegate = to_lower_ascii(j.value("delegate", ""));

    if (j.contains("compiled_latent_frames")) {
        if (!j["compiled_latent_frames"].is_number_integer() ||
            j["compiled_latent_frames"].get<int>() <= 0) {
            out_error = "Field 'compiled_latent_frames' must be a positive integer";
            return false;
        }
        out_meta.compiled_latent_frames = j["compiled_latent_frames"].get<int>();
    }

    auto require_positive_int = [&](const char* field, int& dst) -> bool {
        if (!j.contains(field) || !j[field].is_number()) {
            out_error = std::string("Missing numeric field '") + field + "'";
            return false;
        }
        const int val = j[field].get<int>();
        if (val <= 0) {
            out_error = std::string("Field '") + field + "' must be > 0";
            return false;
        }
        dst = val;
        return true;
    };

    if (!require_positive_int("sample_rate", out_meta.sample_rate) ||
        !require_positive_int("io_channels", out_meta.io_channels) ||
        !require_positive_int("latent_dim", out_meta.latent_dim) ||
        !require_positive_int("downsampling_ratio", out_meta.downsampling_ratio) ||
        !require_positive_int("latent_align", out_meta.latent_align)) {
        return false;
    }

    out_meta.cond_dim = j.value("cond_dim", out_meta.cond_dim);
    out_meta.global_cond_dim = j.value("global_cond_dim", out_meta.global_cond_dim);
    out_meta.input_concat_dim = j.value("input_concat_dim", out_meta.input_concat_dim);
    out_meta.max_duration_sec = j.value("max_duration_sec", out_meta.max_duration_sec);
    out_meta.duration_padding_sec = j.value("duration_padding_sec", out_meta.duration_padding_sec);

    if (!j.contains("diffusion") || !j["diffusion"].is_object()) {
        out_error = "Missing or invalid 'diffusion' section";
        return false;
    }
    const auto& diff = j["diffusion"];
    if (!diff.contains("objective") || !diff["objective"].is_string() ||
        !diff.contains("default_sampler") || !diff["default_sampler"].is_string() ||
        !diff.contains("default_steps") || !diff["default_steps"].is_number_integer()) {
        out_error = "Diffusion section missing required objective/default_sampler/default_steps";
        return false;
    }
    out_meta.objective = diff["objective"].get<std::string>();
    const std::string sampler_name = diff["default_sampler"].get<std::string>();
    if (!parse_sampler_algorithm(sampler_name, out_meta.default_sampler)) {
        out_error = "Unsupported diffusion.default_sampler '" + sampler_name + "'";
        return false;
    }
    out_meta.default_steps = std::max(1, diff["default_steps"].get<int>());
    out_meta.default_cfg_scale = diff.value("default_cfg_scale", 1.0f);
    out_meta.default_apg_scale = diff.value("default_apg_scale", 1.0f);
    out_meta.has_inpaint_concat = diff.value("has_inpaint_concat", out_meta.input_concat_dim > 0);
    if (diff.contains("dist_shift")) {
        parse_dist_shift(diff["dist_shift"], out_meta.dist_shift);
    }
    read_string_array(diff, "cross_attn_cond_ids", out_meta.cross_attn_cond_ids);
    read_string_array(diff, "global_cond_ids", out_meta.global_cond_ids);
    read_string_array(diff, "input_concat_ids", out_meta.input_concat_ids);

    if (!j.contains("tokenizer") || !j["tokenizer"].is_object()) {
        out_error = "Missing or invalid 'tokenizer' section";
        return false;
    }
    const auto& tok = j["tokenizer"];
    out_meta.tokenizer_file = tok.value("file", "tokenizer.json");
    out_meta.tokenizer_path = resolve_path(bundle_dir, out_meta.tokenizer_file);
    out_meta.max_token_length = tok.value("max_length", 128);
    if (out_meta.max_token_length <= 0) {
        out_error = "Invalid 'tokenizer.max_length' (must be > 0)";
        return false;
    }
    out_meta.pad_token_id = tok.value("pad_token_id", 0);
    out_meta.eos_token_id = tok.value("eos_token_id", 1);
    out_meta.add_eos = tok.value("add_eos", true);
    if (check_files && !out_meta.tokenizer_path.empty() &&
        !file_exists_regular(out_meta.tokenizer_path)) {
        out_error = "Tokenizer file not found: " + out_meta.tokenizer_path;
        return false;
    }

    if (!j.contains("components") || !j["components"].is_object()) {
        out_error = "Missing or invalid 'components' section";
        return false;
    }
    const auto& comps = j["components"];

    auto parse_comp = [&](const char* key, std::string& model_out, std::string& sidecar_out,
                          std::string* encode_out = nullptr,
                          std::string* decode_out = nullptr) -> bool {
        if (!comps.contains(key) || !comps[key].is_object()) {
            out_error = std::string("Missing component object '") + key + "'";
            return false;
        }
        const auto& c = comps[key];
        const std::string model_rel = c.value("model", "");
        if (model_rel.empty()) {
            out_error = std::string("Missing component '") + key + ".model'";
            return false;
        }
        model_out = resolve_path(bundle_dir, model_rel);
        if (check_files && !file_exists_regular(model_out)) {
            out_error = std::string("Component model not found (") + key + "): " + model_out;
            return false;
        }

        const std::string sidecar_rel = c.value("sidecar", "");
        sidecar_out = sidecar_rel.empty() ? "" : resolve_path(bundle_dir, sidecar_rel);
        if (check_files && !sidecar_out.empty() && !file_exists_regular(sidecar_out)) {
            out_error = std::string("Component sidecar not found (") + key + "): " + sidecar_out;
            return false;
        }

        if (encode_out != nullptr) {
            const std::string enc_rel = c.value("encode_model", "");
            *encode_out = enc_rel.empty() ? model_out : resolve_path(bundle_dir, enc_rel);
            if (check_files && !file_exists_regular(*encode_out)) {
                out_error = "Autoencoder encode_model not found: " + *encode_out;
                return false;
            }
        }
        if (decode_out != nullptr) {
            const std::string dec_rel = c.value("decode_model", "");
            *decode_out = dec_rel.empty() ? model_out : resolve_path(bundle_dir, dec_rel);
            if (check_files && !file_exists_regular(*decode_out)) {
                out_error = "Autoencoder decode_model not found: " + *decode_out;
                return false;
            }
        }
        return true;
    };

    if (!parse_comp("text_encoder", out_meta.text_encoder_model, out_meta.text_encoder_sidecar) ||
        !parse_comp("number_cond", out_meta.number_cond_model, out_meta.number_cond_sidecar) ||
        !parse_comp("dit", out_meta.dit_model, out_meta.dit_sidecar) ||
        !parse_comp("autoencoder", out_meta.ae_model, out_meta.ae_sidecar,
                    &out_meta.ae_encode_model, &out_meta.ae_decode_model)) {
        return false;
    }

    return true;
}

auto parse_sampler_bundle_manifest(const std::string& manifest_path,
                                   SamplerBundleMetadata& out_meta, std::string& out_error)
    -> bool {
    std::ifstream ifs(manifest_path);
    if (!ifs.is_open()) {
        out_error = "Cannot open sampler manifest file: " + manifest_path;
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();

    std::error_code ec;
    const fs::path abs_path = fs::weakly_canonical(fs::path(manifest_path), ec);
    const std::string bundle_dir = (ec ? fs::path(manifest_path) : abs_path).parent_path().string();

    if (!parse_sampler_bundle_json(oss.str(), bundle_dir, out_meta, out_error, true)) {
        return false;
    }
    out_meta.manifest_path = manifest_path;
    return true;
}

}  // namespace netbender::sampler
