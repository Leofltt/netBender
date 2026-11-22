#include "CsoundSessionAdapter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <utility>

namespace netbender::csound_fe {

namespace {

auto to_lower(std::string s) -> std::string {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

auto ends_with(const std::string& s, const std::string& suffix) -> bool {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

auto has_supported_model_extension(const std::string& path) -> bool {
    const std::string lower = to_lower(path);
    return ends_with(lower, ".pte") || ends_with(lower, ".onnx") || ends_with(lower, ".ts");
}

auto build_path_candidates(const std::string& raw_path) -> std::vector<std::string> {
    if (raw_path.empty()) {
        return {};
    }
    if (has_supported_model_extension(raw_path)) {
        return {raw_path};
    }
    return {
        raw_path + ".pte",
        raw_path + ".onnx",
        raw_path + ".ts",
        raw_path,
    };
}

auto default_home_models_dir(const std::string& override_dir) -> std::filesystem::path {
    if (!override_dir.empty()) {
        return std::filesystem::path(override_dir);
    }
#if defined(_WIN32)
    if (const char* user_profile = std::getenv("USERPROFILE")) {
        if (user_profile[0] != '\0') {
            return std::filesystem::path(user_profile) / ".netbender" / "models";
        }
    }
#endif
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".netbender" / "models";
        }
    }
    return {};
}

auto is_regular_file_noexcept(const std::filesystem::path& p) -> bool {
    if (p.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::is_regular_file(p, ec);
}

auto swap_xnnpack_to_coreml_if_exists(const std::string& path) -> std::string {
    const std::string lower = to_lower(path);
    if (!ends_with(lower, "_xnnpack.pte")) {
        return path;
    }
    const std::string suffix = "_xnnpack.pte";
    std::string candidate = path.substr(0, path.size() - suffix.size()) + "_coreml.pte";
    if (is_regular_file_noexcept(candidate)) {
        return candidate;
    }
    return path;
}

auto collect_unique_methods(IBackend& backend, const NetBenderSession& session)
    -> std::vector<std::string> {
    std::vector<std::string> merged;
    auto push_unique = [&merged](const std::string& item) {
        if (!item.empty() && std::find(merged.begin(), merged.end(), item) == merged.end()) {
            merged.push_back(item);
        }
    };
    for (const auto& m : session.get_available_modes()) {
        push_unique(m);
    }
    for (const auto& m : backend.get_available_methods()) {
        push_unique(m);
    }
    return merged;
}

template <typename SampleT>
auto zero_outputs(SampleT* const* out_channels, int num_out, int num_frames) -> void {
    if (out_channels == nullptr || num_frames <= 0) {
        return;
    }
    for (int c = 0; c < num_out; ++c) {
        if (out_channels[c] != nullptr) {
            std::fill(out_channels[c], out_channels[c] + num_frames, SampleT{0});
        }
    }
}

}  // namespace

auto resolve_model_path(const std::string& raw_path, const HostFileFinder& host_finder,
                        const std::string& home_models_override) -> std::string {
    const auto candidates = build_path_candidates(raw_path);
    if (candidates.empty()) {
        return {};
    }

    // 1. Exact / relative-to-cwd / absolute path (with .pte/.onnx/.ts probing)
    for (const auto& cand : candidates) {
        if (is_regular_file_noexcept(cand)) {
            return cand;
        }
    }

    // 2. Host file finder (e.g. Csound FindInputFile across SSDIR;SFDIR;INCDIR)
    if (host_finder) {
        for (const auto& cand : candidates) {
            const std::string found = host_finder(cand);
            if (!found.empty() && is_regular_file_noexcept(found)) {
                return found;
            }
        }
    }

    // 3. ~/.netbender/models fallback
    const std::filesystem::path home_models = default_home_models_dir(home_models_override);
    if (!home_models.empty()) {
        for (const auto& cand : candidates) {
            const std::filesystem::path full = home_models / cand;
            if (is_regular_file_noexcept(full)) {
                return full.string();
            }
        }
    }

    return {};
}

auto normalize_method_name(const std::string& method) -> std::string {
    if (method.empty()) {
        return "forward";
    }
    if (method == "decode_gen") {
        return "decode gen";
    }
    return method;
}

auto is_latent_method(const std::string& method) -> bool {
    return method == "encode" || method == "decode";
}

auto is_generative_method(const std::string& method) -> bool {
    return method == "decode gen" || method == "decode_gen" || method == "prior" ||
           method == "generate";
}

auto round_up_pow2(int x) -> int {
    if (x <= 1) {
        return 1;
    }
    int power = 2;
    --x;
    while (x >>= 1) {
        power <<= 1;
    }
    return power;
}

CsoundSessionAdapter::CsoundSessionAdapter(NetBenderSession::BackendCreator creator)
    : m_creator(std::move(creator)), m_has_custom_creator(static_cast<bool>(m_creator)) {}

CsoundSessionAdapter::~CsoundSessionAdapter() {
    close();
}

auto CsoundSessionAdapter::open(const OpenOptions& opts, std::string& error_out) -> bool {
    close();
    error_out.clear();

    if (opts.model_path.empty()) {
        error_out = "netbender_load: model path cannot be empty";
        return false;
    }

    const std::string method = normalize_method_name(opts.method);
    const bool latent_mode = is_latent_method(method);
    const bool gen_mode = is_generative_method(method);

    const bool use_worker =
        (opts.buffer_size == 0)
            ? false
            : ((opts.thread_mode < 0) ? opts.is_realtime_output : (opts.thread_mode != 0));
    const bool enable_sentry = (opts.safety_sentry >= 0) ? (opts.safety_sentry != 0) : !latent_mode;
    const bool enable_dry =
        (opts.dry_delay >= 0) ? (opts.dry_delay != 0) : (!latent_mode && !gen_mode);

    const int safe_ksmps = std::max(1, opts.ksmps);
    const int host_io_block = std::max({256, safe_ksmps, opts.sw_buffer_frames});

    SessionConfig cfg;
    cfg.use_worker_thread = use_worker;
    cfg.enable_dry_delay = enable_dry;
    cfg.enable_safety_sentry = enable_sentry;
    cfg.duplicate_mono_input = (!latent_mode && !gen_mode);
    cfg.duplicate_mono_output = (!latent_mode);
    cfg.resample_mode = ResampleMode::Auto;
    cfg.resample_quality = ResamplerQuality::CatmullRom;
    cfg.min_host_block_size = host_io_block;
    cfg.min_host_channels = std::max(2, opts.max_channels);

    m_session = std::make_unique<NetBenderSession>(cfg);
    if (m_has_custom_creator) {
        m_session->set_backend_creator(m_creator);
    }

    const double host_sr = (opts.host_sample_rate > 0.0) ? opts.host_sample_rate : 44100.0;
    m_session->prepare(host_sr, host_io_block, std::max(2, opts.max_channels));

    std::string target_path = opts.model_path;
    if (!m_has_custom_creator && !is_regular_file_noexcept(target_path)) {
        const std::string resolved = resolve_model_path(target_path);
        if (resolved.empty()) {
            error_out = "netbender_load: could not find model '" + opts.model_path + "'";
            close();
            return false;
        }
        target_path = resolved;
    }

    if (opts.use_gpu) {
        target_path = swap_xnnpack_to_coreml_if_exists(target_path);
        m_session->set_use_gpu(true);
    }

    if (!m_session->load_model(target_path)) {
        error_out = m_session->get_last_error();
        if (error_out.empty()) {
            error_out = "netbender_load: failed to load '" + target_path + "'";
        }
        close();
        return false;
    }

    if (method != m_session->get_method()) {
        if (!m_session->set_method(method)) {
            const auto methods = collect_unique_methods(m_session->backend(), *m_session);
            std::ostringstream oss;
            oss << "netbender_load: method '" << method << "' is not supported by '" << target_path
                << "'. Available methods:";
            for (const auto& m : methods) {
                oss << " " << m;
            }
            error_out = oss.str();
            close();
            return false;
        }
    }

    const int higher_ratio = std::max(1, m_session->backend().get_higher_ratio());
    if (opts.buffer_size > 0) {
        const int eff_buf = round_up_pow2(std::max(opts.buffer_size, higher_ratio));
        m_session->set_buffer_size(eff_buf);
    } else if (opts.buffer_size == 0) {
        const int sync_buf = round_up_pow2(higher_ratio);
        if (sync_buf > 0 && sync_buf != m_session->get_buffer_size()) {
            m_session->set_buffer_size(sync_buf);
        }
    }

    const std::string active_method = m_session->get_method();
    const auto mode_params = m_session->backend().get_mode_params(active_method);
    if (mode_params.size() >= 4) {
        m_in_channels = std::max(0, mode_params[0]);
        m_in_ratio = std::max(1, mode_params[1]);
        m_out_channels = std::max(0, mode_params[2]);
        m_out_ratio = std::max(1, mode_params[3]);
    } else {
        m_in_channels = 0;
        m_in_ratio = 1;
        m_out_channels = 1;
        m_out_ratio = 1;
    }

    if (m_out_channels <= 0) {
        error_out = "netbender_load: method '" + active_method + "' produced 0 output channels";
        close();
        return false;
    }

    // Re-prepare if model needs more channels than initial guess so scratch/dry rings fit cleanly
    const int required_channels = std::max({2, opts.max_channels, m_in_channels, m_out_channels});
    if (required_channels > cfg.min_host_channels) {
        m_session->prepare(host_sr, host_io_block, required_channels);
    }

    m_ksmps = safe_ksmps;
    m_sw_buffer_frames = std::max(0, opts.sw_buffer_frames);
    m_hw_buffer_frames = std::max(0, opts.hw_buffer_frames);
    m_worker_thread_enabled = use_worker;
    m_safety_sentry_enabled = enable_sentry;
    m_dry_delay_enabled = enable_dry;

    m_scratch_channels = std::max(64, required_channels);
    m_scratch_frames = std::max(1024, safe_ksmps);
    m_scaled_in_storage.resize(static_cast<size_t>(m_scratch_channels));
    m_scaled_in_ptrs.resize(static_cast<size_t>(m_scratch_channels), nullptr);
    m_scaled_out_storage.resize(static_cast<size_t>(m_scratch_channels));
    m_scaled_out_ptrs.resize(static_cast<size_t>(m_scratch_channels), nullptr);
    for (int c = 0; c < m_scratch_channels; ++c) {
        m_scaled_in_storage[static_cast<size_t>(c)].assign(static_cast<size_t>(m_scratch_frames),
                                                           0.0f);
        m_scaled_in_ptrs[static_cast<size_t>(c)] =
            m_scaled_in_storage[static_cast<size_t>(c)].data();
        m_scaled_out_storage[static_cast<size_t>(c)].assign(static_cast<size_t>(m_scratch_frames),
                                                            0.0f);
        m_scaled_out_ptrs[static_cast<size_t>(c)] =
            m_scaled_out_storage[static_cast<size_t>(c)].data();
    }

    m_open = true;
    return true;
}

auto CsoundSessionAdapter::close() -> void {
    m_open = false;
    if (m_session) {
        m_session->release();
        m_session->engine().detach_backend();
        m_session.reset();
    }
    m_in_channels = 0;
    m_out_channels = 0;
    m_in_ratio = 1;
    m_out_ratio = 1;
    m_ksmps = 128;
    m_sw_buffer_frames = 0;
    m_hw_buffer_frames = 0;
    m_worker_thread_enabled = false;
    m_safety_sentry_enabled = false;
    m_dry_delay_enabled = false;
    m_scratch_channels = 0;
    m_scratch_frames = 0;
    m_scaled_in_storage.clear();
    m_scaled_in_ptrs.clear();
    m_scaled_out_storage.clear();
    m_scaled_out_ptrs.clear();
}

auto CsoundSessionAdapter::reset() -> void {
    if (m_session) {
        m_session->engine().reset();
    }
}

template <typename SampleT>
auto CsoundSessionAdapter::process_impl(const SampleT* const* in_channels, int num_in,
                                        SampleT* const* out_channels, int num_out, int num_frames,
                                        double zero_dbfs, float dry_wet, bool enabled) -> void {
    if (num_frames <= 0 || out_channels == nullptr || num_out <= 0) {
        return;
    }

    if (!m_open || !m_session || !m_session->is_model_loaded() || !enabled) {
        zero_outputs(out_channels, num_out, num_frames);
        return;
    }

    m_session->set_dry_wet(std::clamp(dry_wet, 0.0f, 1.0f));

    const bool need_scale = (zero_dbfs > 0.0 && std::abs(zero_dbfs - 1.0) > 1e-6);
    if (!need_scale) {
        m_session->process(in_channels, std::max(0, num_in), out_channels, num_out, num_frames);
        return;
    }

    const float inv_scale = static_cast<float>(1.0 / zero_dbfs);
    const float out_scale = static_cast<float>(zero_dbfs);
    const int active_in = std::clamp(num_in, 0, m_scratch_channels);
    const int active_out = std::clamp(num_out, 0, m_scratch_channels);
    const int chunk_cap = std::max(1, m_scratch_frames);

    // Zero any extra output channels beyond preallocated scratch capacity
    for (int c = active_out; c < num_out; ++c) {
        if (out_channels[c] != nullptr) {
            std::fill(out_channels[c], out_channels[c] + num_frames, SampleT{0});
        }
    }

    int offset = 0;
    while (offset < num_frames) {
        const int chunk = std::min(num_frames - offset, chunk_cap);

        for (int c = 0; c < active_in; ++c) {
            float* dst = m_scaled_in_storage[static_cast<size_t>(c)].data();
            const SampleT* src = (in_channels != nullptr) ? in_channels[c] : nullptr;
            if (src != nullptr) {
                for (int i = 0; i < chunk; ++i) {
                    dst[i] = static_cast<float>(src[offset + i]) * inv_scale;
                }
            } else {
                std::fill(dst, dst + chunk, 0.0f);
            }
        }

        m_session->process(m_scaled_in_ptrs.data(), active_in, m_scaled_out_ptrs.data(), active_out,
                           chunk);

        for (int c = 0; c < active_out; ++c) {
            SampleT* dst = out_channels[c];
            if (dst == nullptr) {
                continue;
            }
            const float* src = m_scaled_out_ptrs[static_cast<size_t>(c)];
            for (int i = 0; i < chunk; ++i) {
                dst[offset + i] = static_cast<SampleT>(src[i] * out_scale);
            }
        }

        offset += chunk;
    }
}

auto CsoundSessionAdapter::process(const double* const* in_channels, int num_in,
                                   double* const* out_channels, int num_out, int num_frames,
                                   double zero_dbfs, float dry_wet, bool enabled) -> void {
    process_impl(in_channels, num_in, out_channels, num_out, num_frames, zero_dbfs, dry_wet,
                 enabled);
}

auto CsoundSessionAdapter::process(const float* const* in_channels, int num_in,
                                   float* const* out_channels, int num_out, int num_frames,
                                   double zero_dbfs, float dry_wet, bool enabled) -> void {
    process_impl(in_channels, num_in, out_channels, num_out, num_frames, zero_dbfs, dry_wet,
                 enabled);
}

auto CsoundSessionAdapter::generate(double* const* out_channels, int num_out, int num_frames,
                                    double zero_dbfs, bool enabled) -> void {
    process_impl<double>(nullptr, 0, out_channels, num_out, num_frames, zero_dbfs, 1.0f, enabled);
}

auto CsoundSessionAdapter::generate(float* const* out_channels, int num_out, int num_frames,
                                    double zero_dbfs, bool enabled) -> void {
    process_impl<float>(nullptr, 0, out_channels, num_out, num_frames, zero_dbfs, 1.0f, enabled);
}

auto CsoundSessionAdapter::get_summary() const -> ModelSummary {
    ModelSummary s;
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        return s;
    }
    auto& backend = m_session->backend();
    s.resolved_path = m_session->get_model_path();
    s.backend_name = backend.get_backend_name();
    s.method = m_session->get_method();
    s.in_channels = m_in_channels;
    s.out_channels = m_out_channels;
    s.in_ratio = m_in_ratio;
    s.out_ratio = m_out_ratio;
    s.host_sample_rate = m_session->engine().get_host_sample_rate();
    s.model_sample_rate = m_session->get_model_sample_rate();
    s.resampling_active = m_session->is_resampling_active();
    s.host_latency_samples = m_session->get_latency_samples();
    s.model_buffer_size = m_session->get_buffer_size();
    s.ksmps = m_ksmps;
    s.sw_buffer_frames = m_sw_buffer_frames;
    s.hw_buffer_frames = m_hw_buffer_frames;
    s.worker_thread_enabled = m_worker_thread_enabled;
    s.safety_sentry_enabled = m_safety_sentry_enabled;
    s.dry_delay_enabled = m_dry_delay_enabled;
    s.layers = backend.get_available_layers();
    s.methods = collect_unique_methods(backend, *m_session);
    s.settable_attributes = backend.get_settable_attributes();
    return s;
}

auto CsoundSessionAdapter::format_dump() const -> std::string {
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        return "[netbender] No model loaded.\n";
    }
    const ModelSummary s = get_summary();
    auto& backend = m_session->backend();

    std::ostringstream oss;
    oss << "=== NetBender Model Info ===\n";
    oss << "  path:        " << s.resolved_path << "\n";
    oss << "  backend:     " << s.backend_name << "\n";
    oss << "  method:      " << s.method << " (in_ch=" << s.in_channels
        << ", in_ratio=" << s.in_ratio << ", out_ch=" << s.out_channels
        << ", out_ratio=" << s.out_ratio << ")\n";
    const double kr_hz = (s.ksmps > 0) ? (s.host_sample_rate / static_cast<double>(s.ksmps)) : 0.0;
    oss << "  host:        sr=" << s.host_sample_rate << " Hz, ksmps=" << s.ksmps
        << " (kr=" << kr_hz << " Hz), sw_buf(-b)="
        << (s.sw_buffer_frames > 0 ? std::to_string(s.sw_buffer_frames) + " frames" : "default")
        << ", hw_buf(-B)="
        << (s.hw_buffer_frames > 0 ? std::to_string(s.hw_buffer_frames) + " frames" : "default")
        << "\n";
    oss << "  sample_rate: host=" << s.host_sample_rate << " Hz, model=" << s.model_sample_rate
        << " Hz (resampling=" << (s.resampling_active ? "active" : "bypassed") << ")\n";
    oss << "  streaming:   buffer_size=" << s.model_buffer_size
        << ", latency=" << s.host_latency_samples
        << " samples, mode=" << (s.worker_thread_enabled ? "async_worker" : "sync_inline") << "\n";
    oss << "  safety:      sentry=" << (s.safety_sentry_enabled ? "on" : "off")
        << ", dry_delay=" << (s.dry_delay_enabled ? "on" : "off") << "\n";

    oss << "  methods (" << s.methods.size() << "):";
    for (const auto& m : s.methods) {
        oss << " " << m;
    }
    oss << "\n";

    oss << "  attributes (" << s.settable_attributes.size() << "):";
    if (s.settable_attributes.empty()) {
        oss << " none";
    } else {
        for (const auto& a : s.settable_attributes) {
            oss << " " << a;
        }
    }
    oss << "\n";

    oss << "  bendable layers (" << s.layers.size() << "):\n";
    for (size_t i = 0; i < s.layers.size(); ++i) {
        const auto& name = s.layers[i];
        const auto shape = backend.get_layer_shape(name);
        oss << "    [" << i << "] " << name << " (";
        for (size_t d = 0; d < shape.size(); ++d) {
            if (d > 0) {
                oss << "x";
            }
            oss << shape[d];
        }
        oss << ")\n";
    }
    return oss.str();
}

}  // namespace netbender::csound_fe
