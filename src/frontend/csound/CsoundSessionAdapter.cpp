#include "CsoundSessionAdapter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <utility>
#include <fstream>
#include <nlohmann/json.hpp>

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
    const int initial_channels = std::max(2, opts.max_channels);
    m_session->prepare(host_sr, host_io_block, initial_channels);

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

    m_ksmps = safe_ksmps;
    m_host_sample_rate = host_sr;
    m_host_io_block = host_io_block;
    m_max_channels = initial_channels;
    m_sw_buffer_frames = std::max(0, opts.sw_buffer_frames);
    m_hw_buffer_frames = std::max(0, opts.hw_buffer_frames);
    m_worker_thread_enabled = use_worker;
    m_safety_sentry_enabled = enable_sentry;
    m_dry_delay_enabled = enable_dry;

    if (!sync_mode_metadata(error_out)) {
        close();
        return false;
    }

    m_open = true;
    return true;
}

auto CsoundSessionAdapter::sync_mode_metadata(std::string& error_out) -> bool {
    if (!m_session || !m_session->is_model_loaded()) {
        error_out = "netbender: no model loaded";
        return false;
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
        error_out = "netbender: method '" + active_method + "' produced 0 output channels";
        return false;
    }

    // Re-prepare if model needs more channels than initial guess so scratch/dry rings fit cleanly
    const int required_channels = std::max({2, m_max_channels, m_in_channels, m_out_channels});
    if (required_channels > m_max_channels) {
        m_session->prepare(m_host_sample_rate, m_host_io_block, required_channels);
        m_max_channels = required_channels;
    }

    const int needed_scratch_channels = std::max(64, required_channels);
    const int needed_scratch_frames = std::max(1024, m_ksmps);
    if (m_scratch_channels < needed_scratch_channels || m_scratch_frames < needed_scratch_frames) {
        m_scratch_channels = std::max(m_scratch_channels, needed_scratch_channels);
        m_scratch_frames = std::max(m_scratch_frames, needed_scratch_frames);
        m_scaled_in_storage.resize(static_cast<size_t>(m_scratch_channels));
        m_scaled_in_ptrs.resize(static_cast<size_t>(m_scratch_channels), nullptr);
        m_scaled_out_storage.resize(static_cast<size_t>(m_scratch_channels));
        m_scaled_out_ptrs.resize(static_cast<size_t>(m_scratch_channels), nullptr);
        for (int c = 0; c < m_scratch_channels; ++c) {
            m_scaled_in_storage[static_cast<size_t>(c)].assign(
                static_cast<size_t>(m_scratch_frames), 0.0f);
            m_scaled_in_ptrs[static_cast<size_t>(c)] =
                m_scaled_in_storage[static_cast<size_t>(c)].data();
            m_scaled_out_storage[static_cast<size_t>(c)].assign(
                static_cast<size_t>(m_scratch_frames), 0.0f);
            m_scaled_out_ptrs[static_cast<size_t>(c)] =
                m_scaled_out_storage[static_cast<size_t>(c)].data();
        }
    }

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
    m_host_sample_rate = 44100.0;
    m_host_io_block = 256;
    m_max_channels = 2;
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

auto CsoundSessionAdapter::get_layer_count() const -> int {
    return (m_open && m_session) ? m_session->get_layer_count() : 0;
}

auto CsoundSessionAdapter::find_layer_index(const std::string& name) const -> int {
    return (m_open && m_session) ? m_session->find_layer_index(name) : -1;
}

auto CsoundSessionAdapter::get_layer_name(int index) const -> std::string {
    return (m_open && m_session) ? m_session->get_layer_name(index) : std::string{};
}

auto CsoundSessionAdapter::get_layer_weight_count(int index) const -> size_t {
    return (m_open && m_session) ? m_session->get_layer_weight_count(index) : 0;
}

auto CsoundSessionAdapter::set_layer_bend(int layer_idx, float scale, float offset, float heat,
                                          float memory, int drift_mode, bool frozen) noexcept
    -> void {
    if (m_open && m_session) {
        m_session->set_layer_bend_rt(layer_idx, scale, offset, heat, memory, drift_mode,
                                     frozen ? 1 : 0);
    }
}

auto CsoundSessionAdapter::set_layer_bridge(int target_idx, int source_idx, float depth) noexcept
    -> void {
    if (m_open && m_session) {
        m_session->set_layer_bridge_rt(target_idx, source_idx, depth);
    }
}

auto CsoundSessionAdapter::set_short_circuit(bool active) noexcept -> void {
    if (m_open && m_session) {
        m_session->trigger_short_circuit(active);
    }
}

auto CsoundSessionAdapter::set_trigger(int mode, float attack, float release,
                                       float threshold) noexcept -> void {
    if (!m_open || !m_session) {
        return;
    }
    auto& bm = m_session->bending();
    bm.set_trigger_mode(static_cast<TriggerMode>(std::clamp(mode, 0, 2)));
    if (attack >= 0.0f) {
        bm.set_envelope_attack(attack);
    }
    if (release >= 0.0f) {
        bm.set_envelope_release(release);
    }
    if (threshold >= 0.0f) {
        bm.set_transient_threshold(threshold);
    }
}

auto CsoundSessionAdapter::set_latent(bool enabled, float x, float y, float depth, int mode,
                                      float slew) noexcept -> void {
    if (!m_open || !m_session) {
        return;
    }
    auto& lte = m_session->latent();
    lte.set_enabled(enabled);
    lte.set_coords(x, y);
    lte.set_depth(depth);
    lte.set_mode(static_cast<LatentMode>(std::clamp(mode, 0, 2)));
    lte.set_slew(slew);
}

auto CsoundSessionAdapter::get_sentry_blown() const noexcept -> bool {
    return (m_open && m_session) ? m_session->safety_sentry().is_fuse_blown() : false;
}

auto CsoundSessionAdapter::set_sentry_auto_reset(bool enabled) noexcept -> void {
    if (m_open && m_session) {
        m_session->safety_sentry().set_auto_reset(enabled);
    }
}

auto CsoundSessionAdapter::request_fuse_reset() noexcept -> void {
    if (m_open && m_session) {
        m_session->request_fuse_reset();
    }
}

auto CsoundSessionAdapter::get_layer_weights(int layer_idx, bool original) const
    -> std::vector<float> {
    return (m_open && m_session) ? m_session->get_layer_weights(layer_idx, original)
                                 : std::vector<float>{};
}

auto CsoundSessionAdapter::prepare_layer_weight_staging(int layer_idx) -> bool {
    return (m_open && m_session) ? m_session->prepare_layer_weight_staging(layer_idx) : false;
}

auto CsoundSessionAdapter::set_layer_weights_immediate(int layer_idx, const float* data,
                                                       size_t count) -> bool {
    return (m_open && m_session) ? m_session->set_layer_weights_immediate(layer_idx, data, count)
                                 : false;
}

auto CsoundSessionAdapter::set_layer_weights_rt(int layer_idx, const float* data,
                                                size_t count) noexcept -> bool {
    return (m_open && m_session) ? m_session->set_layer_weights_rt(layer_idx, data, count) : false;
}

auto CsoundSessionAdapter::request_reset_layer_weights(int layer_idx) noexcept -> void {
    if (m_open && m_session) {
        m_session->request_reset_layer_weights(layer_idx);
    }
}

auto CsoundSessionAdapter::request_reset_all_weights() noexcept -> void {
    if (m_open && m_session) {
        m_session->request_reset_all_weights();
    }
}

auto CsoundSessionAdapter::reset_layer_weights_immediate(int layer_idx) -> void {
    if (!m_open || !m_session) {
        return;
    }
    const std::string name = m_session->get_layer_name(layer_idx);
    if (!name.empty()) {
        m_session->clear_layer_bending(name);
    }
}

auto CsoundSessionAdapter::reset_all_weights_immediate() -> void {
    if (m_open && m_session) {
        m_session->clear_all_layer_bending();
    }
}

namespace {

auto filter_attr_names(std::vector<std::string> raw) -> std::vector<std::string> {
    std::vector<std::string> out;
    out.reserve(raw.size());
    for (auto& name : raw) {
        if (!name.empty() && name != "none" &&
            std::find(out.begin(), out.end(), name) == out.end()) {
            out.push_back(std::move(name));
        }
    }
    return out;
}

template <typename Fn>
auto run_catching_backend(const char* op_name, std::string& error_out, Fn&& fn) -> bool {
    try {
        fn();
        return true;
    } catch (const std::string& s) {
        error_out = std::string(op_name) + ": " + s;
    } catch (const char* s) {
        error_out = std::string(op_name) + ": " + (s != nullptr ? s : "unknown error");
    } catch (const std::exception& e) {
        error_out = std::string(op_name) + ": " + e.what();
    } catch (...) {
        error_out = std::string(op_name) + ": unknown backend exception";
    }
    return false;
}

}  // namespace

auto CsoundSessionAdapter::save_state_json(bool include_drawn_weights, bool pretty) const
    -> std::string {
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        return {};
    }
    return m_session->to_json(include_drawn_weights).dump(pretty ? 2 : -1);
}

auto CsoundSessionAdapter::save_state_file(const std::string& path, bool include_drawn_weights,
                                           std::string& error_out) const -> bool {
    error_out.clear();
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        error_out = "netbender_savestate: session is not open or no model is loaded";
        return false;
    }
    if (path.empty()) {
        error_out = "netbender_savestate: file path cannot be empty";
        return false;
    }
    const std::string json_str = save_state_json(include_drawn_weights, /*pretty=*/true);
    if (json_str.empty()) {
        error_out = "netbender_savestate: failed to serialize session state";
        return false;
    }
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        error_out = "netbender_savestate: could not open '" + path + "' for writing";
        return false;
    }
    ofs << json_str << "\n";
    if (!ofs.good()) {
        error_out = "netbender_savestate: write failed for '" + path + "'";
        return false;
    }
    return true;
}

auto CsoundSessionAdapter::load_state_json(const std::string& json_str, bool restore_model,
                                           std::string& error_out,
                                           const HostFileFinder& host_finder) -> bool {
    error_out.clear();
    if (!m_open || !m_session) {
        error_out = "netbender_loadstate: session is not open";
        return false;
    }

    size_t len = json_str.size();
    while (len > 0 && json_str[len - 1] == '\0') {
        --len;
    }
    if (len == 0) {
        error_out = "netbender_loadstate: JSON state string is empty";
        return false;
    }

    nlohmann::json j = nlohmann::json::parse(json_str.data(), json_str.data() + len, nullptr,
                                             /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        error_out = "netbender_loadstate: state is not valid JSON";
        return false;
    }
    if (j.value("format", std::string{}) != NetBenderSession::k_state_format) {
        error_out = "netbender_loadstate: unexpected preset format (expected 'netbender.session')";
        return false;
    }
    if (j.value("version", 0) > NetBenderSession::k_state_version) {
        error_out = "netbender_loadstate: preset version is newer than this build supports";
        return false;
    }

    if (!restore_model) {
        // Erase model section so NetBenderSession::from_json leaves active method/buffer intact.
        j.erase("model");
    } else if (j.contains("model") && j["model"].is_object() && !m_has_custom_creator) {
        const std::string raw_path = j["model"].value("path", std::string{});
        if (!raw_path.empty() && !is_regular_file_noexcept(raw_path)) {
            std::string resolved = resolve_model_path(raw_path, host_finder);
            if (resolved.empty()) {
                const std::string filename = std::filesystem::path(raw_path).filename().string();
                if (!filename.empty() && filename != raw_path) {
                    resolved = resolve_model_path(filename, host_finder);
                }
            }
            if (!resolved.empty()) {
                j["model"]["path"] = resolved;
            }
        }
    }

    if (!m_session->from_json(j, restore_model)) {
        error_out = m_session->get_last_error();
        if (error_out.empty()) {
            error_out = "netbender_loadstate: failed to apply session state";
        }
        return false;
    }

    if (restore_model) {
        if (!sync_mode_metadata(error_out)) {
            return false;
        }
    }
    return true;
}

auto CsoundSessionAdapter::load_state_file(const std::string& raw_path, bool restore_model,
                                           const HostFileFinder& host_finder,
                                           std::string& error_out) -> bool {
    error_out.clear();
    if (raw_path.empty()) {
        error_out = "netbender_loadstate: preset path cannot be empty";
        return false;
    }

    std::string resolved;
    if (is_regular_file_noexcept(raw_path)) {
        resolved = raw_path;
    } else if (host_finder) {
        const std::string cand = host_finder(raw_path);
        if (!cand.empty() && is_regular_file_noexcept(cand)) {
            resolved = cand;
        }
    }
    if (resolved.empty()) {
        error_out = "netbender_loadstate: could not find preset file '" + raw_path + "'";
        return false;
    }

    std::ifstream ifs(resolved, std::ios::binary);
    if (!ifs.is_open()) {
        error_out = "netbender_loadstate: could not open preset file '" + resolved + "'";
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    if (ifs.bad()) {
        error_out = "netbender_loadstate: failed reading preset file '" + resolved + "'";
        return false;
    }
    return load_state_json(ss.str(), restore_model, error_out, host_finder);
}

auto CsoundSessionAdapter::load_state_path_or_json(const std::string& input, bool restore_model,
                                                   const HostFileFinder& host_finder,
                                                   std::string& error_out) -> bool {
    size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first])) != 0) {
        ++first;
    }
    if (first >= input.size()) {
        error_out = "netbender_loadstate: preset path or JSON string cannot be empty";
        return false;
    }
    if (input[first] == '{') {
        return load_state_json(input.substr(first), restore_model, error_out, host_finder);
    }
    return load_state_file(input, restore_model, host_finder, error_out);
}

auto CsoundSessionAdapter::get_available_attributes() const -> std::vector<std::string> {
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        return {};
    }
    std::vector<std::string> attrs;
    std::string ignored_err;
    run_catching_backend("netbender_attrs", ignored_err, [&] {
        attrs = filter_attr_names(m_session->backend().get_available_attributes());
    });
    return attrs;
}

auto CsoundSessionAdapter::get_settable_attributes() const -> std::vector<std::string> {
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        return {};
    }
    std::vector<std::string> attrs;
    std::string ignored_err;
    run_catching_backend("netbender_attrs", ignored_err, [&] {
        attrs = filter_attr_names(m_session->backend().get_settable_attributes());
    });
    return attrs;
}

auto CsoundSessionAdapter::get_attribute(const std::string& attr_name, std::string& value_out,
                                         std::string& error_out) const -> bool {
    error_out.clear();
    value_out.clear();
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        error_out = "netbender_attr_get: session is not open or no model is loaded";
        return false;
    }
    if (attr_name.empty()) {
        error_out = "netbender_attr_get: attribute name cannot be empty";
        return false;
    }

    bool known = false;
    if (!run_catching_backend("netbender_attr_get", error_out, [&] {
            const auto settable = filter_attr_names(m_session->backend().get_settable_attributes());
            if (std::find(settable.begin(), settable.end(), attr_name) != settable.end()) {
                known = true;
            } else {
                const auto avail =
                    filter_attr_names(m_session->backend().get_available_attributes());
                known = (std::find(avail.begin(), avail.end(), attr_name) != avail.end());
            }
        })) {
        return false;
    }
    if (!known) {
        error_out = "netbender_attr_get: unknown attribute '" + attr_name + "'";
        return false;
    }

    return run_catching_backend("netbender_attr_get", error_out, [&] {
        value_out = m_session->backend().get_attribute_as_string(attr_name);
    });
}

auto CsoundSessionAdapter::set_attribute(const std::string& attr_name,
                                         const std::vector<std::string>& args,
                                         std::string& error_out) -> bool {
    error_out.clear();
    if (!m_open || !m_session || !m_session->is_model_loaded()) {
        error_out = "netbender_attr_set: session is not open or no model is loaded";
        return false;
    }
    if (attr_name.empty()) {
        error_out = "netbender_attr_set: attribute name cannot be empty";
        return false;
    }
    if (args.empty()) {
        error_out = "netbender_attr_set: value cannot be empty";
        return false;
    }

    bool settable = false;
    if (!run_catching_backend("netbender_attr_set", error_out, [&] {
            settable = m_session->backend().has_settable_attribute(attr_name);
        })) {
        return false;
    }
    if (!settable) {
        error_out =
            "netbender_attr_set: attribute '" + attr_name + "' is not settable on this model";
        return false;
    }

    return run_catching_backend("netbender_attr_set", error_out, [&] {
        m_session->backend().set_attribute(attr_name, args, IBackend::BufferMap{});
    });
}

auto CsoundSessionAdapter::set_attribute_string(const std::string& attr_name,
                                                const std::string& raw_value,
                                                std::string& error_out) -> bool {
    std::istringstream iss(raw_value);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(std::move(tok));
    }
    if (tokens.empty()) {
        error_out = "netbender_attr_set: value cannot be empty";
        return false;
    }
    return set_attribute(attr_name, tokens, error_out);
}

}  // namespace netbender::csound_fe
