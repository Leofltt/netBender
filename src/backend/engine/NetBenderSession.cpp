#include "NetBenderSession.h"
#include "BackendFactory.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <nlohmann/json.hpp>
#include <cstring>

namespace {

using json = nlohmann::json;

auto to_lower(std::string s) -> std::string {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

auto ends_with(const std::string& s, const std::string& suffix) -> bool {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

auto replace_suffix(const std::string& s, const std::string& from, const std::string& to)
    -> std::string {
    return s.substr(0, s.size() - from.size()) + to;
}

auto drift_mode_to_string(DriftMode mode) -> const char* {
    switch (mode) {
        case DriftMode::Noise:
            return "noise";
        case DriftMode::RandomWalk:
            return "random_walk";
        case DriftMode::ThermalOU:
        default:
            return "thermal_ou";
    }
}

auto drift_mode_from_string(const std::string& s) -> DriftMode {
    if (s == "noise") {
        return DriftMode::Noise;
    }
    if (s == "random_walk") {
        return DriftMode::RandomWalk;
    }
    return DriftMode::ThermalOU;
}

auto trigger_mode_to_string(TriggerMode mode) -> const char* {
    switch (mode) {
        case TriggerMode::Momentary:
            return "momentary";
        case TriggerMode::Transient:
            return "transient";
        case TriggerMode::Continuous:
        default:
            return "continuous";
    }
}

auto trigger_mode_from_string(const std::string& s) -> TriggerMode {
    if (s == "momentary") {
        return TriggerMode::Momentary;
    }
    if (s == "transient") {
        return TriggerMode::Transient;
    }
    return TriggerMode::Continuous;
}

auto latent_mode_to_string(LatentMode mode) -> const char* {
    switch (mode) {
        case LatentMode::LatentSlew:
            return "latent_slew";
        case LatentMode::Quantize:
            return "quantize";
        case LatentMode::FourierOrbit:
        default:
            return "fourier_orbit";
    }
}

auto latent_mode_from_string(const std::string& s) -> LatentMode {
    if (s == "latent_slew") {
        return LatentMode::LatentSlew;
    }
    if (s == "quantize") {
        return LatentMode::Quantize;
    }
    return LatentMode::FourierOrbit;
}

auto weights_differ(const std::vector<float>& a, const std::vector<float>& b) -> bool {
    if (a.size() != b.size()) {
        return true;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-7f) {
            return true;
        }
    }
    return false;
}

auto compute_layer_weight_count(IBackend& backend, const std::string& name) -> size_t {
    const auto shape = backend.get_layer_shape(name);
    if (!shape.empty()) {
        size_t total = 1;
        bool valid = true;
        for (int64_t dim : shape) {
            if (dim <= 0) {
                valid = false;
                break;
            }
            total *= static_cast<size_t>(dim);
        }
        if (valid && total > 0) {
            return total;
        }
    }
    return backend.get_original_layer_weights(name).size();
}

}  // namespace

//==============================================================================
NetBenderSession::NetBenderSession(SessionConfig config)
    : m_config(config), m_backend_creator([](const std::string& path) {
          return BackendFactory::create_for_file(path);
      }) {
    m_backend = BackendFactory::create_default();

    m_engine.set_pre_inference_callback([this](IBackend& backend) {
        drain_pending_rt_bending();
        std::string active_layer;
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            active_layer = m_active_layer;
        }
        if (m_bending_automation_active.load()) {
            m_bending.apply_bending(&backend, active_layer, m_auto_scale.load(),
                                    m_auto_offset.load(), m_auto_heat.load(), m_auto_memory.load(),
                                    m_auto_short.load());
        } else {
            // No host automation: never override the active layer's stored state
            m_bending.apply_bending(&backend, std::string{}, 1.0f, 0.0f, 0.0f, -1.0f, false);
        }
    });

    m_engine.set_latent_hook_provider([this]() -> IBackend::LatentHook {
        std::string method;
        IBackend::LatentHook user_hook;
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            method = m_method;
            user_hook = m_user_latent_hook;
        }
        if (user_hook) {
            return user_hook;
        }
        if (method == "decode gen" || m_latent.is_enabled()) {
            float lx = -999.0f;
            float ly = -999.0f;
            float depth = -999.0f;
            if (m_latent_automation_active.load()) {
                lx = m_auto_latent_x.load();
                ly = m_auto_latent_y.load();
                depth = m_auto_latent_depth.load();
            }
            return [this, lx, ly, depth](IBackend::LatentBuffer& buf) {
                m_latent.apply_hook(buf, lx, ly, depth);
            };
        }
        return nullptr;
    });

    m_engine.set_fuse_reset_callback([this] { request_fuse_reset(); });

    m_engine.set_error_callback([this](const std::string& msg) { set_error(msg); });
}

NetBenderSession::~NetBenderSession() {
    m_engine.detach_backend();
    clear_layer_slots();
}

//==============================================================================
auto NetBenderSession::prepare(double host_sample_rate, int max_block_size, int max_channels)
    -> void {
    m_engine.stop();
    m_host_sample_rate.store(host_sample_rate > 0.0 ? host_sample_rate : 44100.0);
    m_max_block_size.store(std::max(m_config.min_host_block_size, max_block_size));
    m_max_channels.store(std::max(m_config.min_host_channels, max_channels));
    if (m_backend) {
        m_backend->set_sample_rate(m_host_sample_rate.load());
    }
    configure_engine();
}

auto NetBenderSession::release() -> void {
    m_engine.stop();
}

//==============================================================================
auto NetBenderSession::set_backend_creator(BackendCreator creator) -> void {
    if (creator) {
        m_backend_creator = std::move(creator);
        m_custom_backend_creator = true;
    }
}

auto NetBenderSession::load_model(const std::string& path) -> bool {
    m_model_loaded.store(false);
    m_engine.detach_backend();
    clear_layer_slots();

    try {
        auto new_backend = m_backend_creator(path);
        if (!new_backend) {
            set_error("NetBenderSession: unsupported model file '" + path + "'");
            return false;
        }

        const std::string lower = to_lower(path);
        if (ends_with(lower, "_coreml.pte") || ends_with(lower, "_mps.pte")) {
            m_use_gpu.store(true);
        } else if (ends_with(lower, "_xnnpack.pte") || ends_with(lower, "_portable.pte")) {
            m_use_gpu.store(false);
        }

        new_backend->use_gpu(m_use_gpu.load());
        m_backend = std::move(new_backend);

        const int err = m_backend->load(path, m_host_sample_rate.load());
        if (err != 0) {
            set_error("NetBenderSession: failed to load '" + path +
                      "': " + m_backend->get_last_error());
            return false;
        }

        // Default mode: prefer "forward", else first available
        std::string default_mode;
        const auto modes = m_backend->get_plugin_modes();
        if (!modes.empty()) {
            default_mode = modes[0];
            if (std::find(modes.begin(), modes.end(), "forward") != modes.end()) {
                default_mode = "forward";
            }
        }
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            m_model_path = path;
            m_method = default_mode;
            m_active_layer.clear();
            m_last_error.clear();
        }
        m_model_in = 0;
        m_model_out = 0;
        if (!default_mode.empty()) {
            update_mode_channels(default_mode);
        }

        // Auto-configure buffer size to backend preferred size or model ratio
        const int pref_size = m_backend->get_preferred_buffer_size();
        if (pref_size > 0) {
            m_buffer_size.store(pref_size);
        } else {
            const int higher_ratio = m_backend->get_higher_ratio();
            int req_size = 1;
            while (req_size < higher_ratio) {
                req_size <<= 1;
            }
            const int min_buffer_for_ratio = (higher_ratio >= 2048) ? 4096 : 2048;
            m_buffer_size.store(std::max(min_buffer_for_ratio, req_size * 2));
        }

        // Reset latent terrain cached matrices, stale layer bending states, and rebuild RT slots
        m_latent.reset();
        m_bending.clear_all_layer_bending();
        rebuild_layer_slots();

        m_model_loaded.store(true);
        configure_engine();
        return true;
    } catch (const std::exception& e) {
        set_error(std::string("NetBenderSession: exception loading model: ") + e.what());
    } catch (...) {
        set_error("NetBenderSession: unknown exception loading model");
    }
    return false;
}

auto NetBenderSession::get_model_path() const -> std::string {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_model_path;
}

auto NetBenderSession::get_last_error() const -> std::string {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_last_error;
}

auto NetBenderSession::get_available_modes() const -> std::vector<std::string> {
    if (!m_model_loaded.load() || !m_backend) {
        return {};
    }
    return m_backend->get_plugin_modes();
}

auto NetBenderSession::get_layer_names() const -> std::vector<std::string> {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_layer_names;
}

auto NetBenderSession::find_layer_index(const std::string& name) const -> int {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    const auto it = m_layer_name_to_idx.find(name);
    return (it != m_layer_name_to_idx.end()) ? it->second : -1;
}

auto NetBenderSession::get_layer_name(int index) const -> std::string {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    if (index < 0 || static_cast<size_t>(index) >= m_layer_names.size()) {
        return {};
    }
    return m_layer_names[static_cast<size_t>(index)];
}

auto NetBenderSession::get_layer_weight_count(int index) const -> size_t {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    if (index < 0 || static_cast<size_t>(index) >= m_layer_weight_counts.size()) {
        return 0;
    }
    return m_layer_weight_counts[static_cast<size_t>(index)];
}

auto NetBenderSession::set_method(const std::string& method) -> bool {
    if (!m_backend || !m_model_loaded.load()) {
        return false;
    }
    const auto modes = m_backend->get_plugin_modes();
    const bool is_valid = (std::find(modes.begin(), modes.end(), method) != modes.end()) ||
                          m_backend->has_method(method);
    if (!is_valid) {
        return false;
    }

    const int prev_in = m_model_in;
    const int prev_out = m_model_out;
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_method = method;
    }
    update_mode_channels(method);

    if (prev_in != m_model_in || prev_out != m_model_out || !m_engine.is_configured()) {
        configure_engine();
    } else {
        m_engine.set_method(method);
        if (!m_engine.is_worker_running()) {
            m_backend->clear_decode_gen_fifo();
        }
    }
    return true;
}

auto NetBenderSession::get_method() const -> std::string {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_method;
}

auto NetBenderSession::set_buffer_size(int size) -> bool {
    if (size <= 0 || (size & (size - 1)) != 0) {
        return false;
    }
    m_buffer_size.store(size);
    configure_engine();
    return true;
}

auto NetBenderSession::set_use_gpu(bool enabled) -> void {
    if (m_use_gpu.load() == enabled && m_model_loaded.load()) {
        return;
    }

    m_use_gpu.store(enabled);
    const std::string model_path = get_model_path();
    if (!m_model_loaded.load() || model_path.empty()) {
        if (m_backend) {
            m_backend->use_gpu(enabled);
        }
        return;
    }

    const std::string saved_method = get_method();
    const int saved_buffer_size = m_buffer_size.load();
    std::string target_path = model_path;

    // For .pte models, swap between sibling *_xnnpack.pte (CPU) and *_coreml.pte (GPU / ANE)
    const std::string lower = to_lower(model_path);
    std::string candidate;
    if (enabled && ends_with(lower, "_xnnpack.pte")) {
        candidate = replace_suffix(model_path, "_xnnpack.pte", "_coreml.pte");
    } else if (!enabled && ends_with(lower, "_coreml.pte")) {
        candidate = replace_suffix(model_path, "_coreml.pte", "_xnnpack.pte");
    }
    std::error_code ec;
    if (!candidate.empty() && std::filesystem::is_regular_file(candidate, ec)) {
        target_path = candidate;
    }

    if (load_model(target_path)) {
        if (!saved_method.empty() && saved_method != get_method()) {
            set_method(saved_method);
        }
        if (saved_buffer_size > 0 && saved_buffer_size != m_buffer_size.load()) {
            set_buffer_size(saved_buffer_size);
        }
    }
}

//==============================================================================
template <typename SampleT>
auto NetBenderSession::process_impl(const SampleT* const* host_in, int num_in_channels,
                                    SampleT* const* host_out, int num_out_channels,
                                    int num_frames) NETBENDER_NONBLOCKING -> void {
    if (num_frames <= 0) {
        return;
    }

    if (!m_model_loaded.load() || !m_engine.is_configured()) {
        if (host_out != nullptr) {
            for (int c = 0; c < num_out_channels; ++c) {
                if (host_out[c] != nullptr) {
                    std::fill(host_out[c], host_out[c] + num_frames, SampleT(0));
                }
            }
        }
        return;
    }

    // Sidechain audio envelope follower for transient-triggered bending
    if (host_in != nullptr && num_in_channels > 0) {
        SampleT max_peak = SampleT(0);
        for (int c = 0; c < num_in_channels; ++c) {
            const SampleT* ch = host_in[c];
            if (ch == nullptr) {
                continue;
            }
            for (int i = 0; i < num_frames; ++i) {
                const SampleT a = ch[i] < SampleT(0) ? -ch[i] : ch[i];
                max_peak = a > max_peak ? a : max_peak;
            }
        }
        m_bending.update_audio_envelope(static_cast<float>(max_peak));
    }

    m_engine.set_dry_wet(m_dry_wet.load());
    m_engine.process(host_in, num_in_channels, host_out, num_out_channels, num_frames);
}

auto NetBenderSession::process(const float* const* host_in, int num_in_channels,
                               float* const* host_out, int num_out_channels,
                               int num_frames) NETBENDER_NONBLOCKING -> void {
    process_impl<float>(host_in, num_in_channels, host_out, num_out_channels, num_frames);
}

auto NetBenderSession::process(const double* const* host_in, int num_in_channels,
                               double* const* host_out, int num_out_channels,
                               int num_frames) NETBENDER_NONBLOCKING -> void {
    process_impl<double>(host_in, num_in_channels, host_out, num_out_channels, num_frames);
}

auto NetBenderSession::set_dry_wet(float dry_wet) noexcept NETBENDER_NONBLOCKING -> void {
    const float clamped = std::clamp(dry_wet, 0.0f, 1.0f);
    m_dry_wet.store(clamped);
    m_engine.set_dry_wet(clamped);
}

auto NetBenderSession::set_bending_automation(float scale, float offset, float heat, float memory,
                                              bool short_active) noexcept NETBENDER_NONBLOCKING
    -> void {
    m_auto_scale.store(scale);
    m_auto_offset.store(offset);
    m_auto_heat.store(heat);
    m_auto_memory.store(memory);
    m_auto_short.store(short_active);
    m_bending_automation_active.store(true);
}

auto NetBenderSession::set_layer_bend_rt(int layer_idx, float scale, float offset, float heat,
                                         float memory, int drift_mode,
                                         int frozen) noexcept NETBENDER_NONBLOCKING -> void {
    ScopedRtReader reader(m_layer_slots_gate);
    if (!reader.is_acquired() || !m_model_loaded.load(std::memory_order_acquire) || layer_idx < 0 ||
        static_cast<size_t>(layer_idx) >= m_layer_slots.size()) {
        return;
    }

    auto& slot = *m_layer_slots[static_cast<size_t>(layer_idx)];
    const uint64_t seq = m_op_seq.fetch_add(1, std::memory_order_acq_rel) + 1;

    slot.scale.store(scale, std::memory_order_relaxed);
    slot.offset.store(offset, std::memory_order_relaxed);
    slot.bend_seq.store(seq, std::memory_order_relaxed);
    slot.scale_offset_dirty.store(true, std::memory_order_release);

    if (heat >= 0.0f) {
        slot.heat.store(heat, std::memory_order_relaxed);
        slot.heat_seq.store(seq, std::memory_order_relaxed);
        slot.heat_dirty.store(true, std::memory_order_release);
    }
    if (memory >= 0.0f) {
        slot.memory.store(std::clamp(memory, 0.0f, 1.0f), std::memory_order_relaxed);
        slot.memory_seq.store(seq, std::memory_order_relaxed);
        slot.memory_dirty.store(true, std::memory_order_release);
    }
    if (drift_mode >= 0) {
        slot.drift_mode.store(std::clamp(drift_mode, 0, 2), std::memory_order_relaxed);
        slot.drift_seq.store(seq, std::memory_order_relaxed);
        slot.drift_dirty.store(true, std::memory_order_release);
    }
    if (frozen >= 0) {
        slot.frozen.store(frozen > 0 ? 1 : 0, std::memory_order_relaxed);
        slot.frozen_seq.store(seq, std::memory_order_relaxed);
        slot.frozen_dirty.store(true, std::memory_order_release);
    }

    m_any_slot_dirty.store(true, std::memory_order_release);
}

auto NetBenderSession::set_layer_bridge_rt(int target_idx, int source_idx,
                                           float depth) noexcept NETBENDER_NONBLOCKING -> void {
    ScopedRtReader reader(m_layer_slots_gate);
    if (!reader.is_acquired() || !m_model_loaded.load(std::memory_order_acquire) ||
        target_idx < 0 || static_cast<size_t>(target_idx) >= m_layer_slots.size()) {
        return;
    }

    int norm_src = -1;
    if (source_idx >= 0 && static_cast<size_t>(source_idx) < m_layer_slots.size() &&
        source_idx != target_idx) {
        norm_src = source_idx;
    }

    auto& slot = *m_layer_slots[static_cast<size_t>(target_idx)];
    const uint64_t seq = m_op_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    slot.bridge_source_idx.store(norm_src, std::memory_order_relaxed);
    slot.bridge_depth.store(std::clamp(depth, 0.0f, 1.0f), std::memory_order_relaxed);
    slot.bridge_seq.store(seq, std::memory_order_relaxed);
    slot.bridge_dirty.store(true, std::memory_order_release);
    m_any_slot_dirty.store(true, std::memory_order_release);
}

auto NetBenderSession::set_layer_weights_rt(int layer_idx, const float* data,
                                            size_t count) noexcept NETBENDER_NONBLOCKING -> bool {
    ScopedRtReader reader(m_layer_slots_gate);
    if (!reader.is_acquired() || !m_model_loaded.load(std::memory_order_acquire) ||
        data == nullptr || layer_idx < 0 ||
        static_cast<size_t>(layer_idx) >= m_layer_slots.size()) {
        return false;
    }

    auto& slot = *m_layer_slots[static_cast<size_t>(layer_idx)];
    if (!slot.tb_ready.load(std::memory_order_acquire) || slot.tb_buf[0].size() != count) {
        return false;
    }

    std::memcpy(slot.tb_buf[slot.tb_write_idx].data(), data, count * sizeof(float));
    const uint64_t seq = m_op_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    slot.weights_seq.store(seq, std::memory_order_relaxed);

    const int prev = slot.tb_middle.exchange(slot.tb_write_idx | 4, std::memory_order_acq_rel);
    slot.tb_write_idx = prev & 3;
    m_any_slot_dirty.store(true, std::memory_order_release);
    return true;
}

auto NetBenderSession::request_reset_layer_weights(int layer_idx) noexcept NETBENDER_NONBLOCKING
    -> void {
    ScopedRtReader reader(m_layer_slots_gate);
    if (!reader.is_acquired() || !m_model_loaded.load(std::memory_order_acquire) || layer_idx < 0 ||
        static_cast<size_t>(layer_idx) >= m_layer_slots.size()) {
        return;
    }

    auto& slot = *m_layer_slots[static_cast<size_t>(layer_idx)];
    const uint64_t seq = m_op_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    slot.reset_seq.store(seq, std::memory_order_relaxed);
    slot.reset_requested.store(true, std::memory_order_release);
    m_any_slot_dirty.store(true, std::memory_order_release);
}

auto NetBenderSession::request_reset_all_weights() noexcept NETBENDER_NONBLOCKING -> void {
    if (!m_model_loaded.load(std::memory_order_acquire)) {
        return;
    }
    const uint64_t seq = m_op_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    m_reset_all_seq.store(seq, std::memory_order_relaxed);
    m_reset_all_requested.store(true, std::memory_order_release);
    m_any_slot_dirty.store(true, std::memory_order_release);
}

auto NetBenderSession::request_fuse_reset() noexcept NETBENDER_NONBLOCKING -> void {
    m_engine.get_safety_sentry().reset_fuse();
    m_bending.request_audio_thread_reset_all();
    request_reset_all_weights();
}

auto NetBenderSession::set_latent_automation(float x, float y,
                                             float depth) noexcept NETBENDER_NONBLOCKING -> void {
    m_auto_latent_x.store(x);
    m_auto_latent_y.store(y);
    m_auto_latent_depth.store(depth);
    m_latent_automation_active.store(true);
}

//==============================================================================
auto NetBenderSession::set_active_layer(const std::string& name) -> void {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    m_active_layer = name;
}

auto NetBenderSession::get_active_layer() const -> std::string {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_active_layer;
}

auto NetBenderSession::clear_layer_bending(const std::string& name) -> void {
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        const auto it = m_layer_name_to_idx.find(name);
        if (it != m_layer_name_to_idx.end() && it->second >= 0 &&
            static_cast<size_t>(it->second) < m_layer_slots.size()) {
            const uint64_t seq = m_op_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
            m_layer_slots[static_cast<size_t>(it->second)]->reset_seq.store(
                seq, std::memory_order_release);
        }
    }
    m_bending.clear_layer_bending(name);
    if (!m_engine.is_worker_running() && m_backend) {
        m_backend->reset_layer_weights(name);
    }
}

auto NetBenderSession::clear_all_layer_bending() -> void {
    const uint64_t seq = m_op_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    m_reset_all_seq.store(seq, std::memory_order_release);
    m_bending.clear_all_layer_bending();
    if (!m_engine.is_worker_running() && m_backend) {
        m_backend->reset_all_layer_weights();
    }
}

auto NetBenderSession::reset_fuse() -> void {
    m_engine.get_safety_sentry().reset_fuse();
    clear_all_layer_bending();
}

auto NetBenderSession::set_latent_hook(IBackend::LatentHook hook) -> void {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    m_user_latent_hook = std::move(hook);
}

auto NetBenderSession::get_layer_weights(int layer_idx, bool original) const -> std::vector<float> {
    if (!m_model_loaded.load(std::memory_order_acquire) || !m_backend) {
        return {};
    }
    std::string name;
    bool pending_reset = false;
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= m_layer_names.size()) {
            return {};
        }
        name = m_layer_names[static_cast<size_t>(layer_idx)];
        if (!original && static_cast<size_t>(layer_idx) < m_layer_slots.size()) {
            const auto& slot = *m_layer_slots[static_cast<size_t>(layer_idx)];
            const uint64_t pending_reset_seq =
                std::max(m_reset_all_seq.load(std::memory_order_acquire),
                         slot.reset_seq.load(std::memory_order_acquire));
            const bool reset_flag = m_reset_all_requested.load(std::memory_order_acquire) ||
                                    slot.reset_requested.load(std::memory_order_acquire);
            if (reset_flag &&
                slot.weights_seq.load(std::memory_order_acquire) <= pending_reset_seq) {
                pending_reset = true;
            }
        }
    }

    if (original || pending_reset) {
        return m_backend->get_original_layer_weights(name);
    }

    const LayerBendingState st = m_bending.get_layer_state(name);
    if (st.isApplied && !st.scratchFinal.empty()) {
        return st.scratchFinal;
    }
    if (!st.drawnWeights.empty()) {
        return st.drawnWeights;
    }
    return m_backend->get_layer_weights(name);
}

auto NetBenderSession::prepare_layer_weight_staging(int layer_idx) -> bool {
    ScopedRtReader reader(m_layer_slots_gate);
    if (!reader.is_acquired() || !m_model_loaded.load(std::memory_order_acquire) || layer_idx < 0 ||
        static_cast<size_t>(layer_idx) >= m_layer_slots.size()) {
        return false;
    }
    const size_t expected = m_layer_weight_counts[static_cast<size_t>(layer_idx)];
    if (expected == 0) {
        return false;
    }
    auto& slot = *m_layer_slots[static_cast<size_t>(layer_idx)];
    std::lock_guard<std::mutex> tb_lock(slot.tb_init_mutex);
    if (!slot.tb_ready.load(std::memory_order_acquire)) {
        for (auto& buf : slot.tb_buf) {
            buf.assign(expected, 0.0f);
        }
        slot.tb_write_idx = 0;
        slot.tb_read_idx = 1;
        slot.tb_middle.store(2, std::memory_order_relaxed);
        slot.tb_ready.store(true, std::memory_order_release);
    }
    return true;
}

auto NetBenderSession::set_layer_weights_immediate(int layer_idx, const float* data, size_t count)
    -> bool {
    if (!m_model_loaded.load(std::memory_order_acquire) || data == nullptr) {
        return false;
    }
    std::string name;
    size_t expected = 0;
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= m_layer_names.size()) {
            return false;
        }
        expected = m_layer_weight_counts[static_cast<size_t>(layer_idx)];
        if (expected == 0 || count != expected) {
            return false;
        }
        name = m_layer_names[static_cast<size_t>(layer_idx)];
    }
    std::vector<float> weights(data, data + count);
    m_bending.set_layer_drawn_weights(name, weights);
    return true;
}

auto NetBenderSession::set_latency_callback(LatencyCallback cb) -> void {
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_latency_callback = std::move(cb);
    }
    notify_latency();
}

auto NetBenderSession::set_error_callback(ErrorCallback cb) -> void {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    m_error_callback = std::move(cb);
}

//==============================================================================
auto NetBenderSession::configure_engine() -> void {
    if (!m_model_loaded.load() || !m_backend || m_model_out <= 0) {
        m_engine.detach_backend();
        notify_latency();
        return;
    }

    StreamingConfig cfg;
    cfg.method = get_method();
    cfg.dispatch = InferenceDispatch::PluginMode;
    cfg.model_in_channels = m_model_in;
    cfg.model_out_channels = m_model_out;
    cfg.n_batches = 1;
    cfg.model_buffer_size = m_buffer_size.load();
    cfg.host_sample_rate = m_host_sample_rate.load();
    cfg.model_sample_rate = m_backend->get_sample_rate();
    cfg.max_host_block_size = std::max(m_config.min_host_block_size, m_max_block_size.load());
    cfg.max_host_channels = std::max(m_config.min_host_channels, m_max_channels.load());
    cfg.resample_mode = m_config.resample_mode;
    cfg.resample_quality = m_config.resample_quality;
    cfg.use_worker_thread = m_config.use_worker_thread;
    cfg.auto_start_worker = m_config.use_worker_thread;
    cfg.enable_dry_delay = m_config.enable_dry_delay;
    cfg.enable_safety_sentry = m_config.enable_safety_sentry;
    cfg.duplicate_mono_input = m_config.duplicate_mono_input;
    cfg.duplicate_mono_output = m_config.duplicate_mono_output;

    m_engine.configure(m_backend.get(), cfg);
    notify_latency();
}

auto NetBenderSession::update_mode_channels(const std::string& method) -> void {
    if (!m_backend) {
        return;
    }
    const auto params = m_backend->get_mode_params(method);
    if (params.size() >= 4) {
        m_model_in = params[0];
        m_model_out = params[2];
    }
}

auto NetBenderSession::notify_latency() -> void {
    LatencyCallback cb;
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        cb = m_latency_callback;
    }
    if (cb) {
        cb(m_engine.is_configured() ? m_engine.get_latency_samples() : 0);
    }
}

auto NetBenderSession::set_error(const std::string& msg) -> void {
    ErrorCallback cb;
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        m_last_error = msg;
        cb = m_error_callback;
    }
    if (cb) {
        cb(msg);
    }
}

//==============================================================================
auto NetBenderSession::to_json(bool include_drawn_weights) const -> nlohmann::json {
    json j;
    j["format"] = k_state_format;
    j["version"] = k_state_version;

    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        j["model"] = {
            {"path", m_model_path},
            {"method", m_method},
            {"buffer_size", m_buffer_size.load()},
            {"use_gpu", m_use_gpu.load()},
        };
        j["active_layer"] = m_active_layer;
    }

    j["mix"] = {
        {"dry_wet", m_dry_wet.load()},
        {"auto_reset_fuse", m_engine.get_safety_sentry().get_auto_reset()},
    };

    j["trigger"] = {
        {"mode", trigger_mode_to_string(m_bending.get_trigger_mode())},
        {"attack", m_bending.get_envelope_attack()},
        {"release", m_bending.get_envelope_release()},
        {"transient_threshold", m_bending.get_transient_threshold()},
    };

    j["latent"] = {
        {"enabled", m_latent.is_enabled()},
        {"mode", latent_mode_to_string(m_latent.get_mode())},
        {"x", m_latent.get_x()},
        {"y", m_latent.get_y()},
        {"depth", m_latent.get_depth()},
        {"slew", m_latent.get_slew()},
    };

    json layers = json::object();
    for (const auto& [name, s] : m_bending.get_all_layer_states()) {
        json layer = {
            {"scale", s.scale},
            {"offset", s.offset},
            {"heat", s.heat},
            {"memory", s.memory},
            {"drift_mode", drift_mode_to_string(s.driftMode)},
            {"frozen", s.frozen},
            {"bridge_source", s.bridgeSourceLayer},
            {"bridge_depth", s.bridgeDepth},
        };
        if (include_drawn_weights && !s.drawnWeights.empty() &&
            weights_differ(s.drawnWeights, s.originalWeights)) {
            layer["drawn_weights"] = s.drawnWeights;
        }
        layers[name] = std::move(layer);
    }
    j["layers"] = std::move(layers);
    return j;
}

auto NetBenderSession::from_json(const nlohmann::json& state, bool restore_model) -> bool {
    if (!state.is_object()) {
        return false;
    }
    try {
        if (state.value("format", std::string{}) != k_state_format) {
            return false;
        }
        if (state.value("version", 0) > k_state_version) {
            set_error("NetBenderSession: state version is newer than this build supports");
            return false;
        }

        // Clear any prior host automation overrides so the loaded preset controls bending/latent
        // cleanly until the host resumes sending automation values.
        m_bending_automation_active.store(false);
        m_latent_automation_active.store(false);

        // 1. Model (loading resets bending / latent / active layer, so it must come first)
        if (state.contains("model") && state["model"].is_object()) {
            const auto& m = state["model"];
            const bool prev_gpu = m_use_gpu.load();
            const bool target_gpu = m.value("use_gpu", prev_gpu);
            const std::string path = m.value("path", std::string{});
            const bool same_loaded_model = m_model_loaded.load() && !path.empty() &&
                                           path == get_model_path() && prev_gpu == target_gpu;

            m_use_gpu.store(target_gpu);
            std::error_code ec;
            if (restore_model && !path.empty() && !same_loaded_model &&
                (m_custom_backend_creator || std::filesystem::is_regular_file(path, ec))) {
                load_model(path);
            } else if (same_loaded_model) {
                m_latent.reset();
                m_engine.reset();
            }
            const std::string method = m.value("method", std::string{});
            if (!method.empty() && method != get_method()) {
                set_method(method);
            }
            const int buffer_size = m.value("buffer_size", 0);
            if (buffer_size > 0 && buffer_size != m_buffer_size.load()) {
                set_buffer_size(buffer_size);
            }
        }

        // 2. Mix & safety
        if (state.contains("mix") && state["mix"].is_object()) {
            const auto& mix = state["mix"];
            set_dry_wet(mix.value("dry_wet", 1.0f));
            m_engine.get_safety_sentry().set_auto_reset(
                mix.value("auto_reset_fuse", m_engine.get_safety_sentry().get_auto_reset()));
        }

        // 3. Trigger / envelope
        if (state.contains("trigger") && state["trigger"].is_object()) {
            const auto& t = state["trigger"];
            m_bending.set_trigger_mode(
                trigger_mode_from_string(t.value("mode", std::string{"continuous"})));
            m_bending.set_envelope_attack(t.value("attack", 0.25f));
            m_bending.set_envelope_release(t.value("release", 0.85f));
            m_bending.set_transient_threshold(t.value("transient_threshold", 0.15f));
        }

        // 4. Latent terrain
        if (state.contains("latent") && state["latent"].is_object()) {
            const auto& l = state["latent"];
            m_latent.set_enabled(l.value("enabled", false));
            m_latent.set_mode(latent_mode_from_string(l.value("mode", std::string{})));
            m_latent.set_coords(l.value("x", 0.0f), l.value("y", 0.0f));
            m_latent.set_depth(l.value("depth", 0.5f));
            m_latent.set_slew(l.value("slew", 0.0f));
        }

        // 5. Active layer & per-layer bending (clear first so omitted layers do not leak)
        clear_all_layer_bending();
        set_active_layer(state.value("active_layer", std::string{}));

        if (state.contains("layers") && state["layers"].is_object()) {
            for (const auto& [name, l] : state["layers"].items()) {
                if (name.empty() || !l.is_object()) {
                    continue;
                }
                LayerBendingState s = m_bending.get_layer_state(name);
                s.scale = l.value("scale", 1.0f);
                s.offset = l.value("offset", 0.0f);
                s.heat = std::max(0.0f, l.value("heat", 0.0f));
                s.memory = std::clamp(l.value("memory", 0.8f), 0.0f, 1.0f);
                s.driftMode = drift_mode_from_string(l.value("drift_mode", std::string{}));
                s.frozen = l.value("frozen", false);
                s.bridgeSourceLayer = l.value("bridge_source", std::string{});
                s.bridgeDepth = std::clamp(l.value("bridge_depth", 0.0f), 0.0f, 1.0f);
                if (l.contains("drawn_weights") && l["drawn_weights"].is_array()) {
                    s.drawnWeights = l["drawn_weights"].get<std::vector<float>>();
                }
                s.isApplied = false;
                s.lastAppliedEnv = -1.0f;
                m_bending.set_layer_state(name, s);
            }
        }
        return true;
    } catch (const std::exception& e) {
        set_error(std::string("NetBenderSession: invalid state: ") + e.what());
    }
    return false;
}

auto NetBenderSession::save_state(bool include_drawn_weights) const -> std::string {
    return to_json(include_drawn_weights).dump();
}

auto NetBenderSession::load_state(const std::string& serialized, bool restore_model) -> bool {
    size_t len = serialized.size();
    while (len > 0 && serialized[len - 1] == '\0') {
        --len;
    }
    const json j = json::parse(serialized.data(), serialized.data() + len, nullptr,
                               /*allow_exceptions=*/false);
    if (j.is_discarded()) {
        set_error("NetBenderSession: state is not valid JSON");
        return false;
    }
    return from_json(j, restore_model);
}

//==============================================================================
auto NetBenderSession::clear_layer_slots() -> void {
    ScopedReconfigureGate gate(m_layer_slots_gate);
    std::lock_guard<std::mutex> lock(m_state_mutex);
    m_layer_count.store(0, std::memory_order_release);
    m_layer_slots.clear();
    m_layer_names.clear();
    m_layer_name_to_idx.clear();
    m_layer_weight_counts.clear();
    m_op_seq.store(0, std::memory_order_relaxed);
    m_any_slot_dirty.store(false, std::memory_order_relaxed);
    m_reset_all_requested.store(false, std::memory_order_relaxed);
    m_reset_all_seq.store(0, std::memory_order_relaxed);
}

auto NetBenderSession::rebuild_layer_slots() -> void {
    ScopedReconfigureGate gate(m_layer_slots_gate);
    std::lock_guard<std::mutex> lock(m_state_mutex);

    m_layer_slots.clear();
    m_layer_names.clear();
    m_layer_name_to_idx.clear();
    m_layer_weight_counts.clear();

    if (m_backend) {
        m_layer_names = m_backend->get_available_layers();
        m_layer_name_to_idx.reserve(m_layer_names.size());
        m_layer_weight_counts.resize(m_layer_names.size(), 0);
        m_layer_slots.reserve(m_layer_names.size());
        for (size_t i = 0; i < m_layer_names.size(); ++i) {
            m_layer_name_to_idx[m_layer_names[i]] = static_cast<int>(i);
            m_layer_weight_counts[i] = compute_layer_weight_count(*m_backend, m_layer_names[i]);
            m_layer_slots.push_back(std::make_unique<AtomicLayerSlot>());
        }
    }

    m_op_seq.store(0, std::memory_order_relaxed);
    m_any_slot_dirty.store(false, std::memory_order_relaxed);
    m_reset_all_requested.store(false, std::memory_order_relaxed);
    m_reset_all_seq.store(0, std::memory_order_relaxed);
    m_layer_count.store(static_cast<int>(m_layer_names.size()), std::memory_order_release);
}

auto NetBenderSession::drain_pending_rt_bending() -> void {
    ScopedRtReader reader(m_layer_slots_gate);
    if (!reader.is_acquired() || !m_any_slot_dirty.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    uint64_t global_reset_seq = m_reset_all_seq.load(std::memory_order_acquire);
    if (m_reset_all_requested.exchange(false, std::memory_order_acq_rel)) {
        global_reset_seq =
            std::max(global_reset_seq, m_reset_all_seq.load(std::memory_order_acquire));
        m_bending.clear_all_layer_bending();
    }

    for (size_t i = 0; i < m_layer_slots.size(); ++i) {
        auto& slot = *m_layer_slots[i];
        const std::string& name = m_layer_names[i];

        uint64_t eff_reset_seq =
            std::max(global_reset_seq, slot.reset_seq.load(std::memory_order_acquire));
        if (slot.reset_requested.exchange(false, std::memory_order_acq_rel)) {
            const uint64_t layer_rst_seq = slot.reset_seq.load(std::memory_order_acquire);
            eff_reset_seq = std::max(eff_reset_seq, layer_rst_seq);
            if (layer_rst_seq > global_reset_seq) {
                m_bending.clear_layer_bending(name);
            }
        }

        const bool so_dirty = slot.scale_offset_dirty.exchange(false, std::memory_order_acq_rel);
        const bool h_dirty = slot.heat_dirty.exchange(false, std::memory_order_acq_rel);
        const bool m_dirty = slot.memory_dirty.exchange(false, std::memory_order_acq_rel);
        const bool d_dirty = slot.drift_dirty.exchange(false, std::memory_order_acq_rel);
        const bool f_dirty = slot.frozen_dirty.exchange(false, std::memory_order_acq_rel);

        if (so_dirty || h_dirty || m_dirty || d_dirty || f_dirty) {
            const LayerBendingState cur = m_bending.get_layer_state(name);
            if (so_dirty && slot.bend_seq.load(std::memory_order_acquire) > eff_reset_seq) {
                const float scale = slot.scale.load(std::memory_order_relaxed);
                const float offset = slot.offset.load(std::memory_order_relaxed);
                if (std::abs(cur.scale - scale) > 1e-6f) {
                    m_bending.set_layer_scale(name, scale);
                }
                if (std::abs(cur.offset - offset) > 1e-6f) {
                    m_bending.set_layer_offset(name, offset);
                }
            }
            if (h_dirty && slot.heat_seq.load(std::memory_order_acquire) > eff_reset_seq) {
                const float heat = slot.heat.load(std::memory_order_relaxed);
                if (std::abs(cur.heat - heat) > 1e-6f) {
                    m_bending.set_layer_heat(name, heat);
                }
            }
            if (m_dirty && slot.memory_seq.load(std::memory_order_acquire) > eff_reset_seq) {
                const float memory = slot.memory.load(std::memory_order_relaxed);
                if (std::abs(cur.memory - memory) > 1e-6f) {
                    m_bending.set_layer_memory(name, memory);
                }
            }
            if (d_dirty && slot.drift_seq.load(std::memory_order_acquire) > eff_reset_seq) {
                const auto drift_mode =
                    static_cast<DriftMode>(slot.drift_mode.load(std::memory_order_relaxed));
                if (cur.driftMode != drift_mode) {
                    m_bending.set_layer_drift_mode(name, drift_mode);
                }
            }
            if (f_dirty && slot.frozen_seq.load(std::memory_order_acquire) > eff_reset_seq) {
                const bool frozen = (slot.frozen.load(std::memory_order_relaxed) != 0);
                if (cur.frozen != frozen) {
                    m_bending.set_layer_frozen(name, frozen);
                }
            }
        }

        if (slot.bridge_dirty.exchange(false, std::memory_order_acq_rel)) {
            if (slot.bridge_seq.load(std::memory_order_acquire) > eff_reset_seq) {
                const int src_idx = slot.bridge_source_idx.load(std::memory_order_relaxed);
                const std::string src_name =
                    (src_idx >= 0 && static_cast<size_t>(src_idx) < m_layer_names.size())
                        ? m_layer_names[static_cast<size_t>(src_idx)]
                        : std::string{};
                const float depth = slot.bridge_depth.load(std::memory_order_relaxed);

                const LayerBendingState cur = m_bending.get_layer_state(name);
                if (cur.bridgeSourceLayer != src_name) {
                    m_bending.set_layer_bridge_source(name, src_name);
                }
                if (std::abs(cur.bridgeDepth - depth) > 1e-6f) {
                    m_bending.set_layer_bridge_depth(name, depth);
                }
            }
        }

        if (slot.tb_ready.load(std::memory_order_acquire)) {
            if ((slot.tb_middle.load(std::memory_order_acquire) & 4) != 0) {
                const int prev =
                    slot.tb_middle.exchange(slot.tb_read_idx, std::memory_order_acq_rel);
                slot.tb_read_idx = prev & 3;
                if ((prev & 4) != 0 &&
                    slot.weights_seq.load(std::memory_order_acquire) > eff_reset_seq) {
                    m_bending.set_layer_drawn_weights(
                        name, slot.tb_buf[static_cast<size_t>(slot.tb_read_idx)]);
                }
            }
        }
    }
}
