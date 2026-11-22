#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json_fwd.hpp>
#include "IBackend.h"
#include "bending/BendingManager.h"
#include "bending/LatentTerrainEngine.h"
#include "engine/StreamingEngine.h"
#include "dsp/rtsan_utils.h"
#include "dsp/RtGate.h"

/**
 * Host-agnostic session configuration (how the session drives its StreamingEngine).
 */
struct SessionConfig {
    bool use_worker_thread{true};
    bool enable_dry_delay{true};
    bool enable_safety_sentry{true};
    bool duplicate_mono_input{true};
    bool duplicate_mono_output{true};
    ResampleMode resample_mode{ResampleMode::Auto};
    ResamplerQuality resample_quality{ResamplerQuality::CatmullRom};
    int min_host_channels{2};
    int min_host_block_size{256};
};

/**
 * High-level, frontend-agnostic NetBender facade.
 *
 * Composes one IBackend (model runtime), a StreamingEngine (wait-free host <-> model streaming,
 * resampling, dry/wet, SafetySentry), a BendingManager (real-time weight bending) and a
 * LatentTerrainEngine (latent hook) into a single object that any host (JUCE, CLAP, CLI, ...)
 * can drive with a handful of calls:
 *
 *   NetBenderSession session;
 *   session.prepare(48000.0, 512, 2);
 *   session.load_model("models/darbouka_xnnpack.pte");
 *   session.bending().set_layer_scale("decoder.net.0.weight", 1.5f);
 *   // audio thread:
 *   session.process(in, 2, out, 2, 512);
 *   // persistence:
 *   std::string preset = session.save_state();
 *   session.load_state(preset);
 *
 * Threading: lifecycle / model / state methods are message-thread (non-RT) calls. Methods marked
 * NETBENDER_NONBLOCKING are safe to call from the real-time audio thread.
 */
class NetBenderSession {
public:
    using BackendCreator = std::function<std::unique_ptr<IBackend>(const std::string& path)>;
    using LatencyCallback = std::function<void(int latency_samples)>;
    using ErrorCallback = StreamingEngine::ErrorCallback;

    static constexpr int k_state_version = 1;
    static constexpr const char* k_state_format = "netbender.session";

    explicit NetBenderSession(SessionConfig config = {});
    ~NetBenderSession();

    NetBenderSession(const NetBenderSession&) = delete;
    auto operator=(const NetBenderSession&) -> NetBenderSession& = delete;
    NetBenderSession(NetBenderSession&&) = delete;
    auto operator=(NetBenderSession&&) -> NetBenderSession& = delete;

    // ---------------------------------------------------------------------------------------
    // Host lifecycle (non-RT)
    auto prepare(double host_sample_rate, int max_block_size, int max_channels) -> void;
    auto release() -> void;

    // ---------------------------------------------------------------------------------------
    // Model management (non-RT)
    auto set_backend_creator(BackendCreator creator) -> void;  // defaults to BackendFactory
    auto load_model(const std::string& path) -> bool;
    [[nodiscard]] auto is_model_loaded() const -> bool { return m_model_loaded.load(); }
    [[nodiscard]] auto get_model_path() const -> std::string;
    [[nodiscard]] auto get_last_error() const -> std::string;

    // Always valid (a default backend exists before any model is loaded)
    auto backend() -> IBackend& { return *m_backend; }
    [[nodiscard]] auto has_backend() const -> bool { return m_backend != nullptr; }

    [[nodiscard]] auto get_available_modes() const -> std::vector<std::string>;
    auto set_method(const std::string& method) -> bool;
    [[nodiscard]] auto get_method() const -> std::string;
    auto set_buffer_size(int size) -> bool;  // power of two only
    [[nodiscard]] auto get_buffer_size() const -> int { return m_buffer_size.load(); }
    auto set_use_gpu(bool enabled) -> void;
    [[nodiscard]] auto is_using_gpu() const -> bool { return m_use_gpu.load(); }

    [[nodiscard]] auto get_latency_samples() const -> int { return m_engine.get_latency_samples(); }
    [[nodiscard]] auto get_model_sample_rate() const -> double {
        return m_engine.get_model_sample_rate();
    }
    [[nodiscard]] auto is_resampling_active() const -> bool {
        return m_engine.is_resampling_active();
    }

    // Bendable layer introspection (populated on load_model())
    [[nodiscard]] auto get_layer_count() const noexcept NETBENDER_NONBLOCKING -> int {
        return m_layer_count.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_layer_names() const -> std::vector<std::string>;
    [[nodiscard]] auto find_layer_index(const std::string& name) const -> int;
    [[nodiscard]] auto get_layer_name(int index) const -> std::string;
    [[nodiscard]] auto get_layer_weight_count(int index) const -> size_t;

    // ---------------------------------------------------------------------------------------
    // Real-time audio thread
    auto process(const float* const* host_in, int num_in_channels, float* const* host_out,
                 int num_out_channels, int num_frames) NETBENDER_NONBLOCKING -> void;
    auto process(const double* const* host_in, int num_in_channels, double* const* host_out,
                 int num_out_channels, int num_frames) NETBENDER_NONBLOCKING -> void;

    auto set_dry_wet(float dry_wet) noexcept NETBENDER_NONBLOCKING -> void;
    [[nodiscard]] auto get_dry_wet() const noexcept NETBENDER_NONBLOCKING -> float {
        return m_dry_wet.load();
    }

    // Host automation for the *active layer* (e.g. DAW parameters). Once called, these values
    // become authoritative for the active layer on every inference block. memory < 0 keeps the
    // layer's stored memory.
    auto set_bending_automation(float scale, float offset, float heat, float memory,
                                bool short_active) noexcept NETBENDER_NONBLOCKING -> void;
    // Wait-free per-layer real-time bending (safe for concurrent multi-layer modulation from audio
    // thread, e.g. Csound k-rate opcodes, multi-layer 3D waveplot UIs, or DAW automation).
    // Negative sentinel values for heat / memory / drift_mode / frozen preserve the layer's current
    // stored setting so callers can modulate a subset of parameters (e.g. scale & offset only).
    auto set_layer_bend_rt(int layer_idx, float scale, float offset, float heat = -1.0f,
                           float memory = -1.0f, int drift_mode = -1,
                           int frozen = -1) noexcept NETBENDER_NONBLOCKING -> void;
    auto set_layer_bridge_rt(int target_idx, int source_idx,
                             float depth) noexcept NETBENDER_NONBLOCKING -> void;
    auto set_layer_weights_rt(int layer_idx, const float* data,
                              size_t count) noexcept NETBENDER_NONBLOCKING -> bool;
    auto request_reset_layer_weights(int layer_idx) noexcept NETBENDER_NONBLOCKING -> void;
    auto request_reset_all_weights() noexcept NETBENDER_NONBLOCKING -> void;
    auto request_fuse_reset() noexcept NETBENDER_NONBLOCKING -> void;

    // Host automation for the latent terrain hook (overrides LatentTerrainEngine coords/depth)
    auto set_latent_automation(float x, float y, float depth) noexcept NETBENDER_NONBLOCKING
        -> void;
    auto set_decode_gen_smoothness(float smoothness) noexcept NETBENDER_NONBLOCKING -> void;
    [[nodiscard]] auto get_decode_gen_smoothness() const noexcept NETBENDER_NONBLOCKING -> float;
    auto trigger_short_circuit(bool active) noexcept NETBENDER_NONBLOCKING -> void {
        m_bending.trigger_short_circuit(active);
    }

    // ---------------------------------------------------------------------------------------
    // Bending convenience (non-RT)
    auto set_active_layer(const std::string& name) -> void;
    [[nodiscard]] auto get_active_layer() const -> std::string;
    auto clear_layer_bending(const std::string& name) -> void;
    auto clear_all_layer_bending() -> void;
    auto reset_fuse() -> void;
    auto set_latent_hook(IBackend::LatentHook hook) -> void;  // user hook overrides terrain
    [[nodiscard]] auto get_layer_weights(int layer_idx, bool original = false) const
        -> std::vector<float>;
    auto prepare_layer_weight_staging(int layer_idx) -> bool;
    auto set_layer_weights_immediate(int layer_idx, const float* data, size_t count) -> bool;

    // ---------------------------------------------------------------------------------------
    // Subsystems
    auto engine() -> StreamingEngine& { return m_engine; }
    auto bending() -> BendingManager& { return m_bending; }
    [[nodiscard]] auto bending() const -> const BendingManager& { return m_bending; }
    auto latent() -> LatentTerrainEngine& { return m_latent; }
    [[nodiscard]] auto latent() const -> const LatentTerrainEngine& { return m_latent; }
    auto safety_sentry() -> SafetySentry& { return m_engine.get_safety_sentry(); }
    [[nodiscard]] auto safety_sentry() const -> const SafetySentry& {
        return m_engine.get_safety_sentry();
    }

    auto set_latency_callback(LatencyCallback cb) -> void;
    auto set_error_callback(ErrorCallback cb) -> void;

    // ---------------------------------------------------------------------------------------
    // State / preset persistence (JSON, non-RT)
    [[nodiscard]] auto to_json(bool include_drawn_weights = true) const -> nlohmann::json;
    auto from_json(const nlohmann::json& state, bool restore_model = true) -> bool;
    [[nodiscard]] auto save_state(bool include_drawn_weights = true) const -> std::string;
    auto load_state(const std::string& serialized, bool restore_model = true) -> bool;

private:
    struct AtomicLayerSlot {
        std::atomic<float> scale{1.0f};
        std::atomic<float> offset{0.0f};
        std::atomic<bool> scale_offset_dirty{false};
        std::atomic<uint64_t> bend_seq{0};

        std::atomic<float> heat{0.0f};
        std::atomic<bool> heat_dirty{false};
        std::atomic<uint64_t> heat_seq{0};

        std::atomic<float> memory{0.8f};
        std::atomic<bool> memory_dirty{false};
        std::atomic<uint64_t> memory_seq{0};

        std::atomic<int> drift_mode{1};
        std::atomic<bool> drift_dirty{false};
        std::atomic<uint64_t> drift_seq{0};

        std::atomic<int> frozen{0};
        std::atomic<bool> frozen_dirty{false};
        std::atomic<uint64_t> frozen_seq{0};

        std::atomic<int> bridge_source_idx{-1};
        std::atomic<float> bridge_depth{0.0f};
        std::atomic<bool> bridge_dirty{false};
        std::atomic<uint64_t> bridge_seq{0};

        std::atomic<bool> reset_requested{false};
        std::atomic<uint64_t> reset_seq{0};

        // SPSC wait-free triple buffer for RT raw weight injection
        std::mutex tb_init_mutex;
        std::atomic<bool> tb_ready{false};
        std::vector<float> tb_buf[3];
        int tb_write_idx{0};
        int tb_read_idx{1};
        std::atomic<int> tb_middle{2};  // bits 0..1 = buffer index (0..2), bit 2 (4) = dirty flag
        std::atomic<uint64_t> weights_seq{0};
    };

    auto configure_engine() -> void;
    auto update_mode_channels(const std::string& method) -> void;
    auto clear_layer_slots() -> void;
    auto rebuild_layer_slots() -> void;
    auto drain_pending_rt_bending() -> void;
    auto notify_latency() -> void;
    auto set_error(const std::string& msg) -> void;

    template <typename SampleT>
    auto process_impl(const SampleT* const* host_in, int num_in_channels, SampleT* const* host_out,
                      int num_out_channels, int num_frames) NETBENDER_NONBLOCKING -> void;

    SessionConfig m_config;
    BackendCreator m_backend_creator;
    bool m_custom_backend_creator{false};

    std::unique_ptr<IBackend> m_backend;
    std::atomic<bool> m_model_loaded{false};
    std::atomic<bool> m_use_gpu{false};
    std::atomic<int> m_buffer_size{2048};
    int m_model_in{0};
    int m_model_out{0};

    std::atomic<double> m_host_sample_rate{44100.0};
    std::atomic<int> m_max_block_size{2048};
    std::atomic<int> m_max_channels{2};
    std::atomic<float> m_dry_wet{1.0f};

    // Active-layer host automation (written on audio thread, read on inference thread)
    std::atomic<bool> m_bending_automation_active{false};
    std::atomic<float> m_auto_scale{1.0f};
    std::atomic<float> m_auto_offset{0.0f};
    std::atomic<float> m_auto_heat{0.0f};
    std::atomic<float> m_auto_memory{-1.0f};
    std::atomic<bool> m_auto_short{false};

    std::atomic<bool> m_latent_automation_active{false};
    std::atomic<float> m_auto_latent_x{0.0f};
    std::atomic<float> m_auto_latent_y{0.0f};
    std::atomic<float> m_auto_latent_depth{0.5f};
    std::atomic<float> m_decode_gen_smoothness{0.30f};

    mutable std::mutex m_state_mutex;  // guards strings / hooks / callbacks below
    std::string m_model_path;
    std::string m_method;
    std::string m_active_layer;
    std::string m_last_error;
    IBackend::LatentHook m_user_latent_hook{nullptr};
    LatencyCallback m_latency_callback{nullptr};
    ErrorCallback m_error_callback{nullptr};

    // Wait-free per-layer RT bending staging (must outlive m_engine)
    mutable RtGate m_layer_slots_gate;
    std::vector<std::string> m_layer_names;
    std::unordered_map<std::string, int> m_layer_name_to_idx;
    std::vector<size_t> m_layer_weight_counts;
    std::vector<std::unique_ptr<AtomicLayerSlot>> m_layer_slots;
    std::atomic<int> m_layer_count{0};
    std::atomic<uint64_t> m_op_seq{0};
    std::atomic<bool> m_any_slot_dirty{false};
    std::atomic<bool> m_reset_all_requested{false};
    std::atomic<uint64_t> m_reset_all_seq{0};

    BendingManager m_bending;
    LatentTerrainEngine m_latent;

    // Declared last so it is destroyed first (worker stops before backend / bending / slots go
    // away)
    StreamingEngine m_engine;
};
