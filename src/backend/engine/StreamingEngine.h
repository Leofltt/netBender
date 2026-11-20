#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "IBackend.h"
#include "dsp/AudioRingBuffer.h"
#include "dsp/Resampler.h"
#include "dsp/ResamplingRingBuffer.h"
#include "dsp/RtGate.h"
#include "dsp/SafetySentry.h"
#include "dsp/WorkerThread.h"
#include "dsp/rtsan_utils.h"

enum class InferenceDispatch {
    PluginMode = 0,  // Routes "forward", "decode gen", "prior"/"generate" to high-level methods
    RawMethod = 1,   // Always calls IBackend::perform(..., method, ...)
};

struct StreamingConfig {
    std::string method{"forward"};
    InferenceDispatch dispatch{InferenceDispatch::PluginMode};

    int model_in_channels{1};   // Per batch; 0 for generative / prior-only modes
    int model_out_channels{1};  // Per batch
    int n_batches{1};
    int model_buffer_size{2048};

    double host_sample_rate{44100.0};
    double model_sample_rate{44100.0};

    int max_host_block_size{2048};
    int max_host_channels{2};

    ResampleMode resample_mode{ResampleMode::Auto};
    ResamplerQuality resample_quality{ResamplerQuality::CatmullRom};

    bool use_worker_thread{true};
    bool auto_start_worker{true};
    bool enable_dry_delay{false};
    bool enable_safety_sentry{false};
    bool duplicate_mono_input{false};
    bool duplicate_mono_output{false};
};

/**
 * Frontend-agnostic neural audio streaming orchestrator.
 *
 * Bridges variable-rate / variable-size host callbacks (float* or double*) and fixed-rate /
 * fixed-size neural model inference using wait-free SPSC rings, pull-side fractional resampling,
 * lock-free RT reconfiguration gating, semaphore-driven background execution, and SafetySentry
 * output protection.
 */
class StreamingEngine {
public:
    using PreInferenceCallback = std::function<void(IBackend&)>;
    using LatentHookProvider = std::function<IBackend::LatentHook()>;
    using ErrorCallback = std::function<void(const std::string&)>;

    StreamingEngine();
    ~StreamingEngine();

    StreamingEngine(const StreamingEngine&) = delete;
    auto operator=(const StreamingEngine&) -> StreamingEngine& = delete;
    StreamingEngine(StreamingEngine&&) = delete;
    auto operator=(StreamingEngine&&) -> StreamingEngine& = delete;

    // Lifecycle & reconfiguration (non-RT thread safe; acquires ScopedReconfigureGate)
    auto configure(IBackend* backend, const StreamingConfig& config) -> void;
    auto detach_backend() -> void;
    auto start() -> void;
    auto stop() -> void;
    auto reset() -> void;

    auto set_host_sample_rate(double host_sr) -> void;
    auto set_model_sample_rate(double model_sr) -> void;
    auto set_resample_mode(ResampleMode mode) -> void;
    auto set_buffer_size(int model_buffer_size) -> void;
    auto set_method(const std::string& method) -> void;
    auto request_clear_decode_gen_fifo() -> void;

    // Hooks executed around inference on the worker/sync thread (never locked on audio thread)
    auto set_pre_inference_callback(PreInferenceCallback cb) -> void;
    auto set_latent_hook_provider(LatentHookProvider provider) -> void;
    auto set_error_callback(ErrorCallback cb) -> void;

    // Executed from the audio thread via SafetySentry::tick_recovery (must be non-blocking)
    auto set_fuse_reset_callback(std::function<void()> cb) -> void;

    // Real-time audio thread processing (float* and double* host channel arrays)
    auto process(const float* const* host_in, int num_host_in_channels, float* const* host_out,
                 int num_host_out_channels, int num_frames) NETBENDER_NONBLOCKING -> void;

    auto process(const double* const* host_in, int num_host_in_channels, double* const* host_out,
                 int num_host_out_channels, int num_frames) NETBENDER_NONBLOCKING -> void;

    // Controls & introspection
    auto set_dry_wet(float dry_wet) noexcept NETBENDER_NONBLOCKING -> void;
    [[nodiscard]] auto get_dry_wet() const noexcept NETBENDER_NONBLOCKING -> float;

    [[nodiscard]] auto is_configured() const noexcept NETBENDER_NONBLOCKING -> bool {
        return m_configured.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto is_resampling_active() const noexcept NETBENDER_NONBLOCKING -> bool {
        return m_resampling_active.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto is_worker_running() const NETBENDER_NONBLOCKING -> bool {
        return m_worker.is_running();
    }
    [[nodiscard]] auto get_latency_samples() const noexcept NETBENDER_NONBLOCKING -> int {
        return m_host_latency_samples.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_model_latency_samples() const noexcept NETBENDER_NONBLOCKING -> int {
        return m_model_latency_samples.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_host_sample_rate() const noexcept NETBENDER_NONBLOCKING -> double {
        return m_host_sample_rate.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_model_sample_rate() const noexcept NETBENDER_NONBLOCKING -> double {
        return m_model_sample_rate.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_model_buffer_size() const noexcept NETBENDER_NONBLOCKING -> int {
        return m_model_buffer_size.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_config() const -> StreamingConfig;

    auto get_safety_sentry() noexcept -> SafetySentry& { return m_safety_sentry; }
    [[nodiscard]] auto get_safety_sentry() const noexcept -> const SafetySentry& {
        return m_safety_sentry;
    }

    [[nodiscard]] auto get_inference_count() const noexcept NETBENDER_NONBLOCKING -> uint64_t {
        return m_inference_count.load(std::memory_order_relaxed);
    }
    [[nodiscard]] auto get_in_overrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
        return m_in_bank.get_overrun_count();
    }
    [[nodiscard]] auto get_out_underrun_count() const NETBENDER_NONBLOCKING -> uint64_t {
        return m_out_bank.get_underrun_count();
    }

private:
    auto configure_locked() -> void;
    auto clear_and_prefill_locked() -> void;

    [[nodiscard]] auto has_work_and_space() const NETBENDER_NONBLOCKING -> bool;
    auto run_worker_loop() -> void;
    auto run_inference_step() -> void;

    template <typename SampleT>
    auto process_impl(const SampleT* const* host_in, int num_host_in_channels,
                      SampleT* const* host_out, int num_host_out_channels,
                      int num_frames) NETBENDER_NONBLOCKING -> void;

    template <typename SampleT>
    auto apply_dry_wet_and_sentry(SampleT* const* host_out, int num_host_out_channels,
                                  int num_frames) NETBENDER_NONBLOCKING -> void;

    IBackend* m_backend{nullptr};
    StreamingConfig m_config;

    ResamplingRingBufferBank m_in_bank;
    ResamplingRingBufferBank m_out_bank;
    std::vector<AudioRingBuffer> m_dry_delay_buffers;

    std::vector<std::vector<float>> m_worker_in;
    std::vector<std::vector<float>> m_worker_out;
    std::vector<float*> m_worker_in_ptrs;
    std::vector<float*> m_worker_out_ptrs;

    // Pre-allocated RT audio-thread scratch buffers (allocated in configure_locked())
    std::vector<float> m_dry_scratch;
    std::vector<float> m_zero_scratch;
    std::vector<float> m_wet_float_scratch;

    SafetySentry m_safety_sentry;
    WorkerThread m_worker;
    RtGate m_rt_gate;

    mutable std::mutex m_reconfig_mutex;
    mutable std::mutex m_callback_mutex;

    PreInferenceCallback m_pre_inference_cb;
    LatentHookProvider m_latent_hook_provider;
    ErrorCallback m_error_cb;
    std::function<void()> m_fuse_reset_cb;

    std::atomic<bool> m_configured{false};
    std::atomic<bool> m_stopping{false};
    std::atomic<bool> m_resampling_active{false};
    std::atomic<bool> m_pending_clear_fifo{false};
    std::atomic<int> m_model_in_streams{0};
    std::atomic<int> m_model_out_streams{0};
    std::atomic<int> m_model_buffer_size{2048};
    std::atomic<int> m_model_latency_samples{0};
    std::atomic<int> m_host_latency_samples{0};
    std::atomic<double> m_host_sample_rate{44100.0};
    std::atomic<double> m_model_sample_rate{44100.0};
    std::atomic<float> m_dry_wet{1.0f};
    std::atomic<uint64_t> m_inference_count{0};
};
