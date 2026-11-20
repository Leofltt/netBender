#include "StreamingEngine.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <type_traits>
#include <utility>

StreamingEngine::StreamingEngine() : m_worker([this] { run_worker_loop(); }) {}

StreamingEngine::~StreamingEngine() {
    detach_backend();
}

auto StreamingEngine::configure(IBackend* backend, const StreamingConfig& config) -> void {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    const ScopedReconfigureGate rt_gate(m_rt_gate);

    m_stopping.store(true, std::memory_order_release);
    m_worker.stop();

    m_backend = backend;
    m_config = config;
    {
        std::lock_guard<std::mutex> cb_lock(m_callback_mutex);
        if (m_config.method.empty()) {
            m_config.method = "forward";
        }
    }

    configure_locked();
}

auto StreamingEngine::configure_locked() -> void {
    m_config.model_in_channels = std::max(0, m_config.model_in_channels);
    m_config.model_out_channels = std::max(0, m_config.model_out_channels);
    m_config.n_batches = std::max(1, m_config.n_batches);
    m_config.model_buffer_size = std::max(1, m_config.model_buffer_size);
    m_config.host_sample_rate =
        (m_config.host_sample_rate > 0.0) ? m_config.host_sample_rate : 44100.0;
    m_config.model_sample_rate =
        (m_config.model_sample_rate > 0.0) ? m_config.model_sample_rate : m_config.host_sample_rate;
    m_config.max_host_block_size = std::max(256, m_config.max_host_block_size);
    m_config.max_host_channels = std::max(1, m_config.max_host_channels);

    const int total_in = m_config.model_in_channels * m_config.n_batches;
    const int total_out = m_config.model_out_channels * m_config.n_batches;
    const int model_buf = m_config.model_buffer_size;

    const bool ready = (m_backend != nullptr) && (total_in > 0 || total_out > 0);
    if (!ready) {
        m_configured.store(false, std::memory_order_release);
        m_resampling_active.store(false, std::memory_order_release);
        m_model_in_streams.store(0, std::memory_order_release);
        m_model_out_streams.store(0, std::memory_order_release);
        m_model_buffer_size.store(model_buf, std::memory_order_release);
        m_model_latency_samples.store(0, std::memory_order_release);
        m_host_latency_samples.store(0, std::memory_order_release);
        m_host_sample_rate.store(m_config.host_sample_rate, std::memory_order_release);
        m_model_sample_rate.store(m_config.model_sample_rate, std::memory_order_release);
        return;
    }

    const double host_sr = m_config.host_sample_rate;
    const double model_sr = m_config.model_sample_rate;
    bool resampling_active = false;
    switch (m_config.resample_mode) {
        case ResampleMode::Auto:
            resampling_active = std::abs(host_sr - model_sr) > 1e-3;
            break;
        case ResampleMode::ForceEnabled:
            resampling_active = true;
            break;
        case ResampleMode::Disabled:
            resampling_active = false;
            break;
    }

    const double in_ratio = resampling_active ? (host_sr / model_sr) : 1.0;
    const double safe_min_ratio = std::max(0.05, std::min(1.0, in_ratio));

    const int model_latency =
        (total_in > 0) ? (m_config.use_worker_thread ? (model_buf * 2) : model_buf) : 0;
    const int host_latency =
        (total_in > 0) ? (resampling_active ? static_cast<int>(std::round(model_latency * in_ratio))
                                            : model_latency)
                       : 0;

    const int in_cap = std::max({
        16384,
        static_cast<int>(std::ceil(model_buf * std::max(1.0, in_ratio))) * 4,
        m_config.max_host_block_size * 4,
    });
    const int out_cap = std::max({
        8192,
        model_latency + model_buf * 4,
        static_cast<int>(
            std::ceil(static_cast<double>(m_config.max_host_block_size) / safe_min_ratio)) *
            4,
    });
    const int dry_cap = std::max({
        16384,
        host_latency * 2,
        host_latency + m_config.max_host_block_size * 4,
    });

    const int in_scratch = std::max(4096, static_cast<int>(std::ceil(model_buf * in_ratio)) + 64);
    const int out_scratch = std::max(
        4096, static_cast<int>(std::ceil(m_config.max_host_block_size / safe_min_ratio)) + 64);

    m_in_bank.init(total_in, in_cap, host_sr, model_sr, m_config.resample_mode,
                   m_config.resample_quality, in_scratch);
    m_out_bank.init(total_out, out_cap, model_sr, host_sr, m_config.resample_mode,
                    m_config.resample_quality, out_scratch);

    if (m_config.enable_dry_delay) {
        const int n_dry = std::max({2, m_config.max_host_channels, total_out});
        m_dry_delay_buffers.resize(static_cast<size_t>(n_dry));
        for (auto& dry_ring : m_dry_delay_buffers) {
            dry_ring.init_with_capacity(dry_cap);
        }
    } else {
        m_dry_delay_buffers.clear();
    }

    m_worker_in.resize(static_cast<size_t>(total_in));
    m_worker_in_ptrs.resize(static_cast<size_t>(total_in));
    for (int c = 0; c < total_in; ++c) {
        m_worker_in[static_cast<size_t>(c)].assign(static_cast<size_t>(model_buf), 0.0f);
        m_worker_in_ptrs[static_cast<size_t>(c)] = m_worker_in[static_cast<size_t>(c)].data();
    }

    m_worker_out.resize(static_cast<size_t>(total_out));
    m_worker_out_ptrs.resize(static_cast<size_t>(total_out));
    for (int c = 0; c < total_out; ++c) {
        m_worker_out[static_cast<size_t>(c)].assign(static_cast<size_t>(model_buf), 0.0f);
        m_worker_out_ptrs[static_cast<size_t>(c)] = m_worker_out[static_cast<size_t>(c)].data();
    }

    const int rt_scratch_size = std::max(1024, m_config.max_host_block_size);
    m_dry_scratch.assign(static_cast<size_t>(rt_scratch_size), 0.0f);
    m_zero_scratch.assign(static_cast<size_t>(rt_scratch_size), 0.0f);
    m_wet_float_scratch.assign(static_cast<size_t>(rt_scratch_size), 0.0f);

    m_model_in_streams.store(total_in, std::memory_order_release);
    m_model_out_streams.store(total_out, std::memory_order_release);
    m_model_buffer_size.store(model_buf, std::memory_order_release);
    m_model_latency_samples.store(model_latency, std::memory_order_release);
    m_host_latency_samples.store(host_latency, std::memory_order_release);
    m_host_sample_rate.store(host_sr, std::memory_order_release);
    m_model_sample_rate.store(model_sr, std::memory_order_release);
    m_resampling_active.store(resampling_active, std::memory_order_release);
    m_pending_clear_fifo.store(false, std::memory_order_release);

    clear_and_prefill_locked();

    m_configured.store(true, std::memory_order_release);
    m_stopping.store(false, std::memory_order_release);

    if (m_config.use_worker_thread && m_config.auto_start_worker) {
        m_worker.start();
    }
}

auto StreamingEngine::clear_and_prefill_locked() -> void {
    m_in_bank.clear();
    m_out_bank.clear();

    for (auto& ch : m_worker_in) {
        std::fill(ch.begin(), ch.end(), 0.0f);
    }
    for (auto& ch : m_worker_out) {
        std::fill(ch.begin(), ch.end(), 0.0f);
    }

    const int model_latency = m_model_latency_samples.load(std::memory_order_acquire);
    if (model_latency > 0 && m_out_bank.get_num_channels() > 0) {
        m_out_bank.push_zeros_all(model_latency);
    }

    const int host_latency = m_host_latency_samples.load(std::memory_order_acquire);
    for (auto& dry_ring : m_dry_delay_buffers) {
        dry_ring.clear();
        if (host_latency > 0) {
            (void)dry_ring.put_zeros(host_latency);
        }
    }

    if (m_backend != nullptr) {
        m_backend->clear_decode_gen_fifo();
    }
}

auto StreamingEngine::detach_backend() -> void {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    const ScopedReconfigureGate rt_gate(m_rt_gate);

    m_stopping.store(true, std::memory_order_release);
    m_configured.store(false, std::memory_order_release);
    m_worker.stop();

    m_backend = nullptr;
    m_resampling_active.store(false, std::memory_order_release);
    m_model_in_streams.store(0, std::memory_order_release);
    m_model_out_streams.store(0, std::memory_order_release);
    m_model_latency_samples.store(0, std::memory_order_release);
    m_host_latency_samples.store(0, std::memory_order_release);
}

auto StreamingEngine::start() -> void {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    const ScopedReconfigureGate rt_gate(m_rt_gate);

    if (m_configured.load(std::memory_order_acquire) && m_config.use_worker_thread) {
        m_stopping.store(false, std::memory_order_release);
        m_worker.start();
    }
}

auto StreamingEngine::stop() -> void {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    const ScopedReconfigureGate rt_gate(m_rt_gate);

    m_stopping.store(true, std::memory_order_release);
    m_worker.stop();
}

auto StreamingEngine::reset() -> void {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    const ScopedReconfigureGate rt_gate(m_rt_gate);

    const bool should_restart =
        m_worker.is_running() || (m_configured.load(std::memory_order_acquire) &&
                                  m_config.use_worker_thread && m_config.auto_start_worker);

    m_stopping.store(true, std::memory_order_release);
    m_worker.stop();

    m_safety_sentry.reset_fuse();
    clear_and_prefill_locked();

    m_stopping.store(false, std::memory_order_release);
    if (should_restart && m_configured.load(std::memory_order_acquire)) {
        m_worker.start();
    }
}

auto StreamingEngine::set_host_sample_rate(double host_sr) -> void {
    if (host_sr <= 0.0) {
        return;
    }

    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    if (m_configured.load(std::memory_order_acquire) &&
        std::abs(m_config.host_sample_rate - host_sr) <= 1e-6) {
        return;
    }

    m_config.host_sample_rate = host_sr;
    m_host_sample_rate.store(host_sr, std::memory_order_release);

    if (m_backend == nullptr) {
        return;
    }

    const ScopedReconfigureGate rt_gate(m_rt_gate);
    m_stopping.store(true, std::memory_order_release);
    m_worker.stop();
    configure_locked();
}

auto StreamingEngine::set_model_sample_rate(double model_sr) -> void {
    if (model_sr <= 0.0) {
        return;
    }

    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    if (m_configured.load(std::memory_order_acquire) &&
        std::abs(m_config.model_sample_rate - model_sr) <= 1e-6) {
        return;
    }

    m_config.model_sample_rate = model_sr;
    m_model_sample_rate.store(model_sr, std::memory_order_release);

    if (m_backend == nullptr) {
        return;
    }

    const ScopedReconfigureGate rt_gate(m_rt_gate);
    m_stopping.store(true, std::memory_order_release);
    m_worker.stop();
    configure_locked();
}

auto StreamingEngine::set_resample_mode(ResampleMode mode) -> void {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    if (m_configured.load(std::memory_order_acquire) && m_config.resample_mode == mode) {
        return;
    }

    m_config.resample_mode = mode;

    if (m_backend == nullptr) {
        return;
    }

    const ScopedReconfigureGate rt_gate(m_rt_gate);
    m_stopping.store(true, std::memory_order_release);
    m_worker.stop();
    configure_locked();
}

auto StreamingEngine::set_buffer_size(int model_buffer_size) -> void {
    if (model_buffer_size <= 0) {
        return;
    }

    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    if (m_configured.load(std::memory_order_acquire) &&
        m_config.model_buffer_size == model_buffer_size) {
        return;
    }

    m_config.model_buffer_size = model_buffer_size;

    if (m_backend == nullptr) {
        return;
    }

    const ScopedReconfigureGate rt_gate(m_rt_gate);
    m_stopping.store(true, std::memory_order_release);
    m_worker.stop();
    configure_locked();
}

auto StreamingEngine::set_method(const std::string& method) -> void {
    if (method.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> cb_lock(m_callback_mutex);
        m_config.method = method;
    }
    m_pending_clear_fifo.store(true, std::memory_order_release);
}

auto StreamingEngine::request_clear_decode_gen_fifo() -> void {
    m_pending_clear_fifo.store(true, std::memory_order_release);
}

auto StreamingEngine::set_pre_inference_callback(PreInferenceCallback cb) -> void {
    std::lock_guard<std::mutex> cb_lock(m_callback_mutex);
    m_pre_inference_cb = std::move(cb);
}

auto StreamingEngine::set_latent_hook_provider(LatentHookProvider provider) -> void {
    std::lock_guard<std::mutex> cb_lock(m_callback_mutex);
    m_latent_hook_provider = std::move(provider);
}

auto StreamingEngine::set_error_callback(ErrorCallback cb) -> void {
    std::lock_guard<std::mutex> cb_lock(m_callback_mutex);
    m_error_cb = std::move(cb);
}

auto StreamingEngine::set_fuse_reset_callback(std::function<void()> cb) -> void {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    const ScopedReconfigureGate rt_gate(m_rt_gate);
    m_fuse_reset_cb = std::move(cb);
}

auto StreamingEngine::get_config() const -> StreamingConfig {
    std::lock_guard<std::mutex> reconfig_lock(m_reconfig_mutex);
    std::lock_guard<std::mutex> cb_lock(m_callback_mutex);
    return m_config;
}

auto StreamingEngine::set_dry_wet(float dry_wet) noexcept NETBENDER_NONBLOCKING -> void {
    m_dry_wet.store(std::clamp(dry_wet, 0.0f, 1.0f), std::memory_order_relaxed);
}

auto StreamingEngine::get_dry_wet() const noexcept NETBENDER_NONBLOCKING -> float {
    return m_dry_wet.load(std::memory_order_relaxed);
}

auto StreamingEngine::has_work_and_space() const NETBENDER_NONBLOCKING -> bool {
    const int n_in = m_model_in_streams.load(std::memory_order_acquire);
    const int n_out = m_model_out_streams.load(std::memory_order_acquire);
    const int model_buf = m_model_buffer_size.load(std::memory_order_acquire);
    if (model_buf <= 0) {
        return false;
    }

    if (n_out > 0 && m_out_bank.get_free_space_src() < model_buf) {
        return false;
    }

    if (n_in > 0) {
        return m_in_bank.can_pull(model_buf);
    }

    // Generative / prior mode (0 audio inputs): keep healthy synthesis headroom in out_bank.
    return (n_out > 0) && (m_out_bank.get_available_src() < model_buf * 3);
}

auto StreamingEngine::run_worker_loop() -> void {
    while (!m_stopping.load(std::memory_order_acquire) && has_work_and_space()) {
        run_inference_step();
    }
}

auto StreamingEngine::run_inference_step() -> void {
    if (m_backend == nullptr || !m_configured.load(std::memory_order_acquire)) {
        return;
    }

    const int n_in = m_model_in_streams.load(std::memory_order_acquire);
    const int n_out = m_model_out_streams.load(std::memory_order_acquire);
    const int model_buf = m_model_buffer_size.load(std::memory_order_acquire);
    if (model_buf <= 0) {
        return;
    }

    if (n_in > 0) {
        (void)m_in_bank.pull(m_worker_in_ptrs.data(), n_in, model_buf);
    }

    for (auto& out_ch : m_worker_out) {
        std::fill(out_ch.begin(), out_ch.end(), 0.0f);
    }

    {
        // When use_worker_thread == false (synchronous inline mode), run_inference_step() is
        // called inside process() (which is NETBENDER_NONBLOCKING). Scope-disable RTSan around
        // the callback snapshot and backend inference call while keeping all ring-buffer,
        // resampler, RtGate, and SafetySentry paths strictly RTSan-checked.
        const netbender::rtsan::ScopedDisabler rtsan_disabler;

        std::string method;
        InferenceDispatch dispatch = InferenceDispatch::PluginMode;
        int n_batches = 1;
        int model_out_channels = 1;
        PreInferenceCallback pre_cb;
        LatentHookProvider hook_provider;
        ErrorCallback error_cb;
        {
            std::lock_guard<std::mutex> cb_lock(m_callback_mutex);
            method = m_config.method;
            dispatch = m_config.dispatch;
            n_batches = m_config.n_batches;
            model_out_channels = m_config.model_out_channels;
            pre_cb = m_pre_inference_cb;
            hook_provider = m_latent_hook_provider;
            error_cb = m_error_cb;
        }

        if (m_pending_clear_fifo.exchange(false, std::memory_order_acq_rel)) {
            m_backend->clear_decode_gen_fifo();
        }

        try {
            if (pre_cb) {
                pre_cb(*m_backend);
            }

            if (dispatch == InferenceDispatch::PluginMode) {
                if (method == "forward") {
                    IBackend::LatentHook hook = hook_provider ? hook_provider() : nullptr;
                    m_backend->perform_forward(m_worker_in_ptrs, m_worker_out_ptrs, n_batches,
                                               model_out_channels, model_buf, std::move(hook));
                } else if (method == "decode gen" || method == "decode_gen") {
                    IBackend::LatentHook hook = hook_provider ? hook_provider() : nullptr;
                    m_backend->perform_decode_gen(m_worker_out_ptrs, n_batches, model_out_channels,
                                                  model_buf, std::move(hook));
                } else if (method == "prior" || method == "generate") {
                    m_backend->perform_prior_decode(m_worker_out_ptrs, n_batches,
                                                    model_out_channels, model_buf);
                } else {
                    m_backend->perform(m_worker_in_ptrs, m_worker_out_ptrs, method, n_batches,
                                       model_out_channels, model_buf);
                }
            } else {
                m_backend->perform(m_worker_in_ptrs, m_worker_out_ptrs, method, n_batches,
                                   model_out_channels, model_buf);
            }
        } catch (const std::exception& e) {
            for (auto& out_ch : m_worker_out) {
                std::fill(out_ch.begin(), out_ch.end(), 0.0f);
            }
            if (error_cb) {
                error_cb(e.what());
            }
        } catch (...) {
            for (auto& out_ch : m_worker_out) {
                std::fill(out_ch.begin(), out_ch.end(), 0.0f);
            }
            if (error_cb) {
                error_cb("Unknown inference exception");
            }
        }
    }

    for (int c = 0; c < n_out; ++c) {
        (void)m_out_bank.push_channel(c, m_worker_out[static_cast<size_t>(c)].data(), model_buf);
    }
    m_inference_count.fetch_add(1, std::memory_order_relaxed);
}

template <typename SampleT>
auto StreamingEngine::apply_dry_wet_and_sentry(SampleT* const* host_out, int num_host_out_channels,
                                               int num_frames) NETBENDER_NONBLOCKING -> void {
    const bool use_dry = m_config.enable_dry_delay;
    const bool use_sentry = m_config.enable_safety_sentry;
    if ((!use_dry && !use_sentry) || host_out == nullptr || num_frames <= 0) {
        return;
    }

    const float wet_gain =
        use_dry ? std::clamp(m_dry_wet.load(std::memory_order_relaxed), 0.0f, 1.0f) : 1.0f;
    const float dry_gain = use_dry ? (1.0f - wet_gain) : 0.0f;
    const int scratch_cap = static_cast<int>(m_dry_scratch.size());
    if (scratch_cap <= 0) {
        return;
    }

    const int n_dry_buffers = static_cast<int>(m_dry_delay_buffers.size());

    for (int c = 0; c < num_host_out_channels; ++c) {
        SampleT* out_ptr = host_out[c];
        const bool has_own_dry_ring = use_dry && (c < n_dry_buffers);

        int offset = 0;
        while (offset < num_frames) {
            const int chunk = std::min(num_frames - offset, scratch_cap);

            // Drain each dry delay ring at most once per audio callback; channels beyond the
            // allocated dry ring bank fall back to silence without double-draining channel 0.
            const float* dry_ptr = m_zero_scratch.data();
            if (has_own_dry_ring) {
                (void)m_dry_delay_buffers[static_cast<size_t>(c)].get(m_dry_scratch.data(), chunk);
                dry_ptr = m_dry_scratch.data();
            }

            if (out_ptr == nullptr) {
                offset += chunk;
                continue;
            }

            if constexpr (std::is_same_v<SampleT, float>) {
                if (use_sentry) {
                    m_safety_sentry.process_channel(out_ptr + offset, dry_ptr, chunk, wet_gain,
                                                    dry_gain);
                } else {
                    float* dst = out_ptr + offset;
                    for (int i = 0; i < chunk; ++i) {
                        dst[i] = dst[i] * wet_gain + dry_ptr[i] * dry_gain;
                    }
                }
            } else {
                float* wet_tmp = m_wet_float_scratch.data();
                const double* src_d = out_ptr + offset;
                if (use_sentry) {
                    for (int i = 0; i < chunk; ++i) {
                        wet_tmp[i] = static_cast<float>(src_d[i]);
                    }
                    m_safety_sentry.process_channel(wet_tmp, dry_ptr, chunk, wet_gain, dry_gain);
                    double* dst_d = out_ptr + offset;
                    for (int i = 0; i < chunk; ++i) {
                        dst_d[i] = static_cast<double>(wet_tmp[i]);
                    }
                } else {
                    double* dst_d = out_ptr + offset;
                    for (int i = 0; i < chunk; ++i) {
                        dst_d[i] = static_cast<double>(static_cast<float>(src_d[i]) * wet_gain +
                                                       dry_ptr[i] * dry_gain);
                    }
                }
            }

            offset += chunk;
        }
    }

    if (use_dry) {
        for (int c = num_host_out_channels; c < n_dry_buffers; ++c) {
            (void)m_dry_delay_buffers[static_cast<size_t>(c)].drop(num_frames);
        }
    }

    if (use_sentry) {
        m_safety_sentry.tick_recovery(m_fuse_reset_cb);
    }
}

template <typename SampleT>
auto StreamingEngine::process_impl(const SampleT* const* host_in, int num_host_in_channels,
                                   SampleT* const* host_out, int num_host_out_channels,
                                   int num_frames) NETBENDER_NONBLOCKING -> void {
    if (num_frames <= 0 || num_host_out_channels <= 0 || host_out == nullptr) {
        return;
    }

    const ScopedRtReader rt_guard(m_rt_gate);
    if (!rt_guard.is_acquired() || !m_configured.load(std::memory_order_acquire)) {
        for (int c = 0; c < num_host_out_channels; ++c) {
            if (host_out[c] != nullptr) {
                std::fill(host_out[c], host_out[c] + num_frames, SampleT{0});
            }
        }
        return;
    }

    const int n_in = m_model_in_streams.load(std::memory_order_acquire);
    const int n_out = m_model_out_streams.load(std::memory_order_acquire);

    // 1. Push host audio inputs into in_bank (host SR timebase)
    if (n_in > 0) {
        for (int c = 0; c < n_in; ++c) {
            const SampleT* src = nullptr;
            if (host_in != nullptr) {
                if (c < num_host_in_channels) {
                    src = host_in[c];
                } else if (m_config.duplicate_mono_input && num_host_in_channels > 0) {
                    src = host_in[0];
                }
            }
            (void)m_in_bank.push_channel(c, src, num_frames);
        }
    }

    // 2. Push host audio inputs into dry delay alignment rings when enabled
    if (m_config.enable_dry_delay) {
        const int n_dry = static_cast<int>(m_dry_delay_buffers.size());
        for (int c = 0; c < n_dry; ++c) {
            const SampleT* src = nullptr;
            if (host_in != nullptr) {
                if (c < num_host_in_channels) {
                    src = host_in[c];
                } else if (m_config.duplicate_mono_input && num_host_in_channels > 0) {
                    src = host_in[0];
                }
            }
            (void)m_dry_delay_buffers[static_cast<size_t>(c)].put(src, num_frames);
        }
    }

    // 3. Schedule async background inference or run synchronously inline
    if (m_config.use_worker_thread) {
        if (has_work_and_space()) {
            m_worker.trigger();
        }
    } else {
        if (n_in > 0) {
            while (has_work_and_space()) {
                run_inference_step();
            }
        } else if (n_out > 0) {
            const int model_buf = m_model_buffer_size.load(std::memory_order_acquire);
            while (!m_out_bank.can_pull(num_frames) &&
                   m_out_bank.get_free_space_src() >= model_buf) {
                run_inference_step();
            }
        }
    }

    // 4. Pull lockstep resampled (or bypassed) model outputs into host output channels
    (void)m_out_bank.pull(host_out, num_host_out_channels, num_frames);

    if (m_config.duplicate_mono_output && n_out == 1 && num_host_out_channels >= 2 &&
        host_out[0] != nullptr && host_out[1] != nullptr) {
        std::copy_n(host_out[0], num_frames, host_out[1]);
    }

    // 5. Apply optional dry/wet blending and SafetySentry protection
    apply_dry_wet_and_sentry(host_out, num_host_out_channels, num_frames);
}

auto StreamingEngine::process(const float* const* host_in, int num_host_in_channels,
                              float* const* host_out, int num_host_out_channels,
                              int num_frames) NETBENDER_NONBLOCKING -> void {
    process_impl(host_in, num_host_in_channels, host_out, num_host_out_channels, num_frames);
}

auto StreamingEngine::process(const double* const* host_in, int num_host_in_channels,
                              double* const* host_out, int num_host_out_channels,
                              int num_frames) NETBENDER_NONBLOCKING -> void {
    process_impl(host_in, num_host_in_channels, host_out, num_host_out_channels, num_frames);
}
