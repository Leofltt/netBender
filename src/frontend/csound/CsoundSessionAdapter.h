#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "NetBenderSession.h"

namespace netbender::csound_fe {

struct OpenOptions {
    std::string model_path;
    std::string method{"forward"};
    int buffer_size{-1};  // -1 = backend preferred, 0 = synchronous inline, >0 = pow2 size
    bool use_gpu{false};
    int thread_mode{-1};  // -1 = auto (async if realtime else sync), 0 = sync, 1 = async
    int safety_sentry{
        -1};            // -1 = auto (on for audio modes, off for encode/decode), 0 = off, 1 = on
    int dry_delay{-1};  // -1 = auto (on for forward, off for gen/encode/decode), 0 = off, 1 = on
    double host_sample_rate{44100.0};
    int ksmps{128};
    int max_channels{2};
    bool is_realtime_output{true};
};

struct ModelSummary {
    std::string resolved_path;
    std::string backend_name;
    std::string method;
    int in_channels{0};
    int out_channels{0};
    int in_ratio{1};
    int out_ratio{1};
    double host_sample_rate{44100.0};
    double model_sample_rate{44100.0};
    bool resampling_active{false};
    int host_latency_samples{0};
    int model_buffer_size{0};
    bool worker_thread_enabled{false};
    bool safety_sentry_enabled{false};
    bool dry_delay_enabled{false};
    std::vector<std::string> layers;
    std::vector<std::string> methods;
    std::vector<std::string> settable_attributes;
};

using HostFileFinder = std::function<std::string(const std::string& candidate)>;

/**
 * Resolves a model path across:
 *   1. Exact / relative-to-cwd / absolute path (with .pte/.onnx/.ts probing when extensionless)
 *   2. Host file finder callback (e.g. Csound FindInputFile on SSDIR/SFDIR/INCDIR)
 *   3. ~/.netbender/models (or home_models_override when provided for tests)
 * Returns an empty string if no regular file matches.
 */
[[nodiscard]] auto resolve_model_path(const std::string& raw_path,
                                      const HostFileFinder& host_finder = nullptr,
                                      const std::string& home_models_override = {}) -> std::string;

[[nodiscard]] auto normalize_method_name(const std::string& method) -> std::string;
[[nodiscard]] auto is_latent_method(const std::string& method) -> bool;
[[nodiscard]] auto is_generative_method(const std::string& method) -> bool;
[[nodiscard]] auto round_up_pow2(int x) -> int;

/**
 * Pure C++20 session wrapper bridging Csound opcode lifecycles and NetBenderSession.
 * Contains no Csound headers so it can be unit-tested independently with MockBackend.
 */
class CsoundSessionAdapter {
public:
    explicit CsoundSessionAdapter(NetBenderSession::BackendCreator creator = nullptr);
    ~CsoundSessionAdapter();

    CsoundSessionAdapter(const CsoundSessionAdapter&) = delete;
    auto operator=(const CsoundSessionAdapter&) -> CsoundSessionAdapter& = delete;
    CsoundSessionAdapter(CsoundSessionAdapter&&) = delete;
    auto operator=(CsoundSessionAdapter&&) -> CsoundSessionAdapter& = delete;

    auto open(const OpenOptions& opts, std::string& error_out) -> bool;
    auto close() -> void;
    auto reset() -> void;

    auto process(const double* const* in_channels, int num_in, double* const* out_channels,
                 int num_out, int num_frames, double zero_dbfs, float dry_wet, bool enabled)
        -> void;
    auto process(const float* const* in_channels, int num_in, float* const* out_channels,
                 int num_out, int num_frames, double zero_dbfs, float dry_wet, bool enabled)
        -> void;

    auto generate(double* const* out_channels, int num_out, int num_frames, double zero_dbfs,
                  bool enabled) -> void;
    auto generate(float* const* out_channels, int num_out, int num_frames, double zero_dbfs,
                  bool enabled) -> void;

    [[nodiscard]] auto is_open() const -> bool { return m_open; }
    [[nodiscard]] auto get_summary() const -> ModelSummary;
    [[nodiscard]] auto format_dump() const -> std::string;
    [[nodiscard]] auto get_in_channels() const -> int { return m_in_channels; }
    [[nodiscard]] auto get_out_channels() const -> int { return m_out_channels; }
    [[nodiscard]] auto session() -> NetBenderSession* { return m_session.get(); }
    [[nodiscard]] auto session() const -> const NetBenderSession* { return m_session.get(); }

private:
    template <typename SampleT>
    auto process_impl(const SampleT* const* in_channels, int num_in, SampleT* const* out_channels,
                      int num_out, int num_frames, double zero_dbfs, float dry_wet, bool enabled)
        -> void;

    NetBenderSession::BackendCreator m_creator;
    bool m_has_custom_creator{false};
    std::unique_ptr<NetBenderSession> m_session;

    bool m_open{false};
    int m_in_channels{0};
    int m_out_channels{0};
    int m_in_ratio{1};
    int m_out_ratio{1};
    bool m_worker_thread_enabled{false};
    bool m_safety_sentry_enabled{false};
    bool m_dry_delay_enabled{false};

    // Pre-allocated scratch buffers used only when 0dbfs != 1.0
    int m_scratch_channels{0};
    int m_scratch_frames{0};
    std::vector<std::vector<float>> m_scaled_in_storage;
    std::vector<const float*> m_scaled_in_ptrs;
    std::vector<std::vector<float>> m_scaled_out_storage;
    std::vector<float*> m_scaled_out_ptrs;
};

}  // namespace netbender::csound_fe
