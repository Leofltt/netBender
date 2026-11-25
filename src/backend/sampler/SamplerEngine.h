#pragma once

#include "ISamplerBackend.h"
#include "SamplerTypes.h"
#include "TextTokenizer.h"
#include "../dsp/RtGate.h"
#include "../dsp/SafetySentry.h"
#include "../dsp/rtsan_utils.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace netbender::sampler {

/**
 * Frontend-agnostic SA3 Sampler Engine:
 *   - Background worker thread running cancellable `SamplerPrompter` generation
 *   - Atomic `RtGate`-protected `SampleSlot` swap (audio + latents + waveform peaks)
 *   - Wait-free RT audio callback (`OneShot` / `Loop` / `Gate` / MIDI pitch / `SafetySentry`)
 */
class SamplerEngine {
public:
    static constexpr int k_max_slots = 12;
    static constexpr int k_num_slots = k_max_slots;

    SamplerEngine();
    explicit SamplerEngine(std::unique_ptr<ISamplerBackend> backend);
    ~SamplerEngine();

    SamplerEngine(const SamplerEngine&) = delete;
    auto operator=(const SamplerEngine&) -> SamplerEngine& = delete;

    // Non-RT lifecycle & configuration
    auto load_bundle(const std::string& manifest_path, bool use_gpu = true) -> bool;
    auto load_tokenizer_from_json(const std::string& json_text, int max_length, int pad_id,
                                  int eos_id, bool add_eos) -> bool;
    [[nodiscard]] auto is_loaded() const -> bool;
    [[nodiscard]] auto get_metadata() const -> const SamplerBundleMetadata&;
    [[nodiscard]] auto get_last_error() const -> std::string;

    auto prepare(double host_sample_rate, int max_block_size) -> void;

    // Async generation control
    auto request_generate(const GenerationRequest& req) -> void;
    auto request_generate_slots(const GenerationRequest& base_req, uint32_t slot_mask) -> void;
    auto cancel_generation() -> void;

    // Active slot bank size (4, 8, or 12)
    auto set_slot_count(int count) noexcept -> void;
    [[nodiscard]] auto get_slot_count() const noexcept -> int {
        return m_slot_count.load(std::memory_order_acquire);
    }

    [[nodiscard]] auto get_stage() const noexcept -> GenerationProgress::Stage {
        return m_stage.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_current_step() const noexcept -> int {
        return m_current_step.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_total_steps() const noexcept -> int {
        return m_total_steps.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_batch_current_slot() const noexcept -> int {
        return m_batch_current_slot.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_batch_index() const noexcept -> int {
        return m_batch_index.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_batch_total() const noexcept -> int {
        return m_batch_total.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_last_gen_ms() const noexcept -> double {
        return m_last_gen_ms.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_slot_generation_id() const noexcept -> uint64_t {
        return m_slot_generation_id.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_slot_generation_id(int slot_idx) const noexcept -> uint64_t;
    [[nodiscard]] auto get_slot_snapshot() const -> std::shared_ptr<const SampleSlot>;
    [[nodiscard]] auto get_slot_snapshot(int slot_idx) const -> std::shared_ptr<const SampleSlot>;

    // Active/focused slot selection & per-slot loop region
    auto set_active_slot(int slot_idx) noexcept -> void;
    [[nodiscard]] auto get_active_slot() const noexcept -> int {
        return m_active_slot_index.load(std::memory_order_acquire);
    }

    // Lock-free playback parameters (safe from UI or audio thread)
    auto set_playback_mode(SamplerPlaybackMode mode) noexcept -> void {
        m_playback_mode.store(mode, std::memory_order_release);
    }
    [[nodiscard]] auto get_playback_mode() const noexcept -> SamplerPlaybackMode {
        return m_playback_mode.load(std::memory_order_acquire);
    }

    auto set_loop_region(float start_norm, float end_norm) noexcept -> void;
    auto set_loop_region(int slot_idx, float start_norm, float end_norm) noexcept -> void;
    [[nodiscard]] auto get_loop_start() const noexcept -> float {
        return get_loop_start(get_active_slot());
    }
    [[nodiscard]] auto get_loop_end() const noexcept -> float {
        return get_loop_end(get_active_slot());
    }
    [[nodiscard]] auto get_loop_start(int slot_idx) const noexcept -> float;
    [[nodiscard]] auto get_loop_end(int slot_idx) const noexcept -> float;

    auto set_speed(float speed) noexcept -> void;
    [[nodiscard]] auto get_speed() const noexcept -> float {
        return m_speed.load(std::memory_order_acquire);
    }

    auto set_gain(float gain) noexcept -> void;
    [[nodiscard]] auto get_gain() const noexcept -> float {
        return m_gain.load(std::memory_order_acquire);
    }

    [[nodiscard]] auto get_playhead_norm() const noexcept -> float {
        return m_playhead_norm.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto is_playing() const noexcept -> bool {
        return m_playing.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto get_playing_slot() const noexcept -> int {
        return m_playing_slot_index.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto is_slot_playing(int slot_idx) const noexcept -> bool {
        return is_playing() && get_playing_slot() == slot_idx;
    }

    // Wait-free RT triggers & audio callback
    auto trigger_rt(int midi_note = 60, float velocity = 1.0f) NETBENDER_NONBLOCKING -> void;
    auto trigger_slot_rt(int slot_idx, int midi_note = 60,
                         float velocity = 1.0f) NETBENDER_NONBLOCKING -> void;
    auto release_rt(int midi_note = -1) NETBENDER_NONBLOCKING -> void;
    auto stop_rt() NETBENDER_NONBLOCKING -> void;
    auto stop_slot_rt(int slot_idx) NETBENDER_NONBLOCKING -> void;

    auto process_rt(float* const* out_channels, int num_channels,
                    int num_samples) NETBENDER_NONBLOCKING -> void;

private:
    auto worker_loop() -> void;
    static auto compute_waveform_peaks(const std::vector<std::vector<float>>& audio,
                                       int num_samples) -> std::vector<float>;
    static auto compute_spectrogram(const std::vector<std::vector<float>>& audio, int num_samples,
                                    int sample_rate) -> std::vector<float>;
    static auto detect_nonsilent_region(const std::vector<std::vector<float>>& audio,
                                        int num_samples, int sample_rate,
                                        float threshold_db = -84.0f) -> std::pair<float, float>;
    static auto sample_linear(const std::vector<float>& ch, double pos) noexcept -> float;

    std::unique_ptr<ISamplerBackend> m_backend;
    TextTokenizer m_tokenizer;
    mutable std::mutex m_error_mutex;
    std::string m_last_error;

    // Worker thread state
    std::thread m_worker_thread;
    std::mutex m_worker_mutex;
    std::condition_variable m_worker_cv;
    bool m_stop_worker{false};
    bool m_has_pending_request{false};
    GenerationRequest m_pending_request;
    uint32_t m_pending_slot_mask{0x0Fu};
    std::atomic<bool> m_cancel_requested{false};
    std::atomic<int> m_slot_count{4};

    // Progress & slot state
    std::atomic<GenerationProgress::Stage> m_stage{GenerationProgress::Stage::Idle};
    std::atomic<int> m_current_step{0};
    std::atomic<int> m_total_steps{0};
    std::atomic<int> m_batch_current_slot{-1};
    std::atomic<int> m_batch_index{0};
    std::atomic<int> m_batch_total{0};
    std::atomic<double> m_last_gen_ms{0.0};
    std::atomic<uint64_t> m_slot_generation_id{0};
    std::array<std::atomic<uint64_t>, k_num_slots> m_slot_generation_ids{};

    mutable std::mutex m_snapshot_mutex;
    std::array<std::shared_ptr<SampleSlot>, k_num_slots> m_slots;
    RtGate m_slot_gate;

    // Playback parameters
    std::atomic<int> m_active_slot_index{0};
    std::atomic<int> m_playing_slot_index{0};
    std::atomic<SamplerPlaybackMode> m_playback_mode{SamplerPlaybackMode::OneShot};
    std::array<std::atomic<float>, k_num_slots> m_loop_starts{};
    std::array<std::atomic<float>, k_num_slots> m_loop_ends{};
    std::atomic<float> m_speed{1.0f};
    std::atomic<float> m_gain{1.0f};
    std::atomic<double> m_host_sample_rate{44100.0};

    // RT pending trigger state (atomic-only, wait-free)
    std::atomic<bool> m_trigger_pending{false};
    std::atomic<bool> m_release_pending{false};
    std::atomic<bool> m_stop_pending{false};
    std::atomic<int> m_trigger_slot{0};
    std::atomic<int> m_stop_slot{-1};
    std::atomic<int> m_trigger_note{60};
    std::atomic<float> m_trigger_vel{1.0f};
    std::atomic<int> m_release_note{-1};

    // RT audio-thread-only playback state
    std::atomic<bool> m_playing{false};
    std::atomic<float> m_playhead_norm{0.0f};
    double m_playhead{0.0};
    double m_grain_pos[2]{0.0, 0.0};
    double m_grain_phase{0.0};
    int m_active_note{60};
    float m_voice_vel{1.0f};
    bool m_releasing{false};
    float m_env{0.0f};
    std::vector<float> m_zero_scratch;
    SafetySentry m_sentry;
};

}  // namespace netbender::sampler
