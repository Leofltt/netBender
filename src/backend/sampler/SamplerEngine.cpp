#include "SamplerEngine.h"
#include "ExecuTorchSamplerBackend.h"
#include "SampleAnalysis.h"
#include "SamplerPrompter.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace netbender::sampler {

SamplerEngine::SamplerEngine() : SamplerEngine(std::make_unique<ExecuTorchSamplerBackend>()) {}

SamplerEngine::SamplerEngine(std::unique_ptr<ISamplerBackend> backend)
    : m_backend(std::move(backend)) {
    m_zero_scratch.assign(4096, 0.0f);
    for (int i = 0; i < k_max_slots; ++i) {
        m_loop_starts[static_cast<size_t>(i)].store(0.0f, std::memory_order_relaxed);
        m_loop_ends[static_cast<size_t>(i)].store(1.0f, std::memory_order_relaxed);
    }
    m_worker_thread = std::thread(&SamplerEngine::worker_loop, this);
}

SamplerEngine::~SamplerEngine() {
    cancel_generation();
    {
        std::lock_guard<std::mutex> lock(m_worker_mutex);
        m_stop_worker = true;
    }
    m_worker_cv.notify_all();
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
}

auto SamplerEngine::load_bundle(const std::string& manifest_path, bool use_gpu) -> bool {
    cancel_generation();
    if (!m_backend) {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        m_last_error = "No sampler backend configured";
        return false;
    }
    if (m_backend->load_bundle(manifest_path, use_gpu) != 0) {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        m_last_error = m_backend->get_last_error();
        return false;
    }
    const auto& meta = m_backend->get_metadata();
    if (!meta.tokenizer_path.empty()) {
        if (!m_tokenizer.load(meta.tokenizer_path, meta.max_token_length, meta.pad_token_id,
                              meta.eos_token_id, meta.add_eos)) {
            std::lock_guard<std::mutex> lock(m_error_mutex);
            m_last_error = m_tokenizer.get_last_error();
            return false;
        }
    }
    m_stage.store(GenerationProgress::Stage::Idle, std::memory_order_release);
    return true;
}

auto SamplerEngine::load_tokenizer_from_json(const std::string& json_text, int max_length,
                                             int pad_id, int eos_id, bool add_eos) -> bool {
    const bool ok =
        m_tokenizer.load_from_json_string(json_text, max_length, pad_id, eos_id, add_eos);
    if (!ok) {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        m_last_error = m_tokenizer.get_last_error();
    }
    return ok;
}

auto SamplerEngine::is_loaded() const -> bool {
    return m_backend && m_backend->is_loaded() && m_tokenizer.is_loaded();
}

auto SamplerEngine::get_metadata() const -> const SamplerBundleMetadata& {
    return m_backend->get_metadata();
}

auto SamplerEngine::get_last_error() const -> std::string {
    std::lock_guard<std::mutex> lock(m_error_mutex);
    return m_last_error;
}

auto SamplerEngine::prepare(double host_sample_rate, int max_block_size) -> void {
    if (host_sample_rate > 0.0) {
        m_host_sample_rate.store(host_sample_rate, std::memory_order_release);
    }
    if (max_block_size > static_cast<int>(m_zero_scratch.size())) {
        ScopedReconfigureGate gate(m_slot_gate);
        m_zero_scratch.assign(static_cast<size_t>(max_block_size), 0.0f);
    }
}

auto SamplerEngine::request_generate(const GenerationRequest& req) -> void {
    const int active = get_active_slot();
    request_generate_slots(req, 1u << active);
}

auto SamplerEngine::set_slot_count(int count) noexcept -> void {
    const int snapped = (count >= 12) ? 12 : ((count >= 8) ? 8 : 4);
    m_slot_count.store(snapped, std::memory_order_release);
    if (get_active_slot() >= snapped) {
        m_active_slot_index.store(snapped - 1, std::memory_order_release);
    }
}

auto SamplerEngine::request_generate_slots(const GenerationRequest& base_req, uint32_t slot_mask)
    -> void {
    const int count = get_slot_count();
    const uint32_t valid_mask = (count >= 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
    const uint32_t masked = slot_mask & valid_mask;
    if (masked == 0) {
        return;
    }
    m_cancel_requested.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(m_worker_mutex);
        m_pending_request = base_req;
        m_pending_slot_mask = masked;
        m_has_pending_request = true;
        m_stage.store(GenerationProgress::Stage::Tokenizing, std::memory_order_release);
    }
    m_worker_cv.notify_one();
}

auto SamplerEngine::cancel_generation() -> void {
    m_cancel_requested.store(true, std::memory_order_release);
}

auto SamplerEngine::get_slot_generation_id(int slot_idx) const noexcept -> uint64_t {
    if (slot_idx < 0 || slot_idx >= k_max_slots) {
        return 0;
    }
    return m_slot_generation_ids[static_cast<size_t>(slot_idx)].load(std::memory_order_acquire);
}

auto SamplerEngine::get_slot_snapshot() const -> std::shared_ptr<const SampleSlot> {
    return get_slot_snapshot(get_active_slot());
}

auto SamplerEngine::get_slot_snapshot(int slot_idx) const -> std::shared_ptr<const SampleSlot> {
    if (slot_idx < 0 || slot_idx >= k_max_slots) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(m_snapshot_mutex);
    return m_slots[static_cast<size_t>(slot_idx)];
}

auto SamplerEngine::set_active_slot(int slot_idx) noexcept -> void {
    m_active_slot_index.store(std::clamp(slot_idx, 0, k_max_slots - 1), std::memory_order_release);
}

auto SamplerEngine::set_loop_region(float start_norm, float end_norm) noexcept -> void {
    set_loop_region(get_active_slot(), start_norm, end_norm);
}

auto SamplerEngine::set_loop_region(int slot_idx, float start_norm, float end_norm) noexcept
    -> void {
    if (slot_idx < 0 || slot_idx >= k_max_slots) {
        return;
    }
    const float s = std::clamp(start_norm, 0.0f, 0.99f);
    const float e = std::clamp(end_norm, s + 0.01f, 1.0f);
    m_loop_starts[static_cast<size_t>(slot_idx)].store(s, std::memory_order_release);
    m_loop_ends[static_cast<size_t>(slot_idx)].store(e, std::memory_order_release);
}

auto SamplerEngine::get_loop_start(int slot_idx) const noexcept -> float {
    if (slot_idx < 0 || slot_idx >= k_max_slots) {
        return 0.0f;
    }
    return m_loop_starts[static_cast<size_t>(slot_idx)].load(std::memory_order_acquire);
}

auto SamplerEngine::get_loop_end(int slot_idx) const noexcept -> float {
    if (slot_idx < 0 || slot_idx >= k_max_slots) {
        return 1.0f;
    }
    return m_loop_ends[static_cast<size_t>(slot_idx)].load(std::memory_order_acquire);
}

auto SamplerEngine::set_speed(float speed) noexcept -> void {
    m_speed.store(std::clamp(speed, 0.25f, 4.0f), std::memory_order_release);
}

auto SamplerEngine::set_gain(float gain) noexcept -> void {
    m_gain.store(std::clamp(gain, 0.0f, 2.0f), std::memory_order_release);
}

auto SamplerEngine::trigger_rt(int midi_note, float velocity) NETBENDER_NONBLOCKING -> void {
    trigger_slot_rt(get_active_slot(), midi_note, velocity);
}

auto SamplerEngine::trigger_slot_rt(int slot_idx, int midi_note,
                                    float velocity) NETBENDER_NONBLOCKING -> void {
    m_trigger_slot.store(std::clamp(slot_idx, 0, k_max_slots - 1), std::memory_order_relaxed);
    m_trigger_note.store(std::clamp(midi_note, 0, 127), std::memory_order_relaxed);
    m_trigger_vel.store(std::clamp(velocity, 0.0f, 1.0f), std::memory_order_relaxed);
    m_trigger_pending.store(true, std::memory_order_release);
}

auto SamplerEngine::release_rt(int midi_note) NETBENDER_NONBLOCKING -> void {
    m_release_note.store(midi_note, std::memory_order_relaxed);
    m_release_pending.store(true, std::memory_order_release);
}

auto SamplerEngine::stop_rt() NETBENDER_NONBLOCKING -> void {
    m_stop_slot.store(-1, std::memory_order_relaxed);
    m_stop_pending.store(true, std::memory_order_release);
}

auto SamplerEngine::stop_slot_rt(int slot_idx) NETBENDER_NONBLOCKING -> void {
    m_stop_slot.store(std::clamp(slot_idx, 0, k_max_slots - 1), std::memory_order_relaxed);
    m_stop_pending.store(true, std::memory_order_release);
}

auto SamplerEngine::sample_linear(const std::vector<float>& ch, double pos) noexcept -> float {
    if (ch.empty()) {
        return 0.0f;
    }
    const int n = static_cast<int>(ch.size());
    if (pos <= 0.0) {
        return ch[0];
    }
    const int i0 = static_cast<int>(pos);
    if (i0 >= n - 1) {
        return ch[static_cast<size_t>(n - 1)];
    }
    const float frac = static_cast<float>(pos - static_cast<double>(i0));
    return ch[static_cast<size_t>(i0)] +
           frac * (ch[static_cast<size_t>(i0 + 1)] - ch[static_cast<size_t>(i0)]);
}

auto SamplerEngine::worker_loop() -> void {
    while (true) {
        GenerationRequest base_req;
        uint32_t slot_mask = 0;
        {
            std::unique_lock<std::mutex> lock(m_worker_mutex);
            m_worker_cv.wait(lock, [this]() { return m_stop_worker || m_has_pending_request; });
            if (m_stop_worker) {
                break;
            }
            base_req = m_pending_request;
            slot_mask = m_pending_slot_mask;
            m_has_pending_request = false;
            m_cancel_requested.store(false, std::memory_order_release);
        }

        if (!is_loaded()) {
            std::lock_guard<std::mutex> lock(m_error_mutex);
            m_last_error = "Sampler bundle not loaded";
            m_batch_current_slot.store(-1, std::memory_order_release);
            m_stage.store(GenerationProgress::Stage::Error, std::memory_order_release);
            continue;
        }

        std::vector<int> selected_slots;
        selected_slots.reserve(k_max_slots);
        for (int i = 0; i < k_max_slots; ++i) {
            if ((slot_mask & (1u << i)) != 0) {
                selected_slots.push_back(i);
            }
        }
        if (selected_slots.empty()) {
            continue;
        }

        const int default_steps = std::max(1, m_backend->get_metadata().default_steps);
        const int req_steps = base_req.steps > 0 ? base_req.steps : default_steps;
        m_batch_total.store(static_cast<int>(selected_slots.size()), std::memory_order_release);

        SamplerPrompter prompter(*m_backend, m_tokenizer);
        double total_elapsed_ms = 0.0;
        bool batch_ok = true;

        for (size_t k = 0; k < selected_slots.size(); ++k) {
            const int slot_idx = selected_slots[k];
            m_batch_current_slot.store(slot_idx, std::memory_order_release);
            m_batch_index.store(static_cast<int>(k + 1), std::memory_order_release);
            m_total_steps.store(req_steps, std::memory_order_release);
            m_current_step.store(0, std::memory_order_release);
            m_stage.store(GenerationProgress::Stage::Tokenizing, std::memory_order_release);

            GenerationRequest slot_req = base_req;
            slot_req.seed = base_req.seed + static_cast<uint64_t>(slot_idx);

            GeneratedSample generated;
            const bool ok = prompter.generate(
                slot_req, generated, &m_cancel_requested, [this](const GenerationProgress& p) {
                    m_current_step.store(p.current_step, std::memory_order_release);
                    m_total_steps.store(p.total_steps, std::memory_order_release);
                    m_stage.store(p.stage, std::memory_order_release);
                });

            if (!ok) {
                const bool cancelled = m_cancel_requested.load(std::memory_order_acquire);
                if (!cancelled) {
                    std::lock_guard<std::mutex> lock(m_error_mutex);
                    m_last_error = prompter.get_last_error();
                }
                m_batch_current_slot.store(-1, std::memory_order_release);
                m_stage.store(cancelled ? GenerationProgress::Stage::Cancelled
                                        : GenerationProgress::Stage::Error,
                              std::memory_order_release);
                batch_ok = false;
                break;
            }

            total_elapsed_ms += generated.elapsed_ms;

            auto slot = std::make_shared<SampleSlot>();
            slot->num_samples =
                generated.audio.empty() ? 0 : static_cast<int>(generated.audio[0].size());
            slot->waveform_peaks = compute_waveform_peaks(generated.audio, slot->num_samples);
            slot->spectrogram =
                compute_spectrogram(generated.audio, slot->num_samples, generated.sample_rate);
            slot->audio = std::move(generated.audio);
            slot->latents = std::move(generated.latents);
            slot->sample_rate = generated.sample_rate;
            slot->latent_dim = generated.latent_dim;
            slot->latent_frames = generated.latent_frames;
            slot->duration_sec = generated.duration_sec;
            slot->prompt = slot_req.prompt;
            slot->seed = slot_req.seed;

            const auto [auto_start, auto_end] =
                detect_nonsilent_region(slot->audio, slot->num_samples, slot->sample_rate, -84.0f);

            {
                std::lock_guard<std::mutex> snap_lock(m_snapshot_mutex);
                ScopedReconfigureGate gate(m_slot_gate);
                m_slots[static_cast<size_t>(slot_idx)] = std::move(slot);
            }

            set_loop_region(slot_idx, auto_start, auto_end);
            m_slot_generation_ids[static_cast<size_t>(slot_idx)].fetch_add(
                1, std::memory_order_acq_rel);
            m_slot_generation_id.fetch_add(1, std::memory_order_acq_rel);
        }

        if (!batch_ok) {
            continue;
        }

        m_batch_current_slot.store(-1, std::memory_order_release);
        m_last_gen_ms.store(total_elapsed_ms, std::memory_order_release);
        m_stage.store(GenerationProgress::Stage::Done, std::memory_order_release);

        if (m_playback_mode.load(std::memory_order_acquire) == SamplerPlaybackMode::Loop &&
            !is_playing() && !selected_slots.empty()) {
            trigger_slot_rt(selected_slots.front(), 60, 1.0f);
        }
    }
}

auto SamplerEngine::process_rt(float* const* out_channels, int num_channels,
                               int num_samples) NETBENDER_NONBLOCKING -> void {
    if (out_channels == nullptr || num_channels <= 0 || num_samples <= 0) {
        return;
    }

    auto clear_outputs = [&]() {
        for (int c = 0; c < num_channels; ++c) {
            if (out_channels[c] != nullptr) {
                std::memset(out_channels[c], 0, static_cast<size_t>(num_samples) * sizeof(float));
            }
        }
    };

    ScopedRtReader reader(m_slot_gate);
    if (!reader.is_acquired()) {
        clear_outputs();
        return;
    }

    int playing_slot =
        std::clamp(m_playing_slot_index.load(std::memory_order_relaxed), 0, k_max_slots - 1);
    const SamplerPlaybackMode mode = m_playback_mode.load(std::memory_order_acquire);

    if (m_stop_pending.exchange(false, std::memory_order_acquire)) {
        const int stop_slot = m_stop_slot.load(std::memory_order_relaxed);
        if (stop_slot < 0 || stop_slot == playing_slot) {
            m_playing.store(false, std::memory_order_relaxed);
            m_releasing = false;
            m_env = 0.0f;
        }
    }

    bool need_reset_playhead = false;
    if (m_trigger_pending.exchange(false, std::memory_order_acquire)) {
        playing_slot =
            std::clamp(m_trigger_slot.load(std::memory_order_relaxed), 0, k_max_slots - 1);
        m_playing_slot_index.store(playing_slot, std::memory_order_relaxed);
        m_active_note = m_trigger_note.load(std::memory_order_relaxed);
        m_voice_vel = m_trigger_vel.load(std::memory_order_relaxed);
        need_reset_playhead = true;
        m_releasing = false;
        m_env = 1.0f;
        m_playing.store(true, std::memory_order_relaxed);
    }

    if (m_release_pending.exchange(false, std::memory_order_acquire)) {
        const int rel_note = m_release_note.load(std::memory_order_relaxed);
        if (mode == SamplerPlaybackMode::Gate && (rel_note < 0 || rel_note == m_active_note)) {
            m_releasing = true;
        }
    }

    const SampleSlot* slot = m_slots[static_cast<size_t>(playing_slot)].get();
    if (slot == nullptr || slot->audio.empty() || slot->num_samples <= 0) {
        m_playing.store(false, std::memory_order_relaxed);
        m_playhead_norm.store(0.0f, std::memory_order_relaxed);
        clear_outputs();
        return;
    }

    const int n_total = slot->num_samples;
    const float loop_start_norm =
        m_loop_starts[static_cast<size_t>(playing_slot)].load(std::memory_order_relaxed);
    const float loop_end_norm =
        m_loop_ends[static_cast<size_t>(playing_slot)].load(std::memory_order_relaxed);
    const int loop_start = std::clamp(static_cast<int>(loop_start_norm * n_total), 0, n_total - 1);
    const int loop_end =
        std::clamp(static_cast<int>(loop_end_norm * n_total), loop_start + 1, n_total);
    const int loop_len = std::max(1, loop_end - loop_start);

    if (need_reset_playhead) {
        m_playhead = static_cast<double>(loop_start);
        m_grain_pos[0] = m_playhead;
        m_grain_pos[1] = m_playhead;
        m_grain_phase = 0.0;
    }

    if (!m_playing.load(std::memory_order_relaxed)) {
        clear_outputs();
        return;
    }

    const double host_sr = std::max(1.0, m_host_sample_rate.load(std::memory_order_relaxed));
    const double slot_sr = static_cast<double>(std::max(1, slot->sample_rate));
    const double speed = static_cast<double>(m_speed.load(std::memory_order_relaxed));
    const double pitch_ratio = std::exp2(static_cast<double>(m_active_note - 60) / 12.0);
    const double pitch_step = (slot_sr / host_sr) * pitch_ratio;
    const double time_step = pitch_step * speed;
    const bool use_time_stretch = (std::abs(speed - 1.0) >= 1e-3);
    const float gain = m_gain.load(std::memory_order_relaxed) * m_voice_vel;

    const bool loop_active =
        (mode == SamplerPlaybackMode::Loop) || (mode == SamplerPlaybackMode::Gate && !m_releasing);
    const int xfade_len = std::min(256, loop_len / 4);
    const float release_step = 1.0f / 512.0f;
    const int slot_ch = static_cast<int>(slot->audio.size());

    constexpr double kGrainSize = 2048.0;
    constexpr float kTwoPi = 6.28318530717958647692f;
    auto wrap_pos_fn = [&](double p) -> double {
        if (loop_active) {
            double rel =
                std::fmod(p - static_cast<double>(loop_start), static_cast<double>(loop_len));
            if (rel < 0.0) {
                rel += static_cast<double>(loop_len);
            }
            return static_cast<double>(loop_start) + rel;
        }
        return std::clamp(p, static_cast<double>(loop_start), static_cast<double>(loop_end - 1));
    };

    for (int s = 0; s < num_samples; ++s) {
        if (!m_playing.load(std::memory_order_relaxed)) {
            for (int c = 0; c < num_channels; ++c) {
                if (out_channels[c] != nullptr) {
                    out_channels[c][s] = 0.0f;
                }
            }
            continue;
        }

        if (m_releasing) {
            m_env = std::max(0.0f, m_env - release_step);
            if (m_env <= 0.0f) {
                m_playing.store(false, std::memory_order_relaxed);
                m_releasing = false;
                for (int c = 0; c < num_channels; ++c) {
                    if (out_channels[c] != nullptr) {
                        out_channels[c][s] = 0.0f;
                    }
                }
                continue;
            }
        }

        const float env_gain = gain * m_env;
        if (!use_time_stretch) {
            const double xfade_start = static_cast<double>(loop_end - xfade_len);
            const bool in_xfade = loop_active && xfade_len > 0 && m_playhead >= xfade_start;
            float w_curr = 1.0f;
            float w_wrap = 0.0f;
            double wrap_pos = static_cast<double>(loop_start);
            if (in_xfade) {
                const float alpha = static_cast<float>((m_playhead - xfade_start) / xfade_len);
                const float angle = std::clamp(alpha, 0.0f, 1.0f) * 1.5707963f;
                w_curr = std::cos(angle);
                w_wrap = std::sin(angle);
                wrap_pos = static_cast<double>(loop_start) + (m_playhead - xfade_start);
            }

            for (int c = 0; c < num_channels; ++c) {
                if (out_channels[c] == nullptr) {
                    continue;
                }
                const auto& src = slot->audio[static_cast<size_t>(c % slot_ch)];
                float val = sample_linear(src, m_playhead);
                if (in_xfade) {
                    val = w_curr * val + w_wrap * sample_linear(src, wrap_pos);
                }
                out_channels[c][s] = val * env_gain;
            }
        } else {
            const float w0 = 0.5f * (1.0f - std::cos(kTwoPi * static_cast<float>(m_grain_phase)));
            const float w1 = 1.0f - w0;
            const double gp0 = wrap_pos_fn(m_grain_pos[0]);
            const double gp1 = wrap_pos_fn(m_grain_pos[1]);

            for (int c = 0; c < num_channels; ++c) {
                if (out_channels[c] == nullptr) {
                    continue;
                }
                const auto& src = slot->audio[static_cast<size_t>(c % slot_ch)];
                const float val = w0 * sample_linear(src, gp0) + w1 * sample_linear(src, gp1);
                out_channels[c][s] = val * env_gain;
            }

            m_grain_pos[0] = wrap_pos_fn(m_grain_pos[0] + pitch_step);
            m_grain_pos[1] = wrap_pos_fn(m_grain_pos[1] + pitch_step);
        }

        m_playhead += time_step;
        if (m_playhead >= static_cast<double>(loop_end)) {
            if (loop_active) {
                m_playhead = static_cast<double>(loop_start) +
                             std::fmod(m_playhead - static_cast<double>(loop_start),
                                       static_cast<double>(loop_len));
            } else {
                m_playing.store(false, std::memory_order_relaxed);
                m_playhead = static_cast<double>(loop_start);
            }
        }

        if (use_time_stretch) {
            const double prev_phase = m_grain_phase;
            m_grain_phase += 1.0 / kGrainSize;
            if (prev_phase < 0.5 && m_grain_phase >= 0.5) {
                m_grain_pos[1] = m_playhead;
            }
            if (m_grain_phase >= 1.0) {
                m_grain_phase -= 1.0;
                m_grain_pos[0] = m_playhead;
            }
        } else {
            m_grain_pos[0] = m_playhead;
            m_grain_pos[1] = m_playhead;
            m_grain_phase = 0.0;
        }
    }

    m_playhead_norm.store(
        std::clamp(static_cast<float>(m_playhead / static_cast<double>(n_total)), 0.0f, 1.0f),
        std::memory_order_relaxed);

    if (static_cast<int>(m_zero_scratch.size()) >= num_samples) {
        for (int c = 0; c < num_channels; ++c) {
            if (out_channels[c] != nullptr) {
                m_sentry.process_channel(out_channels[c], m_zero_scratch.data(), num_samples, 1.0f,
                                         0.0f);
            }
        }
    }
}

}  // namespace netbender::sampler
