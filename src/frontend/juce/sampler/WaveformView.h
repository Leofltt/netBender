#pragma once

#include <JuceHeader.h>
#include "SamplerTypes.h"

#include <functional>
#include <memory>

/**
 * Interactive waveform & spectrogram display for a single `SampleSlot`.
 * Supports compact grid tile mode (with Play/Stop, Regen, and Spec/Wave pills)
 * and full focus mode (with draggable loop start/end handles).
 */
class SamplerWaveformView final : public juce::Component {
public:
    enum class DisplayMode { Waveform, Spectrum };

    std::function<void(float, float)> on_loop_changed;
    std::function<void(int)> on_focus_requested;
    std::function<void(int)> on_toggle_play;
    std::function<void(int)> on_toggle_regen;
    std::function<void(int, DisplayMode)> on_display_mode_changed;
    std::function<void(int)> on_right_click;

    auto set_slot_index(int idx) -> void;
    auto set_compact_mode(bool compact) -> void;
    auto set_selected_for_gen(bool selected) -> void;
    auto set_generating(bool generating) -> void;
    auto set_focused(bool focused) -> void;
    auto set_display_mode(DisplayMode mode) -> void;
    [[nodiscard]] auto get_display_mode() const noexcept -> DisplayMode { return m_display_mode; }

    auto set_slot(std::shared_ptr<const netbender::sampler::SampleSlot> slot) -> void;
    auto set_playhead(float norm, bool playing) -> void;
    auto set_loop_region(float start_norm, float end_norm) -> void;

    auto paint(juce::Graphics& g) -> void override;
    auto mouseDown(const juce::MouseEvent& e) -> void override;
    auto mouseDrag(const juce::MouseEvent& e) -> void override;
    auto mouseUp(const juce::MouseEvent& e) -> void override;

private:
    [[nodiscard]] auto regen_badge_bounds() const -> juce::Rectangle<float>;
    [[nodiscard]] auto play_badge_bounds() const -> juce::Rectangle<float>;
    [[nodiscard]] auto spec_badge_bounds() const -> juce::Rectangle<float>;
    auto rebuild_spec_image() -> void;

    enum class DragTarget { None, StartHandle, EndHandle };
    int m_slot_index{0};
    bool m_compact_mode{false};
    bool m_selected_for_gen{true};
    bool m_generating{false};
    bool m_focused{false};
    DisplayMode m_display_mode{DisplayMode::Waveform};
    std::shared_ptr<const netbender::sampler::SampleSlot> m_slot;
    juce::Image m_spec_image;
    float m_playhead{0.0f};
    bool m_playing{false};
    float m_loop_start{0.0f};
    float m_loop_end{1.0f};
    DragTarget m_drag{DragTarget::None};
};
