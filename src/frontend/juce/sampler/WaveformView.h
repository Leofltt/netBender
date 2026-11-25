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

    std::function<void(float, float)> onLoopChanged;
    std::function<void(int)> onFocusRequested;
    std::function<void(int)> onTogglePlay;
    std::function<void(int)> onToggleRegen;
    std::function<void(int, DisplayMode)> onDisplayModeChanged;
    std::function<void(int)> onRightClick;

    auto setSlotIndex(int idx) -> void;
    auto setCompactMode(bool compact) -> void;
    auto setSelectedForGen(bool selected) -> void;
    auto setGenerating(bool generating) -> void;
    auto setFocused(bool focused) -> void;
    auto setDisplayMode(DisplayMode mode) -> void;
    [[nodiscard]] auto getDisplayMode() const noexcept -> DisplayMode { return m_displayMode; }

    auto setSlot(std::shared_ptr<const netbender::sampler::SampleSlot> slot) -> void;
    auto setPlayhead(float norm, bool playing) -> void;
    auto setLoopRegion(float start_norm, float end_norm) -> void;

    auto paint(juce::Graphics& g) -> void override;
    auto mouseDown(const juce::MouseEvent& e) -> void override;
    auto mouseDrag(const juce::MouseEvent& e) -> void override;
    auto mouseUp(const juce::MouseEvent& e) -> void override;

private:
    [[nodiscard]] auto regenBadgeBounds() const -> juce::Rectangle<float>;
    [[nodiscard]] auto playBadgeBounds() const -> juce::Rectangle<float>;
    [[nodiscard]] auto specBadgeBounds() const -> juce::Rectangle<float>;
    auto rebuildSpecImage() -> void;

    enum class DragTarget { None, StartHandle, EndHandle };
    int m_slotIndex{0};
    bool m_compactMode{false};
    bool m_selectedForGen{true};
    bool m_generating{false};
    bool m_focused{false};
    DisplayMode m_displayMode{DisplayMode::Waveform};
    std::shared_ptr<const netbender::sampler::SampleSlot> m_slot;
    juce::Image m_specImage;
    float m_playhead{0.0f};
    bool m_playing{false};
    float m_loop_start{0.0f};
    float m_loop_end{1.0f};
    DragTarget m_drag{DragTarget::None};
};
