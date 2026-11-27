#pragma once

#include <JuceHeader.h>
#include "SamplerTypes.h"

#include <array>
#include <functional>
#include <string>

/**
 * Right-hand panel for Phase 3 Axis-A Prompt & Trajectory Bending:
 *   - 2D Semantic Morph Pad (navigates any 1..4 active Prompts A / B / C / D)
 *   - Morph Law selector (SLERP / Linear / TokenSplice / FreqSplit)
 *   - Embedding DSP (Drift + Per-Step toggle, Quantize, Wavefold)
 *   - Trajectory & Guidance bends (Duration Lie, A->B Step Morph, T-Warp, PingPong Noise, CFG)
 */
class PromptBendPanel final : public juce::Component {
public:
    struct State {
        netbender::sampler::PromptBendingConfig config;
        float cfg_scale{1.0f};
        float duration_sec{6.0f};
        int steps{8};
    };

    PromptBendPanel();

    std::function<void()> on_bend_changed;
    std::function<void()> on_state_edited;

    auto set_active_corners(const std::array<bool, 4>& active) -> void;

    [[nodiscard]] auto get_state(const std::array<std::string, 4>& prompts) const -> State;
    auto load_state(const State& state) -> void;

    [[nodiscard]] auto build_config(const std::array<std::string, 4>& prompts) const
        -> netbender::sampler::PromptBendingConfig;
    [[nodiscard]] auto get_cfg_scale() const -> float;
    [[nodiscard]] auto is_bend_enabled() const -> bool;
    [[nodiscard]] auto get_duration_sec() const -> float;
    [[nodiscard]] auto get_steps() const -> int;
    auto set_duration_range(double min_v, double max_v) -> void;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    class MorphPad final : public juce::Component {
    public:
        std::function<void(float, float)> on_changed;
        std::function<void()> on_drag_end;

        auto set_xy(float x, float y) -> void;
        auto set_active_corners(const std::array<bool, 4>& active) -> void;
        [[nodiscard]] auto get_x_norm() const noexcept -> float { return m_x; }
        [[nodiscard]] auto get_y_norm() const noexcept -> float { return m_y; }

        auto paint(juce::Graphics& g) -> void override;
        auto mouseDown(const juce::MouseEvent& e) -> void override;
        auto mouseDrag(const juce::MouseEvent& e) -> void override;
        auto mouseUp(const juce::MouseEvent& e) -> void override;

    private:
        auto update_from_event(const juce::MouseEvent& e) -> void;
        std::array<bool, 4> m_active_corners{true, false, false, false};
        float m_x{0.0f};
        float m_y{0.0f};
    };

    auto reset_all() -> void;
    auto notify_bend_changed() -> void;

    juce::ToggleButton m_enable_toggle{"BEND ON"};
    juce::TextButton m_reset_button{"RESET"};

    MorphPad m_morph_pad;
    juce::Label m_law_label{"", "MORPH"};
    juce::ComboBox m_law_combo;

    juce::Label m_jitter_label{"", "JITTER"};
    juce::Slider m_jitter_slider;
    juce::Label m_tok_drop_label{"", "TOK DROP"};
    juce::Slider m_tok_drop_slider;

    juce::Label m_quant_label{"", "QUANT"};
    juce::Slider m_quant_slider;
    juce::Label m_fold_label{"", "FOLD"};
    juce::Slider m_fold_slider;
    juce::Label m_dur_lie_label{"", "DUR LIE"};
    juce::Slider m_dur_lie_slider;
    juce::Label m_step_morph_label{"", "A->B STEP"};
    juce::Slider m_step_morph_slider;
    juce::Label m_t_warp_label{"", "T-WARP"};
    juce::Slider m_t_warp_slider;
    juce::Label m_noise_label{"", "NOISE"};
    juce::Slider m_noise_slider;
    juce::Label m_cfg_label{"", "CFG"};
    juce::Slider m_cfg_slider;
    juce::Label m_duration_label{"", "DUR (s)"};
    juce::Slider m_duration_slider;
    juce::Label m_steps_label{"", "STEPS"};
    juce::Slider m_steps_slider;
};
