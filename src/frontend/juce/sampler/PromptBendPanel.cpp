#include "PromptBendPanel.h"
#include "../ui/UIConstants.h"

#include <algorithm>
#include <cmath>

// =============================================================================
// MorphPad
// =============================================================================

auto PromptBendPanel::MorphPad::set_xy(float x, float y) -> void {
    m_x = std::clamp(x, 0.0f, 1.0f);
    m_y = std::clamp(y, 0.0f, 1.0f);
    repaint();
}

auto PromptBendPanel::MorphPad::set_active_corners(const std::array<bool, 4>& active) -> void {
    if (m_active_corners != active) {
        m_active_corners = active;
        repaint();
    }
}

auto PromptBendPanel::MorphPad::update_from_event(const juce::MouseEvent& e) -> void {
    const float w = std::max(1.0f, static_cast<float>(getWidth()));
    const float h = std::max(1.0f, static_cast<float>(getHeight()));
    m_x = std::clamp(static_cast<float>(e.x) / w, 0.0f, 1.0f);
    m_y = std::clamp(static_cast<float>(e.y) / h, 0.0f, 1.0f);
    if (on_changed) {
        on_changed(m_x, m_y);
    }
    repaint();
}

auto PromptBendPanel::MorphPad::mouseDown(const juce::MouseEvent& e) -> void {
    update_from_event(e);
}

auto PromptBendPanel::MorphPad::mouseDrag(const juce::MouseEvent& e) -> void {
    update_from_event(e);
}

auto PromptBendPanel::MorphPad::mouseUp(const juce::MouseEvent&) -> void {
    if (on_drag_end) {
        on_drag_end();
    }
}

auto PromptBendPanel::MorphPad::paint(juce::Graphics& g) -> void {
    const auto bounds = getLocalBounds().toFloat();
    g.setColour(UIConstants::Canvas::inner_canvas);
    g.fillRect(bounds);

    // Subtle 4x4 grid
    g.setColour(UIConstants::Canvas::grid_minor);
    for (int i = 1; i < 4; ++i) {
        const float f = static_cast<float>(i) / 4.0f;
        g.drawHorizontalLine(static_cast<int>(bounds.getY() + f * bounds.getHeight()),
                             bounds.getX() + 2.0f, bounds.getRight() - 2.0f);
        g.drawVerticalLine(static_cast<int>(bounds.getX() + f * bounds.getWidth()),
                           bounds.getY() + 2.0f, bounds.getBottom() - 2.0f);
    }

    // Corner labels A, B, C, D (lilac if active, dimmed if empty)
    g.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    auto draw_corner = [&](int idx, const char* label, int x, int y, juce::Justification just) {
        g.setColour(m_active_corners[static_cast<size_t>(idx)]
                        ? UIConstants::Colors::lilac
                        : UIConstants::Colors::text_disabled);
        g.drawText(label, x, y, 16, 14, just);
    };
    draw_corner(0, "A", 6, 4, juce::Justification::centredLeft);
    draw_corner(1, "B", getWidth() - 22, 4, juce::Justification::centredRight);
    draw_corner(2, "C", 6, getHeight() - 18, juce::Justification::centredLeft);
    draw_corner(3, "D", getWidth() - 22, getHeight() - 18, juce::Justification::centredRight);

    // Crosshair & glow
    const float px = bounds.getX() + m_x * bounds.getWidth();
    const float py = bounds.getY() + m_y * bounds.getHeight();
    g.setColour(UIConstants::Colors::violet.withAlpha(0.35f));
    g.drawHorizontalLine(static_cast<int>(py), bounds.getX() + 1.0f, bounds.getRight() - 1.0f);
    g.drawVerticalLine(static_cast<int>(px), bounds.getY() + 1.0f, bounds.getBottom() - 1.0f);

    g.setColour(UIConstants::Colors::lilac);
    g.fillEllipse(px - 5.0f, py - 5.0f, 10.0f, 10.0f);
    g.setColour(UIConstants::Colors::text_white);
    g.drawEllipse(px - 5.0f, py - 5.0f, 10.0f, 10.0f, 1.2f);

    g.setColour(UIConstants::Colors::border_medium);
    g.drawRect(bounds, 1.0f);
}

// =============================================================================
// PromptBendPanel
// =============================================================================

PromptBendPanel::PromptBendPanel() {
    m_enable_toggle.setToggleState(true, juce::dontSendNotification);
    m_enable_toggle.onClick = [this]() { notify_bend_changed(); };
    addAndMakeVisible(m_enable_toggle);

    m_reset_button.onClick = [this]() {
        reset_all();
        notify_bend_changed();
    };
    addAndMakeVisible(m_reset_button);

    m_morph_pad.on_drag_end = [this]() { notify_bend_changed(); };
    addAndMakeVisible(m_morph_pad);

    m_law_label.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
    m_law_label.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    addAndMakeVisible(m_law_label);

    m_law_combo.addItem("SLERP", 1);
    m_law_combo.addItem("Linear", 2);
    m_law_combo.addItem("TokenSplice", 3);
    m_law_combo.addItem("FreqSplit", 4);
    m_law_combo.setSelectedId(1, juce::dontSendNotification);
    m_law_combo.onChange = [this]() { notify_bend_changed(); };
    addAndMakeVisible(m_law_combo);

    auto setup_slider = [this](juce::Label& lbl, juce::Slider& sl, double min_v, double max_v,
                               double step, double init_v) {
        lbl.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
        lbl.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        addAndMakeVisible(lbl);
        sl.setSliderStyle(juce::Slider::LinearHorizontal);
        sl.setTextBoxStyle(juce::Slider::TextBoxRight, false, 46, 18);
        sl.setRange(min_v, max_v, step);
        sl.setValue(init_v, juce::dontSendNotification);
        sl.onDragEnd = [this]() { notify_bend_changed(); };
        addAndMakeVisible(sl);
    };

    setup_slider(m_jitter_label, m_jitter_slider, 0.0, 1.0, 0.01, 0.0);
    setup_slider(m_tok_drop_label, m_tok_drop_slider, 0.0, 0.85, 0.01, 0.0);
    setup_slider(m_quant_label, m_quant_slider, 0.0, 32.0, 1.0, 0.0);
    setup_slider(m_fold_label, m_fold_slider, 0.0, 2.0, 0.02, 0.0);
    setup_slider(m_dur_lie_label, m_dur_lie_slider, 0.0, 20.0, 0.5, 0.0);
    setup_slider(m_step_morph_label, m_step_morph_slider, 0.0, 1.0, 0.02, 0.0);
    setup_slider(m_t_warp_label, m_t_warp_slider, 0.25, 2.5, 0.05, 1.0);
    setup_slider(m_noise_label, m_noise_slider, 0.80, 1.20, 0.01, 1.0);
    setup_slider(m_cfg_label, m_cfg_slider, -3.0, 5.0, 0.1, 1.0);
    setup_slider(m_duration_label, m_duration_slider, 1.0, 11.5, 0.5, 6.0);
    setup_slider(m_steps_label, m_steps_slider, 1.0, 16.0, 1.0, 8.0);
}

auto PromptBendPanel::set_active_corners(const std::array<bool, 4>& active) -> void {
    m_morph_pad.set_active_corners(active);
}

auto PromptBendPanel::notify_bend_changed() -> void {
    if (on_state_edited) {
        on_state_edited();
    }
    if (m_enable_toggle.getToggleState() && on_bend_changed) {
        on_bend_changed();
    }
}

auto PromptBendPanel::get_state(const std::array<std::string, 4>& prompts) const -> State {
    return State{build_config(prompts), get_cfg_scale(), get_duration_sec(), get_steps()};
}

auto PromptBendPanel::load_state(const State& state) -> void {
    const auto& cfg = state.config;
    m_enable_toggle.setToggleState(cfg.enabled, juce::dontSendNotification);
    m_morph_pad.set_xy(cfg.morph_x, cfg.morph_y);

    using Law = netbender::sampler::PromptMorphLaw;
    int law_id = 1;
    if (cfg.morph_law == Law::Linear) {
        law_id = 2;
    } else if (cfg.morph_law == Law::TokenSplice) {
        law_id = 3;
    } else if (cfg.morph_law == Law::FreqSplit) {
        law_id = 4;
    }
    m_law_combo.setSelectedId(law_id, juce::dontSendNotification);

    m_duration_slider.setValue(state.duration_sec, juce::dontSendNotification);
    m_steps_slider.setValue(state.steps, juce::dontSendNotification);
    m_jitter_slider.setValue(cfg.latent_jitter, juce::dontSendNotification);
    m_tok_drop_slider.setValue(cfg.token_dropout, juce::dontSendNotification);
    m_quant_slider.setValue(static_cast<double>(cfg.emb_quantize_levels),
                            juce::dontSendNotification);
    m_fold_slider.setValue(cfg.emb_wavefold, juce::dontSendNotification);
    m_dur_lie_slider.setValue(cfg.duration_lie_sec, juce::dontSendNotification);
    m_step_morph_slider.setValue(cfg.step_morph_ab, juce::dontSendNotification);
    m_t_warp_slider.setValue(cfg.schedule_warp, juce::dontSendNotification);
    m_noise_slider.setValue(std::clamp(cfg.pingpong_noise_scale, 0.80f, 1.20f),
                            juce::dontSendNotification);
    m_cfg_slider.setValue(state.cfg_scale, juce::dontSendNotification);
}

auto PromptBendPanel::reset_all() -> void {
    m_morph_pad.set_xy(0.0f, 0.0f);
    m_law_combo.setSelectedId(1, juce::dontSendNotification);
    m_jitter_slider.setValue(0.0, juce::dontSendNotification);
    m_tok_drop_slider.setValue(0.0, juce::dontSendNotification);
    m_quant_slider.setValue(0.0, juce::dontSendNotification);
    m_fold_slider.setValue(0.0, juce::dontSendNotification);
    m_dur_lie_slider.setValue(0.0, juce::dontSendNotification);
    m_step_morph_slider.setValue(0.0, juce::dontSendNotification);
    m_t_warp_slider.setValue(1.0, juce::dontSendNotification);
    m_noise_slider.setValue(1.0, juce::dontSendNotification);
    m_cfg_slider.setValue(1.0, juce::dontSendNotification);
}

auto PromptBendPanel::is_bend_enabled() const -> bool {
    return m_enable_toggle.getToggleState();
}

auto PromptBendPanel::get_cfg_scale() const -> float {
    return static_cast<float>(m_cfg_slider.getValue());
}

auto PromptBendPanel::get_duration_sec() const -> float {
    return static_cast<float>(m_duration_slider.getValue());
}

auto PromptBendPanel::get_steps() const -> int {
    return static_cast<int>(m_steps_slider.getValue());
}

auto PromptBendPanel::set_duration_range(double min_v, double max_v) -> void {
    m_duration_slider.setRange(min_v, max_v, 0.5);
}

auto PromptBendPanel::build_config(const std::array<std::string, 4>& prompts) const
    -> netbender::sampler::PromptBendingConfig {
    netbender::sampler::PromptBendingConfig cfg;
    cfg.enabled = m_enable_toggle.getToggleState();
    cfg.prompts = prompts;
    cfg.morph_x = m_morph_pad.get_x_norm();
    cfg.morph_y = m_morph_pad.get_y_norm();

    using Law = netbender::sampler::PromptMorphLaw;
    const int lid = m_law_combo.getSelectedId();
    if (lid == 2) {
        cfg.morph_law = Law::Linear;
    } else if (lid == 3) {
        cfg.morph_law = Law::TokenSplice;
    } else if (lid == 4) {
        cfg.morph_law = Law::FreqSplit;
    } else {
        cfg.morph_law = Law::Slerp;
    }

    cfg.latent_jitter = static_cast<float>(m_jitter_slider.getValue());
    cfg.token_dropout = static_cast<float>(m_tok_drop_slider.getValue());
    cfg.emb_quantize_levels = static_cast<int>(m_quant_slider.getValue());
    cfg.emb_wavefold = static_cast<float>(m_fold_slider.getValue());
    cfg.duration_lie_sec = static_cast<float>(m_dur_lie_slider.getValue());
    cfg.step_morph_ab = static_cast<float>(m_step_morph_slider.getValue());
    cfg.schedule_warp = static_cast<float>(m_t_warp_slider.getValue());
    cfg.pingpong_noise_scale = static_cast<float>(m_noise_slider.getValue());
    return cfg;
}

auto PromptBendPanel::paint(juce::Graphics& g) -> void {
    const auto bounds = getLocalBounds().toFloat();
    g.setColour(UIConstants::Colors::surface_panel);
    g.fillRect(bounds);
    g.setColour(UIConstants::Colors::border_subtle);
    g.drawRect(bounds, 1.0f);
}

auto PromptBendPanel::resized() -> void {
    auto area = getLocalBounds().reduced(8);

    auto top = area.removeFromTop(24);
    m_enable_toggle.setBounds(top.removeFromLeft(95));
    m_reset_button.setBounds(top.removeFromRight(64));

    area.removeFromTop(6);
    const int pad_h = std::clamp(area.getHeight() / 4, 84, 130);
    m_morph_pad.setBounds(area.removeFromTop(pad_h));

    area.removeFromTop(6);
    auto law_row = area.removeFromTop(22);
    m_law_label.setBounds(law_row.removeFromLeft(56));
    m_law_combo.setBounds(law_row);

    auto place_slider = [&](juce::Label& lbl, juce::Slider& sl) {
        area.removeFromTop(3);
        auto r = area.removeFromTop(19);
        lbl.setBounds(r.removeFromLeft(66));
        sl.setBounds(r);
    };
    area.removeFromTop(3);
    place_slider(m_duration_label, m_duration_slider);
    place_slider(m_steps_label, m_steps_slider);
    place_slider(m_jitter_label, m_jitter_slider);
    place_slider(m_tok_drop_label, m_tok_drop_slider);
    place_slider(m_quant_label, m_quant_slider);
    place_slider(m_fold_label, m_fold_slider);
    place_slider(m_dur_lie_label, m_dur_lie_slider);
    place_slider(m_step_morph_label, m_step_morph_slider);
    place_slider(m_t_warp_label, m_t_warp_slider);
    place_slider(m_noise_label, m_noise_slider);
    place_slider(m_cfg_label, m_cfg_slider);
}
