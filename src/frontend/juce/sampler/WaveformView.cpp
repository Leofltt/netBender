#include "WaveformView.h"
#include "../ui/UIConstants.h"

#include <algorithm>
#include <cmath>

namespace {

auto spectrum_colour(float v, bool inside_loop) -> juce::Colour {
    const float gated = std::clamp((v - 0.10f) / 0.90f, 0.0f, 1.0f);
    const float t = std::pow(gated, 1.20f);
    const juce::Colour k_slate_indigo{0xff1b2240};
    const juce::Colour k_dusk_violet{0xff4c3d7a};
    const juce::Colour k_warm_amber{0xffb57b52};
    const juce::Colour k_pale_sand{0xfff3dfb0};

    juce::Colour c;
    if (t < 0.32f) {
        c = UIConstants::Canvas::inner_canvas.interpolatedWith(k_slate_indigo, t / 0.32f);
    } else if (t < 0.62f) {
        c = k_slate_indigo.interpolatedWith(k_dusk_violet, (t - 0.32f) / 0.30f);
    } else if (t < 0.85f) {
        c = k_dusk_violet.interpolatedWith(k_warm_amber, (t - 0.62f) / 0.23f);
    } else {
        c = k_warm_amber.interpolatedWith(k_pale_sand, (t - 0.85f) / 0.15f);
    }
    return inside_loop ? c : c.withMultipliedBrightness(0.35f);
}

}  // namespace

auto SamplerWaveformView::set_slot_index(int idx) -> void {
    m_slot_index = idx;
    repaint();
}

auto SamplerWaveformView::set_compact_mode(bool compact) -> void {
    m_compact_mode = compact;
    repaint();
}

auto SamplerWaveformView::set_selected_for_gen(bool selected) -> void {
    if (m_selected_for_gen != selected) {
        m_selected_for_gen = selected;
        repaint();
    }
}

auto SamplerWaveformView::set_generating(bool generating) -> void {
    if (m_generating != generating) {
        m_generating = generating;
        repaint();
    }
}

auto SamplerWaveformView::set_focused(bool focused) -> void {
    if (m_focused != focused) {
        m_focused = focused;
        repaint();
    }
}

auto SamplerWaveformView::set_display_mode(DisplayMode mode) -> void {
    if (m_display_mode != mode) {
        m_display_mode = mode;
        repaint();
    }
}

auto SamplerWaveformView::regen_badge_bounds() const -> juce::Rectangle<float> {
    return {static_cast<float>(getWidth()) - 76.0f, 8.0f, 68.0f, 22.0f};
}

auto SamplerWaveformView::play_badge_bounds() const -> juce::Rectangle<float> {
    return {static_cast<float>(getWidth()) - 76.0f, static_cast<float>(getHeight()) - 30.0f, 68.0f,
            22.0f};
}

auto SamplerWaveformView::spec_badge_bounds() const -> juce::Rectangle<float> {
    if (m_compact_mode) {
        return {8.0f, static_cast<float>(getHeight()) - 30.0f, 54.0f, 22.0f};
    }
    return {static_cast<float>(getWidth()) - 72.0f, 8.0f, 64.0f, 22.0f};
}

auto SamplerWaveformView::rebuild_spec_image() -> void {
    if (!m_slot || m_slot->spectrogram.empty()) {
        m_spec_image = {};
        return;
    }
    constexpr int k_cols = netbender::sampler::SampleSlot::k_spec_cols;
    constexpr int k_bins = netbender::sampler::SampleSlot::k_spec_bins;
    if (static_cast<int>(m_slot->spectrogram.size()) < k_cols * k_bins) {
        m_spec_image = {};
        return;
    }

    if (!m_spec_image.isValid() || m_spec_image.getWidth() != k_cols ||
        m_spec_image.getHeight() != k_bins) {
        m_spec_image = juce::Image(juce::Image::ARGB, k_cols, k_bins, false);
    }

    juce::Image::BitmapData bmp(m_spec_image, juce::Image::BitmapData::writeOnly);
    for (int c = 0; c < k_cols; ++c) {
        const float norm_x = (static_cast<float>(c) + 0.5f) / static_cast<float>(k_cols);
        const bool inside = (norm_x >= m_loop_start && norm_x <= m_loop_end);
        for (int b = 0; b < k_bins; ++b) {
            const float val = m_slot->spectrogram[static_cast<size_t>(c * k_bins + b)];
            const int row = (k_bins - 1) - b;
            bmp.setPixelColour(c, row, spectrum_colour(val, inside));
        }
    }
}

auto SamplerWaveformView::set_slot(std::shared_ptr<const netbender::sampler::SampleSlot> slot)
    -> void {
    m_slot = std::move(slot);
    rebuild_spec_image();
    repaint();
}

auto SamplerWaveformView::set_playhead(float norm, bool playing) -> void {
    if (std::abs(norm - m_playhead) > 1e-4f || playing != m_playing) {
        m_playhead = norm;
        m_playing = playing;
        repaint();
    }
}

auto SamplerWaveformView::set_loop_region(float start_norm, float end_norm) -> void {
    m_loop_start = std::clamp(start_norm, 0.0f, 0.99f);
    m_loop_end = std::clamp(end_norm, m_loop_start + 0.01f, 1.0f);
    rebuild_spec_image();
    repaint();
}

auto SamplerWaveformView::paint(juce::Graphics& g) -> void {
    const auto bounds = getLocalBounds().toFloat();
    g.setColour(UIConstants::Canvas::inner_canvas);
    g.fillRect(bounds);

    // Subtle grid lines
    g.setColour(UIConstants::Canvas::grid_minor);
    const float mid_y = bounds.getCentreY();
    g.drawHorizontalLine(static_cast<int>(mid_y), bounds.getX() + 4.0f, bounds.getRight() - 4.0f);
    const int v_divs = m_compact_mode ? 4 : 8;
    for (int i = 1; i < v_divs; ++i) {
        const float x = bounds.getX() +
                        (bounds.getWidth() * static_cast<float>(i)) / static_cast<float>(v_divs);
        g.drawVerticalLine(static_cast<int>(x), bounds.getY() + 4.0f, bounds.getBottom() - 4.0f);
    }

    // Loop region bounds (dim regions outside [lx0, lx1] so active area stays pure inner_canvas)
    const float lx0 = bounds.getX() + m_loop_start * bounds.getWidth();
    const float lx1 = bounds.getX() + m_loop_end * bounds.getWidth();
    g.setColour(juce::Colours::black.withAlpha(0.35f));
    if (lx0 > bounds.getX()) {
        g.fillRect(bounds.getX(), bounds.getY() + 1.0f, lx0 - bounds.getX(),
                   bounds.getHeight() - 2.0f);
    }
    if (lx1 < bounds.getRight()) {
        g.fillRect(lx1, bounds.getY() + 1.0f, bounds.getRight() - lx1, bounds.getHeight() - 2.0f);
    }

    if (m_slot && m_display_mode == DisplayMode::Spectrum && m_spec_image.isValid()) {
        g.setOpacity(1.0f);
        g.setImageResamplingQuality(juce::Graphics::mediumResamplingQuality);
        g.drawImage(m_spec_image, bounds.reduced(1.0f), juce::RectanglePlacement::stretchToFit);
    } else if (m_slot && !m_slot->waveform_peaks.empty()) {
        const int bins = static_cast<int>(m_slot->waveform_peaks.size());
        const float bin_w = bounds.getWidth() / static_cast<float>(bins);
        const float half_h = (bounds.getHeight() - (m_compact_mode ? 28.0f : 16.0f)) * 0.5f;

        for (int i = 0; i < bins; ++i) {
            const float norm_x = static_cast<float>(i) / static_cast<float>(bins);
            const bool inside = (norm_x >= m_loop_start && norm_x <= m_loop_end);
            g.setColour(inside ? UIConstants::Colors::lilac
                               : UIConstants::Colors::lavender.withAlpha(0.28f));
            const float amp = std::max(0.015f, m_slot->waveform_peaks[static_cast<size_t>(i)]);
            const float bar_h = amp * half_h * 2.0f;
            const float x = bounds.getX() + static_cast<float>(i) * bin_w;
            g.fillRect(x, mid_y - bar_h * 0.5f, std::max(1.0f, bin_w - 0.5f), bar_h);
        }
    } else {
        g.setColour(UIConstants::Colors::text_muted);
        g.setFont(juce::FontOptions(m_compact_mode ? UIConstants::Typography::font_body
                                                   : UIConstants::Typography::font_heading));
        const juce::String empty_msg =
            m_generating
                ? ("Generating Slot " + juce::String(m_slot_index + 1) + "...")
                : (m_compact_mode ? ("Slot " + juce::String(m_slot_index + 1) + " empty")
                                  : "Load an SA3 .nbsampler.json bundle and press GENERATE");
        g.drawText(empty_msg, getLocalBounds(), juce::Justification::centred);
    }

    // Loop Start / End handles (full view)
    if (!m_compact_mode) {
        g.setColour(UIConstants::Colors::magenta);
        g.drawVerticalLine(static_cast<int>(lx0), bounds.getY(), bounds.getBottom());
        g.fillRect(lx0 - 3.0f, bounds.getY(), 7.0f, 12.0f);

        g.setColour(UIConstants::Colors::lavender);
        g.drawVerticalLine(static_cast<int>(lx1), bounds.getY(), bounds.getBottom());
        g.fillRect(lx1 - 4.0f, bounds.getY(), 7.0f, 12.0f);
    }

    // Playhead
    if (m_playing && m_slot) {
        const float px = bounds.getX() + m_playhead * bounds.getWidth();
        g.setColour(UIConstants::Colors::amber_light);
        g.drawLine(px, bounds.getY(), px, bounds.getBottom(), 2.0f);
    }

    // Spectrum / Waveform toggle badge
    if (m_slot && m_slot->num_samples > 0) {
        const auto spec_r = spec_badge_bounds();
        const bool is_spec = (m_display_mode == DisplayMode::Spectrum);
        g.setColour(is_spec ? UIConstants::Colors::violet
                            : UIConstants::Colors::surface_card.withAlpha(0.90f));
        g.fillRect(spec_r);
        g.setColour(UIConstants::Colors::border_medium);
        g.drawRect(spec_r, 1.0f);
        g.setColour(UIConstants::Colors::text_white);
        g.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        g.drawText(is_spec ? "WAVE" : "SPEC", spec_r.toNearestInt(), juce::Justification::centred);
    }

    // Compact tile overlays (Slot badge, Regen pill, Play/Stop pill)
    if (m_compact_mode) {
        juce::String badge = "SLOT " + juce::String(m_slot_index + 1);
        if (m_slot && m_slot->num_samples > 0) {
            badge += "  #" + juce::String(m_slot->seed) + "  " +
                     juce::String(m_slot->duration_sec, 1) + "s";
        }
        const juce::Rectangle<float> badge_r{8.0f, 8.0f, 175.0f, 22.0f};
        g.setColour(UIConstants::Colors::surface_card.withAlpha(0.90f));
        g.fillRect(badge_r);
        g.setColour(UIConstants::Colors::border_subtle);
        g.drawRect(badge_r, 1.0f);
        g.setColour(UIConstants::Colors::text_white);
        g.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        g.drawText(badge, 14, 8, 165, 22, juce::Justification::centredLeft);

        const auto regen_r = regen_badge_bounds();
        g.setColour(m_selected_for_gen ? UIConstants::Colors::violet
                                       : UIConstants::Colors::surface_card);
        g.fillRect(regen_r);
        g.setColour(UIConstants::Colors::border_medium);
        g.drawRect(regen_r, 1.0f);
        g.setColour(m_selected_for_gen ? UIConstants::Colors::text_white
                                       : UIConstants::Colors::text_muted);
        g.drawText(m_selected_for_gen ? "[x] GEN" : "[ ] GEN", regen_r.toNearestInt(),
                   juce::Justification::centred);

        if (m_slot && m_slot->num_samples > 0) {
            const auto play_r = play_badge_bounds();
            g.setColour(m_playing ? UIConstants::Colors::ruby : UIConstants::Colors::forest_green);
            g.fillRect(play_r);
            g.setColour(UIConstants::Colors::border_medium);
            g.drawRect(play_r, 1.0f);
            g.setColour(UIConstants::Colors::text_white);
            g.drawText(m_playing ? "STOP" : "PLAY", play_r.toNearestInt(),
                       juce::Justification::centred);
        }
    }

    juce::Colour border_col = UIConstants::Canvas::border;
    float border_w = 1.0f;
    if (m_generating) {
        border_col = UIConstants::Colors::amber_light;
        border_w = 2.0f;
    } else if (m_playing) {
        border_col = UIConstants::Colors::lilac;
        border_w = 2.0f;
    } else if (m_compact_mode && m_focused) {
        border_col = UIConstants::Colors::violet;
        border_w = 1.5f;
    }
    g.setColour(border_col);
    g.drawRect(bounds.reduced(0.5f), border_w);
}

auto SamplerWaveformView::mouseDown(const juce::MouseEvent& e) -> void {
    if (e.mods.isPopupMenu()) {
        if (on_right_click) {
            on_right_click(m_slot_index);
        }
        return;
    }

    const auto pt = e.position;
    if (m_slot && m_slot->num_samples > 0 && spec_badge_bounds().contains(pt)) {
        m_display_mode = (m_display_mode == DisplayMode::Waveform) ? DisplayMode::Spectrum
                                                                   : DisplayMode::Waveform;
        if (on_display_mode_changed) {
            on_display_mode_changed(m_slot_index, m_display_mode);
        }
        repaint();
        return;
    }

    if (m_compact_mode) {
        if (regen_badge_bounds().contains(pt)) {
            if (on_toggle_regen) {
                on_toggle_regen(m_slot_index);
            }
            return;
        }
        if (m_slot && m_slot->num_samples > 0 && play_badge_bounds().contains(pt)) {
            if (on_toggle_play) {
                on_toggle_play(m_slot_index);
            }
            return;
        }
        if (on_focus_requested) {
            on_focus_requested(m_slot_index);
        }
        return;
    }

    const float w = std::max(1.0f, static_cast<float>(getWidth()));
    const float x = static_cast<float>(e.x);
    const float sx = m_loop_start * w;
    const float ex = m_loop_end * w;

    if (std::abs(x - sx) <= 12.0f && std::abs(x - sx) <= std::abs(x - ex)) {
        m_drag = DragTarget::StartHandle;
    } else if (std::abs(x - ex) <= 12.0f) {
        m_drag = DragTarget::EndHandle;
    } else {
        m_drag = DragTarget::None;
    }
}

auto SamplerWaveformView::mouseDrag(const juce::MouseEvent& e) -> void {
    if (m_compact_mode || m_drag == DragTarget::None) {
        return;
    }
    const float norm = std::clamp(static_cast<float>(e.x) / std::max(1, getWidth()), 0.0f, 1.0f);
    if (m_drag == DragTarget::StartHandle) {
        m_loop_start = std::min(norm, m_loop_end - 0.02f);
    } else if (m_drag == DragTarget::EndHandle) {
        m_loop_end = std::max(norm, m_loop_start + 0.02f);
    }
    if (on_loop_changed) {
        on_loop_changed(m_loop_start, m_loop_end);
    }
    repaint();
}

auto SamplerWaveformView::mouseUp(const juce::MouseEvent&) -> void {
    m_drag = DragTarget::None;
}
