#include "WaveformView.h"
#include "../ui/UIConstants.h"

#include <algorithm>
#include <cmath>

namespace {

auto spectrum_colour(float v, bool inside_loop) -> juce::Colour {
    const float t = std::pow(std::clamp(v, 0.0f, 1.0f), 1.15f);
    juce::Colour c;
    if (t < 0.35f) {
        c = UIConstants::Canvas::inner_canvas.interpolatedWith(UIConstants::Colors::deep_violet,
                                                               t / 0.35f);
    } else if (t < 0.68f) {
        c = UIConstants::Colors::deep_violet.interpolatedWith(UIConstants::Colors::violet,
                                                              (t - 0.35f) / 0.33f);
    } else if (t < 0.90f) {
        c = UIConstants::Colors::violet.interpolatedWith(UIConstants::Colors::lilac,
                                                         (t - 0.68f) / 0.22f);
    } else {
        c = UIConstants::Colors::lilac.interpolatedWith(UIConstants::Colors::magenta,
                                                        (t - 0.90f) / 0.10f);
    }
    return inside_loop ? c : c.withMultipliedBrightness(0.38f);
}

}  // namespace

auto SamplerWaveformView::setSlotIndex(int idx) -> void {
    m_slotIndex = idx;
    repaint();
}

auto SamplerWaveformView::setCompactMode(bool compact) -> void {
    m_compactMode = compact;
    repaint();
}

auto SamplerWaveformView::setSelectedForGen(bool selected) -> void {
    if (m_selectedForGen != selected) {
        m_selectedForGen = selected;
        repaint();
    }
}

auto SamplerWaveformView::setGenerating(bool generating) -> void {
    if (m_generating != generating) {
        m_generating = generating;
        repaint();
    }
}

auto SamplerWaveformView::setFocused(bool focused) -> void {
    if (m_focused != focused) {
        m_focused = focused;
        repaint();
    }
}

auto SamplerWaveformView::setDisplayMode(DisplayMode mode) -> void {
    if (m_displayMode != mode) {
        m_displayMode = mode;
        repaint();
    }
}

auto SamplerWaveformView::regenBadgeBounds() const -> juce::Rectangle<float> {
    return {static_cast<float>(getWidth()) - 76.0f, 8.0f, 68.0f, 22.0f};
}

auto SamplerWaveformView::playBadgeBounds() const -> juce::Rectangle<float> {
    return {static_cast<float>(getWidth()) - 76.0f, static_cast<float>(getHeight()) - 30.0f, 68.0f,
            22.0f};
}

auto SamplerWaveformView::specBadgeBounds() const -> juce::Rectangle<float> {
    if (m_compactMode) {
        return {8.0f, static_cast<float>(getHeight()) - 30.0f, 54.0f, 22.0f};
    }
    return {static_cast<float>(getWidth()) - 72.0f, 8.0f, 64.0f, 22.0f};
}

auto SamplerWaveformView::rebuildSpecImage() -> void {
    if (!m_slot || m_slot->spectrogram.empty()) {
        m_specImage = {};
        return;
    }
    constexpr int cols = netbender::sampler::SampleSlot::k_spec_cols;
    constexpr int bins = netbender::sampler::SampleSlot::k_spec_bins;
    if (static_cast<int>(m_slot->spectrogram.size()) < cols * bins) {
        m_specImage = {};
        return;
    }

    if (!m_specImage.isValid() || m_specImage.getWidth() != cols ||
        m_specImage.getHeight() != bins) {
        m_specImage = juce::Image(juce::Image::ARGB, cols, bins, false);
    }

    juce::Image::BitmapData bmp(m_specImage, juce::Image::BitmapData::writeOnly);
    for (int c = 0; c < cols; ++c) {
        const float norm_x = (static_cast<float>(c) + 0.5f) / static_cast<float>(cols);
        const bool inside = (norm_x >= m_loop_start && norm_x <= m_loop_end);
        for (int b = 0; b < bins; ++b) {
            const float val = m_slot->spectrogram[static_cast<size_t>(c * bins + b)];
            const int row = (bins - 1) - b;
            bmp.setPixelColour(c, row, spectrum_colour(val, inside));
        }
    }
}

auto SamplerWaveformView::setSlot(std::shared_ptr<const netbender::sampler::SampleSlot> slot)
    -> void {
    m_slot = std::move(slot);
    rebuildSpecImage();
    repaint();
}

auto SamplerWaveformView::setPlayhead(float norm, bool playing) -> void {
    if (std::abs(norm - m_playhead) > 1e-4f || playing != m_playing) {
        m_playhead = norm;
        m_playing = playing;
        repaint();
    }
}

auto SamplerWaveformView::setLoopRegion(float start_norm, float end_norm) -> void {
    m_loop_start = std::clamp(start_norm, 0.0f, 0.99f);
    m_loop_end = std::clamp(end_norm, m_loop_start + 0.01f, 1.0f);
    rebuildSpecImage();
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
    const int v_divs = m_compactMode ? 4 : 8;
    for (int i = 1; i < v_divs; ++i) {
        const float x = bounds.getX() +
                        (bounds.getWidth() * static_cast<float>(i)) / static_cast<float>(v_divs);
        g.drawVerticalLine(static_cast<int>(x), bounds.getY() + 4.0f, bounds.getBottom() - 4.0f);
    }

    // Active loop region shading
    const float lx0 = bounds.getX() + m_loop_start * bounds.getWidth();
    const float lx1 = bounds.getX() + m_loop_end * bounds.getWidth();
    g.setColour(UIConstants::Colors::violet.withAlpha(0.14f));
    g.fillRect(lx0, bounds.getY() + 2.0f, std::max(2.0f, lx1 - lx0), bounds.getHeight() - 4.0f);

    if (m_slot && m_displayMode == DisplayMode::Spectrum && m_specImage.isValid()) {
        g.setOpacity(1.0f);
        g.setImageResamplingQuality(juce::Graphics::mediumResamplingQuality);
        g.drawImage(m_specImage, bounds.reduced(1.0f), juce::RectanglePlacement::stretchToFit);
    } else if (m_slot && !m_slot->waveform_peaks.empty()) {
        const int bins = static_cast<int>(m_slot->waveform_peaks.size());
        const float bin_w = bounds.getWidth() / static_cast<float>(bins);
        const float half_h = (bounds.getHeight() - (m_compactMode ? 28.0f : 16.0f)) * 0.5f;

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
        g.setFont(juce::FontOptions(m_compactMode ? UIConstants::Typography::font_body
                                                  : UIConstants::Typography::font_heading));
        const juce::String empty_msg =
            m_generating
                ? ("Generating Slot " + juce::String(m_slotIndex + 1) + "...")
                : (m_compactMode ? ("Slot " + juce::String(m_slotIndex + 1) + " empty")
                                 : "Load an SA3 .nbsampler.json bundle and press GENERATE");
        g.drawText(empty_msg, getLocalBounds(), juce::Justification::centred);
    }

    // Loop Start / End handles (full view)
    if (!m_compactMode) {
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
        const auto spec_r = specBadgeBounds();
        const bool is_spec = (m_displayMode == DisplayMode::Spectrum);
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
    if (m_compactMode) {
        juce::String badge = "SLOT " + juce::String(m_slotIndex + 1);
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

        const auto regen_r = regenBadgeBounds();
        g.setColour(m_selectedForGen ? UIConstants::Colors::violet
                                     : UIConstants::Colors::surface_card);
        g.fillRect(regen_r);
        g.setColour(UIConstants::Colors::border_medium);
        g.drawRect(regen_r, 1.0f);
        g.setColour(m_selectedForGen ? UIConstants::Colors::text_white
                                     : UIConstants::Colors::text_muted);
        g.drawText(m_selectedForGen ? "[x] GEN" : "[ ] GEN", regen_r.toNearestInt(),
                   juce::Justification::centred);

        if (m_slot && m_slot->num_samples > 0) {
            const auto play_r = playBadgeBounds();
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
    } else if (m_compactMode && m_focused) {
        border_col = UIConstants::Colors::violet;
        border_w = 1.5f;
    }
    g.setColour(border_col);
    g.drawRect(bounds.reduced(0.5f), border_w);
}

auto SamplerWaveformView::mouseDown(const juce::MouseEvent& e) -> void {
    if (e.mods.isPopupMenu()) {
        if (onRightClick) {
            onRightClick(m_slotIndex);
        }
        return;
    }

    const auto pt = e.position;
    if (m_slot && m_slot->num_samples > 0 && specBadgeBounds().contains(pt)) {
        m_displayMode = (m_displayMode == DisplayMode::Waveform) ? DisplayMode::Spectrum
                                                                 : DisplayMode::Waveform;
        if (onDisplayModeChanged) {
            onDisplayModeChanged(m_slotIndex, m_displayMode);
        }
        repaint();
        return;
    }

    if (m_compactMode) {
        if (regenBadgeBounds().contains(pt)) {
            if (onToggleRegen) {
                onToggleRegen(m_slotIndex);
            }
            return;
        }
        if (m_slot && m_slot->num_samples > 0 && playBadgeBounds().contains(pt)) {
            if (onTogglePlay) {
                onTogglePlay(m_slotIndex);
            }
            return;
        }
        if (onFocusRequested) {
            onFocusRequested(m_slotIndex);
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
    if (m_compactMode || m_drag == DragTarget::None) {
        return;
    }
    const float norm = std::clamp(static_cast<float>(e.x) / std::max(1, getWidth()), 0.0f, 1.0f);
    if (m_drag == DragTarget::StartHandle) {
        m_loop_start = std::min(norm, m_loop_end - 0.02f);
    } else if (m_drag == DragTarget::EndHandle) {
        m_loop_end = std::max(norm, m_loop_start + 0.02f);
    }
    if (onLoopChanged) {
        onLoopChanged(m_loop_start, m_loop_end);
    }
    repaint();
}

auto SamplerWaveformView::mouseUp(const juce::MouseEvent&) -> void {
    m_drag = DragTarget::None;
}
