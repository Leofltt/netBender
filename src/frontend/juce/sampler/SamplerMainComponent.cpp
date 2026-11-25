#include "SamplerMainComponent.h"
#include "../ui/UIConstants.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>

namespace {

auto format_stage(netbender::sampler::GenerationProgress::Stage stage, int cur, int tot, double ms,
                  int batch_idx, int batch_total) -> juce::String {
    using Stage = netbender::sampler::GenerationProgress::Stage;
    const juce::String prefix =
        (batch_total > 1 && batch_idx > 0)
            ? ("Slot " + juce::String(batch_idx) + "/" + juce::String(batch_total) + " — ")
            : juce::String();
    switch (stage) {
        case Stage::Idle:
            return "Ready";
        case Stage::Tokenizing:
            return prefix + "Tokenizing prompt...";
        case Stage::EncodingPrompt:
            return prefix + "Encoding T5Gemma prompt...";
        case Stage::Diffusing:
            return prefix + "Diffusing step " + juce::String(cur) + " / " + juce::String(tot);
        case Stage::Decoding:
            return prefix + "Decoding SAME-S audio...";
        case Stage::Done:
            return "Generated " + juce::String(std::max(1, batch_total)) + " slot(s) in " +
                   juce::String(ms, 0) + " ms";
        case Stage::Cancelled:
            return "Cancelled";
        case Stage::Error:
            return "Generation error";
    }
    return "Ready";
}

auto sanitize_filename(const std::string& raw) -> juce::String {
    std::string out;
    out.reserve(std::min<size_t>(raw.size(), 40));
    for (unsigned char c : raw) {
        if (out.size() >= 40) {
            break;
        }
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if ((c == ' ' || c == '-' || c == '_') && !out.empty() && out.back() != '_') {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    return out.empty() ? juce::String("netbender_sample") : juce::String(out);
}

auto spectrum_colour(float v, bool inside_loop) -> juce::Colour {
    const float t = std::pow(std::clamp(v, 0.0f, 1.0f), 0.65f);
    juce::Colour c;
    if (t < 0.20f) {
        c = UIConstants::Canvas::inner_canvas.interpolatedWith(UIConstants::Colors::dark_violet,
                                                               t / 0.20f);
    } else if (t < 0.45f) {
        c = UIConstants::Colors::dark_violet.interpolatedWith(UIConstants::Colors::violet,
                                                              (t - 0.20f) / 0.25f);
    } else if (t < 0.70f) {
        c = UIConstants::Colors::violet.interpolatedWith(UIConstants::Colors::magenta,
                                                         (t - 0.45f) / 0.25f);
    } else if (t < 0.90f) {
        c = UIConstants::Colors::magenta.interpolatedWith(UIConstants::Colors::amber_light,
                                                          (t - 0.70f) / 0.20f);
    } else {
        c = UIConstants::Colors::amber_light.interpolatedWith(UIConstants::Colors::text_white,
                                                              (t - 0.90f) / 0.10f);
    }
    return inside_loop ? c : c.withMultipliedBrightness(0.40f);
}

auto write_slot_audio_to_file(const netbender::sampler::SampleSlot& snap, juce::File dest,
                              bool apply_range, float start_norm, float end_norm, int format_id,
                              juce::String& out_error) -> bool {
    const bool is_aiff = (format_id == 4);
    const juce::String ext = is_aiff ? ".aiff" : ".wav";
    if (dest.getFileExtension().isEmpty()) {
        dest = dest.withFileExtension(ext);
    }
    if (dest.existsAsFile()) {
        dest.deleteFile();
    }

    const int total_samples = snap.num_samples;
    if (total_samples <= 0 || snap.audio.empty()) {
        out_error = "Slot has no audio";
        return false;
    }

    const int start_idx =
        apply_range ? std::clamp(static_cast<int>(start_norm * total_samples), 0, total_samples - 1)
                    : 0;
    const int end_idx = apply_range ? std::clamp(static_cast<int>(end_norm * total_samples),
                                                 start_idx + 1, total_samples)
                                    : total_samples;
    const int count = std::max(1, end_idx - start_idx);
    const int num_ch = static_cast<int>(snap.audio.size());

    juce::AudioBuffer<float> out_buf(num_ch, count);
    for (int c = 0; c < num_ch; ++c) {
        const auto& src = snap.audio[static_cast<size_t>(c)];
        const int avail = std::max(0, static_cast<int>(src.size()) - start_idx);
        const int to_copy = std::min(count, avail);
        if (to_copy > 0) {
            out_buf.copyFrom(c, 0, src.data() + start_idx, to_copy);
        }
        if (to_copy < count) {
            out_buf.clear(c, to_copy, count - to_copy);
        }
    }

    std::unique_ptr<juce::AudioFormat> audio_fmt;
    int bits = 24;
    auto sample_fmt = juce::AudioFormatWriterOptions::SampleFormat::automatic;
    if (is_aiff) {
        audio_fmt = std::make_unique<juce::AiffAudioFormat>();
        bits = 24;
    } else {
        audio_fmt = std::make_unique<juce::WavAudioFormat>();
        if (format_id == 2) {
            bits = 16;
        } else if (format_id == 3) {
            bits = 32;
            sample_fmt = juce::AudioFormatWriterOptions::SampleFormat::floatingPoint;
        } else {
            bits = 24;
        }
    }

    std::unique_ptr<juce::OutputStream> out_stream = dest.createOutputStream();
    if (!out_stream) {
        out_error = "Cannot open file for writing";
        return false;
    }

    const auto writer_opts = juce::AudioFormatWriterOptions{}
                                 .withSampleRate(static_cast<double>(std::max(1, snap.sample_rate)))
                                 .withNumChannels(num_ch)
                                 .withBitsPerSample(bits)
                                 .withSampleFormat(sample_fmt);
    auto writer = audio_fmt->createWriterFor(out_stream, writer_opts);
    if (!writer) {
        out_error = "Unsupported audio format options";
        return false;
    }

    if (!writer->writeFromAudioSampleBuffer(out_buf, 0, count)) {
        out_error = "Write error";
        return false;
    }
    return true;
}

class ExportModalContent final : public juce::Component {
public:
    ExportModalContent(
        const netbender::sampler::SamplerEngine& engine, int initial_slot,
        bool select_all_populated, bool initial_apply_range,
        std::function<void(std::vector<int>, bool, float, float, int)> on_export_requested)
        : m_engine(engine),
          m_slotCount(engine.get_slot_count()),
          m_onExport(std::move(on_export_requested)) {
        m_slotsHeaderLabel.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
        m_slotsHeaderLabel.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        addAndMakeVisible(m_slotsHeaderLabel);

        m_selectAllBtn.onClick = [this]() { setAllPopulated(true); };
        addAndMakeVisible(m_selectAllBtn);

        m_selectNoneBtn.onClick = [this]() { setAllPopulated(false); };
        addAndMakeVisible(m_selectNoneBtn);

        for (int i = 0; i < m_slotCount; ++i) {
            auto& tb = m_slotToggles[static_cast<size_t>(i)];
            const auto snap = m_engine.get_slot_snapshot(i);
            const bool has_audio = (snap && snap->num_samples > 0);
            juce::String text = "Slot " + juce::String(i + 1);
            if (has_audio) {
                text += " (" + juce::String(snap->duration_sec, 2) + "s, #" +
                        juce::String(snap->seed) + ")";
            } else {
                text += " (empty)";
            }
            tb.setButtonText(text);
            tb.setEnabled(has_audio);
            const bool checked = has_audio && (select_all_populated || i == initial_slot);
            tb.setToggleState(checked, juce::dontSendNotification);
            tb.onClick = [this]() { refreshRangeAndState(); };
            addAndMakeVisible(tb);
        }

        m_formatLabel.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
        m_formatLabel.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        addAndMakeVisible(m_formatLabel);

        m_formatCombo.addItem("WAV 24-bit PCM", 1);
        m_formatCombo.addItem("WAV 16-bit PCM", 2);
        m_formatCombo.addItem("WAV 32-bit Float", 3);
        m_formatCombo.addItem("AIFF 24-bit", 4);
        m_formatCombo.setSelectedId(1, juce::dontSendNotification);
        addAndMakeVisible(m_formatCombo);

        m_applyRangeToggle.setButtonText("Apply selected range (per-slot loop points for multi)");
        m_applyRangeToggle.setToggleState(initial_apply_range, juce::dontSendNotification);
        m_applyRangeToggle.onClick = [this]() { refreshRangeAndState(); };
        addAndMakeVisible(m_applyRangeToggle);

        auto setup_range_slider = [this](juce::Label& lbl, juce::Slider& sl) {
            lbl.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
            lbl.setFont(juce::FontOptions(UIConstants::Typography::font_small));
            addAndMakeVisible(lbl);
            sl.setSliderStyle(juce::Slider::LinearHorizontal);
            sl.setTextBoxStyle(juce::Slider::TextBoxRight, false, 58, 22);
            sl.setTextValueSuffix(" s");
            addAndMakeVisible(sl);
        };
        setup_range_slider(m_startLabel, m_startSlider);
        setup_range_slider(m_endLabel, m_endSlider);

        m_startSlider.onValueChange = [this]() {
            if (m_startSlider.getValue() >= m_endSlider.getValue() - 0.01) {
                m_endSlider.setValue(m_startSlider.getValue() + 0.01, juce::dontSendNotification);
            }
        };
        m_endSlider.onValueChange = [this]() {
            if (m_endSlider.getValue() <= m_startSlider.getValue() + 0.01) {
                m_startSlider.setValue(m_endSlider.getValue() - 0.01, juce::dontSendNotification);
            }
        };

        m_exportBtn.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::violet);
        m_exportBtn.onClick = [this]() {
            std::vector<int> selected = selectedSlots();
            if (selected.empty()) {
                return;
            }
            const bool apply = m_applyRangeToggle.getToggleState();
            float s_norm = 0.0f;
            float e_norm = 1.0f;
            if (selected.size() == 1 && apply) {
                const auto snap = m_engine.get_slot_snapshot(selected[0]);
                const float dur = snap ? std::max(0.01f, snap->duration_sec) : 1.0f;
                s_norm = static_cast<float>(m_startSlider.getValue()) / dur;
                e_norm = static_cast<float>(m_endSlider.getValue()) / dur;
            }
            const int fmt_id = m_formatCombo.getSelectedId();
            if (auto* dw = findParentComponentOfClass<juce::DialogWindow>()) {
                dw->exitModalState(0);
                dw->setVisible(false);
            }
            if (m_onExport) {
                m_onExport(std::move(selected), apply, s_norm, e_norm, fmt_id);
            }
        };
        addAndMakeVisible(m_exportBtn);

        m_cancelBtn.onClick = [this]() {
            if (auto* dw = findParentComponentOfClass<juce::DialogWindow>()) {
                dw->exitModalState(0);
                dw->setVisible(false);
            }
        };
        addAndMakeVisible(m_cancelBtn);

        refreshRangeAndState();
        const int rows = (m_slotCount + 1) / 2;
        setSize(490, 250 + rows * 24);
    }

    auto paint(juce::Graphics& g) -> void override {
        g.fillAll(UIConstants::Colors::surface_panel);
    }

    auto resized() -> void override {
        auto area = getLocalBounds().reduced(16);

        auto top_row = area.removeFromTop(26);
        m_slotsHeaderLabel.setBounds(top_row.removeFromLeft(140));
        m_selectNoneBtn.setBounds(top_row.removeFromRight(90));
        top_row.removeFromRight(6);
        m_selectAllBtn.setBounds(top_row.removeFromRight(90));

        area.removeFromTop(6);
        const int rows = (m_slotCount + 1) / 2;
        auto grid_area = area.removeFromTop(rows * 24);
        const int col_w = grid_area.getWidth() / 2;
        for (int i = 0; i < m_slotCount; ++i) {
            const int r = i / 2;
            const int c = i % 2;
            m_slotToggles[static_cast<size_t>(i)].setBounds(
                grid_area.getX() + c * col_w, grid_area.getY() + r * 24, col_w - 6, 22);
        }

        area.removeFromTop(10);
        auto row2 = area.removeFromTop(28);
        m_formatLabel.setBounds(row2.removeFromLeft(80));
        m_formatCombo.setBounds(row2);

        area.removeFromTop(10);
        m_applyRangeToggle.setBounds(area.removeFromTop(26));

        area.removeFromTop(6);
        auto row3 = area.removeFromTop(26);
        m_startLabel.setBounds(row3.removeFromLeft(80));
        m_startSlider.setBounds(row3);

        area.removeFromTop(6);
        auto row4 = area.removeFromTop(26);
        m_endLabel.setBounds(row4.removeFromLeft(80));
        m_endSlider.setBounds(row4);

        auto bottom = area.removeFromBottom(32);
        m_exportBtn.setBounds(bottom.removeFromRight(140));
        bottom.removeFromRight(8);
        m_cancelBtn.setBounds(bottom.removeFromRight(90));
    }

private:
    [[nodiscard]] auto selectedSlots() const -> std::vector<int> {
        std::vector<int> out;
        for (int i = 0; i < m_slotCount; ++i) {
            if (m_slotToggles[static_cast<size_t>(i)].getToggleState() &&
                m_slotToggles[static_cast<size_t>(i)].isEnabled()) {
                out.push_back(i);
            }
        }
        return out;
    }

    auto setAllPopulated(bool checked) -> void {
        for (int i = 0; i < m_slotCount; ++i) {
            auto& tb = m_slotToggles[static_cast<size_t>(i)];
            if (tb.isEnabled()) {
                tb.setToggleState(checked, juce::dontSendNotification);
            }
        }
        refreshRangeAndState();
    }

    auto refreshRangeAndState() -> void {
        const auto sel = selectedSlots();
        m_exportBtn.setEnabled(!sel.empty());
        m_exportBtn.setButtonText(sel.size() > 1 ? ("EXPORT (" + juce::String(sel.size()) + ")...")
                                                 : "EXPORT...");

        const bool single_custom = (sel.size() == 1 && m_applyRangeToggle.getToggleState());
        m_startSlider.setEnabled(single_custom);
        m_endSlider.setEnabled(single_custom);

        if (sel.size() == 1) {
            const int s_idx = sel[0];
            const auto snap = m_engine.get_slot_snapshot(s_idx);
            const double dur = (snap && snap->num_samples > 0)
                                   ? std::max(0.05, static_cast<double>(snap->duration_sec))
                                   : 1.0;
            m_startSlider.setRange(0.0, std::max(0.01, dur - 0.01), 0.01);
            m_endSlider.setRange(0.01, dur, 0.01);
            m_startSlider.setValue(m_engine.get_loop_start(s_idx) * dur,
                                   juce::dontSendNotification);
            m_endSlider.setValue(m_engine.get_loop_end(s_idx) * dur, juce::dontSendNotification);
        }
    }

    const netbender::sampler::SamplerEngine& m_engine;
    int m_slotCount{4};
    std::function<void(std::vector<int>, bool, float, float, int)> m_onExport;

    juce::Label m_slotsHeaderLabel{"", "SLOTS TO EXPORT"};
    juce::TextButton m_selectAllBtn{"ALL"};
    juce::TextButton m_selectNoneBtn{"NONE"};
    std::array<juce::ToggleButton, netbender::sampler::SamplerEngine::k_max_slots> m_slotToggles;
    juce::Label m_formatLabel{"", "FORMAT"};
    juce::ComboBox m_formatCombo;
    juce::ToggleButton m_applyRangeToggle;
    juce::Label m_startLabel{"", "START"};
    juce::Slider m_startSlider;
    juce::Label m_endLabel{"", "END"};
    juce::Slider m_endSlider;
    juce::TextButton m_exportBtn{"EXPORT..."};
    juce::TextButton m_cancelBtn{"CANCEL"};
};

}  // namespace

// =============================================================================
// WaveformView
// =============================================================================

auto SamplerMainComponent::WaveformView::setSlotIndex(int idx) -> void {
    m_slotIndex = idx;
    repaint();
}

auto SamplerMainComponent::WaveformView::setCompactMode(bool compact) -> void {
    m_compactMode = compact;
    repaint();
}

auto SamplerMainComponent::WaveformView::setSelectedForGen(bool selected) -> void {
    if (m_selectedForGen != selected) {
        m_selectedForGen = selected;
        repaint();
    }
}

auto SamplerMainComponent::WaveformView::setGenerating(bool generating) -> void {
    if (m_generating != generating) {
        m_generating = generating;
        repaint();
    }
}

auto SamplerMainComponent::WaveformView::setFocused(bool focused) -> void {
    if (m_focused != focused) {
        m_focused = focused;
        repaint();
    }
}

auto SamplerMainComponent::WaveformView::setDisplayMode(DisplayMode mode) -> void {
    if (m_displayMode != mode) {
        m_displayMode = mode;
        repaint();
    }
}

auto SamplerMainComponent::WaveformView::regenBadgeBounds() const -> juce::Rectangle<float> {
    return {static_cast<float>(getWidth()) - 76.0f, 8.0f, 68.0f, 22.0f};
}

auto SamplerMainComponent::WaveformView::playBadgeBounds() const -> juce::Rectangle<float> {
    return {static_cast<float>(getWidth()) - 76.0f, static_cast<float>(getHeight()) - 30.0f, 68.0f,
            22.0f};
}

auto SamplerMainComponent::WaveformView::specBadgeBounds() const -> juce::Rectangle<float> {
    if (m_compactMode) {
        return {8.0f, static_cast<float>(getHeight()) - 30.0f, 54.0f, 22.0f};
    }
    return {static_cast<float>(getWidth()) - 72.0f, 8.0f, 64.0f, 22.0f};
}

auto SamplerMainComponent::WaveformView::rebuildSpecImage() -> void {
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

auto SamplerMainComponent::WaveformView::setSlot(
    std::shared_ptr<const netbender::sampler::SampleSlot> slot) -> void {
    m_slot = std::move(slot);
    rebuildSpecImage();
    repaint();
}

auto SamplerMainComponent::WaveformView::setPlayhead(float norm, bool playing) -> void {
    if (std::abs(norm - m_playhead) > 1e-4f || playing != m_playing) {
        m_playhead = norm;
        m_playing = playing;
        repaint();
    }
}

auto SamplerMainComponent::WaveformView::setLoopRegion(float start_norm, float end_norm) -> void {
    m_loop_start = std::clamp(start_norm, 0.0f, 0.99f);
    m_loop_end = std::clamp(end_norm, m_loop_start + 0.01f, 1.0f);
    rebuildSpecImage();
    repaint();
}

auto SamplerMainComponent::WaveformView::paint(juce::Graphics& g) -> void {
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

    // Spectrum / Waveform toggle badge (shown when slot has audio)
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

auto SamplerMainComponent::WaveformView::mouseDown(const juce::MouseEvent& e) -> void {
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

auto SamplerMainComponent::WaveformView::mouseDrag(const juce::MouseEvent& e) -> void {
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

auto SamplerMainComponent::WaveformView::mouseUp(const juce::MouseEvent&) -> void {
    m_drag = DragTarget::None;
}

// =============================================================================
// SamplerLookAndFeel
// =============================================================================

auto SamplerMainComponent::SamplerLookAndFeel::drawProgressBar(juce::Graphics& g,
                                                               juce::ProgressBar&, int width,
                                                               int height, double progress,
                                                               const juce::String& textToShow)
    -> void {
    const auto bounds =
        juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
    g.setColour(UIConstants::Colors::surface_card);
    g.fillRect(bounds);

    if (progress > 0.0) {
        const float fill_w = static_cast<float>(std::clamp(progress, 0.0, 1.0)) * bounds.getWidth();
        g.setColour(UIConstants::Colors::violet);
        g.fillRect(bounds.getX(), bounds.getY(), fill_w, bounds.getHeight());
    }

    g.setColour(UIConstants::Colors::border_medium);
    g.drawRect(bounds, 1.0f);

    if (textToShow.isNotEmpty()) {
        g.setColour(UIConstants::Colors::text_light);
        g.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        g.drawText(textToShow, 0, 0, width, height, juce::Justification::centred, false);
    }
}

// =============================================================================
// SamplerMainComponent
// =============================================================================

SamplerMainComponent::SamplerMainComponent(const juce::String& initial_bundle_path) {
    m_lookAndFeel.setColour(juce::ProgressBar::backgroundColourId,
                            UIConstants::Colors::surface_card);
    m_lookAndFeel.setColour(juce::ProgressBar::foregroundColourId, UIConstants::Colors::violet);
    m_lookAndFeel.setColour(juce::TextEditor::backgroundColourId,
                            UIConstants::Colors::surface_card);
    m_lookAndFeel.setColour(juce::TextEditor::outlineColourId, UIConstants::Colors::border_medium);
    m_lookAndFeel.setColour(juce::TextEditor::focusedOutlineColourId, UIConstants::Colors::violet);
    m_lookAndFeel.setColour(juce::TextEditor::textColourId, UIConstants::Colors::text_white);
    m_lookAndFeel.setColour(juce::TextEditor::highlightColourId, UIConstants::Colors::deep_violet);
    m_lookAndFeel.setColour(juce::Slider::backgroundColourId, UIConstants::Colors::surface_card);
    m_lookAndFeel.setColour(juce::Slider::trackColourId, UIConstants::Colors::violet);
    m_lookAndFeel.setColour(juce::Slider::thumbColourId, UIConstants::Colors::lilac);
    m_lookAndFeel.setColour(juce::Slider::textBoxTextColourId, UIConstants::Colors::text_secondary);
    m_lookAndFeel.setColour(juce::Slider::textBoxBackgroundColourId,
                            UIConstants::Colors::surface_card);
    m_lookAndFeel.setColour(juce::Slider::textBoxOutlineColourId,
                            UIConstants::Colors::border_subtle);
    juce::LookAndFeel::setDefaultLookAndFeel(&m_lookAndFeel);
    setLookAndFeel(&m_lookAndFeel);
    setWantsKeyboardFocus(true);
    addKeyListener(this);

    // Header
    m_titleLabel.setText("NETBENDER SAMPLER", juce::dontSendNotification);
    m_titleLabel.setFont(juce::FontOptions(UIConstants::Typography::font_title, juce::Font::bold));
    m_titleLabel.setColour(juce::Label::textColourId, UIConstants::Colors::text_white);
    addAndMakeVisible(m_titleLabel);

    m_bundleLabel.setText("No bundle loaded", juce::dontSendNotification);
    m_bundleLabel.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
    addAndMakeVisible(m_bundleLabel);

    m_slotCountCombo.addItem("4 Slots", 4);
    m_slotCountCombo.addItem("8 Slots", 8);
    m_slotCountCombo.addItem("12 Slots", 12);
    m_slotCountCombo.setSelectedId(4, juce::dontSendNotification);
    m_slotCountCombo.onChange = [this]() { setSlotCount(m_slotCountCombo.getSelectedId()); };
    addAndMakeVisible(m_slotCountCombo);

    m_exportButton.onClick = [this]() {
        const int slot_idx = m_engine.get_active_slot();
        const float s = m_engine.get_loop_start(slot_idx);
        const float e = m_engine.get_loop_end(slot_idx);
        const bool has_custom = (s > 0.005f || e < 0.995f);
        openExportModal(slot_idx, true, has_custom);
    };
    addAndMakeVisible(m_exportButton);

    m_loadButton.onClick = [this]() { chooseBundleFile(); };
    addAndMakeVisible(m_loadButton);

    m_audioMidiButton.onClick = [this]() { openAudioSettings(); };
    addAndMakeVisible(m_audioMidiButton);

    // Prompt row
    m_promptEditor.setText("Amen break 174 BPM", false);
    m_promptEditor.setTextToShowWhenEmpty("Enter Stable Audio 3 prompt...",
                                          UIConstants::Colors::text_dimmed);
    m_promptEditor.setColour(juce::TextEditor::backgroundColourId,
                             UIConstants::Colors::surface_card);
    m_promptEditor.setColour(juce::TextEditor::outlineColourId, UIConstants::Colors::border_medium);
    m_promptEditor.onReturnKey = [this]() { triggerGeneration(); };
    addAndMakeVisible(m_promptEditor);

    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        auto& btn = m_slotSelectButtons[static_cast<size_t>(i)];
        btn.setButtonText(juce::String(i + 1));
        btn.setClickingTogglesState(true);
        btn.setToggleState(true, juce::dontSendNotification);
        btn.setColour(juce::TextButton::buttonOnColourId, UIConstants::Colors::violet);
        btn.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::surface_card);
        btn.setTooltip("Include Slot " + juce::String(i + 1) + " in next Generate");
        btn.onClick = [this, i]() {
            syncSlotSelectionUi(i, m_slotSelectButtons[static_cast<size_t>(i)].getToggleState());
        };
        addAndMakeVisible(btn);
    }

    m_generateButton.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::violet);
    m_generateButton.onClick = [this]() {
        using Stage = netbender::sampler::GenerationProgress::Stage;
        const auto st = m_engine.get_stage();
        if (st == Stage::Tokenizing || st == Stage::EncodingPrompt || st == Stage::Diffusing ||
            st == Stage::Decoding) {
            m_engine.cancel_generation();
        } else {
            triggerGeneration();
        }
    };
    addAndMakeVisible(m_generateButton);

    addAndMakeVisible(m_progressBar);
    m_statusLabel.setText("Idle", juce::dontSendNotification);
    m_statusLabel.setColour(juce::Label::textColourId, UIConstants::Colors::text_light);
    addAndMakeVisible(m_statusLabel);

    // Focus bar
    m_backToGridButton.onClick = [this]() { setViewMode(ViewMode::Grid); };
    addChildComponent(m_backToGridButton);
    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        auto& fb = m_focusSlotButtons[static_cast<size_t>(i)];
        fb.setButtonText(juce::String(i + 1));
        fb.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::surface_card);
        fb.onClick = [this, i]() { setViewMode(ViewMode::Focus, i); };
        addChildComponent(fb);
    }
    m_focusInfoLabel.setColour(juce::Label::textColourId, UIConstants::Colors::text_light);
    m_focusInfoLabel.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    addChildComponent(m_focusInfoLabel);

    // Grid Waveforms & Focus Waveform
    m_slotDisplayModes.fill(DisplayMode::Waveform);
    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        auto& gv = m_gridViews[static_cast<size_t>(i)];
        gv.setSlotIndex(i);
        gv.setCompactMode(true);
        gv.setSelectedForGen(true);
        gv.setFocused(i == 0);
        gv.onFocusRequested = [this](int slot_idx) { setViewMode(ViewMode::Focus, slot_idx); };
        gv.onTogglePlay = [this](int slot_idx) {
            m_engine.set_active_slot(slot_idx);
            if (m_engine.is_slot_playing(slot_idx)) {
                m_engine.stop_slot_rt(slot_idx);
            } else {
                m_engine.trigger_slot_rt(slot_idx, 60, 1.0f);
            }
        };
        gv.onToggleRegen = [this](int slot_idx) {
            const bool next = !m_slotSelectButtons[static_cast<size_t>(slot_idx)].getToggleState();
            syncSlotSelectionUi(slot_idx, next);
        };
        gv.onDisplayModeChanged = [this](int slot_idx, DisplayMode mode) {
            m_slotDisplayModes[static_cast<size_t>(slot_idx)] = mode;
            if (slot_idx == m_engine.get_active_slot()) {
                m_focusView.setDisplayMode(mode);
            }
        };
        gv.onRightClick = [this](int slot_idx) { showWaveformContextMenu(slot_idx); };
        addAndMakeVisible(gv);
    }

    m_focusView.setCompactMode(false);
    m_focusView.setSlotIndex(0);
    m_focusView.onLoopChanged = [this](float s, float e) {
        const int active = m_engine.get_active_slot();
        m_engine.set_loop_region(active, s, e);
        m_gridViews[static_cast<size_t>(active)].setLoopRegion(s, e);
        updateFocusInfoLabel();
    };
    m_focusView.onDisplayModeChanged = [this](int slot_idx, DisplayMode mode) {
        m_slotDisplayModes[static_cast<size_t>(slot_idx)] = mode;
        m_gridViews[static_cast<size_t>(slot_idx)].setDisplayMode(mode);
    };
    m_focusView.onRightClick = [this](int slot_idx) { showWaveformContextMenu(slot_idx); };
    addChildComponent(m_focusView);

    setSlotCount(4);

    // Transport & controls
    m_triggerButton.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::forest_green);
    m_triggerButton.onClick = [this]() { m_engine.trigger_rt(60, 1.0f); };
    addAndMakeVisible(m_triggerButton);

    m_stopButton.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::ruby);
    m_stopButton.onClick = [this]() { m_engine.stop_rt(); };
    addAndMakeVisible(m_stopButton);

    addAndMakeVisible(m_modeLabel);
    m_modeCombo.addItem("One-Shot", 1);
    m_modeCombo.addItem("Loop", 2);
    m_modeCombo.addItem("Gate", 3);
    m_modeCombo.setSelectedId(2, juce::dontSendNotification);
    m_engine.set_playback_mode(netbender::sampler::SamplerPlaybackMode::Loop);
    m_modeCombo.onChange = [this]() {
        const int id = m_modeCombo.getSelectedId();
        using Mode = netbender::sampler::SamplerPlaybackMode;
        if (id == 1) {
            m_engine.set_playback_mode(Mode::OneShot);
        } else if (id == 2) {
            m_engine.set_playback_mode(Mode::Loop);
        } else {
            m_engine.set_playback_mode(Mode::Gate);
        }
    };
    addAndMakeVisible(m_modeCombo);

    auto setup_slider = [this](juce::Label& lbl, juce::Slider& sl, double min_v, double max_v,
                               double step, double init_v) {
        lbl.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
        lbl.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        addAndMakeVisible(lbl);
        sl.setSliderStyle(juce::Slider::LinearHorizontal);
        sl.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
        sl.setRange(min_v, max_v, step);
        sl.setValue(init_v, juce::dontSendNotification);
        addAndMakeVisible(sl);
    };

    setup_slider(m_durationLabel, m_durationSlider, 1.0, 11.5, 0.5, 6.0);
    setup_slider(m_stepsLabel, m_stepsSlider, 1.0, 16.0, 1.0, 8.0);
    setup_slider(m_speedLabel, m_speedSlider, 0.25, 4.0, 0.01, 1.0);
    setup_slider(m_gainLabel, m_gainSlider, 0.0, 1.5, 0.01, 1.0);

    m_speedSlider.onValueChange = [this]() {
        m_engine.set_speed(static_cast<float>(m_speedSlider.getValue()));
    };
    m_gainSlider.onValueChange = [this]() {
        m_engine.set_gain(static_cast<float>(m_gainSlider.getValue()));
    };

    m_seedLabel.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
    m_seedLabel.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    addAndMakeVisible(m_seedLabel);
    m_seedEditor.setText("42", false);
    m_seedEditor.setInputRestrictions(12, "0123456789");
    addAndMakeVisible(m_seedEditor);

    m_diceButton.onClick = [this]() {
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        m_seedEditor.setText(juce::String(rng() % 1000000ULL), false);
    };
    addAndMakeVisible(m_diceButton);

    setAudioChannels(0, 2);
    deviceManager.addChangeListener(this);
    refreshMidiCallbacks();
    startTimerHz(30);

    setSize(1040, 600);

    if (initial_bundle_path.isNotEmpty()) {
        loadBundleAsync(initial_bundle_path);
    } else {
        const juce::File default_bundle = juce::File::getCurrentWorkingDirectory().getChildFile(
            "models/sa3_small_music/sa3_small_music.nbsampler.json");
        if (default_bundle.existsAsFile()) {
            loadBundleAsync(default_bundle.getFullPathName());
        }
    }
}

SamplerMainComponent::~SamplerMainComponent() {
    deviceManager.removeChangeListener(this);
    removeKeyListener(this);
    stopTimer();
    if (m_loadThread.joinable()) {
        m_loadThread.join();
    }
    for (const auto& info : juce::MidiInput::getAvailableDevices()) {
        deviceManager.removeMidiInputDeviceCallback(info.identifier, this);
    }
    shutdownAudio();
    juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
    setLookAndFeel(nullptr);
}

auto SamplerMainComponent::prepareToPlay(int samplesPerBlockExpected, double sampleRate) -> void {
    m_engine.prepare(sampleRate, samplesPerBlockExpected);
}

auto SamplerMainComponent::getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)
    -> void {
    auto* buf = bufferToFill.buffer;
    if (buf == nullptr) {
        return;
    }
    const int total_ch = buf->getNumChannels();
    if (total_ch <= 0 || bufferToFill.numSamples <= 0) {
        return;
    }
    float* ptrs[2]{
        buf->getWritePointer(0, bufferToFill.startSample),
        (total_ch > 1) ? buf->getWritePointer(1, bufferToFill.startSample) : nullptr,
    };
    m_engine.process_rt(ptrs, std::min(total_ch, 2), bufferToFill.numSamples);
    for (int c = 2; c < total_ch; ++c) {
        buf->clear(c, bufferToFill.startSample, bufferToFill.numSamples);
    }
}

auto SamplerMainComponent::releaseResources() -> void {}

auto SamplerMainComponent::handleIncomingMidiMessage(juce::MidiInput*,
                                                     const juce::MidiMessage& message) -> void {
    if (message.isNoteOn()) {
        m_engine.trigger_rt(message.getNoteNumber(), message.getFloatVelocity());
    } else if (message.isNoteOff()) {
        m_engine.release_rt(message.getNoteNumber());
    } else if (message.isAllNotesOff() || message.isAllSoundOff()) {
        m_engine.stop_rt();
    }
}

auto SamplerMainComponent::refreshMidiCallbacks() -> void {
    for (const auto& info : juce::MidiInput::getAvailableDevices()) {
        deviceManager.removeMidiInputDeviceCallback(info.identifier, this);
        if (deviceManager.isMidiInputDeviceEnabled(info.identifier)) {
            deviceManager.addMidiInputDeviceCallback(info.identifier, this);
        }
    }
}

auto SamplerMainComponent::changeListenerCallback(juce::ChangeBroadcaster* source) -> void {
    if (source == &deviceManager) {
        refreshMidiCallbacks();
    }
}

auto SamplerMainComponent::openAudioSettings() -> void {
    auto selector = std::make_unique<juce::AudioDeviceSelectorComponent>(deviceManager, 0, 0, 2, 2,
                                                                         true, false, true, false);
    selector->setSize(520, 420);
    juce::DialogWindow::LaunchOptions opts;
    opts.content.setOwned(selector.release());
    opts.dialogTitle = "Audio / MIDI Settings";
    opts.dialogBackgroundColour = UIConstants::Colors::surface_panel;
    opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = true;
    opts.resizable = false;
    opts.launchAsync();
}

auto SamplerMainComponent::chooseBundleFile() -> void {
    m_fileChooser = std::make_unique<juce::FileChooser>(
        "Select Stable Audio 3 Sampler Bundle (.nbsampler.json)",
        juce::File::getCurrentWorkingDirectory().getChildFile("models"), "*.nbsampler.json");
    constexpr auto flags =
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    m_fileChooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
        const auto file = fc.getResult();
        if (file.existsAsFile()) {
            loadBundleAsync(file.getFullPathName());
        }
    });
}

auto SamplerMainComponent::loadBundleAsync(const juce::String& manifest_path) -> void {
    if (m_isLoadingBundle.load()) {
        return;
    }
    if (m_loadThread.joinable()) {
        m_loadThread.join();
    }
    m_isLoadingBundle.store(true);
    m_bundleLoadDone.store(false);
    m_pendingBundleName = juce::File(manifest_path).getFileNameWithoutExtension();
    m_bundleLabel.setText("Loading " + m_pendingBundleName + "...", juce::dontSendNotification);
    m_statusLabel.setText("Loading CoreML / ExecuTorch bundle...", juce::dontSendNotification);

    const std::string path_str = manifest_path.toStdString();
    m_loadThread = std::thread([this, path_str]() {
        const bool ok = m_engine.load_bundle(path_str, true);
        if (!ok) {
            m_loadErrorText = m_engine.get_last_error();
        }
        m_bundleLoadOk.store(ok, std::memory_order_release);
        m_bundleLoadDone.store(true, std::memory_order_release);
        m_isLoadingBundle.store(false, std::memory_order_release);
    });
}

auto SamplerMainComponent::triggerGeneration() -> void {
    if (!m_engine.is_loaded()) {
        m_statusLabel.setText("Load a .nbsampler.json bundle first", juce::dontSendNotification);
        return;
    }
    const int slot_count = m_engine.get_slot_count();
    uint32_t slot_mask = 0;
    for (int i = 0; i < slot_count; ++i) {
        if (m_slotSelectButtons[static_cast<size_t>(i)].getToggleState()) {
            slot_mask |= (1u << i);
        }
    }
    if (slot_mask == 0) {
        m_statusLabel.setText(
            "Select at least one slot (1-" + juce::String(slot_count) + ") to generate",
            juce::dontSendNotification);
        return;
    }

    netbender::sampler::GenerationRequest req;
    req.prompt = m_promptEditor.getText().toStdString();
    req.duration_sec = static_cast<float>(m_durationSlider.getValue());
    req.steps = static_cast<int>(m_stepsSlider.getValue());
    req.seed =
        static_cast<uint64_t>(std::max<int64_t>(0, m_seedEditor.getText().getLargeIntValue()));
    req.decode_audio = true;
    m_engine.request_generate_slots(req, slot_mask);
}

auto SamplerMainComponent::setSlotCount(int count) -> void {
    m_engine.set_slot_count(count);
    const int snapped = m_engine.get_slot_count();
    const bool is_focus = (m_viewMode == ViewMode::Focus);

    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        const bool in_range = (i < snapped);
        m_slotSelectButtons[static_cast<size_t>(i)].setVisible(in_range);
        m_gridViews[static_cast<size_t>(i)].setVisible(!is_focus && in_range);
        m_focusSlotButtons[static_cast<size_t>(i)].setVisible(is_focus && in_range);
    }

    if (is_focus) {
        setViewMode(ViewMode::Focus, m_engine.get_active_slot());
    } else {
        resized();
        repaint();
    }
}

auto SamplerMainComponent::syncSlotSelectionUi(int slot_idx, bool selected) -> void {
    if (slot_idx < 0 || slot_idx >= netbender::sampler::SamplerEngine::k_max_slots) {
        return;
    }
    m_slotSelectButtons[static_cast<size_t>(slot_idx)].setToggleState(selected,
                                                                      juce::dontSendNotification);
    m_gridViews[static_cast<size_t>(slot_idx)].setSelectedForGen(selected);
}

auto SamplerMainComponent::setViewMode(ViewMode mode, int focus_slot) -> void {
    const int slot_count = m_engine.get_slot_count();
    if (focus_slot >= 0 && focus_slot < slot_count) {
        m_engine.set_active_slot(focus_slot);
    }
    m_viewMode = mode;
    const int active = std::clamp(m_engine.get_active_slot(), 0, slot_count - 1);

    const bool is_focus = (m_viewMode == ViewMode::Focus);
    m_backToGridButton.setVisible(is_focus);
    m_focusInfoLabel.setVisible(is_focus);
    m_focusView.setVisible(is_focus);
    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        const bool in_range = (i < slot_count);
        m_gridViews[static_cast<size_t>(i)].setVisible(!is_focus && in_range);
        m_gridViews[static_cast<size_t>(i)].setFocused(i == active);
        auto& fb = m_focusSlotButtons[static_cast<size_t>(i)];
        fb.setVisible(is_focus && in_range);
        fb.setColour(juce::TextButton::buttonColourId, (i == active)
                                                           ? UIConstants::Colors::violet
                                                           : UIConstants::Colors::surface_card);
    }

    if (is_focus) {
        m_focusView.setSlotIndex(active);
        m_focusView.setDisplayMode(m_slotDisplayModes[static_cast<size_t>(active)]);
        m_focusView.setSlot(m_engine.get_slot_snapshot(active));
        m_focusView.setLoopRegion(m_engine.get_loop_start(active), m_engine.get_loop_end(active));
        updateFocusInfoLabel();
    }

    resized();
    repaint();
}

auto SamplerMainComponent::updateFocusInfoLabel() -> void {
    const int active = m_engine.get_active_slot();
    const auto snap = m_engine.get_slot_snapshot(active);
    juce::String info = "SLOT " + juce::String(active + 1);
    if (snap && snap->num_samples > 0) {
        const float dur = snap->duration_sec;
        const float s_sec = m_engine.get_loop_start(active) * dur;
        const float e_sec = m_engine.get_loop_end(active) * dur;
        info += "  •  Seed #" + juce::String(snap->seed) + "  •  " + juce::String(dur, 2) +
                "s  •  Range: " + juce::String(s_sec, 2) + "s - " + juce::String(e_sec, 2) + "s";
    } else {
        info += "  •  (empty)";
    }
    m_focusInfoLabel.setText(info, juce::dontSendNotification);
}

auto SamplerMainComponent::showWaveformContextMenu(int slot_idx) -> void {
    const auto snap = m_engine.get_slot_snapshot(slot_idx);
    const bool has_audio = (snap && snap->num_samples > 0);
    const float s = m_engine.get_loop_start(slot_idx);
    const float e = m_engine.get_loop_end(slot_idx);

    juce::PopupMenu menu;
    if (has_audio) {
        menu.addItem(1, "Export Selection as WAV...");
        menu.addItem(2, "Export Full Sample as WAV...");
        menu.addItem(3, "Export Options...");
        menu.addSeparator();
        menu.addItem(4, m_engine.is_slot_playing(slot_idx) ? "Stop" : "Play");
    } else {
        menu.addItem(10, "Slot " + juce::String(slot_idx + 1) + " is empty", false, false);
    }

    if (m_viewMode == ViewMode::Grid) {
        menu.addItem(5, "Focus Slot " + juce::String(slot_idx + 1));
    } else {
        menu.addItem(6, "Back to 2x2 Grid");
    }

    juce::Component::SafePointer<SamplerMainComponent> safe_this(this);
    menu.showMenuAsync(juce::PopupMenu::Options{}, [safe_this, slot_idx, s, e](int result) {
        if (safe_this == nullptr || result == 0) {
            return;
        }
        if (result == 1) {
            safe_this->exportSlotToFile(slot_idx, true, s, e, ExportFormat::Wav24);
        } else if (result == 2) {
            safe_this->exportSlotToFile(slot_idx, false, 0.0f, 1.0f, ExportFormat::Wav24);
        } else if (result == 3) {
            safe_this->openExportModal(slot_idx, false, true);
        } else if (result == 4) {
            if (safe_this->m_engine.is_slot_playing(slot_idx)) {
                safe_this->m_engine.stop_slot_rt(slot_idx);
            } else {
                safe_this->m_engine.trigger_slot_rt(slot_idx, 60, 1.0f);
            }
        } else if (result == 5) {
            safe_this->setViewMode(ViewMode::Focus, slot_idx);
        } else if (result == 6) {
            safe_this->setViewMode(ViewMode::Grid);
        }
    });
}

auto SamplerMainComponent::openExportModal(int default_slot_idx, bool select_all_populated,
                                           bool default_apply_range) -> void {
    juce::Component::SafePointer<SamplerMainComponent> safe_this(this);
    auto content = std::make_unique<ExportModalContent>(
        m_engine, default_slot_idx, select_all_populated, default_apply_range,
        [safe_this](std::vector<int> slots, bool apply_range, float s_norm, float e_norm,
                    int fmt_id) {
            if (safe_this != nullptr) {
                safe_this->exportSlotsToFiles(std::move(slots), apply_range, s_norm, e_norm,
                                              static_cast<ExportFormat>(fmt_id));
            }
        });

    juce::DialogWindow::LaunchOptions opts;
    opts.content.setOwned(content.release());
    opts.dialogTitle = "Export Sample(s)";
    opts.dialogBackgroundColour = UIConstants::Colors::surface_panel;
    opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = true;
    opts.resizable = false;
    opts.launchAsync();
}

auto SamplerMainComponent::exportSlotToFile(int slot_idx, bool apply_range, float start_norm,
                                            float end_norm, ExportFormat format) -> void {
    exportSlotsToFiles({slot_idx}, apply_range, start_norm, end_norm, format);
}

auto SamplerMainComponent::exportSlotsToFiles(std::vector<int> slot_indices, bool apply_range,
                                              float single_start_norm, float single_end_norm,
                                              ExportFormat format) -> void {
    if (slot_indices.empty()) {
        return;
    }

    auto initial_dir = juce::File::getSpecialLocation(juce::File::userDesktopDirectory);
    if (!initial_dir.exists()) {
        initial_dir = juce::File::getCurrentWorkingDirectory();
    }

    const int fmt_id = static_cast<int>(format);
    const bool is_aiff = (format == ExportFormat::Aiff24);
    const juce::String ext = is_aiff ? ".aiff" : ".wav";
    const juce::String pattern = is_aiff ? "*.aiff;*.aif" : "*.wav";

    juce::Component::SafePointer<SamplerMainComponent> safe_this(this);

    if (slot_indices.size() == 1) {
        const int slot_idx = slot_indices[0];
        const auto snap = m_engine.get_slot_snapshot(slot_idx);
        if (!snap || snap->audio.empty() || snap->num_samples <= 0) {
            m_statusLabel.setText("Slot " + juce::String(slot_idx + 1) + " has no audio to export",
                                  juce::dontSendNotification);
            return;
        }
        const juce::String default_name =
            sanitize_filename(snap->prompt) + "_slot" + juce::String(slot_idx + 1) + ext;

        m_exportChooser = std::make_unique<juce::FileChooser>(
            "Export Audio Sample", initial_dir.getChildFile(default_name), pattern);
        constexpr auto flags = juce::FileBrowserComponent::saveMode |
                               juce::FileBrowserComponent::canSelectFiles |
                               juce::FileBrowserComponent::warnAboutOverwriting;
        m_exportChooser->launchAsync(flags, [safe_this, snap, apply_range, single_start_norm,
                                             single_end_norm, fmt_id](const juce::FileChooser& fc) {
            if (safe_this == nullptr) {
                return;
            }
            const juce::File dest = fc.getResult();
            if (dest == juce::File{}) {
                return;
            }
            juce::String err;
            const bool ok = write_slot_audio_to_file(*snap, dest, apply_range, single_start_norm,
                                                     single_end_norm, fmt_id, err);
            safe_this->m_statusLabel.setText(
                ok ? ("Exported " + dest.getFileName()) : ("Export failed: " + err),
                juce::dontSendNotification);
        });
        return;
    }

    m_exportChooser = std::make_unique<juce::FileChooser>(
        "Select Folder to Export " + juce::String(slot_indices.size()) + " Samples", initial_dir);
    constexpr auto dir_flags =
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories;
    m_exportChooser->launchAsync(
        dir_flags, [safe_this, slot_indices = std::move(slot_indices), apply_range, fmt_id,
                    ext](const juce::FileChooser& fc) {
            if (safe_this == nullptr) {
                return;
            }
            const juce::File dir = fc.getResult();
            if (!dir.isDirectory()) {
                return;
            }
            int exported = 0;
            juce::String last_err;
            for (int slot_idx : slot_indices) {
                const auto snap = safe_this->m_engine.get_slot_snapshot(slot_idx);
                if (!snap || snap->num_samples <= 0) {
                    continue;
                }
                const float s_norm = safe_this->m_engine.get_loop_start(slot_idx);
                const float e_norm = safe_this->m_engine.get_loop_end(slot_idx);
                const juce::String fname =
                    sanitize_filename(snap->prompt) + "_slot" + juce::String(slot_idx + 1) + ext;
                if (write_slot_audio_to_file(*snap, dir.getChildFile(fname), apply_range, s_norm,
                                             e_norm, fmt_id, last_err)) {
                    ++exported;
                }
            }
            if (exported > 0) {
                safe_this->m_statusLabel.setText(
                    "Exported " + juce::String(exported) + " slot(s) to " + dir.getFileName(),
                    juce::dontSendNotification);
            } else {
                safe_this->m_statusLabel.setText("Export failed: " + last_err,
                                                 juce::dontSendNotification);
            }
        });
}

auto SamplerMainComponent::timerCallback() -> void {
    if (m_bundleLoadDone.exchange(false)) {
        if (m_bundleLoadOk.load()) {
            const auto& meta = m_engine.get_metadata();
            m_bundleLabel.setText(m_pendingBundleName + " [" + juce::String(meta.delegate) + "]",
                                  juce::dontSendNotification);
            m_statusLabel.setText("Bundle ready", juce::dontSendNotification);
            if (meta.max_duration_sec > 1.0f) {
                const double max_d = std::min(11.5, static_cast<double>(meta.max_duration_sec));
                m_durationSlider.setRange(1.0, max_d, 0.5);
            }
        } else {
            m_bundleLabel.setText("Load failed", juce::dontSendNotification);
            m_statusLabel.setText(m_loadErrorText, juce::dontSendNotification);
        }
    }

    const int active_slot = m_engine.get_active_slot();
    const int playing_slot = m_engine.get_playing_slot();
    const bool any_playing = m_engine.is_playing();
    const float playhead = m_engine.get_playhead_norm();
    const int gen_slot = m_engine.get_batch_current_slot();

    const int slot_count = m_engine.get_slot_count();
    for (int i = 0; i < slot_count; ++i) {
        const uint64_t sid = m_engine.get_slot_generation_id(i);
        if (sid != m_lastSeenSlotIds[static_cast<size_t>(i)]) {
            m_lastSeenSlotIds[static_cast<size_t>(i)] = sid;
            auto snap = m_engine.get_slot_snapshot(i);
            const float ls = m_engine.get_loop_start(i);
            const float le = m_engine.get_loop_end(i);
            m_gridViews[static_cast<size_t>(i)].setSlot(snap);
            m_gridViews[static_cast<size_t>(i)].setLoopRegion(ls, le);
            if (i == active_slot) {
                m_focusView.setSlot(snap);
                m_focusView.setLoopRegion(ls, le);
                updateFocusInfoLabel();
            }
        }
        const bool slot_playing = any_playing && (playing_slot == i);
        m_gridViews[static_cast<size_t>(i)].setPlayhead(slot_playing ? playhead : 0.0f,
                                                        slot_playing);
        m_gridViews[static_cast<size_t>(i)].setGenerating(gen_slot == i);
    }

    const bool active_playing = any_playing && (playing_slot == active_slot);
    m_focusView.setPlayhead(active_playing ? playhead : 0.0f, active_playing);
    m_focusView.setGenerating(gen_slot == active_slot);

    using Stage = netbender::sampler::GenerationProgress::Stage;
    const Stage stage = m_engine.get_stage();
    const int cur = m_engine.get_current_step();
    const int tot = std::max(1, m_engine.get_total_steps());
    const int batch_idx = m_engine.get_batch_index();
    const int batch_tot = m_engine.get_batch_total();
    const bool busy = (stage == Stage::Tokenizing || stage == Stage::EncodingPrompt ||
                       stage == Stage::Diffusing || stage == Stage::Decoding);

    m_generateButton.setButtonText(busy ? "CANCEL" : "GENERATE");
    if (busy) {
        const double slot_frac =
            (stage == Stage::Decoding) ? 0.95 : static_cast<double>(cur) / static_cast<double>(tot);
        const int b_tot = std::max(1, batch_tot);
        const int b_done = std::max(0, batch_idx - 1);
        m_progressValue = (static_cast<double>(b_done) + slot_frac) / static_cast<double>(b_tot);
        m_statusLabel.setText(
            format_stage(stage, cur, tot, m_engine.get_last_gen_ms(), batch_idx, batch_tot),
            juce::dontSendNotification);
    } else if (!m_isLoadingBundle.load()) {
        m_progressValue = (stage == Stage::Done) ? 1.0 : 0.0;
        if (stage != Stage::Idle) {
            m_statusLabel.setText(stage == Stage::Error
                                      ? juce::String(m_engine.get_last_error())
                                      : format_stage(stage, cur, tot, m_engine.get_last_gen_ms(),
                                                     batch_idx, batch_tot),
                                  juce::dontSendNotification);
        }
    }
}

auto SamplerMainComponent::keyPressed(const juce::KeyPress& key, juce::Component* origin) -> bool {
    if (origin == &m_promptEditor || origin == &m_seedEditor) {
        return false;
    }
    if (key == juce::KeyPress::escapeKey && m_viewMode == ViewMode::Focus) {
        setViewMode(ViewMode::Grid);
        return true;
    }
    if (key == juce::KeyPress::spaceKey) {
        if (m_engine.is_playing()) {
            m_engine.stop_rt();
        } else {
            m_engine.trigger_rt(60, 1.0f);
        }
        return true;
    }
    return false;
}

auto SamplerMainComponent::paint(juce::Graphics& g) -> void {
    const juce::ColourGradient bg(UIConstants::Colors::background_gradient_top, 0.0f, 0.0f,
                                  UIConstants::Colors::background_gradient_bottom, 0.0f,
                                  static_cast<float>(getHeight()), false);
    g.setGradientFill(bg);
    g.fillAll();
}

auto SamplerMainComponent::resized() -> void {
    auto area = getLocalBounds().reduced(14);

    // Header (36 px)
    auto header = area.removeFromTop(36);
    m_titleLabel.setBounds(header.removeFromLeft(220));
    m_audioMidiButton.setBounds(header.removeFromRight(115));
    header.removeFromRight(8);
    m_loadButton.setBounds(header.removeFromRight(115));
    header.removeFromRight(8);
    m_exportButton.setBounds(header.removeFromRight(90));
    header.removeFromRight(8);
    m_slotCountCombo.setBounds(header.removeFromRight(95));
    header.removeFromRight(10);
    m_bundleLabel.setBounds(header);

    area.removeFromTop(10);

    // Prompt row (34 px)
    const int slot_count = m_engine.get_slot_count();
    const int sel_btn_w = (slot_count >= 12) ? 22 : 26;
    auto prompt_row = area.removeFromTop(34);
    m_generateButton.setBounds(prompt_row.removeFromRight(115));
    prompt_row.removeFromRight(6);
    for (int i = slot_count - 1; i >= 0; --i) {
        m_slotSelectButtons[static_cast<size_t>(i)].setBounds(
            prompt_row.removeFromRight(sel_btn_w));
        prompt_row.removeFromRight(2);
    }
    prompt_row.removeFromRight(5);
    m_promptEditor.setBounds(prompt_row);

    area.removeFromTop(8);

    // Progress & status row (22 px)
    auto status_row = area.removeFromTop(22);
    m_progressBar.setBounds(status_row.removeFromLeft(240));
    status_row.removeFromLeft(12);
    m_statusLabel.setBounds(status_row);

    area.removeFromTop(10);

    // Controls (72 px at bottom)
    auto controls = area.removeFromBottom(72);
    area.removeFromBottom(10);

    // Center: Dynamic Grid (2x2, 4x2, 4x3) OR Focus Bar + Focus Waveform
    if (m_viewMode == ViewMode::Grid) {
        constexpr int gap = 8;
        const int cols = (slot_count <= 4) ? 2 : 4;
        const int rows = (slot_count + cols - 1) / cols;
        const int cell_w = std::max(10, (area.getWidth() - gap * (cols - 1)) / cols);
        const int cell_h = std::max(10, (area.getHeight() - gap * (rows - 1)) / rows);
        for (int i = 0; i < slot_count; ++i) {
            const int r = i / cols;
            const int c = i % cols;
            m_gridViews[static_cast<size_t>(i)].setBounds(
                area.getX() + c * (cell_w + gap), area.getY() + r * (cell_h + gap), cell_w, cell_h);
        }
    } else {
        auto focus_bar = area.removeFromTop(28);
        m_backToGridButton.setBounds(focus_bar.removeFromLeft(80));
        focus_bar.removeFromLeft(8);
        const int fb_w = (slot_count >= 12) ? 24 : 28;
        for (int i = 0; i < slot_count; ++i) {
            m_focusSlotButtons[static_cast<size_t>(i)].setBounds(focus_bar.removeFromLeft(fb_w));
            focus_bar.removeFromLeft(3);
        }
        focus_bar.removeFromLeft(8);
        m_focusInfoLabel.setBounds(focus_bar);
        area.removeFromTop(8);
        m_focusView.setBounds(area);
    }

    // Split controls into 2 rows
    auto row1 = controls.removeFromTop(32);
    controls.removeFromTop(8);
    auto row2 = controls.removeFromTop(32);

    m_triggerButton.setBounds(row1.removeFromLeft(95));
    row1.removeFromLeft(8);
    m_stopButton.setBounds(row1.removeFromLeft(75));
    row1.removeFromLeft(16);
    m_modeLabel.setBounds(row1.removeFromLeft(44));
    m_modeCombo.setBounds(row1.removeFromLeft(110));
    row1.removeFromLeft(18);
    m_durationLabel.setBounds(row1.removeFromLeft(52));
    m_durationSlider.setBounds(row1.removeFromLeft(210));
    row1.removeFromLeft(16);
    m_stepsLabel.setBounds(row1.removeFromLeft(46));
    m_stepsSlider.setBounds(row1);

    m_speedLabel.setBounds(row2.removeFromLeft(48));
    m_speedSlider.setBounds(row2.removeFromLeft(220));
    row2.removeFromLeft(16);
    m_gainLabel.setBounds(row2.removeFromLeft(40));
    m_gainSlider.setBounds(row2.removeFromLeft(200));
    row2.removeFromLeft(16);
    m_seedLabel.setBounds(row2.removeFromLeft(40));
    m_seedEditor.setBounds(row2.removeFromLeft(105));
    row2.removeFromLeft(8);
    m_diceButton.setBounds(row2.removeFromLeft(65));
}
