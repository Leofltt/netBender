#include "ExportModal.h"
#include "../ui/UIConstants.h"

#include <algorithm>
#include <cctype>
#include <memory>

auto sanitize_export_filename(const std::string& raw) -> juce::String {
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

auto write_slot_audio_to_file(const netbender::sampler::SampleSlot& snap, juce::File dest,
                              bool apply_range, float start_norm, float end_norm,
                              SamplerExportFormat format, juce::String& out_error) -> bool {
    const bool is_aiff = (format == SamplerExportFormat::Aiff24);
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
        if (format == SamplerExportFormat::Wav16) {
            bits = 16;
        } else if (format == SamplerExportFormat::WavFloat32) {
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

ExportModalContent::ExportModalContent(
    const netbender::sampler::SamplerEngine& engine, int initial_slot, bool select_all_populated,
    bool initial_apply_range,
    std::function<void(std::vector<int>, bool, float, float, SamplerExportFormat)>
        on_export_requested)
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
            text += " (" + juce::String(snap->duration_sec, 2) + "s, #" + juce::String(snap->seed) +
                    ")";
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
        const auto fmt = static_cast<SamplerExportFormat>(m_formatCombo.getSelectedId());
        if (auto* dw = findParentComponentOfClass<juce::DialogWindow>()) {
            dw->exitModalState(0);
            dw->setVisible(false);
        }
        if (m_onExport) {
            m_onExport(std::move(selected), apply, s_norm, e_norm, fmt);
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

auto ExportModalContent::paint(juce::Graphics& g) -> void {
    g.fillAll(UIConstants::Colors::surface_panel);
}

auto ExportModalContent::resized() -> void {
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
        m_slotToggles[static_cast<size_t>(i)].setBounds(grid_area.getX() + c * col_w,
                                                        grid_area.getY() + r * 24, col_w - 6, 22);
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

auto ExportModalContent::selectedSlots() const -> std::vector<int> {
    std::vector<int> out;
    for (int i = 0; i < m_slotCount; ++i) {
        if (m_slotToggles[static_cast<size_t>(i)].getToggleState() &&
            m_slotToggles[static_cast<size_t>(i)].isEnabled()) {
            out.push_back(i);
        }
    }
    return out;
}

auto ExportModalContent::setAllPopulated(bool checked) -> void {
    for (int i = 0; i < m_slotCount; ++i) {
        auto& tb = m_slotToggles[static_cast<size_t>(i)];
        if (tb.isEnabled()) {
            tb.setToggleState(checked, juce::dontSendNotification);
        }
    }
    refreshRangeAndState();
}

auto ExportModalContent::refreshRangeAndState() -> void {
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
        m_startSlider.setValue(m_engine.get_loop_start(s_idx) * dur, juce::dontSendNotification);
        m_endSlider.setValue(m_engine.get_loop_end(s_idx) * dur, juce::dontSendNotification);
    }
}
