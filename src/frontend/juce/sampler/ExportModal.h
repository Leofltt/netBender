#pragma once

#include <JuceHeader.h>
#include "SamplerEngine.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

enum class SamplerExportFormat { Wav24 = 1, Wav16 = 2, WavFloat32 = 3, Aiff24 = 4 };

/** Sanitize a prompt string into a safe lowercase filename stem. */
auto sanitize_export_filename(const std::string& raw) -> juce::String;

/** Write a single `SampleSlot` (full or `[start_norm, end_norm]` slice) to WAV/AIFF on disk. */
auto write_slot_audio_to_file(const netbender::sampler::SampleSlot& snap, juce::File dest,
                              bool apply_range, float start_norm, float end_norm,
                              SamplerExportFormat format, juce::String& out_error) -> bool;

/** Modal dialog content allowing multi-slot selection, format, and range export. */
class ExportModalContent final : public juce::Component {
public:
    ExportModalContent(
        const netbender::sampler::SamplerEngine& engine, int initial_slot,
        bool select_all_populated, bool initial_apply_range,
        std::function<void(std::vector<int>, bool, float, float, SamplerExportFormat)>
            on_export_requested);

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    [[nodiscard]] auto selected_slots() const -> std::vector<int>;
    auto set_all_populated(bool checked) -> void;
    auto refresh_range_and_state() -> void;

    const netbender::sampler::SamplerEngine& m_engine;
    int m_slot_count{4};
    std::function<void(std::vector<int>, bool, float, float, SamplerExportFormat)> m_on_export;

    juce::Label m_slots_header_label{"", "SLOTS TO EXPORT"};
    juce::TextButton m_select_all_btn{"ALL"};
    juce::TextButton m_select_none_btn{"NONE"};
    std::array<juce::ToggleButton, netbender::sampler::SamplerEngine::k_max_slots> m_slot_toggles;
    juce::Label m_format_label{"", "FORMAT"};
    juce::ComboBox m_format_combo;
    juce::ToggleButton m_apply_range_toggle;
    juce::Label m_start_label{"", "START"};
    juce::Slider m_start_slider;
    juce::Label m_end_label{"", "END"};
    juce::Slider m_end_slider;
    juce::TextButton m_export_btn{"EXPORT..."};
    juce::TextButton m_cancel_btn{"CANCEL"};
};
