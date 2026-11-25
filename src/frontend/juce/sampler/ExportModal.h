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
    [[nodiscard]] auto selectedSlots() const -> std::vector<int>;
    auto setAllPopulated(bool checked) -> void;
    auto refreshRangeAndState() -> void;

    const netbender::sampler::SamplerEngine& m_engine;
    int m_slotCount{4};
    std::function<void(std::vector<int>, bool, float, float, SamplerExportFormat)> m_onExport;

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
