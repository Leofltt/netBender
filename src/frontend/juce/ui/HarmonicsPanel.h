#pragma once
#include <JuceHeader.h>

class HarmonicsPanel : public juce::Component {
public:
    HarmonicsPanel();
    ~HarmonicsPanel() override = default;

    auto resized() -> void override;

    juce::ToggleButton harmonic_enable_toggle{"Enable Harmonics"};
    juce::Label harmonic_freq_label{{}, "Frequency"};
    juce::Slider harmonic_freq_slider;
    juce::Label harmonic_partials_label{{}, "Partials"};
    juce::Slider harmonic_partials_slider;
    juce::Label harmonic_morph_label{{}, "Morph Phase"};
    juce::Slider harmonic_morph_slider;
    juce::Label harmonic_depth_label{{}, "Wave Gain"};
    juce::Slider harmonic_depth_slider;
    juce::Label harmonic_mode_label{{}, "Combine Mode:"};
    juce::ComboBox harmonic_mode_combo;
    juce::TextButton harmonic_apply_button{"Stamp Wave"};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HarmonicsPanel)
};
