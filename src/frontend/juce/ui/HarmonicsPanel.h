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

    // Compatibility references
    juce::ToggleButton& harmonicEnableToggle{harmonic_enable_toggle};
    juce::Label& harmonicFreqLabel{harmonic_freq_label};
    juce::Slider& harmonicFreqSlider{harmonic_freq_slider};
    juce::Label& harmonicPartialsLabel{harmonic_partials_label};
    juce::Slider& harmonicPartialsSlider{harmonic_partials_slider};
    juce::Label& harmonicMorphLabel{harmonic_morph_label};
    juce::Slider& harmonicMorphSlider{harmonic_morph_slider};
    juce::Label& harmonicDepthLabel{harmonic_depth_label};
    juce::Slider& harmonicDepthSlider{harmonic_depth_slider};
    juce::Label& harmonicModeLabel{harmonic_mode_label};
    juce::ComboBox& harmonicModeCombo{harmonic_mode_combo};
    juce::TextButton& harmonicApplyButton{harmonic_apply_button};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HarmonicsPanel)
};
