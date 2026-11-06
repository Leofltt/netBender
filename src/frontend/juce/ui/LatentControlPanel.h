#pragma once
#include <JuceHeader.h>

class LatentControlPanel : public juce::Component {
public:
    LatentControlPanel();
    ~LatentControlPanel() override = default;

    auto resized() -> void override;

    juce::ToggleButton latent_enable_button{"Enable Latent Hook"};
    juce::Label latent_depth_label{{}, "Latent Depth"};
    juce::Slider latent_depth_slider;
    juce::Label latent_mode_label{{}, "Orbit Mode:"};
    juce::ComboBox latent_mode_combo;
    juce::Label latent_slew_label{{}, "Slew Speed"};
    juce::Slider latent_slew_slider;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentControlPanel)
};
