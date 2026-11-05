#pragma once
#include <JuceHeader.h>

class MutatePanel : public juce::Component {
public:
    MutatePanel();
    ~MutatePanel() override = default;

    auto resized() -> void override;

    juce::ToggleButton mutate_enable_toggle{"Enable Transform"};
    juce::Label scale_label{{}, "Scale"};
    juce::Slider scale_slider;
    juce::Label offset_label{{}, "Offset"};
    juce::Slider offset_slider;

    juce::ToggleButton drift_enable_toggle{"Enable Thermal Drift"};
    juce::Label heat_label{{}, "Heat (Jitter)"};
    juce::Slider heat_slider;
    juce::Label memory_label{{}, "Memory (Drag)"};
    juce::Slider memory_slider;
    juce::Label drift_mode_label{{}, "Drift Algorithm:"};
    juce::ComboBox drift_mode_combo;
    juce::ToggleButton freeze_button{"Freeze Drift"};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MutatePanel)
};
