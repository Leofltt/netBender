#pragma once
#include <JuceHeader.h>

class BridgePanel : public juce::Component {
public:
    BridgePanel();
    ~BridgePanel() override = default;

    auto resized() -> void override;

    juce::ToggleButton bridge_enable_toggle{"Enable Cross-Talk"};
    juce::Label bridge_label{{}, "Bridge Wire Source:"};
    juce::ComboBox bridge_combo;
    juce::Label bridge_depth_label{{}, "Cross-Talk Depth"};
    juce::Slider bridge_depth_slider;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BridgePanel)
};
