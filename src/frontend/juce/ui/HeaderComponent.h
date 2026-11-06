#pragma once
#include <JuceHeader.h>

class HeaderComponent : public juce::Component {
public:
    HeaderComponent();
    ~HeaderComponent() override = default;

    auto resized() -> void override;

    juce::Label title_label;
    juce::TextButton load_button{"Load Model"};
    juce::Label status_label;

    juce::Label method_label{{}, "Mode:"};
    juce::ComboBox method_combo;

    juce::Label buffer_label{{}, "Buffer:"};
    juce::Label buffer_status_label;
    juce::TextButton gpu_button{"CPU"};

    juce::Label dry_wet_label{{}, "Dry/Wet:"};
    juce::Slider dry_wet_slider;

    juce::TextButton save_model_button{"Save Model"};
    juce::TextButton fuse_button{"FUSE: OK"};
    juce::TextButton short_circuit_button{"SHORT [!]"};
    juce::Label trigger_mode_label{{}, "Trigger:"};
    juce::ComboBox trigger_mode_combo;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HeaderComponent)
};
