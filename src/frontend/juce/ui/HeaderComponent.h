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

    // Compatibility references
    juce::Label& titleLabel{title_label};
    juce::TextButton& loadButton{load_button};
    juce::Label& statusLabel{status_label};
    juce::Label& methodLabel{method_label};
    juce::ComboBox& methodCombo{method_combo};
    juce::Label& bufferLabel{buffer_label};
    juce::Label& bufferStatusLabel{buffer_status_label};
    juce::TextButton& gpuButton{gpu_button};
    juce::Label& dryWetLabel{dry_wet_label};
    juce::Slider& dryWetSlider{dry_wet_slider};
    juce::TextButton& saveModelButton{save_model_button};
    juce::TextButton& fuseButton{fuse_button};
    juce::TextButton& shortCircuitButton{short_circuit_button};
    juce::Label& triggerModeLabel{trigger_mode_label};
    juce::ComboBox& triggerModeCombo{trigger_mode_combo};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HeaderComponent)
};
