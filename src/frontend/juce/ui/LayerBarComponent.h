#pragma once
#include <JuceHeader.h>

class LayerBarComponent : public juce::Component {
public:
    LayerBarComponent();
    ~LayerBarComponent() override = default;

    auto resized() -> void override;

    juce::TextButton view_weights_button{"weights"};
    juce::TextButton view_latent_button{"latent"};

    juce::TextButton display_curve_button{"1D"};
    juce::TextButton display_matrix_button{"2D"};

    juce::Label category_label{{}, "Trace:"};
    juce::ComboBox category_combo;

    juce::Label layer_label{{}, "Layer:"};
    juce::ComboBox layer_combo;

    juce::TextButton reset_layer_button{"Reset Layer"};
    juce::TextButton reset_all_button{"Reset All Layers"};

    // Compatibility references
    juce::TextButton& viewWeightsButton{view_weights_button};
    juce::TextButton& viewLatentButton{view_latent_button};
    juce::TextButton& displayCurveButton{display_curve_button};
    juce::TextButton& displayMatrixButton{display_matrix_button};
    juce::Label& categoryLabel{category_label};
    juce::ComboBox& categoryCombo{category_combo};
    juce::Label& layerLabel{layer_label};
    juce::ComboBox& layerCombo{layer_combo};
    juce::TextButton& resetLayerButton{reset_layer_button};
    juce::TextButton& resetAllButton{reset_all_button};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LayerBarComponent)
};
