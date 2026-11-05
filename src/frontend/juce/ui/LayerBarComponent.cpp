#include "LayerBarComponent.h"
#include "UIConstants.h"

LayerBarComponent::LayerBarComponent() {
    using namespace UIConstants::Colors;

    view_weights_button.setColour(juce::TextButton::buttonColourId, deep_violet);
    view_weights_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(view_weights_button);

    view_latent_button.setColour(juce::TextButton::buttonColourId, button_secondary_bg);
    view_latent_button.setColour(juce::TextButton::textColourOffId, juce::Colours::darkgrey);
    addAndMakeVisible(view_latent_button);

    display_curve_button.setColour(juce::TextButton::buttonColourId, deep_violet);
    display_curve_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(display_curve_button);

    display_matrix_button.setColour(juce::TextButton::buttonColourId, button_secondary_bg);
    display_matrix_button.setColour(juce::TextButton::textColourOffId, juce::Colours::darkgrey);
    addAndMakeVisible(display_matrix_button);

    category_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(category_label);

    category_combo.addItem("All Traces", 1);
    category_combo.addItem("Norm / Dynamics", 2);
    category_combo.addItem("Conv Kerns", 3);
    category_combo.addItem("Attention", 4);
    category_combo.addItem("Linear / Dense", 5);
    category_combo.addItem("Activation (Snake)", 6);
    category_combo.addItem("Recurrent (LSTM)", 7);
    category_combo.addItem("Biases", 8);
    category_combo.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(category_combo);

    layer_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(layer_label);
    addAndMakeVisible(layer_combo);

    reset_layer_button.setColour(juce::TextButton::buttonColourId, ruby);
    reset_layer_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(reset_layer_button);

    reset_all_button.setColour(juce::TextButton::buttonColourId, dark_ruby);
    reset_all_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(reset_all_button);
}

auto LayerBarComponent::resized() -> void {
    using namespace UIConstants::Layout;
    using namespace UIConstants::LayerBar;

    auto layerRow = getLocalBounds();

    view_weights_button.setBounds(layerRow.removeFromLeft(view_weights_width));
    layerRow.removeFromLeft(gap_small);
    view_latent_button.setBounds(layerRow.removeFromLeft(view_latent_width));
    layerRow.removeFromLeft(gap_medium);

    display_curve_button.setBounds(layerRow.removeFromLeft(curve_button_width));
    layerRow.removeFromLeft(gap_tiny);
    display_matrix_button.setBounds(layerRow.removeFromLeft(matrix_button_width));
    layerRow.removeFromLeft(gap_standard);

    category_label.setBounds(layerRow.removeFromLeft(category_label_width));
    category_combo.setBounds(layerRow.removeFromLeft(category_combo_width));
    layerRow.removeFromLeft(gap_medium);

    layer_label.setBounds(layerRow.removeFromLeft(layer_label_width));

    reset_all_button.setBounds(layerRow.removeFromRight(reset_all_width));
    layerRow.removeFromRight(gap_compact);
    reset_layer_button.setBounds(layerRow.removeFromRight(reset_layer_width));
    layerRow.removeFromRight(gap_medium);

    layer_combo.setBounds(layerRow);
}
