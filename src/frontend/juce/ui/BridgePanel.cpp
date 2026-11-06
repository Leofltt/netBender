#include "BridgePanel.h"
#include "UIConstants.h"

BridgePanel::BridgePanel() {
    using namespace UIConstants::Colors;
    using namespace UIConstants::Layout;

    bridge_enable_toggle.setToggleState(false, juce::dontSendNotification);
    bridge_enable_toggle.setColour(juce::ToggleButton::textColourId, copper);
    bridge_enable_toggle.setColour(juce::ToggleButton::tickColourId, copper);
    addAndMakeVisible(bridge_enable_toggle);

    bridge_label.setText("Bridge Wire Source:", juce::dontSendNotification);
    bridge_label.setJustificationType(juce::Justification::centredLeft);
    bridge_label.setColour(juce::Label::textColourId, copper);
    addAndMakeVisible(bridge_label);

    bridge_combo.setTextWhenNothingSelected("Select Target / Source...");
    addAndMakeVisible(bridge_combo);

    bridge_depth_label.setText("Cross-Talk Depth", juce::dontSendNotification);
    bridge_depth_label.setJustificationType(juce::Justification::centredLeft);
    bridge_depth_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(bridge_depth_label);

    bridge_depth_slider.setRange(0.0, 1.0, 0.01);
    bridge_depth_slider.setValue(0.0);
    bridge_depth_slider.setSliderStyle(juce::Slider::LinearBar);
    bridge_depth_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    bridge_depth_slider.setColour(juce::Slider::trackColourId, golden_ochre);
    addAndMakeVisible(bridge_depth_slider);
}

auto BridgePanel::resized() -> void {
    using namespace UIConstants::Layout;

    auto b = getLocalBounds().reduced(gap_standard);
    bridge_enable_toggle.setBounds(b.removeFromTop(row_height_standard));
    b.removeFromTop(gap_standard);
    bridge_label.setBounds(b.removeFromTop(label_height_tall));
    bridge_combo.setBounds(b.removeFromTop(row_height_standard));
    b.removeFromTop(gap_large);
    bridge_depth_label.setBounds(b.removeFromTop(label_height_tall));
    bridge_depth_slider.setBounds(b.removeFromTop(row_height_standard));
}
